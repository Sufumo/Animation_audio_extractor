# 动漫角色语音数据集提取 Pipeline 完整流程报告

> 报告日期：2026-07-17  
> 目标：为流程优化提供可对照的完整技术文档

---

## 1. 项目概述与整体架构

本 Pipeline 的目标是从动漫剧集中自动提取**单一目标角色**的干净语音片段，用于后续的语音模型训练（如 GPT-SoVITS、RVC 等）。

### 1.1 整体架构

采用**两阶段**设计：

| 阶段 | 职责 | 核心外部依赖 |
|------|------|-------------|
| **Stage 1** | 数据清洗：视频→音频、移除 OP/ED、移除背景音乐 | ffmpeg, Mel-Band-Roformer |
| **Stage 2** | 说话人提取：目标说话人增强、ASR 说话人分离、清洗、切分 | UniSE, 阿里云 DashScope (paraformer-v2 + Qwen) |

### 1.2 工程特性

- **配置驱动**：通过 `configs/default.yaml` 控制所有参数
- **断点续传**：`TaskState` 在每个 task 目录下生成 `task_state.yaml`，记录各步骤状态与输出路径，支持 `--resume`
- **组件化**：每个 stage 内的子步骤可通过 `components` 列表独立启用/禁用
- **时间轴映射**：Stage 2 的核心难点是多轮音频处理后的**时间戳回传**（原始音频 → TSE → 静音移除 → ASR 拼接 → 回传切分）

---

## 2. Stage 1: 数据清洗流程

输入：`source_dir` 中的剧集文件（MP4/MP3/WAV）  
输出：`output/stage1/03_cleaned/` 中的 `*_cleaned.wav`

### 2.1 步骤一：源文件转 WAV (`mp4_to_wav`)

**文件**：`src/pipeline/components/mp4_to_wav.py`

- 遍历 `source_dir`，将所有媒体文件统一转为 WAV (PCM)
- 关键参数（来自 `configs/default.yaml`）：
  - `sample_rate`: 44100（默认保留原音质）
  - `mono`: false（默认立体声，为后续保留空间信息）
  - `bit_depth`: 16
- **优化点**：若输入已是同参数 WAV，ffmpeg 会直接 `copy` 音频流，避免重编码
- 输出目录：`output/stage1/01_wav/`

### 2.2 步骤二：OP/ED 移除 (`oped_removal`)

**文件**：`src/pipeline/components/oped_removal.py`

- 输入：`oped_dir` 中的 OP/ED 音频文件
- 原理：通过音频指纹/相似度匹配，在剧集中定位并切除 OP/ED 时间段
- 输出：`output/stage1/02_no_oped/*_no_oped.wav`
- 记录：`removed_segments` 会写入 task_state，记录被切除的时间段

### 2.3 步骤三：背景音乐移除 (`bgm_removal`)

**文件**：`src/pipeline/components/bgm_removal.py`

- **核心模型**：Mel-Band-Roformer（人声分离模型）
- 输入：`02_no_oped/` 中的音频
- 处理：
  1. 长音频按 `bgm_segment_seconds`（默认 360s）切分为 chunk
  2. 每个 chunk 送入 Mel-Band-Roformer 推理
  3. 提取 vocals（人声）轨道
  4. 将所有 vocals chunk 拼接回完整音频
- 输出：`output/stage1/03_cleaned/*_cleaned.wav`
- 该输出是**去除 BGM 后的原剧人声轨**，仍包含多个角色的声音

---

## 3. Stage 2: 目标说话人提取流程（核心）

输入：Stage 1 的 `*_cleaned.wav` + `reference_dir` 中的目标角色参考音频  
输出：`output/stage2/episode_000/clips/` 中的目标角色片段 + `*_merged_output.wav`

### 3.1 步骤一：UniSE TSE v1 (`unise_tse_v1`)

**文件**：`src/pipeline/components/unise_tse.py`

- **作用**：目标说话人提取（Target Speaker Extraction）
- 输入：
  - `mix`：Stage 1 输出的 cleaned.wav（含多角色人声）
  - `enroll`：`reference_dir` 中的参考音频（目标角色的干净样本）
- 处理：
  1. 将输入音频转为 16kHz 单声道
  2. 长音频按 `segment_seconds`（默认 360s）切分为 chunk
  3. 每个 chunk 与参考音频一起送入 UniSE 模型推理
  4. 合并所有 chunk 的输出
- **关键特性**：**UniSE 输出与输入长度完全一致**（时间轴一一对应）。非目标说话人被抑制为接近静音，但时间轴上没有压缩或拉伸。
- 输出：`episode_000/ep000_tse_v1.wav`
- **兼容性处理**：代码包含 PyTorch 2.6+ 的 `weights_only=False` 自动补丁（`test_patched_*.py`），避免 `torch.load` 报错

### 3.2 步骤二：静音移除 (`silence_removal`)

**文件**：`src/pipeline/components/silence_removal.py`

- **作用**：切除 TSE 输出中的静音段，压缩音频长度
- 处理：
  1. 用 `librosa.load` 以 16kHz 单声道加载音频
  2. `librosa.effects.split(wav, top_db=40, ref=np.max)` 检测非静音区间
  3. 合并间隔 < `min_silence_sec`（默认 0.3s）的区间，避免过度碎片化
  4. 将所有非静音区间切出并拼接为新的音频文件
- 输出：
  - `episode_000/ep000_silence_removed.wav`
  - `episode_000/ep000_silence_map.json` ← **关键映射文件**
- **silence_map 结构**：
  ```json
  {
    "sample_rate": 16000,
    "original_duration_sec": 1792.0,
    "output_duration_sec": 685.5,
    "segments": [
      {
        "index": 0,
        "original_start_sec": 2.5,
        "original_end_sec": 18.3,
        "output_start_sec": 0.0,
        "output_end_sec": 15.8,
        "duration_sec": 15.8
      }
    ]
  }
  ```

### 3.3 步骤三：构建 ASR 输入 (`build_asr_input`) —— 新流程核心

**文件**：`src/pipeline/components/audio_mapping.py`

这是用户最近要求优化的核心步骤，用于**避免将全量音频送入阿里云 ASR**。

#### 3.3.1 设计动机

- 全量音频 ASR 的问题：
  - 费用高（按音频时长计费）
  - 剧集中大量时间段是配角台词、背景音、无对白场景，干扰 diarization 准确度
- TSE 预过滤的优势：
  - UniSE 已将非目标说话人压制为静音
  - 通过检测 TSE 输出中的非静音区，即可知道**目标角色在哪些时间段说话**
  - 由于 TSE 输出与原始音频长度一致，这些时间戳可直接用于原始音频切分

#### 3.3.2 处理流程

1. **检测目标活跃段**：
   - 加载 `tse_v1.wav`（16kHz 单声道）
   - `librosa.effects.split(wav, top_db=40)` 获取非静音区间
   - 合并间隔 < `min_silence_sec`（0.3s）的区间
   - 实测：24 分钟剧集检测出约 243 个非静音区间

2. **从原始音频提取对应段落**：
   - 由于 TSE 输出与原始 cleaned.wav 时间轴一致，直接用上述时间戳从原始音频切分
   - 每个段落通过 ffmpeg 转为**单声道**（`-ac 1`）—— **阿里云 diarization 要求单声道**

3. **拼接 ASR 输入**：
   - 用 `merge_audio()` 将段落以 0.5s 静音间隙拼接为单个文件
   - 间隙作用：让 ASR 感知段落边界，避免将所有内容识别为一句话

4. **生成映射表**：
   - `asr_map.json` 记录每个段落在原始音频和 ASR 输入中的起止时间

#### 3.3.3 asr_map 结构

```json
{
  "gap_sec": 0.5,
  "sample_rate": 44100,
  "channels": 1,
  "total_duration_sec": 685.64,
  "segments": [
    {
      "index": 0,
      "original_start_sec": 2.698,
      "original_end_sec": 96.123,
      "asr_start_sec": 0.0,
      "asr_end_sec": 93.425
    }
  ]
}
```

**实测效果**：ASR 输入从原始 1792s 缩减至 ~686s，节省约 **62%** 的 ASR 费用和处理时间。

### 3.4 步骤四：阿里云 ASR 说话人分离 (`aliyun_asr`)

**文件**：`src/pipeline/components/aliyun_asr.py`, `src/aliyun/dashscope_client.py`

- **模型**：`paraformer-v2`（ DashScope 中**唯一支持说话人分离**的模型）
- 调用流程：
  1. `upload_local_to_oss()`：将本地音频上传至阿里云临时 OSS，获取 `oss://` 链接
  2. `create_filetrans_task()`：创建异步转写任务
     - 关键参数：`diarization_enabled=true`, `speaker_count=2`
     - 关键结构：`input.file_urls` 必须为**数组**（`paraformer-v2` 要求）
  3. `wait_filetrans_task()`：轮询任务状态（默认 5s 间隔，1h 超时）
  4. `fetch_transcription_result()`：从 `transcription_url` 下载 JSON 结果
- 输出：`episode_000/ep000_diarization.srt`
- **SRT 格式**：每段文本前带 `[0]` 或 `[1]` 说话人标签
- 实测结果：成功分离出 2 位说话人，不再是 `unknown`

### 3.5 步骤五：说话人声纹校验 (`speaker_verify`)

**文件**：`src/pipeline/components/speaker_verify.py`

- **目的**：纠正 ASR diarization 的 speaker_id 与目标角色不一致的问题（例如时长最长的 speaker[0] 实际是 ED/脏数据）。
- **方法**：
  1. 从 `diarization.srt` 为每个说话人抽取若干段（默认每说话人 8 段，时长 1.5–12s）
  2. 用 `pyannote/wespeaker-voxceleb-resnet34-LM` 提取 embedding
  3. 与 `reference_dir` 参考音频做 cosine 相似度（质心优先）
  4. 相似度最高的 speaker_id 作为后续清洗的 `main_speaker`
- 输出：`episode_000/ep000_speaker_scores.json`
- 失败时回退到「总时长最长」启发式

### 3.6 步骤六：SRT 清洗 (`srt_cleaning`)

**文件**：`src/pipeline/components/srt_cleaning.py`

- **目的**：从 ASR 结果中删除非目标角色的台词和无意义拟声词
- 处理流程：
  1. **确定主说话人**：优先使用 `speaker_verify` 的结果；否则计算每个 `speaker_id` 的总时长，最长者即为 `main_speaker`
  2. **LLM 清洗**（默认启用 `use_llm=True`）：
     - 调用 `qwen3.6-max`
     - System Prompt 要求：
       - 只保留主说话人的字幕
       - 删除无意义 utterance（如 "啊啊啊", "嗯嗯", "哦"）
       - **不得修改保留字幕的文本内容**
       - 保持 SRT 格式，重新编号
  3. **本地兜底**：若 LLM 调用失败，使用本地规则过滤
- 输出：`episode_000/ep000_cleaned.srt`
- **注意**：清洗后的 SRT **不含说话人标签**，仅剩文本和时间戳

### 3.7 步骤七：重建音频片段 (`rebuild_clips`)

**文件**：`src/pipeline/components/audio_mapping.py`

这是时间轴回传的核心步骤。

#### 3.7.1 时间戳映射流程

1. 解析 `cleaned.srt`，得到时间戳列表（基于 ASR 输入时间轴）
2. **映射回原始时间轴**：`map_asr_to_original(timestamps, asr_map)`
   - 将每个 ASR 字幕区间与 `asr_map.segments` 中的所有真实音频段求交
   - 一个字幕若跨过多个拼接段，会拆成多个互不连续的原始音频区间
   - 字幕落在人工静音间隙中的部分直接丢弃
3. **切分音频**：`cut_by_timestamps()`
   - 输入：原始 cleaned.wav + 映射后的时间戳
   - 每段前后加 `clip_padding_sec`（默认 0.15s）padding
   - 过滤时长 < `min_clip_duration_sec`（默认 0.1s）的片段
   - 输出命名：`ep000_target_clip_0000_{start}_{end}.wav`
4. 输出目录：`episode_000/clips/`

#### 3.7.2 关于 silence_map 的回退逻辑

代码中保留了旧逻辑的回退：
- 优先使用 `asr_map`（新流程）
- 若 `asr_map` 不可用，回退到 `silence_map`（旧流程）
- 若两者都不可用，直接按原时间戳切分

### 3.8 步骤八：可选 UniSE TSE v2 (`run_unise_v2`)

- 若 `run_unise_v2: true`，会对每个切分好的片段再次运行 UniSE TSE
- 用于进一步提纯目标角色声音
- 输出目录：`episode_000/clips_v2/`

### 3.9 步骤九：合并输出 (`merge_output`)

**文件**：`src/audio_utils.py` → `merge_with_gaps()`

- 将所有 clips 以 `merge_gap_sec`（默认 0.5s）的静音间隔拼接为单个 WAV
- 自动检测首段音频的采样率和声道数，生成匹配格式的静音间隙
- 使用 ffmpeg concat demuxer + stream copy（无损且快速）
- 输出：`episode_000/ep000_merged_output.wav`

---

## 4. 关键时间轴映射机制详解

Stage 2 涉及**三条时间轴**的转换，是流程中最容易出错的环节：

### 4.1 三条时间轴定义

| 时间轴 | 代表音频 | 说明 |
|--------|---------|------|
| **原始时间轴** | `cleaned.wav` (Stage 1 输出) | 与剧集原始时间完全一致，长度约 1792s |
| **TSE/静音时间轴** | `tse_v1.wav` / `silence_removed.wav` | TSE 输出长度与原始一致；静音移除后长度被压缩 |
| **ASR 时间轴** | `asr_input.wav` | 由原始音频的多个非连续段落拼接而成，含 0.5s 间隙，长度约 686s |

### 4.2 映射关系图

```
原始 cleaned.wav (1792s)
|── [2.5s - 18.3s] 目标角色说话 ──┐
|── [45.2s - 67.1s] 目标角色说话 ├→ build_asr_input → asr_input.wav
|── ...                         │
|                               │
▼                               ▼
tse_v1.wav (1792s)         ASR 时间轴 (686s)
| 非目标被静音                  |── [0s - 15.8s]  (对应原始 2.5-18.3)
|                               |── [16.3s - 38.2s] (对应原始 45.2-67.1, +0.5s gap)
▼                               ▼
非静音检测                       ASR 识别
    │                            │
    └──────────────┬─────────────┘
                   │
                   ▼
            map_asr_to_original()
                   │
                   ▼
            切分原始 cleaned.wav
```

### 4.3 为什么 TSE 输出能直接对应原始时间戳？

UniSE TSE 模型基于时域掩码（time-domain masking），其输出与输入**采样点级别对齐**。也就是说：
- `tse_v1.wav` 的第 N 个采样点对应 `cleaned.wav` 的第 N 个采样点
- 即使某个时间段被静音，该时间段在 TSE 输出中也是静音（采样点为 0 或接近 0），但**时间位置不变**
- 因此 `librosa.effects.split` 在 TSE 输出上检测到的时间戳，可以直接用于原始音频切分

### 4.4 ASR 时间轴回传算法

`map_asr_to_original([start, end], asr_map)` 的逻辑：

1. **字幕与 ASR 真实音频段相交**：逐段计算交集，并分别线性映射  
   `original_t = seg.original_start + (overlap_t - seg.asr_start)`

2. **字幕跨越一个或多个人工静音间隙**：拆成多个原始区间  
   不能只映射字幕的起止点后在原始音频上连续切割，否则会把不属于 TSE
   非静音段的中间对白一并带入最终输出

3. **字幕仅落在人工静音间隙中**：不生成音频区间

---

## 5. 配置文件参数与代码默认值对照

| 参数 | 配置路径 | 代码默认值 | 说明 |
|------|---------|-----------|------|
| `bgm_segment_seconds` | `stage1.bgm_segment_seconds` | 360.0 | Mel-Band-Roformer 分块长度 |
| `sample_rate` | `stage1.sample_rate` | 44100 | Stage 1 输出采样率 |
| `mono` | `stage1.mono` | false | Stage 1 是否输出单声道 |
| `segment_seconds` | `stage2.segment_seconds` | 360.0 | UniSE/ASR 分块长度 |
| `asr_speaker_count` | `stage2.asr_speaker_count` | 2 | ASR 说话人数提示 |
| `srt_model` | `stage2.srt_model` | qwen3.6-max | Qwen 清洗模型 |
| `run_unise_v2` | `stage2.run_unise_v2` | false | 是否对 clips 二次 TSE |
| `asr_gap_sec` | `stage2.asr_gap_sec` | 2.0 | ASR 输入中段间静音间隙 |
| `clip_padding_sec` | `stage2.clip_padding_sec` | 0.15 | 切分片段前后 padding |
| `min_clip_duration_sec` | `stage2.min_clip_duration_sec` | 0.1 | 最小片段时长，低于则丢弃 |
| `merge_gap_sec` | `stage2.merge_gap_sec` | 2.0 | 最终合并输出时段间间隙 |
| `speaker_embedding_model` | `stage2.speaker_embedding_model` | pyannote/wespeaker-voxceleb-resnet34-LM | 说话人校验 embedding 模型 |
| `speaker_verify_samples` | `stage2.speaker_verify_samples` | 8 | 每说话人抽样段数 |
| `top_db` (静音检测) | 硬编码 | 40 | librosa 静音检测阈值 |
| `min_silence_sec` (静音检测) | 硬编码 | 0.3 | 低于此时长的静音视为同一句话 |

---

## 6. 当前已知问题与优化建议

### 6.1 ASR 段落过长（核心优化点）

**现象**：`v2` 流程的 ASR 输出中，单个段落可达 50 秒以上（如 00:00:54 → 00:01:47）。对比直接跑全量音频的 `paraformer` 版本，同样内容被切分为 4 段。

**根因分析**：
- ASR 输入是由 243 个原始段落用 0.5s 间隙拼接而成的
- 0.5s 的静音对 paraformer-v2 而言**过短**，模型将其视为句内停顿而非段落边界
- 当多个原始段落的内容在语义上连续时（如长对话），ASR 倾向于合并为超长段落

**影响**：
- 对语音训练不利：过长段落包含多句话，句间韵律变化大
- Qwen 清洗时，超长文本对 LLM 的上下文理解也带来压力

**优化建议**：
1. **增大 `asr_gap_sec`**：从 0.5s 调至 1.0s ~ 1.5s，给 ASR 更明确的段落边界提示
2. **ASR 后处理切分**：基于强制对齐（forced alignment）或标点符号，将 > 10s 的段落进一步切分
3. **调整 `top_db` 阈值**：若 `top_db=40` 过滤掉了 TSE 输出中目标角色的轻声段，可能导致有效段落被错误合并；可尝试 `top_db=35` 或 `top_db=45` 对比效果

### 6.2 片段数量偏少

**现象**：`v2` 产出 25 段，而直接 ASR 的 `paraformer` 版本产出 37 段。

**可能原因**：
- ASR 合并导致段落数量减少
- Qwen 清洗时过滤了更多内容
- 部分 `speaker[1]` 的内容被错误归类为 `speaker[0]` 并被保留

**优化建议**：
- 在 `srt_cleaning.py` 中增加**置信度阈值**：若 ASR 返回的某段说话人置信度低，直接丢弃
- 增加**角色声纹二次校验**：用参考音频的声纹特征与每个 clip 做相似度对比，过滤相似度低的片段

### 6.3 ED/歌曲被识别为 speaker[1]

**现象**：片尾曲（ED）部分被 ASR 正确识别为 `speaker[1]`，随后被 Qwen 过滤掉。但如果剧中出现合唱、广播、旁白等，也可能被归类为非主说话人。

**优化建议**：
- 在 Stage 1 增加**ED/ED 精准切除**：当前 OP/ED 移除依赖音频指纹匹配，若剧集有多个版本 ED 或特别篇，匹配可能失效
- 在 Stage 2 增加**声纹相似度打分**：不仅依赖 ASR 的 speaker_id，还计算每个片段与参考音频的声纹相似度，设置阈值过滤

### 6.4 硬编码参数缺乏可调性

**现象**：`top_db=40` 和 `min_silence_sec=0.3` 在 `build_asr_input()` 和 `remove_silence()` 中是硬编码的。

**优化建议**：
- 将这两个参数暴露到 `configs/default.yaml` 中
- 不同动漫的录音电平、混响、BGM 残留程度不同，统一阈值难以适配所有情况

### 6.5 对低质量 TSE 输出的鲁棒性

**现象**：若 UniSE 参考音频与剧中目标角色差异较大（如角色变声、喊叫、耳语），TSE 可能无法有效抑制其他说话人。

**优化建议**：
- 在 `build_asr_input()` 中增加**音量门限**：不仅检测非静音，还检测音量是否高于某阈值，避免 TSE 残留的低音量干扰音被送入 ASR
- 提供多段参考音频（当前只取 `reference_dir` 中的第一个文件），对参考音频做拼接或平均处理

### 6.6 UniSE v2 的边际收益

**现状**：`run_unise_v2` 默认关闭。

**建议**：
- 若 Stage 1 的 BGM 移除已经很干净，且参考音频质量高，v1 的 TSE 通常已足够
- v2 是对片段再次处理，可能对片段边缘的截断造成 artifacts，建议根据具体效果 A/B 测试后决定是否开启

---

## 7. 文件目录结构速查

一次完整运行后的 `task_dir` 结构：

```
task_dir/
├── task_state.yaml              # 断点状态
├── output/
│   ├── stage1/
│   │   ├── 01_wav/              # 转码后音频
│   │   ├── 02_no_oped/          # 去除 OP/ED
│   │   └── 03_cleaned/          # 去除 BGM (Mel-Band-Roformer)
│   └── stage2/
│       └── episode_000/
│           ├── ep000_tse_v1.wav
│           ├── ep000_silence_removed.wav
│           ├── ep000_silence_map.json
│           ├── ep000_asr_input.wav      # 单声道，拼接后
│           ├── ep000_asr_input.asr_map.json
│           ├── ep000_diarization.srt    # ASR 原始输出，含 [0]/[1]
│           ├── ep000_cleaned.srt        # Qwen 清洗后
│           ├── clips/                   # 最终目标角色片段
│           │   ├── ep000_target_clip_0000_*.wav
│           │   └── ...
│           └── ep000_merged_output.wav  # 合并输出
```

---

## 8. 外部依赖与版本要求

| 依赖 | 用途 | 版本注意事项 |
|------|------|-------------|
| ffmpeg | 所有音频切分/合并/转码 | 需支持 anullsrc, concat demuxer |
| librosa | 静音检测 | 需配合 numpy |
| PyTorch + PyTorch Lightning | UniSE 推理 | >= 2.6 需要 weights_only 补丁 |
| Mel-Band-Roformer | BGM 移除 | 需独立项目目录 |
| QuarkAudio-UniSE | 目标说话人提取 | 需独立项目目录 + checkpoint |
| DashScope API | ASR + Qwen | `DASHSCOPE_API_KEY` 环境变量 |
| OpenAI SDK | Qwen 调用 | 用于兼容 DashScope 的 OpenAI 接口 |

---

*报告完。如需针对某个具体步骤进行深度调整（如 ASR 后处理切分、声纹相似度过滤、参考音频多段合并等），可继续提出。*
