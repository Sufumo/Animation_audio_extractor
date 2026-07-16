# Anime Voice Training Pipeline — 执行计划表

> 本项目目标：将动画 MP4/MP3 转化为高质量目标角色语音训练数据集。
> 项目路径：`/Users/AITraining/Documents/Personal/train_audio_extract/plugins/anime-voice-training-pipeline`
> 任务管理方式：每个任务拥有独立任务文件夹，内部 `task_state.yaml` 记录各步骤完成状态，支持断点续跑。

---

## 项目总览

| 阶段 | 名称 | 说明 |
|------|------|------|
| Stage 1 | 数据清洗 | MP4→WAV、去除 OP/ED、去除背景音乐，得到高质量角色语音混合音频 |
| Stage 2 | 说话人提取 | UniSE TSE 初提取 → 静音去除 → 阿里云 ASR 说话人分离 → Qwen 清洗 SRT → 二次映射回原始音频 → 可选二次 UniSE |

**输入类型**：
1. `source/`：动画源文件（多集 MP4 或 MP3）
2. `oped/`：动画 OP/ED 等需要识别并去除的音频
3. `reference/`：目标角色参考音频（用于 UniSE TSE）

**输出**：任务文件夹下的 `output/stage2_clips/` 多段 WAV 音频。

---

## 计划项与状态

### 1. 项目骨架与配置
- [x] 创建项目目录结构 (`src/`, `scripts/`, `configs/`, `tests/`)
- [ ] 编写 `README.md`（项目说明、依赖、用法）
- [ ] 编写 `requirements.txt`
- [ ] 编写默认 `configs/default.yaml`
- [ ] 初始化 Git 仓库并准备 `.gitignore`

**产出文件**：
- `README.md`
- `requirements.txt`
- `configs/default.yaml`
- `.gitignore`

---

### 2. 任务状态管理模块
- [ ] 实现 `src/task_state.py`
  - 每个任务文件夹下维护 `task_state.yaml`
  - 记录：任务 ID、输入路径、各 stage 完成状态、输出路径、时间戳、错误信息
  - 提供 `mark_done(step)` / `is_done(step)` / `get_outputs(step)` / `reset(step)` 接口
  - 支持从现有任务文件夹恢复状态

**产出文件**：
- `src/task_state.py`
- `tests/test_task_state.py`

---

### 3. 高质量音频工具模块
- [ ] 实现 `src/audio_utils.py`
  - `convert_to_wav(input_path, output_path, sample_rate=None, mono=False, bit_depth=16)`：优先 copy 音频流，必须重编码时使用 PCM
  - `split_audio(input_path, output_dir, segment_seconds, suffix_fmt="%03d")`：使用 ffmpeg segment，高质量 PCM 输出
  - `merge_audio(segment_list, output_path, concat_with_copy=True)`：使用 ffmpeg concat demuxer + copy
  - `resample(input_path, output_path, target_sr, bit_depth=16)`：使用 soxr 重采样
  - `get_duration(input_path)` / `get_media_info(input_path)`
  - 所有切分/合并操作默认最高质量，避免额外有损编码

**产出文件**：
- `src/audio_utils.py`
- `tests/test_audio_utils.py`

---

### 4. Stage 1：数据清洗
- [ ] 实现 `src/pipeline/components/mp4_to_wav.py`
  - 支持 MP4 / MP3 / WAV 输入
  - 输出统一为 WAV（PCM s16le，保留原采样率或指定 44100/16000）
  - 多集输入分别处理
- [ ] 实现 `src/pipeline/components/oped_removal.py`
  - 使用音频指纹（如 `pyacoustid` / `audfprint` / `chromaprint`）识别每集开头/结尾的 OP/ED
  - 支持从 `oped/` 文件夹中所有音频构建指纹库
  - 识别成功后，切除对应时间段，记录切除位置
  - 若识别失败，可回退到用户配置的时间段硬切
- [ ] 实现 `src/pipeline/components/bgm_removal.py`
  - 封装 Mel-Band-Roformer Vocal Model 调用
  - 输入长音频时自动切分（参考 UniSE_Colab 360s 一段），分别去背景音，再合并
  - 输出 `*_vocals.wav`（保留人声）和 `*_instrumental.wav`（背景音乐）
- [ ] 实现 `src/pipeline/stage1_data_cleaning.py`
  - 编排以上组件
  - 每集输出：`stage1_cleaned/{episode}_cleaned.wav`
  - 在 `task_state.yaml` 中记录进度

**产出文件**：
- `src/pipeline/components/mp4_to_wav.py`
- `src/pipeline/components/oped_removal.py`
- `src/pipeline/components/bgm_removal.py`
- `src/pipeline/stage1_data_cleaning.py`
- `tests/test_stage1.py`

---

### 5. Stage 2：说话人提取
- [ ] 实现 `src/pipeline/components/unise_tse.py`
  - 封装 QuarkAudio-UniSE 的 `test.py`
  - 输入音频按 360s 切分（16000Hz mono）
  - 准备 `mix/`、`enroll/`、`tgt/` 目录
  - 调用 UniSE 生成 `output/` 下各切片的目标说话人音频
  - 合并切片为完整音频
  - 支持 `mode=tse` 目标说话人提取
- [ ] 实现 `src/pipeline/components/silence_removal.py`
  - 使用 `librosa` / `webrtcvad` / `ffmpeg silencedetect` 检测并去除静音段
  - 记录每段非静音音频在原音频中的 `[start, end)` 位置映射
  - 输出 `stage2_silence_removed.wav` + `silence_map.json`
- [ ] 实现 `src/pipeline/components/audio_mapping.py`
  - 提供 `forward_map(audio_path, cuts)`：按切割点切分并记录原位置
  - 提供 `reverse_map(cut_files, map_json)`：将处理后的片段按原位置拼回
  - 保证时间轴精确，便于后续 SRT 时间映射
- [ ] 实现 `src/pipeline/components/aliyun_asr.py`
  - 调用阿里云录音文件识别（`qwen3-asr-flash-filetrans`）
  - 本地文件上传到临时 OSS（参考 `qwen-ai-mcp/src/qwen_ai/tools/audio_transcribe.py`）
  - 启用说话人分离（speaker diarization），输出 SRT / JSON
  - 单集单段处理，降低识别错说话人概率
- [ ] 实现 `src/pipeline/components/srt_cleaning.py`
  - 调用 Qwen 3.6-max（`qwen3.6-max`）模型
  - Prompt 要求：删除意义不明（如“啊啊啊”）的句子；只保留主 speaker，删除其他 speaker
  - 输入 SRT，输出清洗后的 SRT
- [ ] 实现 `src/pipeline/stage2_speaker_extraction.py`
  - 编排流程：
    1. UniSE TSE 初提取 → `stage2_tse_v1.wav`
    2. 静音去除 + 位置映射 → `stage2_silence_removed.wav` + `map.json`
    3. 阿里云 ASR 说话人分离 → `stage2_diarization.srt`
    4. Qwen 清洗 SRT → `stage2_cleaned.srt`
    5. 根据清洗后 SRT 时间范围，反向映射回原始（Stage 1 输出）音频，切出目标片段
    6. （可选，默认关闭）对切出片段再次做 UniSE TSE → `stage2_tse_v2/`
  - 最终输出：`output/stage2_clips/*.wav`

**产出文件**：
- `src/pipeline/components/unise_tse.py`
- `src/pipeline/components/silence_removal.py`
- `src/pipeline/components/audio_mapping.py`
- `src/pipeline/components/aliyun_asr.py`
- `src/pipeline/components/srt_cleaning.py`
- `src/pipeline/stage2_speaker_extraction.py`
- `tests/test_stage2.py`

---

### 6. 阿里云客户端封装
- [ ] 实现 `src/aliyun/dashscope_client.py`
  - 读取 `DASHSCOPE_API_KEY`
  - 提供 ASR 文件识别调用（异步创建 + 轮询）
  - 提供 Qwen Chat 调用（SRT 清洗）
- [ ] 实现 `src/aliyun/oss_uploader.py`
  - 本地文件上传到 DashScope 临时 OSS，返回 `oss://` URL
  - 参考 `qwen-ai-mcp/src/qwen_ai/tools/audio_transcribe.py` 的 `_upload_local_to_oss`

**产出文件**：
- `src/aliyun/dashscope_client.py`
- `src/aliyun/oss_uploader.py`

---

### 7. 主脚本 `main.py`
- [ ] 实现命令行入口
  - `--task-dir`：指定任务文件夹（若不存在则创建）
  - `--config`：配置文件路径
  - `--source` / `--oped` / `--reference`：输入路径
  - `--stage {1,2,all}`：选择运行阶段
  - `--resume`：断点续跑
  - `--components`：可选启用/禁用特定组件
- [ ] 组件在配置中以列表形式存在，可拆卸
  - 例如：`stage1: [mp4_to_wav, oped_removal, bgm_removal]`
- [ ] 本地测试模式
  - 默认读取 `configs/test_local.yaml`
  - 使用 `/Users/AITraining/Documents/Personal/train_audio_extract/data/test.mp4`
  - 只跑少量数据或缩短音频，避免全量长时间运行
  - 对 UniSE 和 BGM 去除步骤使用音频切分，防止爆显存

**产出文件**：
- `main.py`
- `configs/test_local.yaml`

---

### 8. 分脚本 `scripts/convert_anime_to_dataset.py`
- [ ] 实现独立入口脚本，用于“动画 MP4 到高质量语音训练集”的转换
  - 封装 Stage 1 + Stage 2 的简化调用
  - 支持直接传入 `source_dir`、`oped_dir`、`reference_dir`、`output_dir`
  - 可脱离 `main.py` 单独使用

**产出文件**：
- `scripts/convert_anime_to_dataset.py`

---

### 9. IPython Notebook
- [ ] 实现 `anime_voice_training.ipynb`
  - 从 GitHub 仓库 clone 项目
  - 安装依赖
  - 下载 UniSE 与 BiCodec 等模型（参考 UniSE_Colab.ipynb）
  - 上传/挂载数据
  - 配置并运行 `main.py`
  - 下载/保存结果

**产出文件**：
- `anime_voice_training.ipynb`

---

### 10. 本地冒烟测试
- [ ] 使用 `test.mp4` 运行主脚本
  - 只启用快速组件，或限制音频时长
  - 验证 `task_state.yaml` 正确记录
  - 验证输出目录结构正确
  - 验证音频无额外质量损失（采样率、编码格式）
- [ ] 修复测试中发现的问题
- [ ] 完善 `README.md` 与使用示例

**产出文件**：
- 测试日志
- 更新后的文档

---

## 关键技术决策

| 决策点 | 方案 | 原因 |
|--------|------|------|
| 音频切分 | ffmpeg segment / concat demuxer + copy | 避免重编码带来的额外损失 |
| 背景音去除 | Mel-Band-Roformer Vocal Model | 本地可用，已在 Mac 上验证 |
| 说话人提取 | QuarkAudio-UniSE TSE | 用户已有项目，支持参考音频 |
| 长音频处理 | 切分为 360s 片段分别处理 | 防止 UniSE / Roformer 爆显存 |
| ASR + 说话人分离 | 阿里云 `qwen3-asr-flash-filetrans` | 支持长音频、说话人分离、临时 OSS |
| SRT 清洗 | Qwen 3.6-max | 大模型可理解语义并过滤无意义片段 |
| 断点续跑 | 任务文件夹内 `task_state.yaml` | 简单可靠，便于人工检查 |

---

## 风险与待确认事项

1. **OP/ED 识别准确率**：音频指纹对同动画不同集 OP/ED 位置识别可能不够鲁棒，需要准备回退方案。
2. **UniSE 在本地运行速度**：本地 MPS/CPU 运行较慢，Notebook 中建议 GPU 环境。
3. **阿里云 API 费用**：长音频 ASR 和多次 Qwen 调用可能产生费用，需在文档中提示。
4. **音频质量要求**：除背景音乐消除外，其余步骤必须避免重采样/重编码；已规定使用 copy 或 PCM。
5. **二次 UniSE 默认关闭**：根据用户要求，最后一步可选，默认不执行。

---

## 完成定义

- [ ] `main.py` 可运行本地测试并产出预期输出
- [ ] `scripts/convert_anime_to_dataset.py` 可独立运行
- [ ] `anime_voice_training.ipynb` 可在 Colab 环境中运行
- [ ] 所有组件在主配置中可拆卸
- [ ] 每个任务使用 `task_state.yaml` 记录状态，支持断点续跑
- [ ] 项目已推送至 GitHub 仓库
