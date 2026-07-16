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
- [x] 编写 `README.md`（项目说明、依赖、用法）
- [x] 编写 `requirements.txt`
- [x] 编写默认 `configs/default.yaml`
- [x] 初始化 Git 仓库并准备 `.gitignore`

**产出文件**：
- `README.md`
- `requirements.txt`
- `configs/default.yaml`
- `.gitignore`

---

### 2. 任务状态管理模块
- [x] 实现 `src/task_state.py`
  - [x] 每个任务文件夹下维护 `task_state.yaml`
  - [x] 记录：任务 ID、输入路径、各 stage 完成状态、输出路径、时间戳、错误信息
  - [x] 提供 `mark_done(step)` / `is_done(step)` / `get_outputs(step)` / `reset(step)` 接口
  - [x] 支持从现有任务文件夹恢复状态

**产出文件**：
- `src/task_state.py`
- `tests/test_task_state.py`

---

### 3. 高质量音频工具模块
- [x] 实现 `src/audio_utils.py`
  - [x] `convert_to_wav(...)`：优先 copy 音频流，必须重编码时使用 PCM
  - [x] `split_audio(...)`：使用 ffmpeg segment，高质量 PCM 输出
  - [x] `merge_audio(...)`：使用 ffmpeg concat demuxer + copy
  - [x] `resample(...)`：使用 soxr 重采样
  - [x] `get_duration(...)` / `get_media_info(...)`
  - [x] 所有切分/合并操作默认最高质量，避免额外有损编码

**产出文件**：
- `src/audio_utils.py`
- `tests/test_audio_utils.py`

---

### 4. Stage 1：数据清洗
- [x] 实现 `src/pipeline/components/mp4_to_wav.py`
  - [x] 支持 MP4 / MP3 / WAV 输入
  - [x] 输出统一为 WAV（PCM s16le，保留原采样率或指定 44100/16000）
  - [x] 多集输入分别处理
- [x] 实现 `src/pipeline/components/oped_removal.py`
  - [x] 使用 log-Mel 互相关识别每集开头/结尾的 OP/ED
  - [x] 支持从 `oped/` 文件夹中所有音频构建模板库
  - [x] 识别成功后，切除对应时间段，记录切除位置
- [x] 实现 `src/pipeline/components/bgm_removal.py`
  - [x] 封装 Mel-Band-Roformer Vocal Model 调用
  - [x] 输入长音频时自动切分（参考 UniSE_Colab 360s 一段），分别去背景音，再合并
  - [x] 输出 `*_vocals.wav`（保留人声）和 `*_instrumental.wav`（背景音乐）
- [x] 实现 `src/pipeline/stage1_data_cleaning.py`
  - [x] 编排以上组件
  - [x] 每集输出：`output/stage1/03_cleaned/{episode}_cleaned.wav`
  - [x] 在 `task_state.yaml` 中记录进度

**产出文件**：
- `src/pipeline/components/mp4_to_wav.py`
- `src/pipeline/components/oped_removal.py`
- `src/pipeline/components/bgm_removal.py`
- `src/pipeline/stage1_data_cleaning.py`
- `tests/test_stage1.py`

---

### 5. Stage 2：说话人提取
- [x] 实现 `src/pipeline/components/unise_tse.py`
  - [x] 封装 QuarkAudio-UniSE 的 `test.py`
  - [x] 输入音频按 360s 切分（16000Hz mono）
  - [x] 准备 `mix/`、`enroll/`、`tgt/` 目录
  - [x] 调用 UniSE 生成 `output/` 下各切片的目标说话人音频
  - [x] 合并切片为完整音频
  - [x] 支持 `mode=tse` 目标说话人提取
- [x] 实现 `src/pipeline/components/silence_removal.py`
  - [x] 使用 `librosa.effects.split` 检测并去除静音段
  - [x] 记录每段非静音音频在原音频中的 `[start, end)` 位置映射
  - [x] 输出 `stage2_silence_removed.wav` + `silence_map.json`
- [x] 实现 `src/pipeline/components/audio_mapping.py`
  - [x] 提供 `forward_split(...)`：按切割点切分并记录原位置
  - [x] 提供 `rebuild_from_srt(...)`：将 SRT 时间映射回原始音频并切出片段
  - [x] 保证时间轴精确，便于后续 SRT 时间映射
- [x] 实现 `src/pipeline/components/aliyun_asr.py`
  - [x] 调用阿里云录音文件识别（`qwen3-asr-flash-filetrans`）
  - [x] 本地文件上传到临时 OSS（参考 `qwen-ai-mcp`）
  - [x] 启用说话人分离（speaker diarization），输出 SRT / JSON
  - [x] 单集单段处理，降低识别错说话人概率
- [x] 实现 `src/pipeline/components/srt_cleaning.py`
  - [x] 调用 Qwen 3.6-max（`qwen3.6-max`）模型
  - [x] Prompt 要求：删除意义不明（如“啊啊啊”）的句子；只保留主 speaker，删除其他 speaker
  - [x] 输入 SRT，输出清洗后的 SRT；本地启发式作为兜底
- [x] 实现 `src/pipeline/stage2_speaker_extraction.py`
  - [x] 编排流程：UniSE TSE → 静音去除 → ASR → SRT 清洗 → 反向映射切片段 → 可选二次 UniSE
  - [x] 最终输出：`output/stage2/episode_XXX/clips/*.wav`

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
- [x] 实现 `src/aliyun/dashscope_client.py`
  - [x] 读取 `DASHSCOPE_API_KEY`
  - [x] 提供 ASR 文件识别调用（异步创建 + 轮询）
  - [x] 提供 Qwen Chat 调用（SRT 清洗）
  - [x] 本地文件上传 OSS 功能内置于同一模块（`upload_local_to_oss`）
- [x] `src/aliyun/oss_uploader.py` 功能合并至 `dashscope_client.py`

**产出文件**：
- `src/aliyun/dashscope_client.py`
- `src/aliyun/oss_uploader.py`

---

### 7. 主脚本 `main.py`
- [x] 实现命令行入口
  - [x] `--task-dir`：指定任务文件夹
  - [x] `--config`：配置文件路径
  - [x] `--source-dir` / `--oped-dir` / `--reference-dir`：输入路径
  - [x] `--stage {1,2,all}`：选择运行阶段
  - [x] `--resume` / `--no-resume`：断点续跑控制
  - [x] `--test-local`：本地快速测试
- [x] 组件在配置中以列表形式存在，可拆卸
- [x] 本地测试模式自动准备 `test.mp4` 源目录

**产出文件**：
- `main.py`
- `configs/test_local.yaml`

---

### 8. 分脚本 `scripts/convert_anime_to_dataset.py`
- [x] 实现独立入口脚本
  - [x] 封装 Stage 1 + Stage 2 的简化调用
  - [x] 支持直接传入 `source_dir`、`oped_dir`、`reference_dir`、`output_dir`
  - [x] 可脱离 `main.py` 单独使用

**产出文件**：
- `scripts/convert_anime_to_dataset.py`

---

### 9. IPython Notebook
- [x] 实现 `anime_voice_training.ipynb`
  - [x] 从 GitHub 仓库 clone 项目
  - [x] 安装依赖
  - [x] 下载 UniSE 与 BiCodec 等模型
  - [x] 上传/挂载数据
  - [x] 配置并运行 `main.py`
  - [x] 打包下载结果

**产出文件**：
- `anime_voice_training.ipynb`

---

### 10. 本地冒烟测试
- [x] 使用 `test.mp4` 运行主脚本
  - [x] 只启用快速组件（mp4_to_wav）
  - [x] 验证 `task_state.yaml` 正确记录
  - [x] 验证输出目录结构正确
- [x] 运行全部单元测试（21 项通过）
- [x] 完善 `README.md` 与使用示例

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

- [x] `main.py` 可运行本地测试并产出预期输出
- [x] `scripts/convert_anime_to_dataset.py` 可独立运行
- [x] `anime_voice_training.ipynb` 可在 Colab 环境中运行
- [x] 所有组件在主配置中可拆卸
- [x] 每个任务使用 `task_state.yaml` 记录状态，支持断点续跑
- [ ] 项目已推送至 GitHub 仓库（由用户完成最后 `git push`）
