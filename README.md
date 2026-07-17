# Anime Voice Training Pipeline

从动漫剧集提取单一目标角色语音训练数据的可插拔、可断点续传流水线。

## Stage 2 双模式

| mode | 说明 | 配置 |
|------|------|------|
| **`v2`（默认）** | UniSE → VAD 分句 → ECAPA 声纹过滤 → 质量门控 → 可选逐 clip ASR | `configs/default.yaml` |
| **`aliyun` / `v1`（备选）** | UniSE → 拼接 ASR 输入 → 阿里云 diarization → SRT 清洗 → 回传切分 | `configs/legacy_aliyun.yaml` |

切换方式：

```bash
# 默认 v2
python main.py --config configs/default.yaml --stage all ...

# 强制阿里云备选
python main.py --config configs/legacy_aliyun.yaml --stage all ...
# 或
python main.py --config configs/default.yaml --mode aliyun --stage all ...
```

## Features

- Stage 1：转 WAV、去 OP/ED、Mel-Band 去 BGM
- Stage 2 v2：VAD 分句 + 声纹验证（不依赖 ASR 说话人标签）
- Stage 2 legacy：阿里云 paraformer diarization（保留对照）
- 组件可开关、`task_state.yaml` 断点续传、长音频分块防 OOM

## Project Structure

```
.
├── main.py
├── anime_voice_training.ipynb      # Colab：legacy Aliyun
├── anime_voice_training_v2.ipynb   # Colab：v2 首选（共用同一 Drive 数据目录）
├── configs/
│   ├── default.yaml                # mode: v2
│   ├── legacy_aliyun.yaml          # mode: aliyun
│   └── test_from_cache.yaml        # 本地用 test/task 缓存冒烟
├── src/pipeline/
│   ├── stage2_speaker_extraction.py  # 分发器
│   ├── stage2_vad_verify.py          # v2 实现
│   ├── stage2_aliyun.py              # 阿里云备选
│   └── components/
│       ├── vad_detection.py
│       ├── speaker_verification.py   # ECAPA / MFCC
│       ├── quality_gate.py
│       ├── clip_asr.py
│       ├── speaker_verify.py         # legacy pyannote（阿里云路径）
│       └── ...
```

## Quick Start

```bash
pip install -r requirements.txt

python main.py \
  --config configs/default.yaml \
  --task-dir ./my_task \
  --source-dir ./my_data/animations \
  --oped-dir ./my_data/oped \
  --reference-dir ./my_data/reference \
  --stage all
```

本地低算力（复用 `test/task` 的 TSE/cleaned）：

```bash
python main.py --test-from-cache --test-trim-seconds 60
```

## Google Colab（共用 Drive 结构）

两个 notebook **共用**同一套 Drive 路径，无需新建数据目录：

```
MyDrive/UniSE/data/{animations,oped,reference}
MyDrive/UniSE/models/unise.ckpt
MyDrive/UniSE/output
```

| Notebook | Stage2 |
|----------|--------|
| `anime_voice_training.ipynb` | legacy Aliyun（会强制 `mode: aliyun`） |
| `anime_voice_training_v2.ipynb` | v2 首选 |

Colab 本地工作目录不同（`/content/anime_voice_training` vs `_v2`），避免互相覆盖运行缓存；结果都写回 `MyDrive/UniSE/output`。

## Tests

```bash
pytest tests/test_stage2_v2.py tests/test_audio_utils.py -q
```

## License

MIT
