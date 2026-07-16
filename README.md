# Anime Voice Training Pipeline

A pluggable, resumable pipeline that converts animation episodes into a high-quality target-speaker voice training dataset.

## Features

- **Two-stage pipeline**
  - Stage 1: Data cleaning — MP4/MP3 → WAV, OP/ED removal, background-music removal.
  - Stage 2: Speaker extraction — UniSE TSE, silence removal with position mapping, Aliyun ASR speaker diarization, Qwen SRT cleaning, reverse mapping to original audio.
- **Pluggable components**: enable/disable any step via YAML config.
- **Resumable**: each task folder stores `task_state.yaml`; re-run and only pending/failed steps execute.
- **High-quality audio handling**: ffmpeg-based conversion/split/merge, prefers stream copy to avoid extra loss.
- **OOM-safe**: long audio is chunked before heavy models (UniSE, Mel-Band-Roformer).

## Project Structure

```
.
├── main.py                           # Main orchestrator
├── anime_voice_training.ipynb        # Colab notebook
├── scripts/convert_anime_to_dataset.py  # Standalone conversion script
├── configs/
│   ├── default.yaml                  # Default config
│   └── test_local.yaml               # Local smoke-test config
├── src/
│   ├── task_state.py                 # YAML checkpoint manager
│   ├── audio_utils.py                # ffmpeg audio utilities
│   ├── aliyun/dashscope_client.py    # Aliyun DashScope client
│   └── pipeline/
│       ├── stage1_data_cleaning.py
│       ├── stage2_speaker_extraction.py
│       └── components/
│           ├── mp4_to_wav.py
│           ├── oped_removal.py
│           ├── bgm_removal.py
│           ├── unise_tse.py
│           ├── silence_removal.py
│           ├── audio_mapping.py
│           ├── aliyun_asr.py
│           └── srt_cleaning.py
├── tests/                            # Unit tests
└── plan.md                           # Execution plan / status
```

## Requirements

- Python 3.10+
- ffmpeg
- `DASHSCOPE_API_KEY` environment variable for Aliyun ASR / Qwen chat
- Mel-Band-Roformer project + checkpoint for BGM removal
- unified-audio/QuarkAudio-UniSE project + checkpoint for speaker extraction

Install core dependencies:

```bash
pip install -r requirements.txt
```

Install tool-specific dependencies in their own directories:

```bash
cd /path/to/Mel-Band-Roformer-Vocal-Model
pip install -r requirements.txt

cd /path/to/unified-audio/QuarkAudio-UniSE
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare inputs

```
my_data/
├── source/           # episode MP4/MP3 files
├── oped/             # OP/ED audio files (optional)
└── reference/        # target speaker reference audio
```

### 2. Run the pipeline

```bash
python main.py \
    --config configs/default.yaml \
    --task-dir ./my_task \
    --source-dir ./my_data/source \
    --oped-dir ./my_data/oped \
    --reference-dir ./my_data/reference \
    --stage all
```

### 3. Resume a failed run

```bash
python main.py --task-dir ./my_task --stage all
```

## Local Smoke Test

A fast local test that only converts `data/test.mp4` to WAV:

```bash
python main.py --test-local
```

An extended local test that trims `test.mp4` to the first N seconds and runs **BGM removal + UniSE target speaker extraction** (no cloud ASR):

```bash
# default trim is 30 seconds
python main.py --test-local-full

# custom trim length
python main.py --test-local-full --test-trim-seconds 10
```

**Note on UniSE checkpoint:** Some systems or cleanup tools may delete files named `epoch=...ckpt`. If that happens, rename the UniSE checkpoint to `unise.ckpt` and update `configs/default.yaml` accordingly.

## Standalone Script

```bash
python scripts/convert_anime_to_dataset.py \
    --source-dir ./my_data/source \
    --oped-dir ./my_data/oped \
    --reference-dir ./my_data/reference \
    --output-dir ./my_output
```

## Google Colab

Open `anime_voice_training.ipynb` in Colab and run all cells. The notebook expects your data in Google Drive at `MyDrive/UniSE/data`:

```
MyDrive/UniSE/data/
├── source/           # episode MP4/MP3 files
├── oped/             # OP/ED audio files (optional)
└── reference/        # target speaker reference audio
```

Set `DASHSCOPE_API_KEY` in the notebook if you want Stage 2 ASR + SRT cleaning. Stage 1 (data cleaning) and the UniSE target-speaker extraction step do not require cloud API keys.

## Configuration

Key settings in `configs/default.yaml`:

| Setting | Description |
|---------|-------------|
| `stage1.components` | Enabled cleaning steps |
| `stage2.components` | Enabled speaker-extraction steps |
| `stage2.run_unise_v2` | Run a second UniSE pass on final clips (default false) |
| `melband_roformer.*` | Paths to Mel-Band-Roformer project and checkpoint |
| `unise.*` | Paths to UniSE project and checkpoint |
| `aliyun.dashscope_api_key` | API key (or use `DASHSCOPE_API_KEY` env var) |

## Tests

```bash
pytest tests/
```

## Notes

- Audio quality: all internal split/merge operations prefer stream copy or PCM WAV. Background-music removal is the only lossy step and is considered acceptable.
- Cloud costs: Aliyun ASR and Qwen calls may incur charges; process long audio in single-episode segments to reduce repeated speaker mis-identification.

## License

MIT
