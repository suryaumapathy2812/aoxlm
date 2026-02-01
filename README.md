# AOXLM

Smart ASR model combining semantic audio understanding with accurate timestamps.

## Quick Start

```bash
# Install with uv
uv sync

# Install with datagen dependencies
uv sync --extra datagen

# Test Qwen3-ASR transcription
python scripts/data_gen/qwen_transcribe.py audio.mp3
```

## Project Structure

```
src/
├── models/
│   ├── encoder.py      # Semantic audio encoders (WavLM, MMS, etc.)
│   └── ctc_head.py     # CTC projection + decoder
├── data/
│   └── audio.py        # Audio preprocessing
└── inference.py        # Inference pipeline

scripts/
└── data_gen/
    ├── qwen_transcribe.py    # Qwen3-ASR transcription
    ├── gemini_transcribe.py  # Gemini transcription
    └── pipeline.py           # Data generation pipeline
```

## Architecture

See [docs/arch-v1.md](docs/arch-v1.md) for the full architecture design.
