# AOXLM: Audio-Output LLM with Structured Transcription

> **Audio LLM Intelligence + Whisper-Quality Structured Output**

AOXLM combines the deep audio understanding of Audio LLMs (like Qwen2-Audio) with the precise, structured output of ASR systems (like Whisper). Get the best of both worlds: instruction-following, Q&A capabilities, AND word-level timestamps with confidence scores.

## The Problem

| Model Type | Understanding | Instructions | Timestamps | Confidence |
|------------|---------------|--------------|------------|------------|
| **Whisper** | Surface | Limited | ✅ | Partial |
| **Qwen2-Audio** | Deep | ✅ | ❌ | ❌ |
| **AOXLM** | Deep | ✅ | ✅ | ✅ |

**No existing open-source solution combines all four.**

## Features

- **Deep audio understanding** - Powered by Qwen2-Audio
- **Instruction following** - "Transcribe formally", "Translate to Hindi"
- **Word-level timestamps** - Precise timing for every word
- **Confidence scores** - Know which words are uncertain
- **Multi-task** - Transcribe, translate, summarize, Q&A
- **Filler detection** - "uh", "um", pauses (with training)

## Quick Start

```python
from aoxlm import AOXLM

model = AOXLM.from_pretrained("aoxlm-7b")

# Basic transcription with timestamps
result = model.transcribe("audio.wav")
print(result["text"])
# "mister quilter is the apostle of the middle classes"

print(result["words"])
# [
#   {"word": "mister", "start": 0.0, "end": 0.3, "confidence": 0.98},
#   {"word": "quilter", "start": 0.3, "end": 0.7, "confidence": 0.95},
#   ...
# ]

# With instructions
result = model.transcribe("audio.wav", instruction="Transcribe with punctuation")
# "Mister Quilter is the apostle of the middle classes."

# Translation
result = model.transcribe("audio.wav", instruction="Translate to Hindi")
# "मिस्टर क्विल्टर मध्यम वर्ग के प्रेरित हैं।"

# Q&A about audio
answer = model.ask("audio.wav", "What is the speaker's tone?")
# "The speaker has a formal, declarative tone."

# Summarization
summary = model.summarize("long_audio.wav")
# "The speaker discusses Quilter's influence on middle-class values."
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           AOXLM                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Qwen2-Audio-7B (Base Model)                │   │
│   │                                                         │   │
│   │   ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │   │
│   │   │   Audio     │    │  Multimodal │    │    LLM    │  │   │
│   │   │  Encoder    │───▶│  Projector  │───▶│  Decoder  │  │   │
│   │   │             │    │             │    │           │  │   │
│   │   └─────────────┘    └─────────────┘    └───────────┘  │   │
│   │         │                                     │         │   │
│   │         │              Audio                  │ Text    │   │
│   │         │            Embeddings            Output    │   │
│   │         ▼                                     ▼         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Alignment Module (Added)                    │   │
│   │                                                         │   │
│   │   • Cross-attention: text ↔ audio alignment             │   │
│   │   • DTW: optimal monotonic path                         │   │
│   │   • Confidence: from token probabilities                │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Structured Output                       │   │
│   │                                                         │   │
│   │   {                                                     │   │
│   │     "text": "full transcription",                       │   │
│   │     "words": [{"word", "start", "end", "confidence"}],  │   │
│   │     "language": "en",                                   │   │
│   │     "duration": 5.2                                     │   │
│   │   }                                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison with Existing Solutions

| Feature | Whisper | WhisperX | Qwen2-Audio | Gemini | AOXLM |
|---------|---------|----------|-------------|--------|-------|
| Word timestamps | ✅ | ✅ | ❌ | ❌ | ✅ |
| Word confidence | Partial | ✅ | ❌ | ❌ | ✅ |
| Instruction following | ❌ | ❌ | ✅ | ✅ | ✅ |
| Audio Q&A | ❌ | ❌ | ✅ | ✅ | ✅ |
| Translation | Limited | Limited | ✅ | ✅ | ✅ |
| Summarization | ❌ | ❌ | ✅ | ✅ | ✅ |
| Open source | ✅ | ✅ | ✅ | ❌ | ✅ |
| Filler detection | ❌ | ❌ | ❌ | ✅ | ✅ (planned) |

## Use Cases

| Application | How AOXLM Helps |
|-------------|-----------------|
| **Subtitles/Captions** | Word-level timestamps for precise sync |
| **Video editing** | "Jump to where they say X" |
| **Meeting transcription** | Speaker-aware, searchable transcripts |
| **Podcast platforms** | Interactive transcripts with timestamps |
| **Accessibility** | Real-time captions with confidence |
| **Language learning** | Highlight words as spoken |
| **Legal/Medical** | Precise timestamps for records |
| **Call analytics** | Understand tone + exact timing |

## Roadmap

### Phase 1: Foundation (MVP)
- [ ] Integrate Qwen2-Audio as base model
- [ ] Add alignment module for timestamps
- [ ] Implement confidence extraction
- [ ] Basic transcription with structured output
- [ ] HuggingFace integration

### Phase 2: Enhanced Features
- [ ] Instruction-following verification
- [ ] Translation with timestamps
- [ ] Filler word detection ("uh", "um", pauses)
- [ ] Speaker diarization
- [ ] Streaming support

### Phase 3: Production Ready
- [ ] Optimize inference speed
- [ ] Reduce memory footprint
- [ ] ONNX/TensorRT export
- [ ] Comprehensive benchmarks
- [ ] Documentation & examples

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/aoxlm.git
cd aoxlm

# Install dependencies
pip install -e .
# Or with uv
uv sync
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- ~16GB GPU memory (for 7B model)

## Model Variants

| Variant | Base Model | Size | Use Case |
|---------|------------|------|----------|
| `aoxlm-7b` | Qwen2-Audio-7B | ~14GB | Production |
| `aoxlm-2b` | Qwen2-Audio-2B | ~4GB | Edge/Fast inference |

## Training

```bash
# Train alignment module on top of frozen Qwen2-Audio
python scripts/train.py --config configs/aoxlm-7b.yaml

# Fine-tune with LoRA for specific domain
python scripts/train.py --config configs/aoxlm-7b.yaml --lora
```

## Benchmarks

Coming soon...

| Dataset | WER | Timestamp Accuracy | Confidence Calibration |
|---------|-----|-------------------|------------------------|
| LibriSpeech dev-clean | TBD | TBD | TBD |
| CommonVoice | TBD | TBD | TBD |

## Citation

```bibtex
@misc{aoxlm2026,
  title={AOXLM: Audio-Output LLM with Structured Transcription},
  author={Surya Umapathy},
  year={2026},
  url={https://github.com/yourusername/aoxlm}
}
```

## Acknowledgments

- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) - Base audio LLM
- [OpenAI Whisper](https://github.com/openai/whisper) - Timestamp extraction approach
- [VoxLM](https://github.com/suryaumapathy2812/voxlm) - Alignment module design

## License

Apache 2.0
