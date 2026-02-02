# CEFR Assessment System - Quick Reference Card
## Updated: February 2, 2026

## Project Summary

**Goal**: Deterministic CEFR speaking assessment for Indian English learners  
**Output**: 4-class (A1, A2, B1, B2+) scores for 5 dimensions  
**Method**: Classification heads on encoder hidden states (NOT LLM generation)

---

## Classification System

```
A1 (Beginner)     -> <60 WPM, isolated words, <500 vocab
A2 (Elementary)   -> 60-100 WPM, simple sentences, 500-1000 vocab
B1 (Intermediate) -> 100-130 WPM, maintains flow, 1000-2000 vocab
B2+ (Upper+)      -> >130 WPM, fluent, >2000 vocab
```

---

## Architecture

```
                    +---> Whisper-large-v3 ---> WhisperX ---> Transcription + Timestamps
                    |                                              |
Audio (16kHz) ------+                                              v
                    |                                    Fluency Metrics (WPM, pauses)
                    |
                    +---> WavLM/MMS Encoder ---> Hidden States ---> Classification Heads
                                (frozen)              |                    |
                                                      v                    v
                                             [B, T, 1024]         CEFR Scores (A1-B2+)
                                                                  - Fluency
                                                                  - Range
                                                                  - Accuracy
                                                                  - Phonology
                                                                  - Coherence
                                                                  - Overall
```

**Key Insight**: Use hidden states (deterministic), NOT text generation (stochastic)

---

## Why This Architecture?

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Transcription** | Whisper-large-v3 | Robust, well-tested, good on accented English |
| **Timestamps** | WhisperX | No duration limits, accurate word-level alignment |
| **Hidden States** | WavLM/MMS | Pre-trained semantic encoder, ~50Hz frame rate |
| **Classification** | MLP Heads | Deterministic, interpretable, fast |

**No audio duration constraints** - works with any length audio.

---

## Labeling Schema (Phase 1)

| Required Column | Type | Example |
|-----------------|------|---------|
| sample_id | string | "STU001_SES001_001" |
| audio_file | string | "audio/stu001.wav" |
| duration_seconds | float | 45.5 |
| native_language | string | "Tamil" |
| state | string | "Tamil Nadu" |
| task_type | string | "question_response" |
| task_prompt | string | "Where did you grow up?" |
| audio_quality | string | "good" |
| overall_level | string | "A2" |
| fluency_level | string | "A2" |
| range_level | string | "A2" |
| rater_id | string | "rater_01" |

---

## Indian English Patterns (DON'T PENALIZE)

| Pattern | Example | L1 Source |
|---------|---------|-----------|
| Reduplication | "small small" | Tamil, Telugu, Kannada |
| "only" emphasis | "Chennai only" | Tamil, Hindi |
| "itself" emphasis | "Delhi itself" | Hindi |
| Progressive overuse | "I am having" | Hindi |
| Wh-in-situ | "You are doing what?" | Bengali |

---

## Implementation Timeline

| Phase | Weeks | Deliverable | Data Needed |
|-------|-------|-------------|-------------|
| 1 (MVP) | 1-8 | Fluency + Range + Overall | 200 samples |
| 2 | 9-16 | + Accuracy + Phonology | 500 samples |
| 3 | 17-24 | + Coherence + Real-time | 1000 samples |

---

## Key Lessons from VoxLM

1. Encoder + LLM must be trained together (or have trained projector)
2. Freezing both = no adaptation possible
3. LLM generation = inconsistent scores
4. **Hidden states = deterministic, reliable**
5. **Classification heads = small, trainable, interpretable**

---

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| **ASR** | Whisper-large-v3 | Robust, mature |
| **Timestamps** | WhisperX | Word-level alignment |
| **Semantic Encoder** | WavLM-large | 1024-dim, 50Hz |
| **Alternative Encoder** | MMS-300M | Better for code-switching |
| **Coherence LLM** | Qwen3-7B | Phase 3 only |
| **Classification Heads** | Custom MLP | ~100K params each |
| **Hardware** | H100 GPU | Available |

---

## Accuracy Targets

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Exact Match | 60% | 70% | 75% |
| Within 1 Level | 85% | 90% | 95% |

---

## Key Resources

| Resource | Path |
|----------|------|
| Full Knowledge Base | `docs/CEFR_Assessment_System_Knowledge_Base.md` |
| Labeling Template | `docs/CEFR_Labeling_Template.csv` |
| L1 Patterns Reference | `docs/CEFR_L1_Patterns_Reference.md` |
| Encoder Code | `src/models/encoder.py` |
| CTC Head Code | `src/models/ctc_head.py` |

---

## Quick Start Code

```python
# Load semantic encoder
from src.models.encoder import SemanticEncoder

encoder = SemanticEncoder("wavlm-large", freeze=True)
hidden_states = encoder(audio_tensor)  # [B, T, 1024]

# Classification head (example)
import torch.nn as nn

class CEFRHead(nn.Module):
    def __init__(self, input_dim=1024, num_classes=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # Mean pooling
        return self.head(pooled)

# Transcription with timestamps (using WhisperX)
import whisperx

model = whisperx.load_model("large-v3", device="cuda")
result = model.transcribe(audio_path)
result = whisperx.align(result["segments"], model, audio_path)
```

---

## Remember

> "Use encoder hidden states (deterministic) with classification heads, NOT LLM generation (stochastic)."

> "Don't penalize valid Indian English patterns - they're L1 transfer, not errors."

> "Different CEFR dimensions need features from different encoder layers."
