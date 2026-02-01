# AOXLM Architecture v1 - Implementation Plan

## Vision

Build a model that combines:
- **Gemini's understanding**: Knows what the user MEANT, not just what sounds were made
- **Whisper's precision**: Accurate timestamps and verbatim transcription
- **Structured output**: Both what was said and what was meant

## Core Principle

> "The intelligence is in HEARING, not in RESPONDING"

The model should deeply understand audio (like Gemini) but output structured ASR-like data (like Whisper).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AOXLM v1 ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────────┐                              │
│                         │   AUDIO INPUT   │                              │
│                         │  (full file)    │                              │
│                         └────────┬────────┘                              │
│                                  │                                       │
│                                  ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │  SEMANTIC AUDIO ENCODER │                          │
│                    │  (WavLM / Qwen3-Omni)   │                          │
│                    │      ~300-500M          │                          │
│                    └─────────────┬───────────┘                          │
│                                  │                                       │
│                    ┌─────────────┴───────────┐                          │
│                    │                         │                           │
│                    ▼                         ▼                           │
│          ┌─────────────────┐      ┌─────────────────────┐               │
│          │    CTC HEAD     │      │  SEMANTIC DECODER   │               │
│          │   (verbatim)    │      │  (understanding)    │               │
│          │     ~5-10M      │      │     ~200-400M       │               │
│          └────────┬────────┘      └──────────┬──────────┘               │
│                   │                          │                           │
│                   └────────────┬─────────────┘                           │
│                                │                                         │
│                                ▼                                         │
│                    ┌─────────────────────────┐                          │
│                    │   INTELLIGENT MERGER    │                          │
│                    └─────────────┬───────────┘                          │
│                                  │                                       │
│                                  ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │   STREAMING OUTPUT      │                          │
│                    │       (JSONL)           │                          │
│                    └─────────────────────────┘                          │
│                                                                          │
│  TOTAL: ~500-900M params                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Semantic Audio Encoder

**Purpose**: Deeply understand audio content, not just map sounds to phonemes.

**Options (ranked by preference)**:

| Encoder | Params | Type | Availability | Notes |
|---------|--------|------|--------------|-------|
| Qwen3-Omni AuT | ~500M | Semantic | May need extraction | Closest to Google's USM |
| WavLM-large | 300M | Semantic | Open source | Proven, widely used |
| Seamless-M4T encoder | ~300M | Semantic | Open source | Robust to accents/noise |
| MMS (Meta) | 1B | Semantic | Open source | Best multilingual (1000+ langs) |
| HuBERT-large | 300M | Semantic | Open source | Good baseline |

**Why Semantic over Temporal**:
- Temporal (Whisper-style): Maps sounds → text (phonetic)
- Semantic (WavLM-style): Understands meaning → text (contextual)

For language learners, code-switching, mispronunciations → need UNDERSTANDING.

**Output**: Frame-level embeddings `[T × D]` where:
- T = number of frames (typically 50Hz = 20ms per frame)
- D = embedding dimension (768-1280)

**Training**: 
- Phase 1: Frozen (use pretrained)
- Phase 2: Fine-tune with adapter or full (optional)

---

### 2. CTC Head (Verbatim Transcription)

**Purpose**: Exact transcription of what was spoken, with frame-level alignment.

**Architecture**:
```python
class CTCHead(nn.Module):
    def __init__(self, encoder_dim=1024, vocab_size=5000):
        self.projection = nn.Linear(encoder_dim, vocab_size)
    
    def forward(self, encoder_output):
        # encoder_output: [B, T, D]
        logits = self.projection(encoder_output)  # [B, T, vocab_size]
        return logits
```

**Vocab options**:
- Characters (~100 tokens): Fine-grained alignment, slower
- Subwords/BPE (~5000 tokens): Word-level alignment, faster

**Why CTC**:
- Timestamps are COMPUTED from frame predictions, not GENERATED
- Non-autoregressive (instant, parallel)
- Natural alignment: frame index → character/subword

**Output**:
- Frame-level predictions with alignments
- Verbatim text: "I want to um go to the uh libary"
- Timestamps for each token

---

### 3. Semantic Decoder (Understanding)

**Purpose**: Understand what user MEANT, produce clean text.

**Architecture**:
```python
class SemanticDecoder(nn.Module):
    def __init__(self, 
                 encoder_dim=1024, 
                 decoder_dim=768, 
                 num_layers=6,
                 vocab_size=32000):
        self.embed = nn.Embedding(vocab_size, decoder_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=decoder_dim,
                cross_attention=True,  # Attends to encoder
                causal=True            # Autoregressive
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(decoder_dim, vocab_size)
    
    def forward(self, encoder_output, prev_tokens):
        # Cross-attention to encoder output
        # Autoregressive generation
        ...
```

**Key features**:
- Cross-attention to encoder: sees full audio context
- Autoregressive: enables streaming
- Language model: produces fluent, corrected text

**Output**:
- Clean text: "I want to go to the library"
- Token-by-token streaming

---

### 4. Intelligent Merger

**Purpose**: Combine CTC (verbatim + timestamps) with Decoder (understanding).

**Algorithm**:
```python
def merge(ctc_output, decoder_output):
    # ctc_output: [(frame_idx, subword), ...]
    # decoder_output: [token, token, ...]
    
    segments = []
    for decoder_word in decoder_output:
        # Find corresponding CTC tokens
        ctc_match = align_ctc_to_decoder(decoder_word, ctc_output)
        
        # Get timestamps from CTC
        start_frame, end_frame = get_frame_range(ctc_match)
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        
        # Check if correction occurred
        verbatim = ctc_match.text
        corrected = verbatim != decoder_word
        
        # Get confidence from CTC probabilities
        confidence = compute_confidence(ctc_match)
        
        segment = {
            "word": decoder_word,
            "start": start_time,
            "end": end_time,
            "confidence": confidence,
        }
        
        if corrected:
            segment["spoken"] = verbatim
            segment["corrected"] = True
        
        if is_filler(verbatim):
            segment["type"] = "filler"
        
        segments.append(segment)
    
    return segments
```

---

### 5. Streaming Output (JSONL)

**Format**: JSON Lines - one object per word, streamable.

```jsonl
{"w": "I", "s": 0.00, "e": 0.20, "c": 0.99}
{"w": "want", "s": 0.20, "e": 0.50, "c": 0.98}
{"w": "to", "s": 0.50, "e": 0.70, "c": 0.97}
{"w": "um", "s": 0.80, "e": 1.00, "c": 0.95, "t": "filler"}
{"w": "go", "s": 1.10, "e": 1.30, "c": 0.98}
{"w": "to", "s": 1.30, "e": 1.50, "c": 0.97}
{"w": "the", "s": 1.50, "e": 1.70, "c": 0.96}
{"w": "uh", "s": 1.80, "e": 2.00, "c": 0.94, "t": "filler"}
{"w": "library", "s": 2.10, "e": 2.60, "c": 0.95, "spoken": "libary", "cor": true}
{"done": true, "verbatim": "I want to um go to the uh libary", "text": "I want to go to the library", "lang": "en", "dur": 2.6}
```

**Schema**:
- `w`: word
- `s`: start time (seconds)
- `e`: end time (seconds)
- `c`: confidence (0-1)
- `t`: type (optional, e.g., "filler")
- `spoken`: what was actually spoken (if different from `w`)
- `cor`: corrected flag (if word was contextually corrected)

---

## Training Strategy

### Loss Function

```python
loss = alpha * ctc_loss + beta * decoder_loss

# CTC loss: frame-level alignment
ctc_loss = nn.CTCLoss()(ctc_logits, ctc_targets, input_lengths, target_lengths)

# Decoder loss: standard cross-entropy
decoder_loss = nn.CrossEntropyLoss()(decoder_logits, decoder_targets)

# Hyperparameters
alpha = 0.3  # CTC weight
beta = 1.0   # Decoder weight
```

### Training Phases

**Phase 1: Train heads (encoder frozen)**
- Freeze pretrained encoder
- Train CTC head + Decoder from scratch
- Fast convergence, uses pretrained representations
- ~1-2 weeks on 8 GPUs

**Phase 2: Fine-tune encoder (optional)**
- Unfreeze encoder
- Lower learning rate (1e-5 vs 1e-4)
- Fine-tune on target domain (noisy, accented, learner speech)
- ~1-2 weeks additional

### Data Requirements

**From VoxLM dataset, add**:
- Frame-level alignments (from WhisperX or MFA)
- Clean text labels (for decoder)

**Format**:
```json
{
  "audio": "sample.wav",
  "verbatim": "I want to um go to the uh libary",
  "clean": "I want to go to the library",
  "alignments": [
    {"token": "I", "start_frame": 0, "end_frame": 10},
    {"token": "want", "start_frame": 10, "end_frame": 25},
    ...
  ]
}
```

---

## Inference Flow

```
1. Load audio (full file)
   ↓
2. Encode (one-time, ~0.5s for 30s audio)
   encoder_output = encoder(audio)
   ↓
3. CTC forward (instant, parallel, ~50ms)
   ctc_alignment = ctc_decode(ctc_head(encoder_output))
   → All timestamps available immediately
   ↓
4. Decoder streaming (token by token)
   for token in decoder.generate(encoder_output):
       timestamps = get_timestamps(token, ctc_alignment)
       yield {"w": token, "s": start, "e": end, ...}
   ↓
5. Final summary
   yield {"done": true, "verbatim": "...", "text": "..."}
```

**Latency**:
- Encoding: ~0.5s for 30s audio
- CTC: ~50ms (non-autoregressive)
- Streaming starts: ~0.5s after audio received
- Each word: ~20-50ms

---

## Model Sizes

| Configuration | Encoder | CTC | Decoder | Total |
|---------------|---------|-----|---------|-------|
| Minimal | WavLM-base (90M) | 2M | 100M | ~200M |
| Standard | WavLM-large (300M) | 5M | 200M | ~500M |
| Full | Qwen3-AuT (500M) | 10M | 400M | ~900M |

**Recommendation**: Start with Standard (~500M), scale as needed.

---

## Implementation Order

1. **Data pipeline**: Add alignments to VoxLM dataset
2. **Encoder integration**: Load and test WavLM/HuBERT
3. **CTC head**: Implement and test standalone
4. **Decoder**: Implement with cross-attention
5. **Joint training**: Combine losses
6. **Merger**: Align CTC and Decoder outputs
7. **Streaming**: Implement JSONL output
8. **Evaluation**: Test on noisy/accented audio

---

## Success Criteria

1. **Timestamp accuracy**: < 50ms average error (match WhisperX)
2. **Verbatim accuracy**: WER < 10% on clean speech
3. **Understanding**: Correct 80%+ of common mispronunciations
4. **Noise robustness**: WER < 20% on noisy speech (SNR 10dB)
5. **Streaming latency**: First word within 1s of audio end
6. **Filler detection**: 90%+ F1 on filler words (um, uh, etc.)

---

## Open Questions

1. **Encoder choice**: WavLM vs Qwen3-Omni AuT - need to benchmark
2. **CTC vocab**: Characters vs subwords - affects alignment granularity
3. **Correction aggressiveness**: How much should decoder "fix"?
4. **Language mixing**: How to handle code-switching in output?

---

## Next Steps

1. Set up development environment
2. Download and test encoder options
3. Prepare dataset with alignments
4. Implement minimal prototype
5. Train and evaluate
