# AOXLM Architecture

## Overview

AOXLM enhances an existing Audio LLM (Qwen2-Audio) with structured output capabilities (word-level timestamps and confidence scores). Instead of building from scratch, we leverage the pre-trained audio understanding and add a lightweight alignment module.

## Design Philosophy

### Why Not Build From Scratch?

| Approach | Pros | Cons |
|----------|------|------|
| **From scratch** (like VoxLM v1) | Full control | Lose LLM capabilities, massive training needed |
| **Enhance existing** (AOXLM) | Keep all capabilities | Limited by base model architecture |

We choose **enhancement** because:
1. Audio understanding is already solved by Qwen2-Audio
2. Instruction following is already there
3. We only need to add alignment/timestamps
4. Much faster development cycle

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AOXLM                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Audio waveform (16kHz) + Instruction text                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Qwen2-Audio (Frozen/LoRA)                        │ │
│  │                                                                     │ │
│  │   ┌──────────────┐                                                 │ │
│  │   │ Audio        │    Audio features                               │ │
│  │   │ Encoder      │──────────────────┐                              │ │
│  │   │ (Whisper)    │                  │                              │ │
│  │   └──────────────┘                  │                              │ │
│  │                                     ▼                              │ │
│  │   ┌──────────────┐           ┌──────────────┐                     │ │
│  │   │ Text         │           │  Multimodal  │                     │ │
│  │   │ Tokenizer    │──────────▶│  Projector   │                     │ │
│  │   │              │           │              │                     │ │
│  │   └──────────────┘           └──────┬───────┘                     │ │
│  │                                     │                              │ │
│  │                                     ▼                              │ │
│  │                            ┌──────────────┐                        │ │
│  │                            │   Qwen2 LLM  │                        │ │
│  │                            │   Decoder    │                        │ │
│  │                            │              │                        │ │
│  │                            └──────┬───────┘                        │ │
│  │                                   │                                │ │
│  │            ┌──────────────────────┼──────────────────────┐        │ │
│  │            │                      │                      │        │ │
│  │            ▼                      ▼                      ▼        │ │
│  │    Audio Embeddings        Hidden States          Token Logits    │ │
│  │    (for alignment)         (for alignment)        (for text)      │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                 │                      │                      │         │
│                 │                      │                      │         │
│                 ▼                      ▼                      ▼         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Alignment Module (Trainable)                     │ │
│  │                                                                     │ │
│  │   ┌──────────────────────────────────────────────────────────────┐ │ │
│  │   │                  Cross-Attention Layers                       │ │ │
│  │   │                                                               │ │ │
│  │   │   Query: Text hidden states                                   │ │ │
│  │   │   Key/Value: Audio embeddings                                 │ │ │
│  │   │   Output: Alignment weights [text_len, audio_frames]          │ │ │
│  │   │                                                               │ │ │
│  │   └──────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                      │ │
│  │                              ▼                                      │ │
│  │   ┌──────────────────────────────────────────────────────────────┐ │ │
│  │   │                   DTW (Dynamic Time Warping)                  │ │ │
│  │   │                                                               │ │ │
│  │   │   • Find optimal monotonic alignment path                     │ │ │
│  │   │   • Convert attention weights to timestamps                   │ │ │
│  │   │   • Handle word boundaries                                    │ │ │
│  │   │                                                               │ │ │
│  │   └──────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                      │ │
│  │                              ▼                                      │ │
│  │   ┌──────────────────────────────────────────────────────────────┐ │ │
│  │   │               Confidence Extraction                           │ │ │
│  │   │                                                               │ │ │
│  │   │   • Token probability from LLM logits                         │ │ │
│  │   │   • Aggregate to word-level                                   │ │ │
│  │   │   • Calibration (optional)                                    │ │ │
│  │   │                                                               │ │ │
│  │   └──────────────────────────────────────────────────────────────┘ │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                     │                                    │
│                                     ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        Structured Output                            │ │
│  │                                                                     │ │
│  │   {                                                                 │ │
│  │     "text": "mister quilter is the apostle...",                    │ │
│  │     "words": [                                                      │ │
│  │       {"word": "mister", "start": 0.0, "end": 0.3, "conf": 0.98},  │ │
│  │       {"word": "quilter", "start": 0.3, "end": 0.7, "conf": 0.95}, │ │
│  │       ...                                                           │ │
│  │     ],                                                              │ │
│  │     "language": "en",                                               │ │
│  │     "duration": 5.84                                                │ │
│  │   }                                                                 │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Base Model: Qwen2-Audio

**What it provides:**
- Audio encoder (based on Whisper)
- Multimodal projector (audio → LLM space)
- Qwen2 LLM decoder
- Pre-trained weights with audio understanding

**What we need to extract:**
- Audio embeddings (before LLM, for alignment)
- Hidden states (during generation, for alignment)
- Token logits (for confidence)

```python
# Pseudo-code for extracting what we need
class Qwen2AudioWrapper:
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct"):
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name)
    
    def forward_with_alignment_info(self, audio, instruction):
        # Get audio embeddings from encoder
        audio_embeds = self.model.audio_encoder(audio)
        
        # Generate text with hidden states
        outputs = self.model.generate(
            audio_embeds,
            instruction,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        return {
            "audio_embeds": audio_embeds,
            "hidden_states": outputs.hidden_states,
            "scores": outputs.scores,  # For confidence
            "sequences": outputs.sequences  # Generated tokens
        }
```

### 2. Alignment Module

**Purpose:** Compute alignment between generated text and audio frames.

**Architecture:**
```python
class AlignmentModule(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=8):
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, text_hidden, audio_embeds):
        # text_hidden: [batch, text_len, dim]
        # audio_embeds: [batch, audio_frames, dim]
        
        queries = self.query_proj(text_hidden)
        
        for layer in self.cross_attention_layers:
            queries, attn_weights = layer(queries, audio_embeds, audio_embeds)
        
        # attn_weights: [batch, text_len, audio_frames]
        return attn_weights
```

### 3. Timestamp Extraction (DTW)

**Purpose:** Convert soft attention weights to discrete timestamps.

```python
class TimestampExtractor:
    def __init__(self, frame_duration_ms=20):
        self.frame_duration_ms = frame_duration_ms
    
    def __call__(self, alignment_weights, token_ids, tokenizer):
        # alignment_weights: [text_len, audio_frames]
        
        # 1. Apply median filter for smoothness
        weights = median_filter(alignment_weights)
        
        # 2. Run DTW for monotonic alignment
        text_indices, frame_indices = dtw(weights)
        
        # 3. Convert to word timestamps
        words = self.frames_to_words(
            text_indices, frame_indices, token_ids, tokenizer
        )
        
        return words
```

### 4. Confidence Extraction

**Purpose:** Get per-word confidence from token probabilities.

```python
class ConfidenceExtractor:
    @staticmethod
    def from_scores(scores, token_ids, temperature=1.0):
        # scores: List of [batch, vocab_size] from generate()
        # token_ids: [batch, seq_len]
        
        confidences = []
        for i, score in enumerate(scores):
            probs = softmax(score / temperature, dim=-1)
            token_prob = probs[0, token_ids[0, i]]
            confidences.append(token_prob.item())
        
        return confidences
    
    @staticmethod
    def aggregate_to_words(token_confidences, token_ids, tokenizer):
        # Merge subword confidences to word-level
        # Using min aggregation (conservative)
        ...
```

## Training Strategy

### What to Train

| Component | Trainable | Why |
|-----------|-----------|-----|
| Qwen2-Audio encoder | ❌ Frozen | Already has audio understanding |
| Qwen2-Audio LLM | ❄️ LoRA (optional) | Keep capabilities, minor adaptation |
| Alignment module | ✅ Yes | New component, needs training |

### Training Data

**For alignment module:**
- Any dataset with word-level timestamps
- Options:
  - LibriSpeech + forced alignment (WhisperX)
  - Switchboard (has timestamps)
  - MLS (Multilingual LibriSpeech)

**Training objective:**
```python
def alignment_loss(predicted_alignment, target_timestamps, frame_rate):
    # CrisperWhisper-style attention supervision
    # Ground truth: soft target based on word boundaries
    # Loss: 1 - cosine_similarity(predicted, target)
    ...
```

### Training Pipeline

```
1. Load Qwen2-Audio (frozen)
2. Initialize alignment module (random)
3. For each batch:
   a. Forward through Qwen2-Audio → get audio_embeds, hidden_states
   b. Forward through alignment module → get predicted alignment
   c. Compute alignment loss with ground truth timestamps
   d. Backprop only through alignment module
4. Save alignment module weights
```

## Inference Pipeline

```python
def transcribe(audio, instruction="Transcribe this audio"):
    # 1. Get base model outputs
    outputs = qwen2_audio.generate(
        audio, 
        instruction,
        output_hidden_states=True,
        output_scores=True
    )
    
    # 2. Get alignment
    alignment = alignment_module(
        text_hidden=outputs.hidden_states[-1],
        audio_embeds=audio_encoder(audio)
    )
    
    # 3. Extract timestamps
    timestamps = timestamp_extractor(alignment, outputs.sequences)
    
    # 4. Extract confidence
    confidence = confidence_extractor(outputs.scores, outputs.sequences)
    
    # 5. Combine into structured output
    return {
        "text": tokenizer.decode(outputs.sequences),
        "words": merge(timestamps, confidence),
        "duration": audio_duration
    }
```

## Challenges & Solutions

### Challenge 1: Extracting Audio Embeddings

**Problem:** Qwen2-Audio doesn't expose intermediate audio embeddings easily.

**Solutions:**
1. Hook into the model to capture embeddings
2. Use encoder output before projection
3. Re-compute from raw audio using the encoder directly

### Challenge 2: Hidden State Alignment

**Problem:** Hidden states during generation are sequential, need to align with audio.

**Solutions:**
1. Use cross-attention during generation
2. Post-hoc alignment after generation
3. Cache audio embeddings and align at the end

### Challenge 3: Different Tokenizers

**Problem:** Qwen2-Audio uses its own tokenizer, timestamps need word-level.

**Solutions:**
1. Merge subword tokens to words (like VoxLM does)
2. Use tokenizer's word boundaries
3. Post-process with simple rules

## File Structure

```
aoxlm/
├── src/
│   ├── __init__.py
│   ├── model.py              # Main AOXLM class
│   ├── alignment.py          # Alignment module (from VoxLM)
│   ├── timestamp_extractor.py
│   ├── confidence_extractor.py
│   └── qwen2_audio_wrapper.py
├── scripts/
│   ├── train.py              # Train alignment module
│   ├── inference.py          # Run inference
│   └── evaluate.py           # Benchmark
├── configs/
│   ├── aoxlm-7b.yaml
│   └── aoxlm-2b.yaml
├── tests/
│   └── test_alignment.py
└── docs/
    ├── ARCHITECTURE.md       # This file
    └── TRAINING.md
```

## Next Steps

1. **Verify Qwen2-Audio integration** - Can we extract audio embeddings?
2. **Port alignment module** from VoxLM
3. **Create training pipeline** for alignment
4. **Benchmark** on LibriSpeech
5. **Optimize** for inference speed
