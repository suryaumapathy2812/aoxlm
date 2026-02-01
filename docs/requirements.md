# AOXLM Requirements Document

## What We Need and Why

### Core Requirements

| Requirement | Why We Need It | Priority |
|-------------|----------------|----------|
| Semantic audio understanding | Understand INTENT, not just sounds. Handle mispronunciations, accents, language learners | Critical |
| Accurate word-level timestamps | Enable precise audio-text alignment, word highlighting, editing | Critical |
| Verbatim transcription | Know exactly what was said, including fillers | Critical |
| Clean text output | Know what user MEANT (corrected, no fillers) | Critical |
| Confidence scores | Know which words are uncertain | High |
| Streaming output | Real-time UI updates, progressive rendering | High |
| Noise robustness | Work with real-world audio, not just clean recordings | High |
| Small model size | Deployable, runnable on reasonable hardware | Medium |
| Open source | Customizable, no API dependency | Medium |

---

## What We Need

### 1. Semantic Audio Encoder

**What**: A neural network that converts audio waveforms into rich, semantic embeddings.

**Why**:
- Traditional ASR encoders (like Whisper) map sounds → phonemes
- They don't UNDERSTAND what the user means
- For language learners who mispronounce, code-switch, use fillers → need understanding
- Gemini works for noisy audio because it UNDERSTANDS, not just transcribes

**Options**:
- WavLM-large (recommended): Semantic, open source, 300M params
- Qwen3-Omni AuT: Most capable, similar to Google's USM
- HuBERT: Semantic, widely used
- MMS: Best for multilingual

**Technical Requirements**:
- Output: Frame-level embeddings (not just sequence-level)
- Frame rate: 50Hz (20ms per frame) - needed for timestamps
- Dimension: 768-1280
- Must preserve temporal structure for alignment

---

### 2. CTC Head for Timestamps

**What**: A simple projection layer that predicts characters/subwords per frame.

**Why**:
- Timestamps must be COMPUTED, not GENERATED (Gemini's problem)
- CTC naturally gives frame → token alignment
- Non-autoregressive = instant, all timestamps at once
- Confidence from softmax probabilities

**Technical Requirements**:
- Input: Encoder output [T, D]
- Output: Logits [T, vocab_size]
- Vocab: Characters (~100) or subwords (~5000)
- Decoding: CTC beam search or greedy

---

### 3. Cross-Attention Decoder

**What**: A transformer decoder that attends to the encoder output.

**Why**:
- Cross-attention lets decoder "see" full audio context
- Can understand intent and produce clean text
- Autoregressive = streamable
- Language model capabilities = fluent output

**Technical Requirements**:
- 6-12 transformer layers
- Cross-attention to encoder output
- Causal self-attention (autoregressive)
- Vocab: BPE ~32K tokens

---

### 4. Alignment Merger

**What**: Logic to combine CTC output (verbatim + timestamps) with Decoder output (clean text).

**Why**:
- CTC gives accurate timestamps but may have errors
- Decoder gives clean text but no direct timestamps
- Need to merge: use CTC timestamps with Decoder text
- Detect corrections, fillers, etc.

**Technical Requirements**:
- Align decoder tokens to CTC tokens
- Handle insertions (decoder adds punctuation)
- Handle deletions (decoder removes fillers)
- Handle substitutions (decoder corrects words)

---

### 5. Streaming Output Format

**What**: JSONL format - one JSON object per word.

**Why**:
- Standard JSON can't stream (must wait for complete object)
- JSONL: each line is independently parseable
- Client can render words in real-time as they arrive
- Widely supported format

**Format**:
```jsonl
{"w": "hello", "s": 0.0, "e": 0.45, "c": 0.98}
{"w": "world", "s": 0.5, "e": 0.95, "c": 0.96}
{"done": true, "text": "hello world"}
```

---

## What We DON'T Need

### 1. Streaming Audio Input

**What**: Processing audio as it arrives in chunks.

**Why We Don't Need It**:
- Adds significant complexity (chunking, context carryover)
- Not required for our use case (full audio available)
- Can be added later if needed
- Batch processing with text streaming is sufficient

**Alternative**: Full audio in → Stream text out

---

### 2. Complex Instructions

**What**: "Summarize this audio", "Extract action items", "What is the sentiment?"

**Why We Don't Need It**:
- These are UNDERSTANDING tasks, not ASR
- Our model UNDERSTANDS audio, but outputs ASR-format data
- Understanding is used internally (for corrections), not exposed as features
- Keeps model focused and simpler

**What We Do Instead**: Model understands context internally to produce better transcriptions

---

### 3. Multi-Modal Input

**What**: Processing images, video alongside audio.

**Why We Don't Need It**:
- Audio-only use case
- Adding modalities increases complexity and model size
- No benefit for pure transcription
- Distracts from core capability

---

### 4. Speech Synthesis / TTS

**What**: Generating audio from text.

**Why We Don't Need It**:
- Completely different task
- Would double model complexity
- Not part of ASR use case

---

### 5. Real-Time Dialogue

**What**: Back-and-forth conversation with the model.

**Why We Don't Need It**:
- Our model is not a chatbot
- Input: audio, Output: structured transcript
- No dialogue, no conversation

---

### 6. Extremely Long Audio (>10 minutes)

**What**: Processing hour-long recordings in one go.

**Why We Don't Need It (v1)**:
- Typical use case: utterances, sentences, short clips
- Long audio can be chunked and processed
- Optimizing for long context adds complexity
- Can be addressed in v2 if needed

**Current Approach**: Process up to ~5 minutes, chunk longer audio

---

## Output Format Specification

### Streaming Format (JSONL)

Each line is a valid JSON object:

```jsonl
{"w": "I", "s": 0.00, "e": 0.20, "c": 0.99}
{"w": "want", "s": 0.20, "e": 0.50, "c": 0.98}
{"w": "to", "s": 0.50, "e": 0.70, "c": 0.97}
{"w": "um", "s": 0.80, "e": 1.00, "c": 0.95, "t": "filler"}
{"w": "go", "s": 1.10, "e": 1.30, "c": 0.98}
{"w": "library", "s": 2.10, "e": 2.60, "c": 0.95, "spoken": "libary", "cor": true}
{"done": true, "verbatim": "I want to um go to the libary", "text": "I want to go to the library", "lang": "en", "dur": 2.6}
```

### Field Definitions

**Per-word fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `w` | string | Yes | Word (cleaned/corrected) |
| `s` | float | Yes | Start time in seconds |
| `e` | float | Yes | End time in seconds |
| `c` | float | Yes | Confidence (0.0 - 1.0) |
| `t` | string | No | Type: "filler", "partial", "foreign" |
| `spoken` | string | No | What was actually spoken (if different from `w`) |
| `cor` | bool | No | True if word was corrected |
| `lang` | string | No | Language code (if different from main) |

**Final summary fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `done` | bool | Yes | Always `true` |
| `verbatim` | string | Yes | Exact transcription (what was said) |
| `text` | string | Yes | Clean transcription (what was meant) |
| `lang` | string | Yes | Primary language code |
| `dur` | float | Yes | Audio duration in seconds |

---

## Technical Constraints

### Hardware Requirements (Inference)

| Configuration | GPU VRAM | CPU RAM | Notes |
|---------------|----------|---------|-------|
| Minimal | 4GB | 8GB | Quantized model |
| Recommended | 8GB | 16GB | FP16 model |
| Full | 16GB | 32GB | FP32, large batch |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Latency to first word | < 1s | After audio received |
| Throughput | > 2x realtime | 10s audio in < 5s |
| WER (clean) | < 10% | Standard test sets |
| WER (noisy) | < 20% | 10dB SNR |
| Timestamp error | < 50ms | Average absolute error |

### Model Size

| Configuration | Size | Notes |
|---------------|------|-------|
| Target | 500-800M | Competitive with Whisper-large |
| Maximum | 1B | Still deployable |
| Minimum viable | 200M | May sacrifice quality |

---

## Use Cases

### Primary Use Case: Language Learner Transcription

**User**: Language learner practicing speaking

**Input**: 
- Audio of learner speaking (possibly with mispronunciations, fillers, code-switching)

**Output**:
- What they said (verbatim, with timestamps)
- What they meant (corrected)
- Filler detection
- Confidence scores

**Why current ASR fails**:
- Whisper: Transcribes sounds literally, doesn't understand intent
- Gemini: Understands but doesn't give accurate timestamps

### Secondary Use Cases

1. **Meeting transcription**: Accurate timestamps for navigation
2. **Podcast editing**: Word-level alignment for editing
3. **Voice notes**: Quick transcription with key phrases
4. **Accessibility**: Captions with timing information

---

## Success Metrics

### Quality Metrics

1. **Word Error Rate (WER)**: < 10% on clean speech
2. **Timestamp Accuracy**: < 50ms average error
3. **Correction Accuracy**: 80%+ of common mispronunciations detected
4. **Filler Detection F1**: > 90%
5. **Noise Robustness**: WER < 20% at 10dB SNR

### Performance Metrics

1. **Latency**: First word < 1s after audio received
2. **Throughput**: > 2x realtime
3. **Memory**: < 8GB GPU VRAM

### User Experience Metrics

1. **Streaming smoothness**: No visible gaps in output
2. **Correction helpfulness**: Users prefer corrected text
3. **Timestamp usefulness**: Can navigate audio by clicking words

---

## Dependencies

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0 | Model framework |
| transformers | >= 4.30 | Encoder loading |
| torchaudio | >= 2.0 | Audio processing |
| numpy | >= 1.24 | Numerical ops |
| soundfile | >= 0.12 | Audio I/O |

### Pretrained Models

| Model | Source | Purpose |
|-------|--------|---------|
| WavLM-large | Microsoft | Encoder option |
| Qwen3-Omni | Alibaba | Encoder option |
| HuBERT-large | Meta | Encoder option |

### Data

| Dataset | Purpose | Status |
|---------|---------|--------|
| VoxLM dataset | Base training | Available |
| LibriSpeech | Evaluation | Available |
| Common Voice | Multilingual | Available |
| MUSAN | Noise augmentation | Available |

---

## Non-Functional Requirements

### Reliability
- Model should handle edge cases gracefully (silence, noise-only, very short audio)
- Fallback behavior defined for uncertain inputs

### Maintainability
- Modular architecture (encoder, CTC, decoder can be swapped)
- Clear interfaces between components
- Comprehensive logging

### Scalability
- Batch processing support
- Easy to add new languages (via encoder fine-tuning)
- Architecture allows future extensions

### Security
- No external API calls during inference
- Model runs fully offline
- No user data leaves the system
