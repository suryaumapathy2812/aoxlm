# CEFR Assessment System - Complete Knowledge Base

**Project Codename:** VoxLM Evolution  
**Version:** 1.0  
**Last Updated:** February 2, 2026  
**Status:** Architecture Finalized, Ready for Implementation  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Context & Background](#2-project-context--background)
3. [The Core Problem: Why LLM Prompting Fails](#3-the-core-problem-why-llm-prompting-fails)
4. [CEFR Framework Deep Dive](#4-cefr-framework-deep-dive)
5. [Audio-LLM Architecture Fundamentals](#5-audio-llm-architecture-fundamentals)
6. [Encoder Taxonomy (2026 State-of-the-Art)](#6-encoder-taxonomy-2026-state-of-the-art)
7. [Semantic Encoder Selection](#7-semantic-encoder-selection)
8. [VoxLM: What Went Wrong & Lessons Learned](#8-voxlm-what-went-wrong--lessons-learned)
9. [The Solution: Classification Heads on Hidden States](#9-the-solution-classification-heads-on-hidden-states)
10. [Multi-Tap Architecture for CEFR](#10-multi-tap-architecture-for-cefr)
11. [Indian English L1 Transfer Patterns](#11-indian-english-l1-transfer-patterns)
12. [Final Architecture Decision](#12-final-architecture-decision)
13. [Labeling Schema & Guidelines](#13-labeling-schema--guidelines)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Technical Specifications](#15-technical-specifications)
16. [Risk Mitigation](#16-risk-mitigation)
17. [References & Resources](#17-references--resources)
18. [Glossary](#18-glossary)

---

## 1. Executive Summary

### What We're Building

A deterministic, production-grade CEFR (Common European Framework of Reference) speaking assessment system for Indian English learners across all major L1 (first language) backgrounds.

### Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Classification System | **4-class** (A1, A2, B1, B2+) | Matches business need, student distribution |
| Scoring Method | **Classification heads** (not LLM generation) | Deterministic, consistent scores |
| Base Encoder | **WavLM-large** (frozen) | Semantic features for classification |
| ASR | **Whisper-large-v3** | Accurate transcription |
| Timestamps | **WhisperX** | Word-level alignment, no duration limits |
| L1 Strategy | **Pan-Indian, pattern-aware** | Don't penalize valid Indian English |
| Coherence Approach | **LLM hidden states** (Phase 3) | Needs semantic understanding |
| Real-time Approach | **Separate lightweight model** | Latency constraint |

### Business Requirements

- **Placement Test**: Accurate level assessment (A1-C2 → simplified to A1/A2/B1/B2+)
- **Progress Tracking**: Show improvement over time
- **Real-time Feedback**: During speaking practice
- **Scale**: 10,000+ students across all Indian states
- **Accuracy**: Within 1 CEFR level of human raters (>90%)
- **Timeline**: MVP in 3 months, full product in 6 months

### Target Accuracy

| Phase | Exact Match | Within 1 Level |
|-------|-------------|----------------|
| Phase 1 (MVP) | 60% | 85% |
| Phase 2 | 70% | 90% |
| Phase 3 | 75% | 95% |

---

## 2. Project Context & Background

### Company Context

- EdTech company focused on English language learning
- Students from ALL Indian states (not just Tamil Nadu)
- Most workforce needs B1/B2 level English proficiency
- Existing VoxLM model (Whisper encoder + Qwen2 LLM) had inconsistency issues

### Developer Context

- Primary developer: Node.js background (React/backend)
- Intermediate ML complexity preferred
- Need for web development analogies to understand ML concepts
- Access to H100 GPUs for training/inference

### Historical Research Sessions

This document synthesizes learnings from multiple research sessions:

1. **Session 1**: Smart ASR with timestamps - Researched Moshi, Qwen3-ASR, Canary-Qwen, WhisperX
2. **Session 2**: Whisper semantic encoding + CEFR assessment methods
3. **Session 3**: Audio-LLM architecture deep dive, VoxLM failure analysis
4. **Session 4**: Architecture review meeting, labeling schema finalization

---

## 3. The Core Problem: Why LLM Prompting Fails

### The Inconsistency Problem

When using LLM prompting for CEFR scoring:

```
Run 1: "Based on the speech, I would rate this as B1..."
Run 2: "The speaker demonstrates A2 level proficiency..."
Run 3: "This appears to be B1, possibly approaching B2..."
```

**Same input → Different outputs every time**

### Why This Happens

LLMs have **TWO types of outputs**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. HIDDEN STATES (Deterministic)                           │
│     ─────────────────────────────────                       │
│     • Internal representations at each layer                │
│     • Contains the model's "understanding"                  │
│     • Same input = Same hidden states ALWAYS                │
│     • Pure matrix multiplication, no randomness             │
│                                                             │
│  2. GENERATED TOKENS (Stochastic)                           │
│     ─────────────────────────────────                       │
│     • Sampled from probability distribution                 │
│     • Even temperature=0 has floating-point variance        │
│     • Same input = Different outputs sometimes              │
│     • Randomness is fundamental to generation               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Solution

**Use hidden states (deterministic) with classification heads, NOT text generation (stochastic).**

```python
# WRONG: LLM Generation (Stochastic)
response = llm.generate("Rate this speech on CEFR scale...")
# Different every time!

# RIGHT: Classification Head (Deterministic)
hidden_states = llm.get_hidden_states(audio_features)
logits = classification_head(hidden_states)
score = argmax(softmax(logits))
# Same every time!
```

---

## 4. CEFR Framework Deep Dive

### The 5 Speaking Dimensions

| Dimension | What It Measures | Primary Signal Source |
|-----------|------------------|----------------------|
| **Fluency** | Speech rate, pauses, hesitations, self-corrections, smoothness | Acoustic (timestamps, pauses) |
| **Range** | Vocabulary breadth, grammatical structures variety | Text (transcript analysis) |
| **Accuracy** | Grammatical correctness, word choice precision | Text (grammar checking) |
| **Phonology** | Pronunciation, intonation, stress patterns, intelligibility | Acoustic (encoder features) |
| **Coherence** | Logical flow, discourse markers, topic development | Semantic (LLM understanding) |

### Grouped by Signal Type

```
LINGUISTIC DIMENSIONS (need text understanding):
├─ Range: Vocabulary analysis
├─ Accuracy: Grammar analysis
└─ Coherence: Discourse structure

DELIVERY DIMENSIONS (need acoustic analysis):
├─ Fluency: Timing patterns
└─ Phonology: Sound patterns

PRAGMATIC DIMENSIONS (need both):
└─ Interaction: Turn-taking, clarification (not in scope for monologue tasks)
```

### 4-Class Simplification (Business Decision)

**Original 6-class**: A1, A2, B1, B2, C1, C2

**Simplified 4-class**: A1, A2, B1, B2+

**Rationale**:
- Industry research shows workforce needs primarily B1/B2
- We have very few C1/C2 students in training data
- B2/C1/C2 distinction is subtle and unnecessary for placement
- 4-class is easier to achieve high accuracy

### CEFR Level Descriptions

#### A1 - Beginner
- **Fluency**: Very slow (<60 WPM), long pauses, isolated words only
- **Range**: <500 word vocabulary, basic needs only
- **Accuracy**: Only memorized phrases accurate, systematic errors
- **Example**: "I... name... [pause]... Ravi. I... Chennai. Work... office."

#### A2 - Elementary
- **Fluency**: Slow but continuous (60-100 WPM), noticeable hesitations
- **Range**: 500-1000 words, everyday topics
- **Accuracy**: Simple structures sometimes correct, frequent errors
- **Example**: "I grew up in Chennai only. It was small small place. I like because everything is there."

#### B1 - Intermediate
- **Fluency**: Reasonable rate (100-130 WPM), can maintain flow
- **Range**: 1000-2000 words, familiar topics with detail
- **Accuracy**: Good control of simple structures, some complex errors
- **Example**: "I grew up in Chennai, which is a big city in South India. When I was young, we lived in a small neighborhood."

#### B2+ - Upper Intermediate and Above
- **Fluency**: Natural rate (>130 WPM), smooth and spontaneous
- **Range**: >2000 words, abstract/specialized terms
- **Accuracy**: High grammatical control, rare errors
- **Example**: "Growing up in Chennai was a fascinating experience that shaped who I am today. The city underwent tremendous transformation during my childhood."

---

## 5. Audio-LLM Architecture Fundamentals

### Web Development Analogies

For developers coming from web development, here's how to think about audio-LLM components:

```
AUDIO-LLM ARCHITECTURE ↔ FULL-STACK WEB APP

Audio Encoder      ↔  Frontend (React)
                      • Processes raw input (audio ↔ user interactions)
                      • Transforms to structured data (features ↔ state)
                      • Multiple layers of processing

Projector/Connector ↔  API Layer / BFF (Backend-for-Frontend)
                      • Translates format between encoder and LLM
                      • Dimensionality adaptation
                      • Protocol translation

LLM (Thinker)      ↔  Backend Server (Node.js/Express)
                      • Core reasoning and logic
                      • Decision making
                      • State management

Classification Heads ↔ Response Handlers / Controllers
                      • Different endpoints for different outputs
                      • /api/fluency, /api/accuracy, etc.
                      • Task-specific processing

Pooling            ↔  Array.reduce()
                      • Combines many items into summary
                      • mean([1,2,3,4]) → 2.5
```

### Layer Types Explained

```javascript
// LINEAR LAYER
// Like: array.map(x => x * weight + bias)
output = input * weights + bias

// SELF-ATTENTION
// Like: array.map((item, i, arr) => weightedSum(arr, relevanceTo(item)))
// Each element attends to all other elements
for each token:
    attention_weights = softmax(query @ keys.T)
    output = attention_weights @ values

// CROSS-ATTENTION
// Like: SQL JOIN between two tables
// Query from one source, Keys/Values from another
query = from_text_tokens
keys, values = from_audio_features
output = attend(query, keys, values)

// FEED-FORWARD NETWORK (FFN/MLP)
// Like: pipe(linear1, relu, linear2)
output = linear2(relu(linear1(input)))

// CONVOLUTION
// Like: array.map((_, i) => processWindow(array.slice(i, i+3)))
// Local receptive field processing

// LAYER NORMALIZATION
// Like: (x - mean) / std
// Stabilizes training

// POOLING
// Like: array.reduce()
// mean_pooling: array.reduce((a,b) => a+b) / length
// max_pooling: array.reduce((a,b) => Math.max(a,b))
```

### Encoder Types Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODER TYPE COMPARISON                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Frame-level (MFCC)     ↔  Plain HTML                                       │
│  • Fixed windows (~20ms)   • No state management                            │
│  • No cross-frame context  • Static content                                 │
│                                                                             │
│  Convolutional          ↔  jQuery                                           │
│  • Local receptive fields  • Local state, event bubbling                    │
│  • Hierarchical patterns   • DOM traversal                                  │
│                                                                             │
│  Transformer (HuBERT)   ↔  React with Context                               │
│  • Self-attention          • Components see sibling state                   │
│  • Full sequence context   • Prop drilling avoided                          │
│                                                                             │
│  LLM-Integrated (Qwen3) ↔  Next.js + Redux                                  │
│  • Cross-modal reasoning   • Full app state                                 │
│  • Language understanding  • Server-side logic                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Encoder Taxonomy (2026 State-of-the-Art)

### By Processing Paradigm

| Type | Context | Examples | Use Case |
|------|---------|----------|----------|
| Frame-level Acoustic | None | MFCC, FilterBank | Feature extraction |
| Convolutional | Local (~100-500ms) | Wav2Vec CNN frontend | Preprocessing |
| Self-Supervised Transformer | Full sequence | HuBERT, WavLM, wav2vec2 | Pre-training |
| Supervised ASR Encoder | Full sequence | Whisper, Canary, AuT | Speech recognition |
| Causal/Streaming | Past only | Conformer-CTC | Real-time |
| LLM-Integrated | Full + reasoning | Qwen2-Audio, Qwen3-Omni | Understanding |

### By What They Capture

```
ACOUSTIC (raw signal, spectral):
└─ MFCC, FilterBank, early conv layers

PHONETIC (sound units, pronunciation):
└─ HuBERT layers 1-4, WavLM early layers

SEMANTIC-PHONETIC (word-like units):
└─ HuBERT layers 4-8, WavLM mid layers

LINGUISTIC (text-aligned meaning):
└─ Whisper late layers, HuBERT layers 8-12

CONTEXTUAL-SEMANTIC (full reasoning):
└─ Qwen3-Omni Thinker, Audio-LLMs
```

### Layer-wise Information Distribution

```
HuBERT/WavLM (12 layers):
├─ Layers 1-4:  Speaker identity, acoustic properties
├─ Layers 4-8:  Phonetic content, pronunciation
├─ Layers 8-12: Semantic content, linguistic meaning
└─ Layer 12+:   High-level linguistic features

Whisper Encoder (32 layers):
├─ Early:  Spectral patterns, acoustic features
├─ Middle: Phoneme sequences, word boundaries
└─ Late:   Text-aligned semantic representations

Qwen3 Thinker (32 layers):
├─ Layers 1-8:   Token processing, basic patterns
├─ Layers 8-16:  Semantic relationships, meaning
├─ Layers 16-24: Discourse structure, argument flow
└─ Layers 24-32: High-level reasoning, synthesis
```

### Key Insight: Whisper is Already Semantic

Unlike HuBERT/WavLM (self-supervised on audio only), **Whisper was trained on (audio, text) pairs**. This means:

- Whisper encoder outputs are **already aligned with linguistic content**
- Late layers encode **what was said**, not just how it sounded
- No need for separate semantic encoder for most tasks
- But: Whisper is **weak on speaker identity** (content-focused training)

---

## 7. Semantic Encoder Selection

### Why WavLM for CEFR Assessment

For extracting hidden states to feed into classification heads, we use **semantic audio encoders** from the Wav2Vec2 family:

| Encoder | Parameters | Hidden Dim | Best For |
|---------|------------|------------|----------|
| **WavLM-large** | 316M | 1024 | General semantic understanding, noise robustness |
| **MMS-300M** | 300M | 1024 | Code-switching, multilingual (1000+ languages) |
| **XLS-R-300M** | 300M | 1024 | Cross-lingual, accents, 128 languages |
| **HuBERT-large** | 316M | 1024 | Baseline semantic encoder |

### Why Not Use Whisper Encoder Directly?

While Whisper produces excellent transcriptions, its encoder is optimized for ASR, not semantic feature extraction:

| Aspect | Whisper Encoder | WavLM/MMS |
|--------|----------------|-----------|
| **Training objective** | ASR (audio → text) | Self-supervised (audio → audio) |
| **Frame rate** | ~25Hz (40ms) | ~50Hz (20ms) - better resolution |
| **Hidden states** | Text-aligned | Phonetic/semantic layers |
| **For classification** | Less studied | Well-established for downstream tasks |

**Our approach**: Use Whisper for transcription, WavLM for hidden states.

### Encoder Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAVLM-LARGE ENCODER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Audio (16kHz)                                                  │
│       │                                                         │
│       ▼                                                         │
│  CNN Feature Extractor (7 conv layers)                          │
│       │                                                         │
│       ▼ [B, T, 512]                                             │
│  Projection Layer                                               │
│       │                                                         │
│       ▼ [B, T, 1024]                                            │
│  24 Transformer Layers                                          │
│       │                                                         │
│       ├─── Layer 1-8:  Acoustic/phonetic features               │
│       ├─── Layer 9-16: Phonetic/semantic features               │
│       └─── Layer 17-24: Semantic/linguistic features            │
│       │                                                         │
│       ▼ [B, T, 1024]                                            │
│  Output: Frame-level embeddings at ~50Hz                        │
│                                                                 │
│  Key Properties:                                                │
│  • 50 frames per second (20ms per frame)                        │
│  • 1024-dimensional embeddings                                  │
│  • Different layers capture different information               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Layer Feature Extraction

Different CEFR dimensions benefit from different encoder layers:

```python
# Example: Multi-tap feature extraction
class MultiLayerFeatureExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def extract(self, audio):
        # Get all hidden states
        outputs = self.encoder(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of 25 tensors
        
        # Different layers for different dimensions
        features = {
            'phonology': hidden_states[4:8].mean(dim=0),    # Early: acoustic
            'fluency': hidden_states[8:16].mean(dim=0),     # Mid: timing
            'semantic': hidden_states[16:24].mean(dim=0),   # Late: meaning
        }
        return features
```

### Code Example

```python
from src.models.encoder import SemanticEncoder

# Load encoder (frozen for inference)
encoder = SemanticEncoder("wavlm-large", freeze=True, device="cuda")

# Extract hidden states
audio = load_audio("student.wav")  # [samples] at 16kHz
hidden_states = encoder(audio)     # [1, T, 1024] where T = samples/320

# Pool for classification
pooled = hidden_states.mean(dim=1)  # [1, 1024]

# Feed to classification head
logits = cefr_head(pooled)  # [1, 4] for A1/A2/B1/B2+
```

---

## 8. VoxLM: What Went Wrong & Lessons Learned

### What VoxLM Was

An attempt to create a custom audio-LLM by combining:
- **Encoder**: Whisper encoder (frozen)
- **LLM**: Qwen2 decoder (frozen)
- **Training**: Fixed instruction, audio→text pairs

### Why It Failed

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOXLM FAILURE ANALYSIS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem: Whisper encoder + Qwen2 LLM were NOT trained together             │
│                                                                             │
│  ┌─────────────┐         ┌─────────────┐                                   │
│  │   Whisper   │  ─?─>   │   Qwen2     │                                   │
│  │   Encoder   │         │    LLM      │                                   │
│  │             │         │             │                                   │
│  │ Outputs:    │         │ Expects:    │                                   │
│  │ Format A    │         │ Format B    │                                   │
│  │ (audio-     │         │ (text       │                                   │
│  │  aligned)   │         │  tokens)    │                                   │
│  └─────────────┘         └─────────────┘                                   │
│                                                                             │
│  No projector/connector was trained to translate!                           │
│  Qwen2 was frozen, so it couldn't learn the new format.                     │
│                                                                             │
│  Web Dev Analogy:                                                           │
│  ─────────────────                                                          │
│  React frontend (Whisper) connected to new Express backend (Qwen2)          │
│  but API contract wasn't established. Frontend sends JSON, backend          │
│  expects XML, no middleware to translate, backend was read-only.            │
│                                                                             │
│  Result:                                                                    │
│  ────────                                                                   │
│  Model learned audio→text mapping (transcription worked)                    │
│  But did NOT learn instruction-following (couldn't do CEFR scoring)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Lessons

1. **Encoder and LLM must be trained together** or have a trained projector
2. **Freezing both components** means no adaptation can happen
3. **Fixed instruction training** doesn't teach instruction-following
4. **Generation-based scoring** will always be inconsistent
5. **Hidden states** are the reliable output, not generated text

### What We Should Have Done

```python
# Option A: Train a projector
projector = nn.Linear(whisper_dim, qwen_dim)
# Train projector while encoder and LLM are frozen

# Option B: Use pre-aligned models
# Qwen2-Audio, Qwen3-Omni already have trained audio→LLM connections

# Option C: Don't use generation for scoring
# Extract hidden states, add classification heads
hidden = model.get_hidden_states(audio)
score = classification_head(hidden)  # Deterministic!
```

---

## 9. The Solution: Classification Heads on Hidden States

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION HEAD APPROACH                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio Input                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ENCODER (Whisper/AuT)                            │   │
│  │                         FROZEN                                       │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                          Hidden States                                      │
│                          (Deterministic!)                                   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CLASSIFICATION HEADS                              │   │
│  │                         TRAINED                                      │   │
│  │                                                                      │   │
│  │    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │    │ Fluency  │  │  Range   │  │ Accuracy │  │ Overall  │          │   │
│  │    │  Head    │  │  Head    │  │   Head   │  │   Head   │          │   │
│  │    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │   │
│  │         │             │             │             │                 │   │
│  │         ▼             ▼             ▼             ▼                 │   │
│  │     A1/A2/B1/B2+  A1/A2/B1/B2+  A1/A2/B1/B2+  A1/A2/B1/B2+         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Key Properties:                                                            │
│  ───────────────                                                            │
│  ✓ Same input → Same output (always)                                        │
│  ✓ No sampling, no randomness                                               │
│  ✓ Interpretable confidence scores                                          │
│  ✓ Fast inference                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Classification Head Implementation

```python
class CEFRClassificationHead(nn.Module):
    """
    MLP head for CEFR dimension classification
    
    Input: Encoder hidden states (pooled)
    Output: 4-class probabilities (A1, A2, B1, B2+)
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=4):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Optional: Ordinal regression for continuous score
        self.ordinal = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values
        ordinal_score = self.ordinal(x).squeeze(-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'ordinal_score': ordinal_score
        }
```

### Pooling Strategies

| Method | Formula | Best For |
|--------|---------|----------|
| CLS Token | `output[0]` | If model has CLS token |
| Mean Pooling | `mean(output, dim=1)` | General purpose, CEFR |
| Max Pooling | `max(output, dim=1)` | Salient feature detection |
| Attention Pooling | `softmax(W @ output) @ output` | Learned importance |
| Last Token | `output[-1]` | Causal/decoder models |

**Recommendation for CEFR**: Mean Pooling or Attention Pooling

---

## 10. Multi-Tap Architecture for CEFR

### The Core Insight

Different CEFR dimensions need features from different layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-TAP ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio → Encoder → Projector → LLM                                          │
│              │                   │                                          │
│         ┌────┴────┐         ┌────┴────┐                                    │
│         │         │         │         │                                     │
│    Layer 4-8  Layer 16   Layer 16  Layer 24                                │
│         │         │         │         │                                     │
│         ▼         ▼         ▼         ▼                                     │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                         │
│    │Phonology│ │ Fluency │ │  Range  │ │Coherence│                         │
│    │  Head   │ │  Head   │ │  Head   │ │  Head   │                         │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘                         │
│                                                                             │
│  Why Different Layers:                                                      │
│  ─────────────────────                                                      │
│  • Encoder early (4-8): Acoustic patterns → Phonology, partial Fluency     │
│  • Encoder late (16+): Linguistic content → Range, partial Accuracy        │
│  • LLM mid (8-16): Semantic meaning → Accuracy, Range                      │
│  • LLM late (16-24): Discourse structure → Coherence                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Coherence is Hard

Coherence requires understanding **relationships across the utterance**:

```
COHERENT:
"I think education is important. For example, it helps people get jobs.
Moreover, educated people contribute to society. Therefore, everyone
should have access to education."

LLM hidden states encode:
├─ "For example" → Elaboration of previous
├─ "Moreover" → Addition to argument
├─ "Therefore" → Conclusion from premises
└─ Smooth transitions = HIGH coherence

INCOHERENT:
"I think education is important. My cat is fluffy. The weather is nice.
Pizza is delicious."

LLM hidden states encode:
├─ "My cat" → Topic shift, unrelated
├─ "The weather" → Another topic shift
├─ "Pizza" → No connection
└─ Abrupt changes = LOW coherence
```

**Acoustic features alone CANNOT capture this.** Need LLM-level understanding.

### Coherence Solution

```
Option A: LLM Hidden States (Recommended)
─────────────────────────────────────────
Transcript → Qwen3 (frozen) → Extract layers 16-24 → Coherence Head

The LLM's hidden states already encode discourse structure.
Just need a small classification head to read it.

Option B: Feature Engineering (Simpler but less accurate)
─────────────────────────────────────────────────────────
Extract from transcript:
├─ Discourse marker count and variety
├─ Sentence embedding similarity flow
├─ Topic word repetition
└─ Coreference chains
```

---

## 11. Indian English L1 Transfer Patterns

### The Challenge

Students come from ALL Indian states, each with different L1 patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INDIAN L1 TRANSFER PATTERNS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HINDI L1 (UP, MP, Bihar, Rajasthan):                                       │
│  ├─ "I am having" (progressive overuse)                                     │
│  ├─ "Delhi itself" (emphasis marker)                                        │
│  ├─ Retroflex consonants                                                    │
│  └─ SOV influence in complex sentences                                      │
│                                                                             │
│  TAMIL L1 (Tamil Nadu):                                                     │
│  ├─ "Chennai only" (emphasis marker)                                        │
│  ├─ "small small" (reduplication)                                           │
│  ├─ Topic-comment structure                                                 │
│  └─ Specific retroflex patterns                                             │
│                                                                             │
│  TELUGU/KANNADA L1 (Andhra, Karnataka):                                     │
│  ├─ Similar to Tamil (Dravidian family)                                     │
│  ├─ Reduplication patterns                                                  │
│  └─ Different retroflex sounds                                              │
│                                                                             │
│  BENGALI L1 (West Bengal):                                                  │
│  ├─ "What you are doing?" (wh-in-situ)                                      │
│  ├─ Aspiration patterns                                                     │
│  └─ No gender agreement transfer                                            │
│                                                                             │
│  MALAYALAM L1 (Kerala):                                                     │
│  ├─ Unique vowel patterns                                                   │
│  └─ "No?" tag questions                                                     │
│                                                                             │
│  MARATHI L1 (Maharashtra):                                                  │
│  ├─ Gender agreement issues                                                 │
│  └─ Aspiration patterns                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Strategy: Don't Penalize Valid Patterns

**Decision**: Treat all common Indian English patterns as VALID, not errors.

```python
# Example: L1 pattern detection (don't penalize these)

INDIAN_ENGLISH_PATTERNS = {
    'reduplication': r'\b(\w+)\s+\1\b',  # "small small"
    'only_emphasis': r'\b\w+\s+only\b',   # "Chennai only"
    'itself_emphasis': r'\b\w+\s+itself\b',  # "Delhi itself"
    'progressive_overuse': r'am\s+having|is\s+having',  # "I am having"
    'wh_in_situ': r'you\s+are\s+\w+ing\s+what',  # "You are doing what?"
}

def is_l1_pattern(text, pattern_type):
    """
    Returns True if text matches a valid L1 transfer pattern.
    These should NOT be penalized in accuracy scoring.
    """
    pattern = INDIAN_ENGLISH_PATTERNS.get(pattern_type)
    if pattern:
        return bool(re.search(pattern, text, re.IGNORECASE))
    return False
```

### L1 Sampling Requirements

```
MINIMUM REPRESENTATION IN TRAINING DATA (200 samples):

Hindi:      50 samples (25%)   ← Largest speaker population
Tamil:      40 samples (20%)   ← Current focus
Telugu:     30 samples (15%)   ← Large population
Bengali:    20 samples (10%)
Marathi:    20 samples (10%)
Kannada:    15 samples (7.5%)
Malayalam:  15 samples (7.5%)
Other:      10 samples (5%)    ← Gujarati, Punjabi, etc.

Track L1 distribution during labeling.
Alert if any L1 < 10% of dataset.
```

---

## 12. Final Architecture Decision

### Model Selection

Based on our requirements (no audio duration constraints, mature ecosystem), we use:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Transcription** | Whisper-large-v3 | Robust, mature, good on accented English |
| **Timestamps** | WhisperX | No duration limits, accurate word-level alignment |
| **Semantic Encoder** | WavLM-large | Pre-trained, 1024-dim, ~50Hz, proven for downstream tasks |
| **Classification** | MLP heads | Deterministic, interpretable, small |
| **Coherence (Phase 3)** | Qwen3-7B | LLM hidden states for discourse understanding |

### Phase 1 Architecture (MVP)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1 ARCHITECTURE (MVP)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio Input (any duration)                                                 │
│       │                                                                     │
│       ├─────────────────────────────────┐                                  │
│       │                                 │                                   │
│       ▼                                 ▼                                   │
│  ┌─────────────────────┐      ┌─────────────────────────────────┐          │
│  │  WHISPER-LARGE-V3   │      │        WAVLM-LARGE              │          │
│  │      (ASR)          │      │    (Semantic Encoder)           │          │
│  │                     │      │         FROZEN                  │          │
│  │  Output:            │      │                                 │          │
│  │  └─ Transcription   │      │  Output:                        │          │
│  └──────────┬──────────┘      │  └─ Hidden states [B, T, 1024]  │          │
│             │                 └─────────────────┬───────────────┘          │
│             ▼                                   │                           │
│  ┌─────────────────────┐                       │                           │
│  │     WHISPERX        │                       │                           │
│  │   (Alignment)       │                       │                           │
│  │                     │                       │                           │
│  │  Output:            │                       │                           │
│  │  └─ Word timestamps │                       │                           │
│  └──────────┬──────────┘                       │                           │
│             │                                   │                           │
│             ▼                                   ▼                           │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │  ACOUSTIC FEATURES  │           │  ENCODER FEATURES   │                 │
│  │     (32 dim)        │           │    (1024 dim)       │                 │
│  │                     │           │                     │                 │
│  │  • WPM              │           │  • Pooled hidden    │                 │
│  │  • Pause ratio      │           │    states (mean)    │                 │
│  │  • Hesitations      │           │                     │                 │
│  │  • L1 pattern flags │           │                     │                 │
│  └──────────┬──────────┘           └──────────┬──────────┘                 │
│             │                                  │                            │
│  ┌──────────┴──────────┐           ┌──────────┴──────────┐                 │
│  │   TEXT FEATURES     │           │                     │                 │
│  │     (24 dim)        │           │                     │                 │
│  │                     │           │                     │                 │
│  │  • TTR              │           │                     │                 │
│  │  • Vocab level      │           │                     │                 │
│  │  • Error patterns   │           │                     │                 │
│  └──────────┬──────────┘           │                     │                 │
│             │                      │                      │                 │
│             └──────────────────────┴──────────────────────┘                 │
│                                    │                                        │
│                         Concatenate (1080 dim)                              │
│                                    │                                        │
│            ┌───────────────────────┼───────────────────────┐               │
│            │                       │                       │                │
│            ▼                       ▼                       ▼                │
│     ┌───────────┐           ┌───────────┐           ┌───────────┐          │
│     │  FLUENCY  │           │   RANGE   │           │  OVERALL  │          │
│     │   HEAD    │           │   HEAD    │           │   HEAD    │          │
│     │  (~100K)  │           │  (~100K)  │           │  (~100K)  │          │
│     └─────┬─────┘           └─────┬─────┘           └─────┬─────┘          │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│       A1/A2/B1/B2+            A1/A2/B1/B2+            A1/A2/B1/B2+          │
│                                                                             │
│  PHASE 1 SCOPE:                                                            │
│  ├─ Dimensions: Fluency, Range, Overall                                    │
│  ├─ Use case: Placement test only                                          │
│  ├─ Latency: 5-10 seconds (acceptable)                                     │
│  ├─ Data needed: 200-300 labeled samples                                   │
│  ├─ Timeline: 8 weeks                                                      │
│  └─ No audio duration constraints                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Full System Architecture (Phase 3)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FULL SYSTEM ARCHITECTURE (PHASE 3)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio Input                                                                │
│       │                                                                     │
│       ├─────────────────────────────────────┐                              │
│       │                                     │                               │
│       ▼                                     ▼                               │
│  ┌────────────────────────┐        ┌────────────────────────┐              │
│  │  WHISPER + WHISPERX    │        │       WAVLM-LARGE      │              │
│  │   (ASR + Timestamps)   │        │   (Semantic Encoder)   │              │
│  │                        │        │                        │              │
│  │  Outputs:              │        │  Output:               │              │
│  │  ├─ Transcription ─────┼──┐     │  └─ Hidden states      │              │
│  │  └─ Word timestamps    │  │     │     [B, T, 1024]       │              │
│  └───────────┬────────────┘  │     └───────────┬────────────┘              │
│              │               │                  │                           │
│              │               ▼                  │                           │
│              │     ┌────────────────────────┐   │                           │
│              │     │      QWEN3-7B          │   │                           │
│              │     │   (For Coherence)      │   │                           │
│              │     │                        │   │                           │
│              │     │  Input: Transcript     │   │                           │
│              │     │  Output: Hidden states │   │                           │
│              │     │         (layers 16-24) │   │                           │
│              │     └───────────┬────────────┘   │                           │
│              │                 │                │                           │
│              ▼                 ▼                ▼                           │
│  ┌────────────────────────┐  ┌──────────┐  ┌────────────────────────┐      │
│  │   ACOUSTIC + TEXT      │  │   LLM    │  │    ENCODER FEATURES    │      │
│  │   FEATURES             │  │ FEATURES │  │                        │      │
│  │                        │  │          │  │  • Pooled WavLM        │      │
│  │  • Fluency metrics     │  │  Discourse│  │    hidden states      │      │
│  │  • Range metrics       │  │  structure│  │                        │      │
│  │  • Accuracy metrics    │  │          │  │                        │      │
│  │  • Phonology metrics   │  │          │  │                        │      │
│  └───────────┬────────────┘  └────┬─────┘  └───────────┬────────────┘      │
│              │                    │                     │                   │
│              └────────────────────┴─────────────────────┘                   │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │           │             │             │           │               │
│         ▼           ▼             ▼             ▼           ▼               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ FLUENCY  │ │  RANGE   │ │ ACCURACY │ │PHONOLOGY │ │COHERENCE │         │
│  │   HEAD   │ │   HEAD   │ │   HEAD   │ │   HEAD   │ │   HEAD   │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
│                                                                             │
│                           ┌──────────┐                                     │
│                           │ OVERALL  │                                     │
│                           │   HEAD   │                                     │
│                           └──────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Decision Summary Table

| Component | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| **ASR** | Whisper-large-v3 | Whisper-large-v3 | Whisper-large-v3 |
| **Timestamps** | WhisperX | WhisperX | WhisperX |
| **Semantic Encoder** | WavLM-large | WavLM-large | WavLM-large |
| **Dimensions** | Fluency, Range, Overall | + Accuracy, Phonology | + Coherence |
| **Features** | Acoustic + Text + Encoder | + Grammar + Pronunciation | + LLM hidden states |
| **Coherence LLM** | - | - | Qwen3-7B |
| **Use Cases** | Placement | Placement | Placement + Progress |
| **Data** | 200 samples | 500 samples | 1000+ samples |
| **Timeline** | Weeks 1-8 | Weeks 9-16 | Weeks 17-24 |

---

## 13. Labeling Schema & Guidelines

### Phase 1 Minimum Schema (11 Required Columns)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sample_id` | string | Unique identifier | "STU001_SES003_001" |
| `audio_file` | string | Path to audio | "audio/stu001_001.wav" |
| `duration_seconds` | float | Audio length | 45.5 |
| `native_language` | string | Student's L1 | "Tamil" |
| `state` | string | Indian state | "Tamil Nadu" |
| `task_type` | string | Task category | "question_response" |
| `task_prompt` | string | Question asked | "Where did you grow up?" |
| `audio_quality` | string | Quality flag | "good" |
| `overall_level` | string | CEFR level | "A2" |
| `fluency_level` | string | Fluency CEFR | "A2" |
| `range_level` | string | Range CEFR | "A2" |
| `rater_id` | string | Rater identifier | "rater_01" |

### Optional but Recommended

| Column | Type | Description |
|--------|------|-------------|
| `transcript` | string | Full transcription |
| `rating_confidence` | string | "high", "medium", "low" |
| `notes` | string | Rater observations |

### Inter-Rater Reliability (20% of samples)

| Column | Type | Description |
|--------|------|-------------|
| `rater2_id` | string | Second rater |
| `rater2_overall` | string | Second rating |
| `rater2_fluency` | string | Second fluency |
| `rater2_range` | string | Second range |
| `adjudicated_level` | string | Final decision |
| `adjudicator` | string | Who resolved |

### Labeling Dropdown Values

```
native_language: [Tamil, Hindi, Telugu, Bengali, Marathi, 
                  Kannada, Malayalam, Gujarati, Punjabi, Odia, Other]

state: [All 28 Indian states + 8 UTs]

task_type: [question_response, monologue, picture_description, read_aloud]

audio_quality: [good, acceptable, poor, unusable]

overall_level: [A1, A2, B1, B2+]

fluency_level: [A1, A2, B1, B2+]

range_level: [A1, A2, B1, B2+]

rating_confidence: [high, medium, low]
```

### Adjudication Process

```
1. DOUBLE-RATE 20% of samples (every 5th)
   └─ Two independent raters, blind to each other

2. MEASURE AGREEMENT
   ├─ Exact agreement target: >70%
   ├─ Adjacent agreement target: >90%
   └─ Cohen's Kappa target: >0.6

3. HANDLE DISAGREEMENTS
   ├─ Same level → Use agreed level
   ├─ Differ by 1 (A2 vs B1) → Senior rater adjudicates
   └─ Differ by 2+ (A1 vs B1) → Third rater + discussion

4. WEEKLY CALIBRATION SESSIONS
   └─ All raters review 5 samples together
```

---

## 14. Implementation Roadmap

### Phase 1: MVP (Weeks 1-8)

```
WEEK 1-2: SETUP
├─ Create labeling Google Sheet
├─ Recruit and train raters
├─ Set up audio collection pipeline
├─ Write labeling guidelines
└─ Create L1 patterns reference

WEEK 2-4: DATA COLLECTION
├─ Collect 200 audio samples
├─ Label with Fluency + Range + Overall
├─ Double-rate 20% for IRR
├─ Calibration session at Week 3
└─ Transcribe all samples

WEEK 5-6: MODEL DEVELOPMENT
├─ Set up Whisper inference pipeline
├─ Implement feature extractors
│   ├─ Acoustic features (pause, WPM, reps)
│   └─ Text features (TTR, vocab level)
├─ Train classification heads
└─ Validate on held-out 20%

WEEK 7-8: EVALUATION & ITERATION
├─ Calculate accuracy metrics
├─ Error analysis
├─ Iterate on features/heads
└─ Package for deployment
```

### Phase 2: Expansion (Weeks 9-16)

```
WEEK 9-10: DATA EXPANSION
├─ Expand to 500 samples
├─ Add Accuracy and Phonology labels
└─ More L1 diversity

WEEK 11-12: NEW FEATURES
├─ Grammar error detection
├─ Word-level pronunciation features
└─ Accuracy and Phonology heads

WEEK 13-14: MULTI-TASK TRAINING
├─ Joint training of all heads
├─ Multi-task loss balancing
└─ End-to-end fine-tuning (optional)

WEEK 15-16: EVALUATION
├─ Full 4-dimension evaluation
├─ Compare to human raters
└─ Production readiness
```

### Phase 3: Advanced (Weeks 17-24)

```
WEEK 17-18: COHERENCE
├─ Add Coherence labels to data
├─ Integrate Qwen3 for hidden states
└─ Train Coherence head

WEEK 19-20: REAL-TIME
├─ Implement streaming Whisper
├─ Lightweight feedback model
└─ WebSocket API

WEEK 21-24: PRODUCTION
├─ Model optimization (quantization)
├─ API development
├─ Dashboard UI
└─ Load testing and deployment
```

---

## 15. Technical Specifications

### Model Specifications

| Component | Model | Parameters | Memory | Inference Time |
|-----------|-------|------------|--------|----------------|
| Audio Encoder | Whisper-large-v3 | 1.5B | ~6GB | 2-3s (30s audio) |
| Classification Heads | Custom MLP | ~100K each | ~1MB | <10ms |
| Text LLM (Phase 3) | Qwen3-7B | 7B | ~14GB | 1-2s |

### Feature Dimensions

| Feature Set | Dimensions | Source |
|-------------|------------|--------|
| Acoustic Features | 32 | Timestamps + audio analysis |
| Text Features | 24 | Transcript analysis |
| Encoder Features | 512 | Whisper layer 16 (pooled) |
| **Total (Phase 1)** | **568** | Concatenated |
| LLM Features (Phase 3) | 1024 | Qwen3 layer 20 (pooled) |

### Training Configuration

```python
# Phase 1 Training Config
config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 50,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'loss': 'CrossEntropyLoss',
    'weight_decay': 0.01,
    'dropout': 0.2,
    
    'frozen': ['whisper_encoder'],  # 1.5B params
    'trainable': ['classification_heads'],  # ~300K params
}
```

### Hardware Requirements

| Phase | GPU | Memory | Storage |
|-------|-----|--------|---------|
| Phase 1 | H100 (1x) | 80GB | 100GB |
| Phase 2 | H100 (1x) | 80GB | 200GB |
| Phase 3 | H100 (2x) | 160GB | 500GB |

### API Specifications

```yaml
# Placement Test API
POST /api/v1/assess
Input:
  - audio: binary (WAV/MP3)
  - task_type: string
  - task_prompt: string
Output:
  - overall: {level: "A2", confidence: 0.85, score: 0.45}
  - fluency: {level: "A2", confidence: 0.82, reasons: [...]}
  - range: {level: "A2", confidence: 0.88, reasons: [...]}
  - transcript: string
  - word_timestamps: [{word, start, end}, ...]
Latency: <10s

# Real-time Feedback API (Phase 3)
WebSocket /api/v1/realtime
Input: streaming audio chunks (500ms)
Output: streaming feedback
  - fluency_hint: "Good pace" | "Slow down" | "Speed up"
  - current_wpm: number
Latency: <500ms
```

---

## 16. Risk Mitigation

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Not enough labeled data | Medium | High | Start with 200, rule-based fallback |
| L1 patterns misclassified | High | Medium | Tamil-aware guidelines, pattern detector |
| Coherence too hard | High | Medium | Defer to Phase 3, ship without |
| Real-time too slow | Medium | High | Separate lightweight model |
| B2+ accuracy poor | High | Medium | Augment with public corpora |
| Inter-rater disagreement | Medium | Medium | Calibration sessions, adjudication |
| Model overfits to train data | Medium | High | Cross-validation, regularization |

### Fallback Plans

```
IF ML doesn't achieve accuracy targets:

FALLBACK 1: Rule-Based Scoring
──────────────────────────────
Use thresholds on engineered features:
- WPM < 60 → A1 fluency
- WPM 60-100 → A2 fluency
- WPM 100-130 → B1 fluency
- WPM > 130 → B2+ fluency

Not as accurate, but deterministic and explainable.

FALLBACK 2: Human-in-the-Loop
─────────────────────────────
- Low confidence scores → flagged for human review
- High confidence scores → auto-graded
- Hybrid system until model improves

FALLBACK 3: Ensemble with LLM
─────────────────────────────
- Classification head + LLM prompting
- Average scores when they agree
- Human review when they disagree
```

---

## 17. References & Resources

### Models

| Model | Source | Use Case |
|-------|--------|----------|
| Whisper-large-v3 | `openai/whisper-large-v3` | Transcription |
| WhisperX | `m-bain/whisperX` | Word-level timestamps |
| WavLM-large | `microsoft/wavlm-large` | Semantic encoder for classification |
| MMS-300M | `facebook/mms-300m` | Alternative encoder (multilingual) |
| Qwen3-7B | `Qwen/Qwen3-7B` | Coherence (Phase 3 only) |

### Datasets (for augmentation)

| Dataset | Description | Access |
|---------|-------------|--------|
| TEEMI | Taiwan English learners (A1-B2) | Academic |
| Linguaskill | Cambridge assessment data | Commercial |
| Speak & Improve 2025 | Cambridge open corpus | Academic |
| CEFR-SP | Spanish speakers, CEFR labeled | Academic |

### Research Papers

1. **Whisper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
2. **WavLM**: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (Chen et al., 2022)
3. **WhisperX**: "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio" (Bain et al., 2023)
4. **CEFR Assessment**: "Automated CEFR Proficiency Assessment" (various)
5. **Indian English**: "Phonological Features of Indian English" (linguistics research)

### Tools

| Tool | Purpose | Install |
|------|---------|---------|
| transformers | Model loading | `pip install transformers` |
| torchaudio | Audio processing | `pip install torchaudio` |
| librosa | Audio features | `pip install librosa` |
| language-tool-python | Grammar checking | `pip install language-tool-python` |
| montreal-forced-aligner | Word alignment | `conda install -c conda-forge montreal-forced-aligner` |

---

## 18. Glossary

| Term | Definition |
|------|------------|
| **CEFR** | Common European Framework of Reference for Languages. Standard for language proficiency (A1-C2). |
| **L1** | First language / native language / mother tongue. |
| **L2** | Second language / language being learned. |
| **L1 Transfer** | Patterns from native language appearing in L2 speech. |
| **ASR** | Automatic Speech Recognition. Converting audio to text. |
| **Encoder** | Neural network that converts raw input to feature representations. |
| **Hidden States** | Internal representations at each layer of a neural network. |
| **Classification Head** | Small neural network that takes features and outputs class probabilities. |
| **Pooling** | Reducing variable-length sequences to fixed-length vectors. |
| **MLP** | Multi-Layer Perceptron. Simple feed-forward neural network. |
| **Frozen** | Model parameters that are not updated during training. |
| **Fine-tuning** | Updating pre-trained model parameters on new data. |
| **WPM** | Words per minute. Measure of speech rate. |
| **TTR** | Type-Token Ratio. Vocabulary diversity measure. |
| **IRR** | Inter-Rater Reliability. Agreement between human raters. |
| **MoE** | Mixture of Experts. Architecture with multiple specialized sub-networks. |
| **RoPE** | Rotary Position Embedding. Position encoding method for transformers. |
| **Reduplication** | Repeating words for emphasis ("small small"). Common in Indian English. |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2, 2026 | Initial comprehensive documentation |

---

## How to Use This Document

### For Continuing Development

1. Read Sections 1-4 for context and requirements
2. Read Section 12 for the final architecture decision
3. Read Section 13 for labeling schema
4. Read Section 14 for implementation roadmap

### For Onboarding New LLMs/Tools

1. Share the entire document as context
2. Highlight Section 8 (VoxLM lessons) to avoid repeating mistakes
3. Reference Section 15 for technical specifications

### For Technical Deep-Dives

1. Section 5-7 for audio-LLM architecture fundamentals
2. Section 9-10 for classification head approach
3. Section 11 for Indian English specifics

---

*This document serves as the single source of truth for the CEFR Assessment System project. All architectural decisions, technical learnings, and implementation details are captured here for continuity across sessions and tools.*