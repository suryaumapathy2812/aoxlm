# Research: Phonology Metrics & Transcription Alternatives

## Current State

Our phonology assessment uses:
1. **Whisper word confidence** - Measures transcription certainty, not pronunciation quality
2. **Basic pitch/energy** - From librosa, but not properly utilized

### Problems Identified
- Gemini noted "flat intonation" and "staccato rhythm" that we're not capturing
- Confidence scores don't reflect actual pronunciation quality
- Missing: stress patterns, intonation contours, rhythm regularity

---

## Part 1: Improved Phonology Metrics

### 1.1 Pitch Contour Analysis (Intonation)

**What it measures:** Rising/falling patterns that convey meaning and naturalness

**Implementation with librosa:**
```python
import librosa
import numpy as np

def analyze_intonation(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract F0 using pyin (probabilistic YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C6'),  # ~1047 Hz
        sr=sr,
    )
    
    f0_voiced = f0[voiced_flag]
    
    # Key metrics
    metrics = {
        "pitch_mean": np.nanmean(f0_voiced),
        "pitch_std": np.nanstd(f0_voiced),
        "pitch_range": np.nanmax(f0_voiced) - np.nanmin(f0_voiced),
        "pitch_cv": np.nanstd(f0_voiced) / np.nanmean(f0_voiced),  # Coefficient of variation
    }
    
    # Intonation assessment
    cv = metrics["pitch_cv"]
    if cv < 0.10:
        metrics["intonation_pattern"] = "monotone"  # Flat, needs work
    elif cv < 0.15:
        metrics["intonation_pattern"] = "limited"   # Some variation
    elif cv <= 0.35:
        metrics["intonation_pattern"] = "natural"   # Good range
    else:
        metrics["intonation_pattern"] = "erratic"   # Too much variation
    
    return metrics
```

**CEFR Benchmarks for Intonation:**
- A1-A2: CV < 0.12 (monotone, limited expression)
- B1: CV 0.12-0.20 (developing natural patterns)
- B2+: CV 0.15-0.35 (natural, expressive)

---

### 1.2 Rhythm Analysis (nPVI - normalized Pairwise Variability Index)

**What it measures:** How regular/irregular the timing between syllables is

**Background:**
- English is a "stress-timed" language (irregular syllable duration)
- Spanish/French are "syllable-timed" (regular duration)
- nPVI captures this: higher = more stress-timed

**Implementation:**
```python
def calculate_npvi(durations):
    """
    Calculate normalized Pairwise Variability Index.
    
    nPVI = 100 * (sum of |d_k - d_{k+1}| / average(d_k, d_{k+1})) / (n-1)
    
    Higher values = more variable (stress-timed like English)
    Lower values = more regular (syllable-timed)
    
    Native English: nPVI ~55-65
    L2 English learners: often lower (~40-50) - "syllable-timed" accent
    """
    if len(durations) < 2:
        return 0.0
    
    n = len(durations)
    total = 0.0
    
    for k in range(n - 1):
        d_k = durations[k]
        d_k1 = durations[k + 1]
        avg = (d_k + d_k1) / 2
        if avg > 0:
            total += abs(d_k - d_k1) / avg
    
    return 100 * total / (n - 1)

def analyze_rhythm(words):
    """
    Analyze speech rhythm from word timestamps.
    
    Args:
        words: List of dicts with 'start', 'end' keys
    
    Returns:
        Rhythm metrics including nPVI
    """
    # Calculate word durations
    durations = [w['end'] - w['start'] for w in words if w.get('end') and w.get('start')]
    
    if len(durations) < 3:
        return {"npvi": 0, "rhythm_regularity": 0.5}
    
    npvi = calculate_npvi(durations)
    
    # Calculate inter-onset intervals (IOI)
    ioi = []
    for i in range(1, len(words)):
        interval = words[i]['start'] - words[i-1]['start']
        if interval > 0:
            ioi.append(interval)
    
    ioi_npvi = calculate_npvi(ioi) if len(ioi) >= 2 else 0
    
    return {
        "npvi_duration": npvi,           # Word duration variability
        "npvi_ioi": ioi_npvi,            # Inter-onset interval variability
        "mean_word_duration": np.mean(durations),
        "rhythm_score": _npvi_to_score(npvi),
    }

def _npvi_to_score(npvi):
    """Convert nPVI to 0-100 score for CEFR."""
    # Native English ~55-65
    # Learners often ~35-50
    if npvi >= 55:
        return 90  # Native-like
    elif npvi >= 50:
        return 75  # Good rhythm
    elif npvi >= 45:
        return 60  # Developing
    elif npvi >= 40:
        return 45  # Syllable-timed accent
    else:
        return 30  # Very regular/choppy
```

**CEFR Benchmarks for Rhythm (nPVI):**
- A1-A2: nPVI < 45 (choppy, syllable-timed)
- B1: nPVI 45-52 (developing stress-timing)
- B2+: nPVI 52-65 (natural English rhythm)

---

### 1.3 Stress Pattern Analysis

**What it measures:** Whether stressed syllables are properly emphasized

**Implementation approach:**
```python
def analyze_stress_patterns(words, audio_path):
    """
    Analyze stress patterns by comparing energy/duration of syllables.
    
    In stressed syllables:
    - Higher energy (louder)
    - Longer duration
    - Higher/more variable pitch
    """
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Get RMS energy per word
    word_energies = []
    for w in words:
        start_sample = int(w['start'] * sr)
        end_sample = int(w['end'] * sr)
        segment = y[start_sample:end_sample]
        if len(segment) > 0:
            rms = np.sqrt(np.mean(segment ** 2))
            word_energies.append({
                'word': w['word'],
                'energy': rms,
                'duration': w['end'] - w['start'],
            })
    
    # Calculate energy variation (content words should be louder)
    energies = [w['energy'] for w in word_energies]
    energy_cv = np.std(energies) / np.mean(energies) if energies else 0
    
    # Higher CV = better stress differentiation
    return {
        "energy_cv": energy_cv,
        "stress_differentiation": "good" if energy_cv > 0.3 else "weak",
    }
```

---

## Part 2: Transcription Alternatives

### 2.1 Current: faster-whisper (large-v3)

**Pros:**
- Fast (CTranslate2 optimized)
- Good accuracy on Indian English
- Word-level timestamps
- Confidence scores (limited usefulness)

**Cons:**
- Confidence doesn't reflect pronunciation quality
- May normalize/correct speech (hides errors)
- Limited word-level confidence granularity

---

### 2.2 Option A: MMS (Massively Multilingual Speech)

**Model:** `facebook/mms-1b-all`

**Pros:**
- 1000+ languages support
- Better for code-switching
- CTC-based = frame-level alignments possible
- More granular confidence scores

**Cons:**
- Larger model (1B params)
- May need language adapter switching
- Less tested on Indian English specifically

**Code example:**
```python
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# Set language to English
processor.tokenizer.set_target_lang("eng")
model.load_adapter("eng")

# Transcribe
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Get confidence from softmax probabilities
probs = torch.softmax(logits, dim=-1)
max_probs = torch.max(probs, dim=-1).values
frame_confidences = max_probs[0].numpy()  # Per-frame confidence!

# Decode
ids = torch.argmax(logits, dim=-1)[0]
transcription = processor.decode(ids)
```

---

### 2.3 Option B: Whisper + wav2vec2 Hybrid

Use Whisper for transcription, wav2vec2 for pronunciation scoring.

**Approach:**
1. Whisper → transcription + timestamps
2. wav2vec2 → frame-level features
3. Align and score pronunciation

**Pros:**
- Best of both worlds
- Can use specialized pronunciation models

---

### 2.4 Option C: Gemini API (for comparison)

Use Gemini for ground-truth assessment comparison.

**Pros:**
- Most accurate assessment (multimodal understanding)
- Can provide detailed feedback

**Cons:**
- API cost
- Latency
- Not for production (but good for validation)

---

## Recommended Implementation Plan

### Phase 1: Improve Phonology (1-2 days)

1. **Add pitch contour analysis** to phonology.py
   - Use librosa.pyin()
   - Calculate CV for intonation assessment
   - Add "monotone" detection

2. **Add rhythm analysis (nPVI)**
   - Calculate from word timestamps
   - Compare to native English benchmarks

3. **Improve stress detection**
   - Use energy variation across words
   - Content vs function word emphasis

### Phase 2: Test MMS Transcription (1 day)

1. Create `src/transcription/mms.py`
2. Run comparison on sample audio
3. Compare:
   - Transcription accuracy
   - Frame-level confidence granularity
   - Speed

### Phase 3: Validate & Decide (1 day)

1. Run both systems on 10-20 samples
2. Compare with Gemini assessments
3. Decide on best approach

---

## References

1. nPVI: Grabe & Low (2002) - "Durational Variability in Speech"
2. MMS: Pratap et al. (2023) - "Scaling Speech Technology to 1000+ Languages"
3. Whisper: Radford et al. (2022) - "Robust Speech Recognition via Large-Scale Weak Supervision"
4. librosa: McFee et al. (2015) - Audio analysis library
