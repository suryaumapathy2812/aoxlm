"""
Phonology Assessment Module for CEFR Speech Assessment

Analyzes pronunciation quality using Whisper confidence scores and audio features.

CEFR Phonology Criteria:
- A1: Pronunciation may be unclear, strong L1 influence
- A2: Generally clear pronunciation, some errors don't impede understanding
- B1: Clear pronunciation, occasional errors, natural intonation developing
- B2+: Clear, natural pronunciation with appropriate stress and intonation

Features Extracted:
- Word confidence scores (from Whisper)
- Pitch variation (monotone vs expressive)
- Speech energy patterns
- Rhythm regularity
- Pronunciation consistency

Usage:
    >>> from src.features.phonology import PhonologyAssessor
    >>> assessor = PhonologyAssessor()
    >>> result = assessor.assess(words_with_confidence, audio_path="audio.mp3")
    >>> print(result.level)  # "B1"
    >>> print(result.score)  # 65.5

Example:
    >>> extractor = PhonologyFeatureExtractor()
    >>> features = extractor.extract(words, audio_path="audio.mp3")
    >>> print(features.mean_confidence)  # 0.85
    >>> print(features.pitch_variation)  # 45.2
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import pure calculation functions
from .calculations import score_to_cefr_level


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PhonologyFeatures:
    """Features extracted for phonology assessment."""

    # Confidence-based (from Whisper)
    mean_confidence: float = 0.0  # Average word confidence
    min_confidence: float = 0.0  # Minimum confidence (worst word)
    confidence_std: float = 0.0  # Consistency of pronunciation
    low_confidence_ratio: float = 0.0  # % of words with confidence < 0.7
    low_confidence_words: List[str] = field(default_factory=list)

    # Pitch-based (from audio) - INTONATION
    pitch_mean: float = 0.0  # Average pitch (Hz)
    pitch_std: float = 0.0  # Pitch variation
    pitch_range: float = 0.0  # Max - min pitch
    pitch_cv: float = 0.0  # Coefficient of variation (key for intonation)
    intonation_pattern: str = "unknown"  # monotone/limited/natural/erratic

    # Energy-based - STRESS
    energy_mean: float = 0.0  # Average energy
    energy_std: float = 0.0  # Energy variation
    energy_cv: float = 0.0  # Coefficient of variation (key for stress)
    stress_pattern: str = "unknown"  # weak/moderate/good

    # Rhythm-based - nPVI
    npvi_duration: float = 0.0  # nPVI for word durations
    npvi_ioi: float = 0.0  # nPVI for inter-onset intervals
    rhythm_pattern: str = "unknown"  # choppy/developing/natural
    speech_rate_consistency: float = 0.0  # How consistent is the pace
    mean_word_duration: float = 0.0  # Average word duration

    # Legacy (kept for compatibility)
    pitch_variation_ratio: float = 0.0  # Same as pitch_cv
    energy_variation_ratio: float = 0.0  # Same as energy_cv
    syllable_timing_regularity: float = 0.0  # Rhythm regularity

    # Meta
    word_count: int = 0
    duration: float = 0.0
    audio_analyzed: bool = False

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Limit low confidence words in output
        result["low_confidence_words"] = result["low_confidence_words"][:10]
        return result


@dataclass
class PhonologyScore:
    """Phonology assessment result."""

    score: float  # 0-100
    level: str  # CEFR level
    confidence: float  # 0-1
    features: PhonologyFeatures = field(default_factory=PhonologyFeatures)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "level": self.level,
            "confidence": self.confidence,
            "features": self.features.to_dict(),
            "sub_scores": self.sub_scores,
            "feedback": self.feedback,
        }


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================


class PhonologyFeatureExtractor:
    """
    Extract phonology features from transcription and audio.

    Uses:
    1. Whisper word-level confidence scores (primary signal)
    2. Audio features via librosa (pitch, energy) for additional context

    Example:
        >>> extractor = PhonologyFeatureExtractor()
        >>> features = extractor.extract(words, audio_path="audio.mp3")
        >>> features.mean_confidence
        0.85
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.7,
        sample_rate: int = 16000,
    ):
        """
        Initialize extractor.

        Args:
            low_confidence_threshold: Confidence below this = pronunciation issue
            sample_rate: Audio sample rate for analysis
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.sample_rate = sample_rate

    def extract(
        self,
        words: List[Dict],
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> PhonologyFeatures:
        """
        Extract phonology features.

        Args:
            words: List of dicts with "word", "start", "end", and optionally "confidence"
            audio_path: Path to audio file (optional, for pitch/energy analysis)
            audio_array: Pre-loaded audio array (optional)

        Returns:
            PhonologyFeatures
        """
        if not words:
            return PhonologyFeatures()

        # Extract confidence-based features
        confidence_features = self._extract_confidence_features(words)

        # Extract rhythm features (nPVI) from word timestamps
        rhythm_features = self._extract_rhythm_features(words)

        # Extract audio features if available
        audio_features = {}
        audio_analyzed = False

        if audio_path or audio_array is not None:
            try:
                audio_features = self._extract_audio_features(
                    audio_path=audio_path,
                    audio_array=audio_array,
                )
                audio_analyzed = True
            except Exception as e:
                # Audio analysis failed, continue with confidence only
                print(f"Warning: Audio feature extraction failed: {e}")

        # Calculate duration
        duration = 0.0
        if words:
            duration = words[-1].get("end", 0) - words[0].get("start", 0)

        # Get pitch CV and intonation pattern
        pitch_cv = audio_features.get("pitch_cv", 0.0)
        intonation_pattern = self._classify_intonation(pitch_cv)

        # Get energy CV and stress pattern
        energy_cv = audio_features.get("energy_cv", 0.0)
        stress_pattern = self._classify_stress(energy_cv)

        return PhonologyFeatures(
            # Confidence features
            mean_confidence=confidence_features.get("mean", 0.0),
            min_confidence=confidence_features.get("min", 0.0),
            confidence_std=confidence_features.get("std", 0.0),
            low_confidence_ratio=confidence_features.get("low_ratio", 0.0),
            low_confidence_words=confidence_features.get("low_words", []),
            # Pitch/Intonation
            pitch_mean=audio_features.get("pitch_mean", 0.0),
            pitch_std=audio_features.get("pitch_std", 0.0),
            pitch_range=audio_features.get("pitch_range", 0.0),
            pitch_cv=pitch_cv,
            intonation_pattern=intonation_pattern,
            # Energy/Stress
            energy_mean=audio_features.get("energy_mean", 0.0),
            energy_std=audio_features.get("energy_std", 0.0),
            energy_cv=energy_cv,
            stress_pattern=stress_pattern,
            # Rhythm (nPVI)
            npvi_duration=rhythm_features.get("npvi_duration", 0.0),
            npvi_ioi=rhythm_features.get("npvi_ioi", 0.0),
            rhythm_pattern=rhythm_features.get("rhythm_pattern", "unknown"),
            speech_rate_consistency=confidence_features.get("rate_consistency", 0.0),
            mean_word_duration=rhythm_features.get("mean_word_duration", 0.0),
            # Legacy compatibility
            pitch_variation_ratio=pitch_cv,
            energy_variation_ratio=energy_cv,
            syllable_timing_regularity=confidence_features.get(
                "timing_regularity", 0.0
            ),
            # Meta
            word_count=len(words),
            duration=duration,
            audio_analyzed=audio_analyzed,
        )

    def _classify_intonation(self, pitch_cv: float) -> str:
        """Classify intonation pattern based on pitch coefficient of variation."""
        if pitch_cv == 0:
            return "unknown"
        elif pitch_cv < 0.10:
            return "monotone"  # Flat, needs work
        elif pitch_cv < 0.15:
            return "limited"  # Some variation, developing
        elif pitch_cv <= 0.35:
            return "natural"  # Good, expressive
        else:
            return "erratic"  # Too much variation

    def _classify_stress(self, energy_cv: float) -> str:
        """Classify stress pattern based on energy coefficient of variation."""
        if energy_cv == 0:
            return "unknown"
        elif energy_cv < 0.25:
            return "weak"  # Little differentiation between stressed/unstressed
        elif energy_cv <= 0.50:
            return "moderate"  # Some stress differentiation
        else:
            return "good"  # Clear stress patterns

    def _extract_confidence_features(self, words: List[Dict]) -> Dict:
        """Extract features from word confidence scores."""
        # Get confidence scores (default to 1.0 if not available)
        confidences = []
        low_words = []

        for w in words:
            conf = w.get("confidence", w.get("probability", 1.0))
            if conf is None:
                conf = 1.0
            confidences.append(conf)

            if conf < self.low_confidence_threshold:
                low_words.append(w.get("word", ""))

        if not confidences:
            return {
                "mean": 0.0,
                "min": 0.0,
                "std": 0.0,
                "low_ratio": 0.0,
                "low_words": [],
            }

        confidences = np.array(confidences)

        # Calculate rate consistency (based on word timing)
        rate_consistency = self._calculate_rate_consistency(words)
        timing_regularity = self._calculate_timing_regularity(words)

        return {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "std": float(np.std(confidences)),
            "low_ratio": len(low_words) / len(words) if words else 0.0,
            "low_words": low_words,
            "rate_consistency": rate_consistency,
            "timing_regularity": timing_regularity,
        }

    def _extract_rhythm_features(self, words: List[Dict]) -> Dict:
        """
        Extract rhythm features including nPVI (normalized Pairwise Variability Index).

        nPVI measures the variability in duration between successive elements.
        - Higher nPVI = more stress-timed (like native English, ~55-65)
        - Lower nPVI = more syllable-timed (like Spanish, or L2 English learners ~35-45)

        References:
        - Grabe & Low (2002): "Durational Variability in Speech"
        """
        if len(words) < 3:
            return {
                "npvi_duration": 0.0,
                "npvi_ioi": 0.0,
                "mean_word_duration": 0.0,
                "rhythm_pattern": "unknown",
            }

        # Calculate word durations
        durations = []
        for w in words:
            start = w.get("start", 0)
            end = w.get("end", 0)
            if end > start:
                durations.append(end - start)

        # Calculate inter-onset intervals (IOI)
        ioi = []
        for i in range(1, len(words)):
            prev_start = words[i - 1].get("start", 0)
            curr_start = words[i].get("start", 0)
            interval = curr_start - prev_start
            if interval > 0:
                ioi.append(interval)

        # Calculate nPVI for durations
        npvi_duration = self._calculate_npvi(durations)

        # Calculate nPVI for IOI
        npvi_ioi = self._calculate_npvi(ioi)

        # Mean word duration
        mean_duration = np.mean(durations) if durations else 0.0

        # Classify rhythm pattern based on nPVI
        # Native English speakers: nPVI ~55-65
        # L2 learners often have lower values (~35-50)
        rhythm_pattern = self._classify_rhythm(npvi_duration)

        return {
            "npvi_duration": float(npvi_duration),
            "npvi_ioi": float(npvi_ioi),
            "mean_word_duration": float(mean_duration),
            "rhythm_pattern": rhythm_pattern,
        }

    def _calculate_npvi(self, durations: List[float]) -> float:
        """
        Calculate normalized Pairwise Variability Index (nPVI).

        Formula: nPVI = 100 * (1/(n-1)) * sum(|d_k - d_{k+1}| / ((d_k + d_{k+1})/2))

        Args:
            durations: List of durations (word lengths or intervals)

        Returns:
            nPVI value (typically 30-70 for speech)
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

    def _classify_rhythm(self, npvi: float) -> str:
        """
        Classify rhythm pattern based on nPVI.

        CEFR Benchmarks:
        - A1-A2: nPVI < 45 (choppy, syllable-timed)
        - B1: nPVI 45-52 (developing stress-timing)
        - B2+: nPVI 52-65 (natural English rhythm)
        """
        if npvi == 0:
            return "unknown"
        elif npvi < 40:
            return "choppy"  # Very syllable-timed, staccato
        elif npvi < 50:
            return "developing"  # Moving toward stress-timing
        elif npvi <= 65:
            return "natural"  # Good English rhythm
        else:
            return "variable"  # Highly variable (could be expressive or inconsistent)

    def _calculate_rate_consistency(self, words: List[Dict]) -> float:
        """Calculate speech rate consistency across the recording."""
        if len(words) < 5:
            return 0.5

        # Calculate local rates in 3-second windows
        window_size = 3.0
        start_time = words[0].get("start", 0)
        end_time = words[-1].get("end", 0)

        if end_time - start_time < window_size:
            return 0.5

        rates = []
        current = start_time

        while current + window_size <= end_time:
            window_words = [
                w for w in words if current <= w.get("start", 0) < current + window_size
            ]
            if window_words:
                rate = len(window_words) / window_size
                rates.append(rate)
            current += window_size / 2  # 50% overlap

        if len(rates) < 2:
            return 0.5

        # Lower coefficient of variation = more consistent
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        cv = std_rate / mean_rate if mean_rate > 0 else 0

        # Convert to 0-1 score (lower CV = higher consistency)
        consistency = max(0, 1 - cv)
        return float(consistency)

    def _calculate_timing_regularity(self, words: List[Dict]) -> float:
        """Calculate regularity of word timing (rhythm)."""
        if len(words) < 3:
            return 0.5

        # Calculate inter-word intervals
        intervals = []
        for i in range(1, len(words)):
            prev_end = words[i - 1].get("end", 0)
            curr_start = words[i].get("start", 0)
            interval = curr_start - prev_end
            if interval > 0:  # Only positive intervals
                intervals.append(interval)

        if len(intervals) < 2:
            return 0.5

        intervals = np.array(intervals)

        # Lower coefficient of variation = more regular rhythm
        mean_int = np.mean(intervals)
        std_int = np.std(intervals)
        cv = std_int / mean_int if mean_int > 0 else 0

        # Convert to 0-1 score
        regularity = max(0, 1 - min(cv, 1))
        return float(regularity)

    def _extract_audio_features(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Extract pitch and energy features from audio.

        Pitch CV (coefficient of variation) is key for intonation:
        - CV < 0.10: monotone
        - CV 0.10-0.15: limited variation
        - CV 0.15-0.35: natural intonation
        - CV > 0.35: erratic

        Energy CV is key for stress patterns:
        - CV < 0.25: weak stress differentiation
        - CV 0.25-0.50: moderate
        - CV > 0.50: good stress patterns
        """
        import librosa

        # Load audio
        if audio_array is not None:
            y = audio_array
            sr = self.sample_rate
        elif audio_path:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        else:
            return {}

        features = {}

        # Pitch (F0) extraction for INTONATION analysis
        try:
            # Use pyin for pitch tracking (handles speech well)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                fmax=float(librosa.note_to_hz("C6")),  # ~1047 Hz
                sr=sr,
            )

            # Filter to voiced frames only
            f0_voiced = f0[voiced_flag]

            if len(f0_voiced) > 0:
                pitch_mean = float(np.nanmean(f0_voiced))
                pitch_std = float(np.nanstd(f0_voiced))

                features["pitch_mean"] = pitch_mean
                features["pitch_std"] = pitch_std
                features["pitch_range"] = float(
                    np.nanmax(f0_voiced) - np.nanmin(f0_voiced)
                )

                # Pitch CV (coefficient of variation) - KEY METRIC FOR INTONATION
                if pitch_mean > 0:
                    features["pitch_cv"] = pitch_std / pitch_mean
                else:
                    features["pitch_cv"] = 0.0
        except Exception:
            pass

        # Energy (RMS) extraction for STRESS analysis
        try:
            rms = librosa.feature.rms(y=y)[0]

            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))

            features["energy_mean"] = energy_mean
            features["energy_std"] = energy_std

            # Energy CV - KEY METRIC FOR STRESS PATTERNS
            if energy_mean > 0:
                features["energy_cv"] = energy_std / energy_mean
            else:
                features["energy_cv"] = 0.0
        except Exception:
            pass

        return features


# =============================================================================
# SCORER
# =============================================================================


class PhonologyScorer:
    """
    Score phonology features to produce CEFR level.

    Uses multiple signals:
    1. Whisper confidence (pronunciation clarity)
    2. Pitch CV / Intonation (monotone vs expressive)
    3. Energy CV / Stress patterns
    4. nPVI / Rhythm (stress-timing)

    CEFR Benchmarks:
    - A1-A2: monotone, choppy rhythm, weak stress
    - B1: developing intonation and rhythm
    - B2+: natural intonation, good rhythm, clear stress
    """

    def __init__(self):
        """Initialize scorer with default weights."""
        self.weights = {
            "confidence": 0.30,  # Whisper confidence (pronunciation clarity)
            "intonation": 0.25,  # Pitch variation (monotone vs expressive)
            "rhythm": 0.25,  # nPVI (stress-timing)
            "stress": 0.20,  # Energy variation (stress patterns)
        }

    def score(self, features: PhonologyFeatures) -> PhonologyScore:
        """
        Score phonology features.

        Args:
            features: Extracted phonology features

        Returns:
            PhonologyScore with level, score, and feedback
        """
        if features.word_count == 0:
            return PhonologyScore(
                score=50.0,
                level="A2",
                confidence=0.0,
                features=features,
                sub_scores={},
                feedback=["Insufficient data for phonology assessment."],
            )

        # Calculate sub-scores
        sub_scores = {
            "confidence": self._score_confidence(features),
            "intonation": self._score_intonation(features),
            "rhythm": self._score_rhythm_npvi(features),
            "stress": self._score_stress(features),
        }

        # Weighted average
        total_score = sum(sub_scores[k] * self.weights[k] for k in self.weights.keys())

        # Determine level
        level = score_to_cefr_level(total_score)

        # Calculate confidence in assessment
        assessment_confidence = self._calculate_confidence(features)

        # Generate feedback
        feedback = self._generate_feedback(features, sub_scores, level)

        return PhonologyScore(
            score=total_score,
            level=level,
            confidence=assessment_confidence,
            features=features,
            sub_scores=sub_scores,
            feedback=feedback,
        )

    def _score_confidence(self, features: PhonologyFeatures) -> float:
        """
        Score based on Whisper confidence.

        Higher confidence = clearer pronunciation.
        """
        mean_conf = features.mean_confidence

        if mean_conf >= 0.90:
            return 95
        elif mean_conf >= 0.85:
            return 85 + (mean_conf - 0.85) * 200  # 85-95
        elif mean_conf >= 0.75:
            return 70 + (mean_conf - 0.75) * 150  # 70-85
        elif mean_conf >= 0.65:
            return 50 + (mean_conf - 0.65) * 200  # 50-70
        elif mean_conf >= 0.50:
            return 30 + (mean_conf - 0.50) * 133  # 30-50
        else:
            return max(0, mean_conf * 60)  # 0-30

    def _score_intonation(self, features: PhonologyFeatures) -> float:
        """
        Score intonation based on pitch coefficient of variation.

        CEFR Benchmarks:
        - A1-A2: CV < 0.12 (monotone, score 30-50)
        - B1: CV 0.12-0.20 (developing, score 55-70)
        - B2+: CV 0.15-0.35 (natural, score 75-95)
        """
        if not features.audio_analyzed:
            return 60.0  # Neutral if no audio analysis

        cv = features.pitch_cv
        pattern = features.intonation_pattern

        if pattern == "natural":
            return 85  # Good expressive intonation
        elif pattern == "limited":
            return 65  # Developing, some variation
        elif pattern == "monotone":
            return 40  # Flat, needs work
        elif pattern == "erratic":
            return 55  # Too variable, needs smoothing
        else:
            # Fallback to CV-based scoring
            if cv >= 0.15 and cv <= 0.35:
                return 85
            elif cv >= 0.10:
                return 65
            elif cv < 0.10:
                return 40
            else:
                return 55

    def _score_rhythm_npvi(self, features: PhonologyFeatures) -> float:
        """
        Score rhythm based on nPVI (normalized Pairwise Variability Index).

        Native English: nPVI ~55-65 (stress-timed)
        L2 learners: often ~35-50 (syllable-timed)

        CEFR Benchmarks:
        - A1-A2: nPVI < 45 (choppy, score 30-50)
        - B1: nPVI 45-52 (developing, score 55-70)
        - B2+: nPVI 52-65 (natural, score 75-90)
        """
        npvi = features.npvi_duration
        pattern = features.rhythm_pattern

        if pattern == "natural":
            return 85  # Good English rhythm
        elif pattern == "developing":
            return 65  # Moving toward stress-timing
        elif pattern == "choppy":
            return 40  # Syllable-timed, staccato
        elif pattern == "variable":
            return 70  # High variability (could be expressive)
        else:
            # Fallback to nPVI-based scoring
            if npvi >= 52 and npvi <= 65:
                return 85
            elif npvi >= 45:
                return 65
            elif npvi >= 35:
                return 50
            else:
                return 35

    def _score_stress(self, features: PhonologyFeatures) -> float:
        """
        Score stress patterns based on energy coefficient of variation.

        Good stress patterns show clear differentiation between
        stressed and unstressed syllables/words.

        CEFR Benchmarks:
        - A1-A2: CV < 0.25 (weak, score 35-50)
        - B1: CV 0.25-0.40 (moderate, score 55-70)
        - B2+: CV > 0.40 (good, score 75-90)
        """
        if not features.audio_analyzed:
            return 60.0  # Neutral if no audio analysis

        pattern = features.stress_pattern

        if pattern == "good":
            return 85  # Clear stress differentiation
        elif pattern == "moderate":
            return 65  # Some stress patterns
        elif pattern == "weak":
            return 45  # Little differentiation
        else:
            # Fallback to CV-based scoring
            cv = features.energy_cv
            if cv >= 0.40:
                return 85
            elif cv >= 0.25:
                return 65
            else:
                return 45

    def _calculate_confidence(self, features: PhonologyFeatures) -> float:
        """Calculate confidence in the assessment."""
        confidence = 0.4  # Base

        # More words = more confidence
        if features.word_count >= 50:
            confidence += 0.25
        elif features.word_count >= 20:
            confidence += 0.15

        # Audio analysis = more confidence
        if features.audio_analyzed:
            confidence += 0.2

        # Longer duration = more confidence
        if features.duration >= 30:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_feedback(
        self,
        features: PhonologyFeatures,
        sub_scores: Dict[str, float],
        level: str,
    ) -> List[str]:
        """Generate actionable feedback based on new metrics."""
        feedback = []

        # INTONATION feedback (pitch CV)
        if features.audio_analyzed:
            if features.intonation_pattern == "monotone":
                feedback.append(
                    "Speech sounds monotone (flat pitch). "
                    "Practice varying your pitch - go up at questions, down at statements."
                )
            elif features.intonation_pattern == "limited":
                feedback.append(
                    "Intonation is developing. Try to be more expressive with pitch variation."
                )
            elif features.intonation_pattern == "erratic":
                feedback.append(
                    "Pitch variation is uneven. Aim for smoother, more controlled intonation."
                )
            elif features.intonation_pattern == "natural":
                feedback.append("Good intonation - natural pitch variation.")

        # RHYTHM feedback (nPVI)
        if features.rhythm_pattern == "choppy":
            feedback.append(
                "Speech rhythm is choppy/staccato. "
                "Practice 'chunking' - grouping words together instead of saying them one by one."
            )
        elif features.rhythm_pattern == "developing":
            feedback.append(
                "Rhythm is developing. Work on connecting words more smoothly."
            )
        elif features.rhythm_pattern == "natural":
            feedback.append("Good speech rhythm - natural English stress-timing.")

        # STRESS feedback (energy CV)
        if features.audio_analyzed:
            if features.stress_pattern == "weak":
                feedback.append(
                    "Word stress is weak. Practice emphasizing important words "
                    "(nouns, verbs) more than function words (the, a, is)."
                )
            elif features.stress_pattern == "good":
                feedback.append("Good stress patterns - clear emphasis on key words.")

        # PRONUNCIATION CLARITY (confidence)
        if features.mean_confidence < 0.70:
            feedback.append(
                "Pronunciation clarity needs work. Practice speaking slowly and clearly."
            )
        elif features.mean_confidence >= 0.85:
            feedback.append("Good pronunciation clarity - words are well articulated.")

        # Low confidence words (filter out common words and punctuation)
        if features.low_confidence_words:
            # Filter out very common words and punctuation artifacts
            common_words = {
                "i",
                "a",
                "the",
                "to",
                "is",
                "it",
                "and",
                "of",
                "in",
                "that",
            }
            filtered_words = [
                w.strip(".,!?")
                for w in features.low_confidence_words
                if w.strip(".,!?").lower() not in common_words
                and len(w.strip(".,!?")) > 1
            ]
            if filtered_words:
                words = ", ".join(filtered_words[:5])
                feedback.append(f"Words to practice: {words}")

        return feedback


# =============================================================================
# COMBINED ASSESSOR
# =============================================================================


class PhonologyAssessor:
    """
    Complete phonology assessment pipeline.

    Combines feature extraction and scoring.

    Example:
        >>> assessor = PhonologyAssessor()
        >>> result = assessor.assess(words, audio_path="audio.mp3")
        >>> print(f"Level: {result.level}, Score: {result.score:.1f}")
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.7,
        sample_rate: int = 16000,
    ):
        """Initialize assessor."""
        self.extractor = PhonologyFeatureExtractor(
            low_confidence_threshold=low_confidence_threshold,
            sample_rate=sample_rate,
        )
        self.scorer = PhonologyScorer()

    def assess(
        self,
        words: List[Dict],
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> PhonologyScore:
        """
        Assess phonology from transcription and optionally audio.

        Args:
            words: List of dicts with "word", "start", "end", "confidence"
            audio_path: Path to audio file (optional)
            audio_array: Pre-loaded audio array (optional)

        Returns:
            PhonologyScore with level, score, features, and feedback
        """
        features = self.extractor.extract(
            words=words,
            audio_path=audio_path,
            audio_array=audio_array,
        )
        return self.scorer.score(features)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def assess_phonology(
    words: List[Dict],
    audio_path: Optional[str] = None,
) -> PhonologyScore:
    """
    Quick phonology assessment.

    Args:
        words: Word list with confidence scores
        audio_path: Optional path to audio file

    Returns:
        PhonologyScore
    """
    assessor = PhonologyAssessor()
    return assessor.assess(words, audio_path=audio_path)
