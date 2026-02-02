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

    # Pitch-based (from audio)
    pitch_mean: float = 0.0  # Average pitch (Hz)
    pitch_std: float = 0.0  # Pitch variation
    pitch_range: float = 0.0  # Max - min pitch
    pitch_variation_ratio: float = 0.0  # Relative variation

    # Energy-based
    energy_mean: float = 0.0  # Average energy
    energy_std: float = 0.0  # Energy variation
    energy_variation_ratio: float = 0.0  # Relative variation

    # Rhythm-based
    speech_rate_consistency: float = 0.0  # How consistent is the pace
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

        return PhonologyFeatures(
            # Confidence features
            mean_confidence=confidence_features.get("mean", 0.0),
            min_confidence=confidence_features.get("min", 0.0),
            confidence_std=confidence_features.get("std", 0.0),
            low_confidence_ratio=confidence_features.get("low_ratio", 0.0),
            low_confidence_words=confidence_features.get("low_words", []),
            # Audio features
            pitch_mean=audio_features.get("pitch_mean", 0.0),
            pitch_std=audio_features.get("pitch_std", 0.0),
            pitch_range=audio_features.get("pitch_range", 0.0),
            pitch_variation_ratio=audio_features.get("pitch_variation_ratio", 0.0),
            energy_mean=audio_features.get("energy_mean", 0.0),
            energy_std=audio_features.get("energy_std", 0.0),
            energy_variation_ratio=audio_features.get("energy_variation_ratio", 0.0),
            # Rhythm
            speech_rate_consistency=confidence_features.get("rate_consistency", 0.0),
            syllable_timing_regularity=confidence_features.get(
                "timing_regularity", 0.0
            ),
            # Meta
            word_count=len(words),
            duration=duration,
            audio_analyzed=audio_analyzed,
        )

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
        """Extract pitch and energy features from audio."""
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

        # Pitch (F0) extraction
        try:
            # Use pyin for pitch tracking (handles speech well)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),  # ~65 Hz
                fmax=librosa.note_to_hz("C6"),  # ~1047 Hz
                sr=sr,
            )

            # Filter to voiced frames only
            f0_voiced = f0[voiced_flag]

            if len(f0_voiced) > 0:
                features["pitch_mean"] = float(np.nanmean(f0_voiced))
                features["pitch_std"] = float(np.nanstd(f0_voiced))
                features["pitch_range"] = float(
                    np.nanmax(f0_voiced) - np.nanmin(f0_voiced)
                )

                # Pitch variation ratio (normalized by mean)
                if features["pitch_mean"] > 0:
                    features["pitch_variation_ratio"] = (
                        features["pitch_std"] / features["pitch_mean"]
                    )
                else:
                    features["pitch_variation_ratio"] = 0.0
        except Exception:
            pass

        # Energy (RMS) extraction
        try:
            rms = librosa.feature.rms(y=y)[0]

            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))

            if features["energy_mean"] > 0:
                features["energy_variation_ratio"] = (
                    features["energy_std"] / features["energy_mean"]
                )
            else:
                features["energy_variation_ratio"] = 0.0
        except Exception:
            pass

        return features


# =============================================================================
# SCORER
# =============================================================================


class PhonologyScorer:
    """
    Score phonology features to produce CEFR level.

    Primary signal: Whisper confidence scores
    Secondary: Pitch/energy variation (natural speech patterns)

    CEFR Benchmarks (confidence scores):
    - A1: < 0.65 mean confidence
    - A2: 0.65-0.75 mean confidence
    - B1: 0.75-0.85 mean confidence
    - B2+: > 0.85 mean confidence
    """

    def __init__(self):
        """Initialize scorer with default weights."""
        self.weights = {
            "confidence": 0.50,  # Whisper confidence (primary)
            "consistency": 0.20,  # Pronunciation consistency
            "prosody": 0.20,  # Pitch/energy variation
            "rhythm": 0.10,  # Speech rhythm
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
            "consistency": self._score_consistency(features),
            "prosody": self._score_prosody(features),
            "rhythm": self._score_rhythm(features),
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

    def _score_consistency(self, features: PhonologyFeatures) -> float:
        """Score pronunciation consistency (low std = consistent)."""
        # Lower std relative to mean = more consistent
        if features.mean_confidence == 0:
            return 50.0

        cv = features.confidence_std / features.mean_confidence

        # Also penalize high ratio of low-confidence words
        low_ratio_penalty = features.low_confidence_ratio * 30

        if cv < 0.1:
            base = 90
        elif cv < 0.2:
            base = 75
        elif cv < 0.3:
            base = 60
        else:
            base = 45

        return max(0, base - low_ratio_penalty)

    def _score_prosody(self, features: PhonologyFeatures) -> float:
        """
        Score prosody (pitch/energy variation).

        Natural speech has moderate variation.
        Too little = monotone, too much = erratic.
        """
        if not features.audio_analyzed:
            return 60.0  # Neutral if no audio analysis

        pitch_var = features.pitch_variation_ratio
        energy_var = features.energy_variation_ratio

        # Ideal pitch variation: 0.15-0.35 (relative)
        if 0.15 <= pitch_var <= 0.35:
            pitch_score = 85
        elif 0.10 <= pitch_var <= 0.40:
            pitch_score = 70
        elif pitch_var < 0.10:  # Monotone
            pitch_score = 50
        else:  # Too variable
            pitch_score = 55

        # Ideal energy variation: 0.3-0.6
        if 0.3 <= energy_var <= 0.6:
            energy_score = 80
        elif 0.2 <= energy_var <= 0.7:
            energy_score = 65
        else:
            energy_score = 50

        return (pitch_score + energy_score) / 2

    def _score_rhythm(self, features: PhonologyFeatures) -> float:
        """Score speech rhythm regularity."""
        consistency = features.speech_rate_consistency
        regularity = features.syllable_timing_regularity

        # Higher consistency and regularity = better rhythm
        rhythm_score = (consistency + regularity) / 2 * 100

        return min(100, max(0, rhythm_score))

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
        """Generate actionable feedback."""
        feedback = []

        # Confidence-based feedback
        if features.mean_confidence < 0.65:
            feedback.append(
                "Pronunciation clarity needs improvement. "
                "Practice speaking slowly and clearly."
            )
        elif features.mean_confidence < 0.75:
            feedback.append(
                "Some pronunciation issues detected. "
                "Focus on problematic sounds and words."
            )

        # Low confidence words feedback
        if features.low_confidence_words:
            words = ", ".join(features.low_confidence_words[:5])
            feedback.append(f"Words needing pronunciation practice: {words}")

        # Consistency feedback
        if features.low_confidence_ratio > 0.2:
            feedback.append(
                f"{features.low_confidence_ratio:.0%} of words have unclear pronunciation. "
                "Work on consistent articulation."
            )

        # Prosody feedback
        if features.audio_analyzed:
            if features.pitch_variation_ratio < 0.10:
                feedback.append(
                    "Speech sounds monotone. Try varying your pitch for natural intonation."
                )
            elif features.pitch_variation_ratio > 0.40:
                feedback.append(
                    "Pitch variation is high. Aim for smoother intonation patterns."
                )

        # Positive feedback
        if features.mean_confidence >= 0.85:
            feedback.append("Good pronunciation clarity - words are well articulated.")

        if sub_scores.get("rhythm", 0) >= 70:
            feedback.append("Good speech rhythm - natural pacing.")

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
