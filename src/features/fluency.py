"""
Feature-Based Fluency Assessment

Extracts interpretable fluency features from audio transcription and
converts them to CEFR levels based on established research benchmarks.

Fluency Sub-dimensions:
1. Speech Rate - Words per minute, articulation rate
2. Pause Patterns - Frequency, duration, placement
3. Hesitation Markers - Fillers, repetitions, false starts
4. Flow & Rhythm - Rate consistency, smoothness

References:
- CEFR Companion Volume (2020)
- Tavakoli & Skehan (2005) - Fluency measures
- De Jong et al. (2012) - Automatic fluency assessment

Example:
    >>> from src.features.fluency import FluencyAssessor
    >>> assessor = FluencyAssessor()
    >>> result = assessor.assess(words_with_timestamps, duration=30.0)
    >>> print(result["level"])  # "B1"
    >>> print(result["sub_scores"])  # {"speech_rate": 72.5, "pauses": 65.0, ...}
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# Import pure calculation functions from calculations.py
from .calculations import (
    calculate_wpm,
    calculate_articulation_rate,
    calculate_syllables_per_second,
    detect_pauses,
    calculate_pause_metrics,
    calculate_speech_time,
    detect_fillers,
    calculate_filler_rate,
    detect_repetitions,
    detect_false_starts,
    calculate_speech_rate_variability,
    calculate_mean_run_length,
    count_syllables,
    score_to_cefr_level,
    FILLER_WORDS_EN,
)


# CEFR Fluency Benchmarks (based on research)
CEFR_WPM_BENCHMARKS = {
    "A1": {"min": 40, "max": 80, "typical": 60},
    "A2": {"min": 80, "max": 110, "typical": 95},
    "B1": {"min": 110, "max": 140, "typical": 125},
    "B2+": {"min": 140, "max": 180, "typical": 160},
}

# Pause thresholds (in seconds)
PAUSE_THRESHOLDS = {
    "micro": 0.15,  # < 150ms - not a real pause
    "short": 0.25,  # 250ms - short pause
    "medium": 0.5,  # 500ms - medium pause
    "long": 1.0,  # 1s - long pause
    "very_long": 2.0,  # 2s+ - very long pause (disfluency)
}


@dataclass
class FluencyFeatures:
    """Extracted fluency features from speech."""

    # Speech rate features
    wpm: float = 0.0  # Words per minute
    syllables_per_second: float = 0.0  # Estimated syllables/sec
    articulation_rate: float = 0.0  # WPM excluding pauses

    # Pause features
    num_pauses: int = 0  # Total pause count
    num_long_pauses: int = 0  # Pauses > 1 second
    total_pause_time: float = 0.0  # Total time pausing
    mean_pause_duration: float = 0.0  # Average pause length
    pause_ratio: float = 0.0  # Pause time / total time
    pause_rate: float = 0.0  # Pauses per minute

    # Hesitation features
    filler_count: int = 0  # Number of filler words
    filler_rate: float = 0.0  # Fillers per minute
    repetition_count: int = 0  # Repeated words/phrases
    false_start_count: int = 0  # Abandoned utterances

    # Flow features
    speech_rate_variability: float = 0.0  # Std dev of local speech rates
    mean_run_length: float = 0.0  # Mean words between pauses

    # Meta
    word_count: int = 0
    duration: float = 0.0
    speech_time: float = 0.0  # Duration minus pauses

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FluencyScore:
    """CEFR fluency assessment result."""

    level: str  # A1, A2, B1, B2+
    score: float  # Overall score (0-100)
    confidence: float  # Confidence in assessment

    sub_scores: Dict[str, float] = field(default_factory=dict)
    features: FluencyFeatures = field(default_factory=FluencyFeatures)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "features": self.features.to_dict(),
            "feedback": self.feedback,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Fluency Level: {self.level} (score: {self.score:.1f}/100)",
            f"Confidence: {self.confidence:.1%}",
            "",
            "Sub-scores:",
        ]
        for name, score in self.sub_scores.items():
            lines.append(f"  {name:20}: {score:.1f}/100")

        if self.feedback:
            lines.extend(["", "Feedback:"])
            for fb in self.feedback:
                lines.append(f"  • {fb}")

        return "\n".join(lines)


class FluencyFeatureExtractor:
    """
    Extract fluency features from transcription with word timestamps.

    Uses pure calculation functions from calculations.py for all computations.

    Example:
        >>> extractor = FluencyFeatureExtractor()
        >>> words = [
        ...     {"word": "I", "start": 0.0, "end": 0.2},
        ...     {"word": "want", "start": 0.3, "end": 0.6},
        ...     {"word": "um", "start": 1.0, "end": 1.2},
        ...     {"word": "to", "start": 1.3, "end": 1.5},
        ... ]
        >>> features = extractor.extract(words, duration=2.0)
        >>> print(features.wpm, features.filler_count)
    """

    def __init__(
        self,
        language: str = "en",
        pause_threshold: float = 0.25,
        long_pause_threshold: float = 1.0,
    ):
        self.language = language
        self.pause_threshold = pause_threshold
        self.long_pause_threshold = long_pause_threshold
        self.fillers = FILLER_WORDS_EN  # From calculations.py

    def extract(
        self,
        words: List[Dict],
        duration: float,
    ) -> FluencyFeatures:
        """
        Extract fluency features from word timestamps.

        Args:
            words: List of dicts with "word", "start", "end" keys
            duration: Total audio duration in seconds

        Returns:
            FluencyFeatures dataclass
        """
        if not words or duration <= 0:
            return FluencyFeatures(duration=duration)

        # Basic counts
        word_count = len(words)

        # 1. SPEECH RATE (using calculations.py)
        wpm = calculate_wpm(word_count, duration)
        syllables = sum(count_syllables(w.get("word", "")) for w in words)
        syllables_per_sec = calculate_syllables_per_second(syllables, duration)

        # 2. PAUSE ANALYSIS (using calculations.py)
        pauses = detect_pauses(words, threshold_seconds=self.pause_threshold)
        pause_metrics = calculate_pause_metrics(
            pauses, duration, long_pause_threshold=self.long_pause_threshold
        )
        speech_time = calculate_speech_time(duration, pause_metrics["total_time"])
        articulation_rate = calculate_articulation_rate(word_count, speech_time)

        # 3. HESITATION MARKERS (using calculations.py)
        filler_count, _ = detect_fillers(words, self.fillers)
        filler_rate = calculate_filler_rate(filler_count, duration)
        repetition_count = detect_repetitions(words, self.fillers)
        false_start_count = detect_false_starts(words, self.long_pause_threshold)

        # 4. FLOW ANALYSIS (using calculations.py)
        rate_variability = calculate_speech_rate_variability(words)
        mean_run = calculate_mean_run_length(words, self.pause_threshold)

        return FluencyFeatures(
            # Speech rate
            wpm=wpm,
            syllables_per_second=syllables_per_sec,
            articulation_rate=articulation_rate,
            # Pauses
            num_pauses=pause_metrics["count"],
            num_long_pauses=pause_metrics["long_pause_count"],
            total_pause_time=pause_metrics["total_time"],
            mean_pause_duration=float(pause_metrics["mean_duration"]),
            pause_ratio=pause_metrics["pause_ratio"],
            pause_rate=pause_metrics["pause_rate"],
            # Hesitations
            filler_count=filler_count,
            filler_rate=filler_rate,
            repetition_count=repetition_count,
            false_start_count=false_start_count,
            # Flow
            speech_rate_variability=float(rate_variability),
            mean_run_length=float(mean_run),
            # Meta
            word_count=word_count,
            duration=duration,
            speech_time=speech_time,
        )


class FluencyScorer:
    """
    Convert fluency features to CEFR level using research-based thresholds.

    Scoring dimensions:
    1. Speech Rate (30%) - WPM and articulation rate
    2. Pause Patterns (30%) - Frequency, duration, ratio
    3. Hesitations (20%) - Fillers, repetitions
    4. Flow (20%) - Consistency, run length

    Example:
        >>> scorer = FluencyScorer()
        >>> features = FluencyFeatures(wpm=120, pause_ratio=0.15, filler_rate=3.0, ...)
        >>> result = scorer.score(features)
        >>> print(result.level)  # "B1"
    """

    # Scoring weights
    DEFAULT_WEIGHTS = {
        "speech_rate": 0.30,
        "pauses": 0.30,
        "hesitations": 0.20,
        "flow": 0.20,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def score(self, features: FluencyFeatures) -> FluencyScore:
        """
        Score fluency features and return CEFR level.

        Args:
            features: Extracted fluency features

        Returns:
            FluencyScore with level, confidence, and breakdown
        """
        # Calculate sub-scores (0-100)
        speech_rate_score = self._score_speech_rate(features)
        pause_score = self._score_pauses(features)
        hesitation_score = self._score_hesitations(features)
        flow_score = self._score_flow(features)

        sub_scores = {
            "speech_rate": speech_rate_score,
            "pauses": pause_score,
            "hesitations": hesitation_score,
            "flow": flow_score,
        }

        # Weighted overall score
        overall_score = sum(self.weights[dim] * sub_scores[dim] for dim in self.weights)

        # Map to CEFR level
        level = self._score_to_level(overall_score)

        # Calculate confidence
        confidence = self._calculate_confidence(overall_score, features)

        # Generate feedback
        feedback = self._generate_feedback(features, sub_scores, level)

        return FluencyScore(
            level=level,
            score=overall_score,
            confidence=confidence,
            sub_scores=sub_scores,
            features=features,
            feedback=feedback,
        )

    def _score_speech_rate(self, features: FluencyFeatures) -> float:
        """
        Score speech rate (0-100).

        Based on CEFR benchmarks:
        - A1: 40-80 WPM
        - A2: 80-110 WPM
        - B1: 110-140 WPM
        - B2+: 140-180 WPM
        """
        wpm = features.wpm

        if wpm < 30:
            return 5.0
        elif wpm < 40:
            return 10.0 + (wpm - 30) * 1.5  # 10-25
        elif wpm < 80:
            return 25.0 + (wpm - 40) * 0.625  # 25-50
        elif wpm < 110:
            return 50.0 + (wpm - 80) * 0.667  # 50-70
        elif wpm < 140:
            return 70.0 + (wpm - 110) * 0.5  # 70-85
        elif wpm < 180:
            return 85.0 + (wpm - 140) * 0.375  # 85-100
        else:
            return 100.0

    def _score_pauses(self, features: FluencyFeatures) -> float:
        """
        Score pause patterns (0-100).

        Considers:
        - Pause ratio (ideal: < 0.15)
        - Mean pause duration (ideal: < 0.5s)
        - Long pause count (ideal: 0)
        """
        # Pause ratio score (0-40)
        ratio = features.pause_ratio
        if ratio < 0.10:
            ratio_score = 40
        elif ratio < 0.20:
            ratio_score = 40 - (ratio - 0.10) * 200  # 40-20
        elif ratio < 0.35:
            ratio_score = 20 - (ratio - 0.20) * 100  # 20-5
        else:
            ratio_score = 5

        # Mean duration score (0-35)
        mean_dur = features.mean_pause_duration
        if mean_dur < 0.3:
            duration_score = 35
        elif mean_dur < 0.5:
            duration_score = 35 - (mean_dur - 0.3) * 50  # 35-25
        elif mean_dur < 1.0:
            duration_score = 25 - (mean_dur - 0.5) * 30  # 25-10
        else:
            duration_score = max(0, 10 - (mean_dur - 1.0) * 5)

        # Long pause penalty (0-25)
        long_pauses = features.num_long_pauses
        duration = features.duration / 60  # in minutes
        long_pause_rate = long_pauses / duration if duration > 0 else 0

        if long_pause_rate < 1:
            long_pause_score = 25
        elif long_pause_rate < 3:
            long_pause_score = 25 - (long_pause_rate - 1) * 7.5  # 25-10
        else:
            long_pause_score = max(0, 10 - (long_pause_rate - 3) * 2)

        return ratio_score + duration_score + long_pause_score

    def _score_hesitations(self, features: FluencyFeatures) -> float:
        """
        Score hesitation markers (0-100).

        Considers:
        - Filler rate (ideal: < 2/min)
        - Repetitions (ideal: 0)
        - False starts (ideal: 0)
        """
        # Filler score (0-50)
        filler_rate = features.filler_rate
        if filler_rate < 2:
            filler_score = 50
        elif filler_rate < 5:
            filler_score = 50 - (filler_rate - 2) * 10  # 50-20
        elif filler_rate < 10:
            filler_score = 20 - (filler_rate - 5) * 3  # 20-5
        else:
            filler_score = 5

        # Repetition score (0-30)
        duration_min = features.duration / 60
        rep_rate = features.repetition_count / duration_min if duration_min > 0 else 0

        if rep_rate < 1:
            rep_score = 30
        elif rep_rate < 3:
            rep_score = 30 - (rep_rate - 1) * 10  # 30-10
        else:
            rep_score = max(0, 10 - (rep_rate - 3) * 2)

        # False start score (0-20)
        fs_rate = features.false_start_count / duration_min if duration_min > 0 else 0

        if fs_rate < 0.5:
            fs_score = 20
        elif fs_rate < 2:
            fs_score = 20 - (fs_rate - 0.5) * 10  # 20-5
        else:
            fs_score = 5

        return filler_score + rep_score + fs_score

    def _score_flow(self, features: FluencyFeatures) -> float:
        """
        Score speech flow and consistency (0-100).

        Considers:
        - Speech rate variability (ideal: low)
        - Mean run length (ideal: > 8 words)
        """
        # Variability score (0-50)
        variability = features.speech_rate_variability
        if variability < 15:
            var_score = 50
        elif variability < 30:
            var_score = 50 - (variability - 15) * 2  # 50-20
        elif variability < 50:
            var_score = 20 - (variability - 30) * 0.5  # 20-10
        else:
            var_score = 10

        # Run length score (0-50)
        run_length = features.mean_run_length
        if run_length >= 10:
            run_score = 50
        elif run_length >= 6:
            run_score = 30 + (run_length - 6) * 5  # 30-50
        elif run_length >= 3:
            run_score = 15 + (run_length - 3) * 5  # 15-30
        else:
            run_score = run_length * 5  # 0-15

        return var_score + run_score

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to CEFR level."""
        return score_to_cefr_level(score)

    def _calculate_confidence(
        self,
        score: float,
        features: FluencyFeatures,
    ) -> float:
        """
        Calculate confidence in the assessment.

        Higher confidence when:
        - Score is far from level boundaries
        - Sufficient speech duration
        - Consistent sub-scores
        """
        # Distance from boundaries
        boundaries = [0, 35, 55, 75, 100]
        min_distance = min(abs(score - b) for b in boundaries)
        boundary_confidence = min(0.4, min_distance / 25)

        # Duration confidence (more speech = more confident)
        duration = features.duration
        if duration >= 60:
            duration_confidence = 0.3
        elif duration >= 30:
            duration_confidence = 0.2
        elif duration >= 15:
            duration_confidence = 0.1
        else:
            duration_confidence = 0.05

        # Base confidence
        base_confidence = 0.3

        return min(0.95, base_confidence + boundary_confidence + duration_confidence)

    def _generate_feedback(
        self,
        features: FluencyFeatures,
        sub_scores: Dict[str, float],
        level: str,
    ) -> List[str]:
        """Generate actionable feedback based on scores.

        Note: Feedback should be either positive OR negative for each aspect,
        not both. We use sub_scores as the primary indicator to avoid contradictions.
        """
        feedback = []

        # SPEECH RATE feedback (use sub_score to avoid contradictions)
        speech_rate_score = sub_scores.get("speech_rate", 50)
        if speech_rate_score < 50:
            if features.wpm < 80:
                feedback.append(
                    f"Speech rate is slow ({features.wpm:.0f} WPM). "
                    "Try to speak more continuously without long pauses."
                )
            elif features.wpm > 180:
                feedback.append(
                    f"Speech rate is very fast ({features.wpm:.0f} WPM). "
                    "Consider slowing down for clarity."
                )
            else:
                feedback.append(
                    "Speech rate could be improved for more natural delivery."
                )
        elif speech_rate_score >= 70:
            feedback.append("Good speech rate - natural pace.")

        # PAUSE feedback (use sub_score to avoid contradictions)
        pause_score = sub_scores.get("pauses", 50)
        if pause_score < 50:
            if features.pause_ratio > 0.25:
                feedback.append(
                    f"High pause ratio ({features.pause_ratio:.0%}). "
                    "Practice speaking in longer phrases without stopping."
                )
            elif features.num_long_pauses > 3 and features.duration >= 30:
                feedback.append(
                    f"Several long pauses detected ({features.num_long_pauses}). "
                    "Try to reduce hesitation by preparing your thoughts."
                )
            else:
                feedback.append(
                    "Pause patterns could be improved for smoother delivery."
                )
        elif pause_score >= 70:
            feedback.append("Good pause patterns - speech flows well.")

        # HESITATION feedback (use sub_score to avoid contradictions)
        hesitation_score = sub_scores.get("hesitations", 50)
        if hesitation_score < 50:
            if features.filler_rate > 5:
                feedback.append(
                    f"Frequent filler words ({features.filler_rate:.1f}/min). "
                    "Practice pausing silently instead of using 'um' or 'uh'."
                )
            elif features.repetition_count > 2:
                feedback.append(
                    f"Word repetitions detected ({features.repetition_count}). "
                    "Take a breath and think before speaking."
                )
            else:
                feedback.append(
                    "Some hesitation markers detected. Practice for smoother delivery."
                )
        elif hesitation_score >= 80:
            feedback.append("Minimal hesitations - confident delivery.")

        return feedback


class FluencyAssessor:
    """
    Complete fluency assessment pipeline.

    Combines feature extraction and scoring into a single interface.

    Example:
        >>> assessor = FluencyAssessor()
        >>>
        >>> # From Whisper output
        >>> words = [{"word": "hello", "start": 0.0, "end": 0.5}, ...]
        >>> result = assessor.assess(words, duration=30.0)
        >>>
        >>> print(result.level)  # "B1"
        >>> print(result.summary())
    """

    def __init__(
        self,
        language: str = "en",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.extractor = FluencyFeatureExtractor(language=language)
        self.scorer = FluencyScorer(weights=weights)

    def assess(
        self,
        words: List[Dict],
        duration: float,
    ) -> FluencyScore:
        """
        Assess fluency from word timestamps.

        Args:
            words: List of dicts with "word", "start", "end" keys
            duration: Total audio duration in seconds

        Returns:
            FluencyScore with level, confidence, and detailed breakdown
        """
        # Extract features
        features = self.extractor.extract(words, duration)

        # Score and return
        return self.scorer.score(features)

    def assess_from_whisper(
        self,
        whisper_result: Dict,
        duration: Optional[float] = None,
    ) -> FluencyScore:
        """
        Assess fluency from Whisper transcription result.

        Args:
            whisper_result: Whisper output with segments/words
            duration: Override duration (optional)

        Returns:
            FluencyScore
        """
        # Extract words from Whisper format
        words = []

        for segment in whisper_result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append(
                    {
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0),
                        "end": word_info.get("end", 0),
                    }
                )

        # Get duration
        if duration is None:
            if words:
                duration = float(words[-1].get("end", 0))
            else:
                duration = 0.0

        return self.assess(words, duration)


# Convenience function
def assess_fluency(
    words: List[Dict],
    duration: float,
    language: str = "en",
) -> FluencyScore:
    """
    Quick fluency assessment.

    Args:
        words: List of dicts with "word", "start", "end" keys
        duration: Total audio duration in seconds
        language: Language code (default: "en")

    Returns:
        FluencyScore
    """
    assessor = FluencyAssessor(language=language)
    return assessor.assess(words, duration)


if __name__ == "__main__":
    # Demo with sample data
    print("Fluency Assessment Demo")
    print("=" * 60)

    # Simulated word timestamps (B1-level speaker)
    sample_words = [
        {"word": "I", "start": 0.0, "end": 0.15},
        {"word": "think", "start": 0.2, "end": 0.5},
        {"word": "that", "start": 0.55, "end": 0.7},
        {"word": "um", "start": 1.0, "end": 1.2},  # filler
        {"word": "learning", "start": 1.4, "end": 1.8},
        {"word": "English", "start": 1.85, "end": 2.3},
        {"word": "is", "start": 2.35, "end": 2.5},
        {"word": "very", "start": 2.55, "end": 2.8},
        {"word": "important", "start": 2.85, "end": 3.4},
        {"word": "for", "start": 3.8, "end": 4.0},  # pause before
        {"word": "my", "start": 4.05, "end": 4.2},
        {"word": "career", "start": 4.25, "end": 4.7},
        {"word": "because", "start": 5.0, "end": 5.4},
        {"word": "uh", "start": 5.6, "end": 5.8},  # filler
        {"word": "many", "start": 5.9, "end": 6.2},
        {"word": "companies", "start": 6.25, "end": 6.8},
        {"word": "require", "start": 6.9, "end": 7.3},
        {"word": "it", "start": 7.35, "end": 7.5},
    ]

    duration = 8.0  # 8 seconds

    # Assess
    assessor = FluencyAssessor()
    result = assessor.assess(sample_words, duration)

    print(result.summary())
    print("\n" + "=" * 60)
    print("\nDetailed Features:")
    for key, value in result.features.to_dict().items():
        print(f"  {key:25}: {value}")
