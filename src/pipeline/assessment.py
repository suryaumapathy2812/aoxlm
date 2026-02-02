"""
CEFR Assessment Pipeline

Combines transcription and feature-based assessment into a unified pipeline.
Modular design allows swapping transcription backends and assessment methods.

Usage:
    from src.pipeline import AssessmentPipeline

    # Create pipeline
    pipeline = AssessmentPipeline(
        transcriber="faster-whisper",
        model_size="large-v3",
        device="cuda",
    )

    # Assess audio file
    result = pipeline.assess("audio.mp3")

    # Access results
    print(result.level)           # "B1"
    print(result.score)           # 68.5
    print(result.fluency)         # FluencyScore
    print(result.range)           # RangeScore
    print(result.transcription)   # TranscriptionResult
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..transcription import (
    get_transcriber,
    BaseTranscriber,
    TranscriptionResult,
)
from ..features.fluency import FluencyAssessor, FluencyScore
from ..features.range import RangeAssessor, RangeScore


@dataclass
class AssessmentResult:
    """
    Complete CEFR assessment result.

    Combines transcription, fluency, and range assessments
    into a single unified result.
    """

    # Overall assessment
    level: str
    score: float
    confidence: float

    # Component scores
    fluency: FluencyScore
    range: RangeScore

    # Weights used
    fluency_weight: float = 0.6
    range_weight: float = 0.4

    # Source data
    transcription: Optional[TranscriptionResult] = None
    audio_path: Optional[str] = None

    # Metadata
    pipeline_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "fluency_weight": self.fluency_weight,
            "range_weight": self.range_weight,
            "fluency": self.fluency.to_dict(),
            "range": self.range.to_dict(),
            "transcription": self.transcription.to_dict()
            if self.transcription
            else None,
            "audio_path": self.audio_path,
            "pipeline_version": self.pipeline_version,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "CEFR ASSESSMENT RESULT",
            "=" * 60,
            f"",
            f"Overall Level: {self.level}",
            f"Overall Score: {self.score:.1f}/100",
            f"Confidence: {self.confidence:.1%}",
            f"",
            f"Breakdown:",
            f"  Fluency ({self.fluency_weight:.0%}): {self.fluency.level} ({self.fluency.score:.1f})",
            f"  Range ({self.range_weight:.0%}):   {self.range.level} ({self.range.score:.1f})",
        ]

        if self.fluency.feedback:
            lines.extend(["", "Fluency Feedback:"])
            for fb in self.fluency.feedback[:3]:
                lines.append(f"  - {fb}")

        if self.range.feedback:
            lines.extend(["", "Range Feedback:"])
            for fb in self.range.feedback[:3]:
                lines.append(f"  - {fb}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class AssessmentPipeline:
    """
    Complete CEFR assessment pipeline.

    Orchestrates:
    1. Audio transcription (pluggable backend)
    2. Fluency assessment (from word timestamps)
    3. Range assessment (from transcription text)
    4. Combined CEFR scoring

    Example:
        pipeline = AssessmentPipeline(
            transcriber="faster-whisper",
            model_size="large-v3",
            device="cuda",
        )
        result = pipeline.assess("audio.mp3")
        print(result.summary())
    """

    def __init__(
        self,
        transcriber: str = "faster-whisper",
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "en",
        fluency_weight: float = 0.6,
        range_weight: float = 0.4,
        **transcriber_kwargs,
    ):
        """
        Initialize assessment pipeline.

        Args:
            transcriber: Transcription backend ("faster-whisper", etc.)
            model_size: Model size for transcription
            device: Device to use (cuda, cpu)
            language: Target language
            fluency_weight: Weight for fluency in overall score (default: 0.6)
            range_weight: Weight for range in overall score (default: 0.4)
            **transcriber_kwargs: Additional transcriber configuration
        """
        self.transcriber = get_transcriber(
            backend=transcriber,
            model_size=model_size,
            device=device,
            language=language,
            **transcriber_kwargs,
        )

        self.fluency_assessor = FluencyAssessor(language=language)
        self.range_assessor = RangeAssessor(language=language)

        self.fluency_weight = fluency_weight
        self.range_weight = range_weight
        self.language = language

    def assess(
        self,
        audio_path: str,
        transcription_result: Optional[TranscriptionResult] = None,
        **transcribe_kwargs,
    ) -> AssessmentResult:
        """
        Assess CEFR level from audio file.

        Args:
            audio_path: Path to audio file
            transcription_result: Pre-computed transcription (optional)
            **transcribe_kwargs: Additional transcription options

        Returns:
            AssessmentResult with level, scores, and feedback
        """
        audio_path = str(Path(audio_path).resolve())

        # 1. Transcribe (or use provided)
        if transcription_result is None:
            transcription_result = self.transcriber.transcribe(
                audio_path,
                **transcribe_kwargs,
            )

        # 2. Assess fluency from word timestamps
        words = transcription_result.get_words_as_dicts()
        fluency_score = self.fluency_assessor.assess(
            words,
            duration=transcription_result.duration,
        )

        # 3. Assess range from transcription text
        range_score = self.range_assessor.assess(transcription_result.text)

        # 4. Calculate combined score
        combined_score = (
            fluency_score.score * self.fluency_weight
            + range_score.score * self.range_weight
        )

        # 5. Map to CEFR level
        level = self._score_to_level(combined_score)

        # 6. Calculate combined confidence
        combined_confidence = (
            fluency_score.confidence * self.fluency_weight
            + range_score.confidence * self.range_weight
        )

        return AssessmentResult(
            level=level,
            score=combined_score,
            confidence=combined_confidence,
            fluency=fluency_score,
            range=range_score,
            fluency_weight=self.fluency_weight,
            range_weight=self.range_weight,
            transcription=transcription_result,
            audio_path=audio_path,
        )

    def assess_text(
        self,
        text: str,
        words: Optional[List[Dict]] = None,
        duration: Optional[float] = None,
    ) -> AssessmentResult:
        """
        Assess CEFR level from text (without audio).

        Useful for testing or when transcription is already done.

        Args:
            text: Transcription text
            words: Word timestamps (optional, for fluency)
            duration: Audio duration (optional, for fluency)

        Returns:
            AssessmentResult (fluency may be limited without timestamps)
        """
        # Assess range
        range_score = self.range_assessor.assess(text)

        # Assess fluency if timestamps provided
        if words and duration:
            fluency_score = self.fluency_assessor.assess(words, duration)
        else:
            # Create placeholder fluency score
            from ..features.fluency import FluencyScore, FluencyFeatures

            fluency_score = FluencyScore(
                level="N/A",
                score=0.0,
                confidence=0.0,
                feedback=["Fluency requires word timestamps"],
            )

        # Calculate combined score (range only if no fluency)
        if fluency_score.score > 0:
            combined_score = (
                fluency_score.score * self.fluency_weight
                + range_score.score * self.range_weight
            )
            combined_confidence = (
                fluency_score.confidence * self.fluency_weight
                + range_score.confidence * self.range_weight
            )
        else:
            combined_score = range_score.score
            combined_confidence = range_score.confidence

        level = self._score_to_level(combined_score)

        return AssessmentResult(
            level=level,
            score=combined_score,
            confidence=combined_confidence,
            fluency=fluency_score,
            range=range_score,
            fluency_weight=self.fluency_weight if fluency_score.score > 0 else 0.0,
            range_weight=self.range_weight if fluency_score.score > 0 else 1.0,
        )

    def _score_to_level(self, score: float) -> str:
        """Map numeric score to CEFR level."""
        if score < 35:
            return "A1"
        elif score < 55:
            return "A2"
        elif score < 75:
            return "B1"
        else:
            return "B2+"

    def assess_batch(
        self,
        audio_paths: List[str],
        **transcribe_kwargs,
    ) -> List[AssessmentResult]:
        """
        Assess multiple audio files.

        Args:
            audio_paths: List of audio file paths
            **transcribe_kwargs: Transcription options

        Returns:
            List of AssessmentResult
        """
        results = []
        for path in audio_paths:
            result = self.assess(path, **transcribe_kwargs)
            results.append(result)
        return results


# Convenience function
def assess_audio(
    audio_path: str,
    transcriber: str = "faster-whisper",
    model_size: str = "large-v3",
    device: str = "cuda",
    **kwargs,
) -> AssessmentResult:
    """
    Quick assessment of an audio file.

    Args:
        audio_path: Path to audio file
        transcriber: Transcription backend
        model_size: Model size
        device: Device to use
        **kwargs: Additional options

    Returns:
        AssessmentResult

    Example:
        result = assess_audio("speech.mp3", device="cuda")
        print(result.level)  # "B1"
    """
    pipeline = AssessmentPipeline(
        transcriber=transcriber,
        model_size=model_size,
        device=device,
    )
    return pipeline.assess(audio_path, **kwargs)
