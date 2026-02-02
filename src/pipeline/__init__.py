"""
CEFR Assessment Pipeline

Unified pipeline for speech assessment combining:
- Transcription (pluggable backends)
- Fluency assessment (speech rate, pauses, hesitations)
- Range assessment (vocabulary diversity)

Usage:
    from src.pipeline import AssessmentPipeline, assess_audio

    # Quick assessment
    result = assess_audio("audio.mp3", device="cuda")
    print(result.summary())

    # Custom pipeline
    pipeline = AssessmentPipeline(
        transcriber="faster-whisper",
        model_size="large-v3",
        device="cuda",
        fluency_weight=0.6,
        range_weight=0.4,
    )
    result = pipeline.assess("audio.mp3")
"""

from .assessment import (
    AssessmentPipeline,
    AssessmentResult,
    assess_audio,
)

__all__ = [
    "AssessmentPipeline",
    "AssessmentResult",
    "assess_audio",
]
