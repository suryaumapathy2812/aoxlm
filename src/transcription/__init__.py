"""
Transcription Module

Pluggable transcription backends with a unified interface.

Supported Backends:
- FasterWhisperTranscriber: CTranslate2-optimized Whisper (recommended)

Coming Soon:
- OpenAIWhisperTranscriber: Original OpenAI Whisper
- GeminiTranscriber: Google Gemini API
- AzureTranscriber: Azure Speech Services

Usage:
    from src.transcription import FasterWhisperTranscriber

    # Initialize transcriber
    transcriber = FasterWhisperTranscriber(
        model_size="large-v3",
        device="cuda",
    )

    # Transcribe audio
    result = transcriber.transcribe("audio.mp3")

    # Access results
    print(result.text)           # Full transcription
    print(result.words)          # Word-level timestamps
    print(result.duration)       # Audio duration

    # Get words as dicts (for feature extraction)
    words = result.get_words_as_dicts()
"""

from .base import (
    BaseTranscriber,
    TranscriptionResult,
    Segment,
    Word,
)

from .faster_whisper import (
    FasterWhisperTranscriber,
    transcribe_with_faster_whisper,
)


def get_transcriber(
    backend: str = "faster-whisper",
    **kwargs,
) -> BaseTranscriber:
    """
    Factory function to get a transcriber by name.

    Args:
        backend: Backend name ("faster-whisper", "whisper", etc.)
        **kwargs: Backend-specific configuration

    Returns:
        Configured transcriber instance

    Example:
        transcriber = get_transcriber("faster-whisper", model_size="large-v3")
    """
    backends = {
        "faster-whisper": FasterWhisperTranscriber,
        "fasterwhisper": FasterWhisperTranscriber,
    }

    backend_lower = backend.lower().replace("_", "-")

    if backend_lower not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend: {backend}. Available: {available}")

    return backends[backend_lower](**kwargs)


__all__ = [
    # Base classes
    "BaseTranscriber",
    "TranscriptionResult",
    "Segment",
    "Word",
    # Implementations
    "FasterWhisperTranscriber",
    "transcribe_with_faster_whisper",
    # Factory
    "get_transcriber",
]
