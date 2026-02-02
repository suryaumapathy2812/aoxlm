"""
Faster-Whisper Transcription Backend

Uses CTranslate2-optimized Whisper for fast, accurate transcription.
Supports GPU acceleration and various model sizes.

Usage:
    from src.transcription import FasterWhisperTranscriber

    transcriber = FasterWhisperTranscriber(
        model_size="large-v3",
        device="cuda",
    )
    result = transcriber.transcribe("audio.mp3")
    print(result.text)
    print(result.words)  # Word-level timestamps
"""

from typing import Optional, List
from .base import (
    BaseTranscriber,
    TranscriptionResult,
    Segment,
    Word,
)


class FasterWhisperTranscriber(BaseTranscriber):
    """
    Faster-Whisper transcription backend.

    Uses CTranslate2 for optimized inference. Significantly faster
    than OpenAI's original Whisper implementation.

    Model sizes:
    - tiny: Fastest, lowest accuracy
    - base: Fast, good for simple audio
    - small: Balanced speed/accuracy
    - medium: Good accuracy, moderate speed
    - large-v2: High accuracy
    - large-v3: Best accuracy (recommended for accents)

    Attributes:
        model_size: Whisper model size
        device: cuda or cpu
        compute_type: float16, int8, int8_float16
    """

    AVAILABLE_MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: Optional[str] = None,
        language: str = "en",
    ):
        """
        Initialize Faster-Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (cuda, cpu)
            compute_type: Compute type (float16, int8, int8_float16).
                          Auto-selected based on device if None.
            language: Target language code
        """
        super().__init__(language=language, device=device)

        self.model_size = model_size

        # Auto-select compute type
        if compute_type is None:
            self.compute_type = "float16" if device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        # Lazy-load model
        self._model = None

    @property
    def name(self) -> str:
        return f"faster-whisper-{self.model_size}"

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper

            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        condition_on_previous_text: bool = False,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio file with faster-whisper.

        Args:
            audio_path: Path to audio file
            initial_prompt: Optional prompt to guide transcription
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to filter silence with VAD
            vad_parameters: VAD configuration
            condition_on_previous_text: If False, prevents repetition loops
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            TranscriptionResult with text, segments, words
        """
        model = self._load_model()

        # Default VAD parameters
        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            }

        # Transcribe
        segments_iter, info = model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            **kwargs,
        )

        # Convert to our format
        segments = []
        full_text = []

        for seg in segments_iter:
            # Convert words
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            text=w.word,
                            start=w.start,
                            end=w.end,
                            confidence=w.probability
                            if hasattr(w, "probability")
                            else 1.0,
                        )
                    )

            segments.append(
                Segment(
                    id=seg.id,
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=words,
                )
            )
            full_text.append(seg.text)

        return TranscriptionResult(
            text="".join(full_text),
            segments=segments,
            duration=info.duration,
            language=info.language,
            language_confidence=info.language_probability,
            model=self.model_size,
            backend="faster-whisper",
        )

    def unload_model(self):
        """Unload model to free memory."""
        self._model = None


# Convenience function
def transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    language: str = "en",
    **kwargs,
) -> TranscriptionResult:
    """
    Quick transcription with faster-whisper.

    Args:
        audio_path: Path to audio file
        model_size: Model size
        device: Device to use
        language: Target language
        **kwargs: Additional transcription options

    Returns:
        TranscriptionResult
    """
    transcriber = FasterWhisperTranscriber(
        model_size=model_size,
        device=device,
        language=language,
    )
    return transcriber.transcribe(audio_path, **kwargs)
