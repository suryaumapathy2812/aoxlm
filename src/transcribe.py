"""
Transcription Module - Whisper + WhisperX

Provides transcription with word-level timestamps using:
- Whisper-large-v3 for transcription
- WhisperX for word-level alignment

No audio duration constraints - works with any length audio.

Example:
    >>> from src.transcribe import Transcriber
    >>> transcriber = Transcriber()
    >>> result = transcriber.transcribe("audio.wav")
    >>> print(result["text"])
    >>> for word in result["words"]:
    ...     print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import torch

try:
    import whisperx

    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: whisperx not installed. Install with: pip install whisperx")


@dataclass
class WordTimestamp:
    """A single word with timing information."""

    word: str
    start: float
    end: float
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Result from transcription."""

    text: str
    words: List[WordTimestamp]
    language: str
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "duration": self.duration,
        }

    def get_fluency_features(self) -> Dict[str, float]:
        """Extract fluency features from timestamps."""
        if not self.words:
            return {
                "wpm": 0.0,
                "pause_ratio": 0.0,
                "mean_pause_duration": 0.0,
                "num_pauses": 0,
            }

        # Words per minute
        word_count = len(self.words)
        wpm = (word_count / self.duration) * 60 if self.duration > 0 else 0

        # Pause analysis (gaps > 0.3s between words)
        pauses = []
        for i in range(1, len(self.words)):
            gap = self.words[i].start - self.words[i - 1].end
            if gap > 0.3:  # Significant pause
                pauses.append(gap)

        total_pause_time = sum(pauses)
        pause_ratio = total_pause_time / self.duration if self.duration > 0 else 0
        mean_pause = sum(pauses) / len(pauses) if pauses else 0

        return {
            "wpm": round(wpm, 1),
            "pause_ratio": round(pause_ratio, 3),
            "mean_pause_duration": round(mean_pause, 3),
            "num_pauses": len(pauses),
        }


class Transcriber:
    """
    Transcriber using Whisper + WhisperX for word-level timestamps.

    Example:
        >>> transcriber = Transcriber(model_size="large-v3")
        >>> result = transcriber.transcribe("audio.wav")
        >>> print(result.text)
        >>> print(result.get_fluency_features())
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float16",
        language: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu). Auto-detected if None.
            compute_type: Compute type (float16, int8, float32)
            language: Force language (None for auto-detection)
        """
        if not WHISPERX_AVAILABLE:
            raise ImportError(
                "whisperx not installed. Install with: pip install whisperx"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.language = language
        self.model_size = model_size

        print(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=compute_type,
            language=language,
        )

        # Alignment model (loaded lazily per language)
        self._align_model = None
        self._align_metadata = None
        self._align_language = None

        print("Transcriber ready!")

    def _load_align_model(self, language_code: str):
        """Load alignment model for a specific language."""
        if self._align_language != language_code:
            print(f"Loading alignment model for {language_code}...")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            self._align_language = language_code

    def transcribe(
        self,
        audio: Union[str, Path],
        language: Optional[str] = None,
        batch_size: int = 16,
    ) -> TranscriptionResult:
        """
        Transcribe audio with word-level timestamps.

        Args:
            audio: Path to audio file
            language: Override language (None for auto-detection)
            batch_size: Batch size for transcription

        Returns:
            TranscriptionResult with text, words, language, duration
        """
        audio_path = str(audio)

        # Load audio
        audio_array = whisperx.load_audio(audio_path)
        duration = len(audio_array) / 16000  # Assuming 16kHz

        # Transcribe
        result = self.model.transcribe(
            audio_array,
            batch_size=batch_size,
            language=language or self.language,
        )

        detected_language = result.get("language", "en")

        # Align for word timestamps
        self._load_align_model(detected_language)

        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio_array,
            self.device,
            return_char_alignments=False,
        )

        # Extract words
        words = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                word = WordTimestamp(
                    word=word_info.get("word", ""),
                    start=word_info.get("start", 0.0),
                    end=word_info.get("end", 0.0),
                    score=word_info.get("score"),
                )
                words.append(word)

        # Build full text
        full_text = " ".join(seg.get("text", "") for seg in result["segments"])

        return TranscriptionResult(
            text=full_text.strip(),
            words=words,
            language=detected_language,
            duration=duration,
        )

    def transcribe_with_features(
        self,
        audio: Union[str, Path],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe and extract fluency features in one call.

        Args:
            audio: Path to audio file
            language: Override language

        Returns:
            Dict with transcription result and fluency features
        """
        result = self.transcribe(audio, language)
        features = result.get_fluency_features()

        return {
            **result.to_dict(),
            "fluency_features": features,
        }


def transcribe(
    audio: Union[str, Path],
    model_size: str = "large-v3",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> TranscriptionResult:
    """
    Simple one-shot transcription function.

    Args:
        audio: Path to audio file
        model_size: Whisper model size
        language: Force language (None for auto-detection)
        device: Device to run on

    Returns:
        TranscriptionResult
    """
    transcriber = Transcriber(
        model_size=model_size,
        device=device,
        language=language,
    )
    return transcriber.transcribe(audio)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.transcribe <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"\nTranscribing: {audio_path}")
    print("=" * 50)

    transcriber = Transcriber(model_size="large-v3")
    result = transcriber.transcribe(audio_path)

    print(f"\nLanguage: {result.language}")
    print(f"Duration: {result.duration:.1f}s")
    print(f"\nText:\n{result.text}")

    print(f"\nWords ({len(result.words)}):")
    for w in result.words[:10]:
        print(f"  [{w.start:6.2f} - {w.end:6.2f}] {w.word}")
    if len(result.words) > 10:
        print(f"  ... and {len(result.words) - 10} more words")

    features = result.get_fluency_features()
    print(f"\nFluency Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
