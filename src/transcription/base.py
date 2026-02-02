"""
Abstract Base Class for Transcription Backends

Defines the interface that all transcription backends must implement.
This allows swapping between different transcription services:
- faster-whisper (local)
- OpenAI Whisper (local)
- Google Gemini (cloud)
- OpenAI API (cloud)
- Azure Speech (cloud)

Usage:
    from src.transcription import FasterWhisperTranscriber

    transcriber = FasterWhisperTranscriber(model_size="large-v3")
    result = transcriber.transcribe("audio.mp3")

    # Result format is standardized:
    # {
    #     "text": "full transcription",
    #     "segments": [...],
    #     "words": [...],
    #     "duration": 123.4,
    #     "language": "en",
    # }
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Word:
    """Single word with timestamp."""

    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "word": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class Segment:
    """Transcription segment (sentence/phrase)."""

    id: int
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionResult:
    """
    Standardized transcription result.

    All transcription backends return this format.
    """

    text: str
    segments: List[Segment]
    duration: float
    language: str = "en"
    language_confidence: float = 1.0

    # Metadata
    model: str = ""
    backend: str = ""

    @property
    def words(self) -> List[Word]:
        """Flatten all words from segments."""
        all_words = []
        for seg in self.segments:
            all_words.extend(seg.words)
        return all_words

    @property
    def word_count(self) -> int:
        """Total word count."""
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Whisper-compatible format)."""
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "duration": self.duration,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "model": self.model,
            "backend": self.backend,
            "word_count": self.word_count,
        }

    def get_words_as_dicts(self) -> List[Dict]:
        """Get words as list of dicts (for feature extraction)."""
        return [w.to_dict() for w in self.words]


class BaseTranscriber(ABC):
    """
    Abstract base class for transcription backends.

    Subclasses must implement:
    - transcribe(): Main transcription method
    - is_available(): Check if backend is available

    Optional overrides:
    - transcribe_batch(): For batch processing
    """

    def __init__(
        self,
        language: str = "en",
        device: str = "cuda",
    ):
        """
        Initialize transcriber.

        Args:
            language: Target language code
            device: Device to use (cuda, cpu)
        """
        self.language = language
        self.device = device

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Backend-specific options

        Returns:
            TranscriptionResult with text, segments, words
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this transcription backend is available.

        Returns:
            True if backend can be used
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for identification."""
        pass

    def transcribe_batch(
        self,
        audio_paths: List[str],
        **kwargs,
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Default implementation processes sequentially.
        Override for parallel processing.

        Args:
            audio_paths: List of audio file paths
            **kwargs: Backend-specific options

        Returns:
            List of TranscriptionResult
        """
        results = []
        for path in audio_paths:
            result = self.transcribe(path, **kwargs)
            results.append(result)
        return results
