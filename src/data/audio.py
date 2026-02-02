"""
Audio Processing Utilities

Handles audio loading, preprocessing, and validation.
All audio is normalized to 16kHz mono for consistency with encoders.

Supports both torchaudio and librosa backends for maximum compatibility.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import numpy as np

# Try to import audio backends
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class AudioProcessor:
    """
    Audio loading and preprocessing.

    Handles:
    - Loading from file or URL
    - Resampling to 16kHz
    - Converting to mono
    - Normalizing amplitude
    - Chunking long audio

    Example:
        >>> processor = AudioProcessor()
        >>> audio, sr = processor.load("speech.wav")
        >>> print(audio.shape, sr)  # [samples], 16000
    """

    TARGET_SAMPLE_RATE = 16000  # All encoders expect 16kHz

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        normalize: bool = True,
        max_duration: Optional[float] = 300.0,  # 5 minutes max
    ):
        """
        Initialize audio processor.

        Args:
            target_sample_rate: Output sample rate (default 16kHz)
            normalize: Whether to normalize amplitude to [-1, 1]
            max_duration: Maximum audio duration in seconds (None for unlimited)
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.max_duration = max_duration

        # Lazy-loaded resampler cache
        self._resamplers = {}

    def _get_resampler(self, orig_sr: int):
        """Get or create a resampler for the given sample rate."""
        if not TORCHAUDIO_AVAILABLE:
            return None
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sample_rate,
            )
        return self._resamplers[orig_sr]

    def load(
        self,
        path: Union[str, Path],
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file.

        Args:
            path: Path to audio file
            start: Start time in seconds (optional)
            duration: Duration in seconds (optional)

        Returns:
            Tuple of (audio_tensor, sample_rate)
            audio_tensor: [samples] float32 tensor, mono, 16kHz
            sample_rate: Always 16000
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Try librosa first (more compatible), then torchaudio
        if LIBROSA_AVAILABLE:
            return self._load_with_librosa(path, start, duration)
        elif TORCHAUDIO_AVAILABLE:
            return self._load_with_torchaudio(path, start, duration)
        else:
            raise ImportError(
                "No audio backend available. Install librosa or torchaudio: "
                "pip install librosa"
            )

    def _load_with_librosa(
        self,
        path: Path,
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio using librosa backend."""
        # librosa.load handles resampling and mono conversion
        audio_np, sr = librosa.load(
            str(path),
            sr=self.target_sample_rate,
            mono=True,
            offset=start or 0.0,
            duration=duration,
        )

        # Convert to tensor
        audio = torch.from_numpy(audio_np).float()

        # Normalize if needed
        if self.normalize:
            audio = self._normalize(audio)

        # Truncate if too long
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.target_sample_rate)
            if audio.shape[0] > max_samples:
                print(
                    f"Warning: Audio truncated from {audio.shape[0] / self.target_sample_rate:.1f}s "
                    f"to {self.max_duration}s"
                )
                audio = audio[:max_samples]

        return audio, self.target_sample_rate

    def _load_with_torchaudio(
        self,
        path: Path,
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio using torchaudio backend."""
        # Load audio
        if start is not None or duration is not None:
            # Load specific segment
            info = torchaudio.info(str(path))
            orig_sr = info.sample_rate

            frame_offset = int((start or 0) * orig_sr)
            num_frames = int(duration * orig_sr) if duration else -1

            audio, sr = torchaudio.load(
                str(path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        else:
            audio, sr = torchaudio.load(str(path))

        # Process
        audio = self._process(audio, sr)

        return audio, self.target_sample_rate

    def load_from_array(
        self,
        array: np.ndarray,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from numpy array.

        Args:
            array: Audio array [samples] or [channels, samples]
            sample_rate: Original sample rate

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Convert to tensor
        if isinstance(array, np.ndarray):
            audio = torch.from_numpy(array).float()
        else:
            audio = array.float()

        # Ensure 2D [channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Process
        audio = self._process(audio, sample_rate)

        return audio, self.target_sample_rate

    def _process(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Process audio: resample, mono, normalize, truncate.

        Args:
            audio: Audio tensor [channels, samples]
            sample_rate: Original sample rate

        Returns:
            Processed audio [samples]
        """
        # Convert to mono by averaging channels
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            if resampler is not None:
                audio = resampler(audio)
            elif LIBROSA_AVAILABLE:
                # Fallback to librosa resampling
                audio_np = audio.squeeze(0).numpy()
                audio_np = librosa.resample(
                    audio_np, orig_sr=sample_rate, target_sr=self.target_sample_rate
                )
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)

        # Remove channel dimension
        audio = audio.squeeze(0)

        # Normalize amplitude
        if self.normalize:
            audio = self._normalize(audio)

        # Truncate if too long
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.target_sample_rate)
            if audio.shape[0] > max_samples:
                print(
                    f"Warning: Audio truncated from {audio.shape[0] / self.target_sample_rate:.1f}s "
                    f"to {self.max_duration}s"
                )
                audio = audio[:max_samples]

        return audio

    def _normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def get_duration(self, audio: torch.Tensor) -> float:
        """Get duration of audio in seconds."""
        return audio.shape[-1] / self.target_sample_rate

    def chunk(
        self,
        audio: torch.Tensor,
        chunk_duration: float = 30.0,
        overlap: float = 1.0,
    ) -> list:
        """
        Split audio into overlapping chunks.

        Args:
            audio: Audio tensor [samples]
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            List of (chunk_audio, start_time, end_time) tuples
        """
        chunk_samples = int(chunk_duration * self.target_sample_rate)
        overlap_samples = int(overlap * self.target_sample_rate)
        step_samples = chunk_samples - overlap_samples

        chunks = []
        total_samples = audio.shape[0]

        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            start_time = start / self.target_sample_rate
            end_time = end / self.target_sample_rate

            chunks.append((chunk, start_time, end_time))

            start += step_samples

            # Stop if remaining is too short
            if total_samples - start < chunk_samples // 2:
                break

        return chunks

    @staticmethod
    def info(path: Union[str, Path]) -> dict:
        """
        Get audio file information.

        Args:
            path: Path to audio file

        Returns:
            Dict with sample_rate, num_channels, num_frames, duration
        """
        path = Path(path)

        if LIBROSA_AVAILABLE:
            # Use librosa to get duration
            duration = librosa.get_duration(path=str(path))
            # Get sample rate without loading full file
            import soundfile as sf

            info = sf.info(str(path))
            return {
                "sample_rate": info.samplerate,
                "num_channels": info.channels,
                "num_frames": info.frames,
                "duration": duration,
                "encoding": info.subtype,
                "bits_per_sample": None,
            }
        elif TORCHAUDIO_AVAILABLE:
            info = torchaudio.info(str(path))
            return {
                "sample_rate": info.sample_rate,
                "num_channels": info.num_channels,
                "num_frames": info.num_frames,
                "duration": info.num_frames / info.sample_rate,
                "encoding": info.encoding,
                "bits_per_sample": info.bits_per_sample,
            }
        else:
            raise ImportError("No audio backend available")


if __name__ == "__main__":
    # Demo
    print("AudioProcessor demo")
    print("=" * 50)

    processor = AudioProcessor()

    # Create dummy audio
    dummy_audio = torch.randn(2, 32000)  # 2 channels, 2 seconds at 16kHz
    print(f"Input shape: {dummy_audio.shape}")

    audio, sr = processor.load_from_array(dummy_audio.numpy(), 16000)
    print(f"Output shape: {audio.shape}")
    print(f"Output sample rate: {sr}")
    print(f"Duration: {processor.get_duration(audio):.2f}s")

    # Test chunking
    long_audio = torch.randn(160000)  # 10 seconds
    chunks = processor.chunk(long_audio, chunk_duration=3.0, overlap=0.5)
    print(
        f"\nChunked {processor.get_duration(long_audio):.1f}s audio into {len(chunks)} chunks:"
    )
    for i, (chunk, start, end) in enumerate(chunks):
        print(f"  Chunk {i + 1}: {start:.1f}s - {end:.1f}s ({chunk.shape[0]} samples)")
