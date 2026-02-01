"""
AOXLM Inference Pipeline (Phase 1)

Simple inference: Audio -> Encoder -> CTC -> Verbatim + Timestamps

This is Phase 1 - encoder + CTC only. Guaranteed to work because
it's the same architecture as Wav2Vec2-CTC.

Example:
    >>> from aoxlm.inference import transcribe
    >>> result = transcribe("speech.wav")
    >>> print(result["text"])
    >>> for word in result["words"]:
    ...     print(f"{word['w']}: {word['s']:.2f}s - {word['e']:.2f}s")
"""

import json
from pathlib import Path
from typing import Generator, Optional, Union
import torch

from .models.encoder import SemanticEncoder, EncoderName, ENCODER_CONFIGS
from .models.ctc_head import CTCHead, CTCDecoder, DEFAULT_VOCAB, CTCOutput
from .data.audio import AudioProcessor


class AOXLMTranscriber:
    """
    AOXLM Phase 1 Transcription Pipeline.

    Encoder + CTC head for verbatim transcription with accurate timestamps.

    Example:
        >>> transcriber = AOXLMTranscriber("wavlm-large")
        >>> result = transcriber.transcribe("speech.wav")
        >>> print(result["text"])
        >>> print(result["words"])
    """

    def __init__(
        self,
        encoder_name: EncoderName = "wavlm-large",
        vocab: Optional[list] = None,
        device: Optional[str] = None,
        ctc_checkpoint: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            encoder_name: Which encoder to use (see ENCODER_CONFIGS)
            vocab: Custom vocabulary (default: character vocab)
            device: Device to run on (default: auto-detect)
            ctc_checkpoint: Path to trained CTC head weights (optional)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab or DEFAULT_VOCAB

        print(f"Initializing AOXLM Transcriber...")
        print(f"  Encoder: {encoder_name}")
        print(f"  Device: {self.device}")
        print(f"  Vocab size: {len(self.vocab)}")

        # Load encoder
        self.encoder = SemanticEncoder(
            encoder_name=encoder_name,
            freeze=True,
            device=self.device,
        )

        # Initialize CTC head
        self.ctc_head = CTCHead(
            encoder_dim=self.encoder.hidden_size,
            vocab_size=len(self.vocab),
        ).to(self.device)

        # Load CTC weights if provided
        if ctc_checkpoint:
            print(f"Loading CTC weights from {ctc_checkpoint}")
            self.ctc_head.load_state_dict(
                torch.load(ctc_checkpoint, map_location=self.device)
            )

        # Initialize decoder
        self.ctc_decoder = CTCDecoder(vocab=self.vocab)

        # Audio processor
        self.audio_processor = AudioProcessor()

        print("Ready!")

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with:
                - text: Transcribed text
                - words: List of words with timestamps
                - duration: Audio duration in seconds
        """
        # Load audio
        audio, sr = self.audio_processor.load(audio_path)
        duration = self.audio_processor.get_duration(audio)

        # Encode
        with torch.no_grad():
            encoder_output = self.encoder(audio.unsqueeze(0))

        # CTC forward
        logits = self.ctc_head(encoder_output)

        # Decode
        ctc_output = self.ctc_decoder.decode(
            logits,
            frame_rate=self.encoder.frame_rate,
        )

        # Convert to output format
        words = ctc_output.to_jsonl()

        return {
            "text": ctc_output.text,
            "words": words,
            "duration": duration,
            "encoder": self.encoder.encoder_name,
        }

    def transcribe_array(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Transcribe from audio tensor.

        Args:
            audio: Audio tensor [samples] or [batch, samples]
            sample_rate: Audio sample rate

        Returns:
            Same as transcribe()
        """
        # Process audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        duration = audio.shape[-1] / sample_rate

        # Encode
        with torch.no_grad():
            encoder_output = self.encoder(audio, sampling_rate=sample_rate)

        # CTC forward
        logits = self.ctc_head(encoder_output)

        # Decode
        ctc_output = self.ctc_decoder.decode(
            logits,
            frame_rate=self.encoder.frame_rate,
        )

        # Convert to output format
        words = ctc_output.to_jsonl()

        return {
            "text": ctc_output.text,
            "words": words,
            "duration": duration,
            "encoder": self.encoder.encoder_name,
        }

    def transcribe_streaming(
        self,
        audio_path: Union[str, Path],
    ) -> Generator[str, None, None]:
        """
        Transcribe with JSONL streaming output.

        Yields one JSON line per word, then a final summary.

        Args:
            audio_path: Path to audio file

        Yields:
            JSON lines (one per word)
        """
        result = self.transcribe(audio_path)

        # Yield each word
        for word in result["words"]:
            yield json.dumps(word)

        # Yield final summary
        summary = {
            "done": True,
            "text": result["text"],
            "dur": round(result["duration"], 2),
        }
        yield json.dumps(summary)


def transcribe(
    audio_path: Union[str, Path],
    encoder: EncoderName = "wavlm-large",
    device: Optional[str] = None,
) -> dict:
    """
    Simple one-shot transcription function.

    Args:
        audio_path: Path to audio file
        encoder: Which encoder to use
        device: Device to run on

    Returns:
        Transcription result dict
    """
    transcriber = AOXLMTranscriber(encoder_name=encoder, device=device)
    return transcriber.transcribe(audio_path)


def list_encoders():
    """Print available encoders."""
    from .models.encoder import list_encoders as _list

    _list()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m aoxlm.inference <audio_file> [encoder]")
        print("\nAvailable encoders:")
        for name, config in ENCODER_CONFIGS.items():
            print(f"  {name}: {config.best_for}")
        sys.exit(1)

    audio_path = sys.argv[1]
    encoder_name = sys.argv[2] if len(sys.argv) > 2 else "wavlm-large"

    print(f"\nTranscribing: {audio_path}")
    print(f"Encoder: {encoder_name}")
    print("=" * 50)

    transcriber = AOXLMTranscriber(encoder_name=encoder_name)

    print("\nStreaming output:")
    print("-" * 50)
    for line in transcriber.transcribe_streaming(audio_path):
        print(line)
