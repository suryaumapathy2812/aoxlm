"""
CTC Head for Verbatim Transcription

A simple projection layer that maps encoder features to vocabulary logits.
CTC decoding gives us:
1. Verbatim transcription (what was spoken)
2. Frame-level alignments (timestamps are COMPUTED, not generated)
3. Confidence scores (from softmax probabilities)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CTCAlignment:
    """A single aligned token with timing information."""

    token: str
    token_id: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float  # 0-1


@dataclass
class CTCOutput:
    """Output from CTC decoding."""

    text: str  # Decoded text
    alignments: List[CTCAlignment]  # Per-token alignments
    frame_rate: float  # Frames per second

    def to_jsonl(self) -> List[dict]:
        """Convert to JSONL format (one dict per word)."""
        words = []
        current_word = ""
        word_start = None
        word_alignments = []

        for align in self.alignments:
            if align.token.startswith("##") or align.token.startswith("_"):
                # Subword continuation
                current_word += align.token.replace("##", "").replace("_", "")
                word_alignments.append(align)
            elif align.token in [" ", ""]:
                # Space or empty - emit current word
                if current_word and word_alignments:
                    words.append(
                        {
                            "w": current_word,
                            "s": round(word_alignments[0].start_time, 2),
                            "e": round(word_alignments[-1].end_time, 2),
                            "c": round(
                                sum(a.confidence for a in word_alignments)
                                / len(word_alignments),
                                2,
                            ),
                        }
                    )
                current_word = ""
                word_alignments = []
            else:
                # New word
                if current_word and word_alignments:
                    words.append(
                        {
                            "w": current_word,
                            "s": round(word_alignments[0].start_time, 2),
                            "e": round(word_alignments[-1].end_time, 2),
                            "c": round(
                                sum(a.confidence for a in word_alignments)
                                / len(word_alignments),
                                2,
                            ),
                        }
                    )
                current_word = align.token
                word_start = align.start_time
                word_alignments = [align]

        # Don't forget the last word
        if current_word and word_alignments:
            words.append(
                {
                    "w": current_word,
                    "s": round(word_alignments[0].start_time, 2),
                    "e": round(word_alignments[-1].end_time, 2),
                    "c": round(
                        sum(a.confidence for a in word_alignments)
                        / len(word_alignments),
                        2,
                    ),
                }
            )

        return words


class CTCHead(nn.Module):
    """
    CTC projection head.

    Maps encoder features [B, T, D] to vocabulary logits [B, T, V].
    Simple but effective - this is the standard Wav2Vec2-CTC architecture.

    Example:
        >>> ctc = CTCHead(encoder_dim=1024, vocab_size=32)
        >>> encoder_output = torch.randn(1, 500, 1024)
        >>> logits = ctc(encoder_output)
        >>> print(logits.shape)  # [1, 500, 32]
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        vocab_size: int = 32,  # Characters by default
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size

        # Optional projection layer (can help with training stability)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(encoder_dim, vocab_size)

        # Initialize weights
        nn.init.normal_(self.projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project encoder output to vocabulary logits.

        Args:
            encoder_output: Encoder features [B, T, D]

        Returns:
            Logits [B, T, V] where V = vocab_size
        """
        x = self.dropout(encoder_output)
        logits = self.projection(x)
        return logits

    def get_log_probs(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for CTC loss."""
        logits = self.forward(encoder_output)
        return F.log_softmax(logits, dim=-1)


class CTCDecoder:
    """
    CTC decoder with alignment extraction.

    Decodes CTC logits to text and extracts frame-level alignments.
    Supports greedy decoding (fast) and beam search (better quality).

    Example:
        >>> decoder = CTCDecoder(vocab=list("abcdefghijklmnopqrstuvwxyz "))
        >>> logits = torch.randn(1, 100, 28)  # 100 frames, 28 vocab
        >>> output = decoder.decode(logits, frame_rate=50.0)
        >>> print(output.text)
        >>> print(output.alignments[0])
    """

    def __init__(
        self,
        vocab: List[str],
        blank_id: int = 0,
    ):
        """
        Initialize CTC decoder.

        Args:
            vocab: List of vocabulary tokens (index 0 should be blank)
            blank_id: ID of the blank token (default 0)
        """
        self.vocab = vocab
        self.blank_id = blank_id
        self.id_to_token = {i: t for i, t in enumerate(vocab)}
        self.token_to_id = {t: i for i, t in enumerate(vocab)}

    def decode(
        self,
        logits: torch.Tensor,
        frame_rate: float = 50.0,
        method: str = "greedy",
    ) -> CTCOutput:
        """
        Decode CTC logits to text with alignments.

        Args:
            logits: CTC logits [B, T, V] or [T, V]
            frame_rate: Frames per second (for time calculation)
            method: "greedy" or "beam" (beam not implemented yet)

        Returns:
            CTCOutput with text and alignments
        """
        if logits.dim() == 3:
            logits = logits[0]  # Take first batch item

        if method == "greedy":
            return self._greedy_decode(logits, frame_rate)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def _greedy_decode(
        self,
        logits: torch.Tensor,
        frame_rate: float,
    ) -> CTCOutput:
        """
        Greedy CTC decoding with alignment extraction.

        For each frame, take the most likely token.
        Collapse repeated tokens and remove blanks.
        Track frame boundaries for alignments.
        """
        # Get predictions and probabilities
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)  # [T]
        confidences = probs.max(dim=-1).values  # [T]

        # Convert to lists
        predictions = predictions.cpu().tolist()
        confidences = confidences.cpu().tolist()

        # Decode with alignment tracking
        alignments: List[CTCAlignment] = []
        prev_token_id = None
        token_start_frame = 0
        token_confidences = []

        frame_duration = 1.0 / frame_rate

        for frame_idx, (token_id, conf) in enumerate(zip(predictions, confidences)):
            if token_id == prev_token_id:
                # Same token continues
                token_confidences.append(conf)
            else:
                # Token changed - emit previous token if not blank
                if prev_token_id is not None and prev_token_id != self.blank_id:
                    token = self.id_to_token.get(prev_token_id, "?")
                    avg_conf = (
                        sum(token_confidences) / len(token_confidences)
                        if token_confidences
                        else 0
                    )

                    alignments.append(
                        CTCAlignment(
                            token=token,
                            token_id=prev_token_id,
                            start_frame=token_start_frame,
                            end_frame=frame_idx - 1,
                            start_time=token_start_frame * frame_duration,
                            end_time=(frame_idx - 1) * frame_duration,
                            confidence=avg_conf,
                        )
                    )

                # Start new token
                prev_token_id = token_id
                token_start_frame = frame_idx
                token_confidences = [conf]

        # Don't forget the last token
        if prev_token_id is not None and prev_token_id != self.blank_id:
            token = self.id_to_token.get(prev_token_id, "?")
            avg_conf = (
                sum(token_confidences) / len(token_confidences)
                if token_confidences
                else 0
            )

            alignments.append(
                CTCAlignment(
                    token=token,
                    token_id=prev_token_id,
                    start_frame=token_start_frame,
                    end_frame=len(predictions) - 1,
                    start_time=token_start_frame * frame_duration,
                    end_time=(len(predictions) - 1) * frame_duration,
                    confidence=avg_conf,
                )
            )

        # Build text
        text = "".join(a.token for a in alignments)

        return CTCOutput(
            text=text,
            alignments=alignments,
            frame_rate=frame_rate,
        )


def create_character_vocab(include_space: bool = True) -> List[str]:
    """
    Create a simple character vocabulary.

    Returns:
        List of characters with blank at index 0
    """
    # Blank token first (required for CTC)
    vocab = ["<blank>"]

    # Lowercase letters
    vocab.extend(list("abcdefghijklmnopqrstuvwxyz"))

    # Common punctuation
    vocab.extend(list(".,!?'-"))

    # Space
    if include_space:
        vocab.append(" ")

    # Numbers
    vocab.extend(list("0123456789"))

    return vocab


# Default vocabulary
DEFAULT_VOCAB = create_character_vocab()


if __name__ == "__main__":
    # Demo
    print("Default vocabulary:")
    print(DEFAULT_VOCAB)
    print(f"Size: {len(DEFAULT_VOCAB)}")

    # Test CTC head
    print("\nTesting CTCHead...")
    ctc = CTCHead(encoder_dim=1024, vocab_size=len(DEFAULT_VOCAB))
    dummy_encoder = torch.randn(1, 100, 1024)
    logits = ctc(dummy_encoder)
    print(f"Input shape: {dummy_encoder.shape}")
    print(f"Output shape: {logits.shape}")

    # Test decoder
    print("\nTesting CTCDecoder...")
    decoder = CTCDecoder(vocab=DEFAULT_VOCAB)
    output = decoder.decode(logits, frame_rate=50.0)
    print(f"Decoded text: '{output.text}'")
    print(f"Alignments: {len(output.alignments)} tokens")
    if output.alignments:
        print(f"First alignment: {output.alignments[0]}")
