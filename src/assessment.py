"""
CEFR Assessment Pipeline

End-to-end pipeline for CEFR speaking assessment:
    Audio -> Transcription + Semantic Encoding -> CEFR Level Predictions

Components:
- Whisper + WhisperX: Transcription with word-level timestamps
- WavLM: Semantic encoder for hidden state extraction
- CEFR Heads: Classification heads for each assessment dimension

Example:
    >>> from src.assessment import CEFRAssessor
    >>> assessor = CEFRAssessor()
    >>> result = assessor.assess("audio.wav")
    >>> print(result.overall_level)  # 'B1'
    >>> print(result.dimension_scores)  # {'fluency': 'A2', 'range': 'B1', ...}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
import torch
import torchaudio

from .transcribe import Transcriber, TranscriptionResult
from .models.encoder import SemanticEncoder, ENCODER_CONFIGS
from .models.cefr_heads import (
    CEFRMultiDimensionHeads,
    create_cefr_heads,
    CEFR_LEVELS,
    CEFR_DIMENSIONS,
)


@dataclass
class CEFRScore:
    """Score for a single CEFR dimension."""

    dimension: str
    level: str  # A1, A2, B1, B2+
    confidence: float
    probabilities: Dict[str, float]  # {A1: 0.1, A2: 0.3, B1: 0.5, B2+: 0.1}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
        }


@dataclass
class AssessmentResult:
    """Complete CEFR assessment result."""

    # Audio info
    audio_path: str
    duration: float

    # Transcription
    transcription: str
    language: str
    word_count: int

    # Fluency features (from timestamps)
    fluency_features: Dict[str, float]

    # CEFR scores by dimension
    dimension_scores: Dict[str, CEFRScore]

    # Overall level (from 'overall' head or computed)
    overall_level: str
    overall_confidence: float

    # Raw data for debugging
    _hidden_states_shape: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "duration": round(self.duration, 2),
            "transcription": self.transcription,
            "language": self.language,
            "word_count": self.word_count,
            "fluency_features": self.fluency_features,
            "dimension_scores": {
                dim: score.to_dict() for dim, score in self.dimension_scores.items()
            },
            "overall_level": self.overall_level,
            "overall_confidence": round(self.overall_confidence, 3),
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"CEFR Assessment Result",
            f"=" * 40,
            f"Audio: {self.audio_path}",
            f"Duration: {self.duration:.1f}s | Words: {self.word_count}",
            f"",
            f"Overall Level: {self.overall_level} (confidence: {self.overall_confidence:.1%})",
            f"",
            f"Dimension Scores:",
        ]
        for dim, score in self.dimension_scores.items():
            lines.append(f"  {dim:12}: {score.level} ({score.confidence:.1%})")

        lines.extend(
            [
                f"",
                f"Fluency Features:",
                f"  WPM: {self.fluency_features.get('wpm', 0):.1f}",
                f"  Pause ratio: {self.fluency_features.get('pause_ratio', 0):.1%}",
                f"  Num pauses: {self.fluency_features.get('num_pauses', 0)}",
            ]
        )

        return "\n".join(lines)


class CEFRAssessor:
    """
    End-to-end CEFR speaking assessor.

    Combines transcription (Whisper+WhisperX), semantic encoding (WavLM),
    and classification (CEFR heads) into a single pipeline.

    Example:
        >>> assessor = CEFRAssessor(phase=1)  # fluency, range, overall
        >>> result = assessor.assess("student_response.wav")
        >>> print(result.overall_level)
        >>> print(result.summary())

    Phases:
        - Phase 1: fluency, range, overall
        - Phase 2: + accuracy, phonology
        - Phase 3: + coherence (full 6 dimensions)
    """

    def __init__(
        self,
        phase: Literal[1, 2, 3] = 1,
        encoder_name: str = "wavlm-large",
        whisper_model: str = "large-v3",
        device: Optional[str] = None,
        heads_checkpoint: Optional[str] = None,
    ):
        """
        Initialize the CEFR assessor.

        Args:
            phase: Assessment phase (1, 2, or 3) - determines which dimensions
            encoder_name: Semantic encoder to use (default: wavlm-large)
            whisper_model: Whisper model size (default: large-v3)
            device: Device to run on (auto-detected if None)
            heads_checkpoint: Path to trained CEFR heads checkpoint (optional)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.phase = phase

        print(f"Initializing CEFR Assessor (Phase {phase}) on {self.device}")
        print("-" * 50)

        # 1. Transcriber (Whisper + WhisperX)
        print("\n[1/3] Loading Transcriber...")
        self.transcriber = Transcriber(
            model_size=whisper_model,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "float32",
        )

        # 2. Semantic Encoder (WavLM)
        print("\n[2/3] Loading Semantic Encoder...")
        self.encoder = SemanticEncoder(
            encoder_name=encoder_name,
            freeze=True,
            device=self.device,
        )

        # 3. CEFR Classification Heads
        print("\n[3/3] Loading CEFR Heads...")
        self.heads = create_cefr_heads(
            phase=phase,
            input_dim=self.encoder.hidden_size,
        )
        self.heads.to(self.device)

        # Load checkpoint if provided
        if heads_checkpoint:
            print(f"Loading heads from checkpoint: {heads_checkpoint}")
            state_dict = torch.load(heads_checkpoint, map_location=self.device)
            self.heads.load_state_dict(state_dict)

        self.heads.eval()

        print("\n" + "=" * 50)
        print("CEFR Assessor ready!")
        print(f"  Dimensions: {self.heads.dimensions}")
        print(f"  Encoder: {encoder_name} ({self.encoder.hidden_size}D)")
        print("=" * 50)

    def _load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # [samples]

    def assess(
        self,
        audio: Union[str, Path],
        language: Optional[str] = None,
        return_hidden_states: bool = False,
    ) -> AssessmentResult:
        """
        Perform full CEFR assessment on an audio file.

        Args:
            audio: Path to audio file
            language: Override language detection (None for auto)
            return_hidden_states: Whether to include hidden states in result

        Returns:
            AssessmentResult with all scores and metadata
        """
        audio_path = str(audio)

        # Step 1: Transcribe with word-level timestamps
        transcription = self.transcriber.transcribe(audio_path, language=language)
        fluency_features = transcription.get_fluency_features()

        # Step 2: Load audio and get encoder hidden states
        waveform = self._load_audio(audio_path)
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, samples]

        with torch.no_grad():
            hidden_states = self.encoder(waveform, preprocess=True)  # [1, T, D]

        # Step 3: Get CEFR predictions from all heads
        with torch.no_grad():
            head_outputs = self.heads(hidden_states)

        # Step 4: Build dimension scores
        dimension_scores = {}
        for dim_name, output in head_outputs.items():
            probs = output["probs"][0].cpu().tolist()  # First (only) batch item
            prob_dict = {level: probs[i] for i, level in enumerate(CEFR_LEVELS)}

            dimension_scores[dim_name] = CEFRScore(
                dimension=dim_name,
                level=output["predicted_level"][0],
                confidence=output["confidence"][0].item(),
                probabilities=prob_dict,
            )

        # Get overall level
        if "overall" in dimension_scores:
            overall_level = dimension_scores["overall"].level
            overall_confidence = dimension_scores["overall"].confidence
        else:
            # Fallback: average of dimension confidences
            overall_level = max(
                dimension_scores.values(), key=lambda x: x.confidence
            ).level
            overall_confidence = sum(
                s.confidence for s in dimension_scores.values()
            ) / len(dimension_scores)

        # Build result
        result = AssessmentResult(
            audio_path=audio_path,
            duration=transcription.duration,
            transcription=transcription.text,
            language=transcription.language,
            word_count=len(transcription.words),
            fluency_features=fluency_features,
            dimension_scores=dimension_scores,
            overall_level=overall_level,
            overall_confidence=overall_confidence,
            _hidden_states_shape=tuple(hidden_states.shape)
            if return_hidden_states
            else None,
        )

        return result

    def assess_batch(
        self,
        audio_files: List[Union[str, Path]],
        language: Optional[str] = None,
    ) -> List[AssessmentResult]:
        """
        Assess multiple audio files.

        Note: Currently processes sequentially. Batch encoding could be added
        for better GPU utilization.

        Args:
            audio_files: List of paths to audio files
            language: Override language detection

        Returns:
            List of AssessmentResult objects
        """
        results = []
        for i, audio_path in enumerate(audio_files):
            print(f"Processing {i + 1}/{len(audio_files)}: {audio_path}")
            try:
                result = self.assess(audio_path, language=language)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        return results

    def get_dimension_names(self) -> List[str]:
        """Get the names of dimensions being assessed."""
        return self.heads.dimensions.copy()


def assess(
    audio: Union[str, Path],
    phase: Literal[1, 2, 3] = 1,
    encoder: str = "wavlm-large",
    whisper: str = "large-v3",
) -> AssessmentResult:
    """
    Quick one-shot assessment function.

    Args:
        audio: Path to audio file
        phase: Assessment phase (1, 2, or 3)
        encoder: Encoder name
        whisper: Whisper model size

    Returns:
        AssessmentResult
    """
    assessor = CEFRAssessor(
        phase=phase,
        encoder_name=encoder,
        whisper_model=whisper,
    )
    return assessor.assess(audio)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.assessment <audio_file> [phase]")
        print("  phase: 1 (fluency, range, overall)")
        print("         2 (+ accuracy, phonology)")
        print("         3 (+ coherence)")
        sys.exit(1)

    audio_path = sys.argv[1]
    phase = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"\nAssessing: {audio_path}")
    print(f"Phase: {phase}")
    print("=" * 50)

    assessor = CEFRAssessor(phase=phase)
    result = assessor.assess(audio_path)

    print("\n")
    print(result.summary())

    print("\n\nFull JSON output:")
    import json

    print(json.dumps(result.to_dict(), indent=2))
