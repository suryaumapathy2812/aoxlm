"""
Semantic Audio Encoder

Unified interface for loading semantic audio encoders:
- WavLM (microsoft/wavlm-large)
- MMS (facebook/mms-300m)
- XLS-R (facebook/wav2vec2-xls-r-300m)
- UniSpeech-SAT (microsoft/unispeech-sat-large)
- HuBERT (facebook/hubert-large-ls960-ft)

All encoders output frame-level embeddings [B, T, D] at ~50Hz (20ms per frame).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
    HubertModel,
    AutoModel,
    AutoFeatureExtractor,
)


@dataclass
class EncoderConfig:
    """Configuration for a semantic encoder."""

    name: str
    hf_id: str
    hidden_size: int
    num_layers: int
    frame_rate: float  # Hz (frames per second)
    model_class: str  # Which HF class to use
    best_for: str  # What this encoder is best for


# Encoder configurations
ENCODER_CONFIGS = {
    "wavlm-large": EncoderConfig(
        name="WavLM-large",
        hf_id="microsoft/wavlm-large",
        hidden_size=1024,
        num_layers=24,
        frame_rate=50.0,  # 20ms per frame
        model_class="WavLMModel",
        best_for="General semantic understanding, noise robustness",
    ),
    "wavlm-base": EncoderConfig(
        name="WavLM-base",
        hf_id="microsoft/wavlm-base-plus",
        hidden_size=768,
        num_layers=12,
        frame_rate=50.0,
        model_class="WavLMModel",
        best_for="Smaller/faster option",
    ),
    "mms-300m": EncoderConfig(
        name="MMS-300M",
        hf_id="facebook/mms-300m",
        hidden_size=1024,
        num_layers=24,
        frame_rate=50.0,
        model_class="Wav2Vec2Model",
        best_for="Code-switching, 1000+ languages",
    ),
    "mms-1b": EncoderConfig(
        name="MMS-1B",
        hf_id="facebook/mms-1b",
        hidden_size=1280,
        num_layers=48,
        frame_rate=50.0,
        model_class="Wav2Vec2Model",
        best_for="Maximum multilingual coverage",
    ),
    "xls-r-300m": EncoderConfig(
        name="XLS-R-300M",
        hf_id="facebook/wav2vec2-xls-r-300m",
        hidden_size=1024,
        num_layers=24,
        frame_rate=50.0,
        model_class="Wav2Vec2Model",
        best_for="Cross-lingual, accents, 128 languages",
    ),
    "xls-r-1b": EncoderConfig(
        name="XLS-R-1B",
        hf_id="facebook/wav2vec2-xls-r-1b",
        hidden_size=1280,
        num_layers=48,
        frame_rate=50.0,
        model_class="Wav2Vec2Model",
        best_for="Maximum cross-lingual scale",
    ),
    "unispeech-sat": EncoderConfig(
        name="UniSpeech-SAT-large",
        hf_id="microsoft/unispeech-sat-large",
        hidden_size=1024,
        num_layers=24,
        frame_rate=50.0,
        model_class="Wav2Vec2Model",  # Compatible architecture
        best_for="Speaker variation, different speaking styles",
    ),
    "hubert-large": EncoderConfig(
        name="HuBERT-large",
        hf_id="facebook/hubert-large-ls960-ft",
        hidden_size=1024,
        num_layers=24,
        frame_rate=50.0,
        model_class="HubertModel",
        best_for="Baseline semantic encoder",
    ),
}

EncoderName = Literal[
    "wavlm-large",
    "wavlm-base",
    "mms-300m",
    "mms-1b",
    "xls-r-300m",
    "xls-r-1b",
    "unispeech-sat",
    "hubert-large",
]


class SemanticEncoder(nn.Module):
    """
    Unified semantic audio encoder.

    Wraps different HuggingFace models with a consistent interface.
    All outputs are frame-level embeddings [B, T, D] at ~50Hz.

    Example:
        >>> encoder = SemanticEncoder("wavlm-large")
        >>> audio = torch.randn(1, 16000 * 10)  # 10 seconds at 16kHz
        >>> features = encoder(audio)
        >>> print(features.shape)  # [1, 500, 1024] - 500 frames for 10s
    """

    def __init__(
        self,
        encoder_name: EncoderName = "wavlm-large",
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(
                f"Unknown encoder: {encoder_name}. "
                f"Available: {list(ENCODER_CONFIGS.keys())}"
            )

        self.config = ENCODER_CONFIGS[encoder_name]
        self.encoder_name = encoder_name
        self.freeze = freeze
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and feature extractor
        self.model, self.feature_extractor = self._load_model()

        if freeze:
            self._freeze_model()

        self.to(self.device)

    def _load_model(self) -> Tuple[nn.Module, Wav2Vec2FeatureExtractor]:
        """Load the appropriate HuggingFace model and feature extractor."""
        hf_id = self.config.hf_id
        model_class = self.config.model_class

        print(f"Loading {self.config.name} from {hf_id}...")

        # Load feature extractor (same for all Wav2Vec2-family models)
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id)
        except Exception:
            # Fallback for some models
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_id)

        # Load model based on class
        if model_class == "WavLMModel":
            model = WavLMModel.from_pretrained(hf_id)
        elif model_class == "HubertModel":
            model = HubertModel.from_pretrained(hf_id)
        elif model_class == "Wav2Vec2Model":
            model = Wav2Vec2Model.from_pretrained(hf_id)
        else:
            # Generic fallback
            model = AutoModel.from_pretrained(hf_id)

        print(f"Loaded {self.config.name}: {self._count_params(model):.1f}M params")

        return model, feature_extractor

    def _freeze_model(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print(f"Encoder frozen (requires_grad=False)")

    def _count_params(self, model: nn.Module) -> float:
        """Count parameters in millions."""
        return sum(p.numel() for p in model.parameters()) / 1e6

    @property
    def hidden_size(self) -> int:
        """Output dimension of the encoder."""
        return self.config.hidden_size

    @property
    def frame_rate(self) -> float:
        """Output frames per second (Hz)."""
        return self.config.frame_rate

    @property
    def frame_duration(self) -> float:
        """Duration of each frame in seconds."""
        return 1.0 / self.config.frame_rate

    def preprocess(
        self,
        audio: torch.Tensor,
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Preprocess audio for the encoder.

        Args:
            audio: Raw audio tensor [B, samples] or [samples]
            sampling_rate: Audio sampling rate (default 16kHz)

        Returns:
            Preprocessed input values tensor
        """
        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Convert to numpy for feature extractor
        audio_np = audio.cpu().numpy()

        # Process through feature extractor
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        return inputs.input_values.to(self.device)

    def forward(
        self,
        audio: torch.Tensor,
        sampling_rate: int = 16000,
        preprocess: bool = True,
    ) -> torch.Tensor:
        """
        Encode audio to frame-level semantic embeddings.

        Args:
            audio: Audio tensor [B, samples] or preprocessed [B, samples]
            sampling_rate: Audio sampling rate (only used if preprocess=True)
            preprocess: Whether to run preprocessing

        Returns:
            Frame-level embeddings [B, T, D] where:
                - B = batch size
                - T = number of frames (~50 frames per second)
                - D = hidden_size (768-1280 depending on encoder)
        """
        if preprocess:
            input_values = self.preprocess(audio, sampling_rate)
        else:
            input_values = audio.to(self.device)

        # Forward through encoder
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model(input_values)

        # Get last hidden state [B, T, D]
        hidden_states = outputs.last_hidden_state

        return hidden_states

    def get_output_length(self, input_length: int, sampling_rate: int = 16000) -> int:
        """
        Calculate output sequence length for a given input length.

        Args:
            input_length: Number of input samples
            sampling_rate: Audio sampling rate

        Returns:
            Number of output frames
        """
        # Wav2Vec2-family models downsample by ~320x (16000/50 = 320)
        # This is approximate - actual may vary slightly
        duration = input_length / sampling_rate
        return int(duration * self.frame_rate)

    def __repr__(self) -> str:
        return (
            f"SemanticEncoder(\n"
            f"  name={self.config.name},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  frame_rate={self.frame_rate}Hz,\n"
            f"  frozen={self.freeze},\n"
            f"  best_for='{self.config.best_for}'\n"
            f")"
        )


def list_encoders() -> None:
    """Print available encoders and their properties."""
    print("Available Semantic Encoders:")
    print("=" * 80)
    for name, config in ENCODER_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  HuggingFace: {config.hf_id}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Best for: {config.best_for}")
    print()


if __name__ == "__main__":
    list_encoders()
