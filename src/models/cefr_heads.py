"""
CEFR Classification Heads

Classification heads for CEFR level prediction from encoder hidden states.

Each head outputs 4-class probabilities: A1, A2, B1, B2+

Example:
    >>> from src.models.cefr_heads import CEFRClassificationHead, CEFRMultiDimensionHeads
    >>>
    >>> # Single head
    >>> head = CEFRClassificationHead(input_dim=1024)
    >>> hidden_states = encoder(audio)  # [B, T, 1024]
    >>> output = head(hidden_states)
    >>> print(output["predicted_class"])  # 0=A1, 1=A2, 2=B1, 3=B2+
    >>>
    >>> # Multi-dimension heads
    >>> heads = CEFRMultiDimensionHeads(input_dim=1024)
    >>> outputs = heads(hidden_states)
    >>> print(outputs["fluency"]["predicted_class"])
    >>> print(outputs["range"]["predicted_class"])
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# CEFR level labels
CEFR_LEVELS = ["A1", "A2", "B1", "B2+"]
CEFR_LEVEL_TO_IDX = {level: idx for idx, level in enumerate(CEFR_LEVELS)}
CEFR_IDX_TO_LEVEL = {idx: level for level, idx in CEFR_LEVEL_TO_IDX.items()}

# CEFR dimensions
CEFR_DIMENSIONS = ["fluency", "range", "accuracy", "phonology", "coherence", "overall"]


@dataclass
class CEFRPrediction:
    """Prediction output from a CEFR classification head."""

    logits: torch.Tensor  # [B, 4] raw logits
    probs: torch.Tensor  # [B, 4] softmax probabilities
    predicted_idx: torch.Tensor  # [B] predicted class index (0-3)
    predicted_level: List[str]  # [B] predicted CEFR level string
    confidence: torch.Tensor  # [B] confidence of prediction

    def to_dict(self) -> Dict:
        return {
            "logits": self.logits,
            "probs": self.probs,
            "predicted_idx": self.predicted_idx,
            "predicted_level": self.predicted_level,
            "confidence": self.confidence,
        }


class PoolingLayer(nn.Module):
    """Pooling layer for variable-length sequences."""

    def __init__(
        self,
        method: Literal["mean", "max", "attention", "last"] = "mean",
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.method = method

        if method == "attention" and input_dim is not None:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.Tanh(),
                nn.Linear(input_dim // 4, 1),
            )
        else:
            self.attention = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool variable-length sequences to fixed-length vectors.

        Args:
            x: Input tensor [B, T, D]
            mask: Optional attention mask [B, T]

        Returns:
            Pooled tensor [B, D]
        """
        if self.method == "mean":
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)

        elif self.method == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        elif self.method == "attention":
            if self.attention is None:
                raise ValueError("Attention pooling requires input_dim")

            # Compute attention weights
            attn_weights = self.attention(x).squeeze(-1)  # [B, T]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, T]

            # Weighted sum
            return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, D]

        elif self.method == "last":
            return x[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.method}")


class CEFRClassificationHead(nn.Module):
    """
    Classification head for a single CEFR dimension.

    Takes encoder hidden states and outputs 4-class predictions (A1, A2, B1, B2+).

    Architecture:
        Pool -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Softmax

    Example:
        >>> head = CEFRClassificationHead(input_dim=1024)
        >>> hidden_states = torch.randn(2, 100, 1024)  # [B, T, D]
        >>> output = head(hidden_states)
        >>> print(output["predicted_level"])  # ['A2', 'B1']
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Pooling layer
        self.pooling = PoolingLayer(method=pooling, input_dim=input_dim)

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder hidden states [B, T, D]
            attention_mask: Optional mask [B, T]

        Returns:
            Dict with logits, probs, predicted_idx, predicted_level, confidence
        """
        # Pool to fixed length
        pooled = self.pooling(hidden_states, attention_mask)  # [B, D]

        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        probs = F.softmax(logits, dim=-1)

        # Get predictions
        confidence, predicted_idx = probs.max(dim=-1)
        predicted_level = [CEFR_IDX_TO_LEVEL[idx.item()] for idx in predicted_idx]

        return {
            "logits": logits,
            "probs": probs,
            "predicted_idx": predicted_idx,
            "predicted_level": predicted_level,
            "confidence": confidence,
        }

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            hidden_states: Encoder hidden states [B, T, D]
            labels: Ground truth labels [B] (0-3 for A1-B2+)
            attention_mask: Optional mask [B, T]
            class_weights: Optional class weights for imbalanced data

        Returns:
            Loss tensor
        """
        output = self.forward(hidden_states, attention_mask)
        logits = output["logits"]

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        return loss_fn(logits, labels)


class CEFRMultiDimensionHeads(nn.Module):
    """
    Multi-head classifier for all CEFR dimensions.

    Contains separate classification heads for:
    - Fluency
    - Range
    - Accuracy (Phase 2)
    - Phonology (Phase 2)
    - Coherence (Phase 3)
    - Overall

    Example:
        >>> heads = CEFRMultiDimensionHeads(input_dim=1024)
        >>> hidden_states = torch.randn(2, 100, 1024)
        >>> outputs = heads(hidden_states)
        >>> print(outputs["fluency"]["predicted_level"])
        >>> print(outputs["overall"]["confidence"])
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",
        dimensions: Optional[List[str]] = None,
    ):
        super().__init__()

        self.dimensions = dimensions or ["fluency", "range", "overall"]

        # Create a head for each dimension
        self.heads = nn.ModuleDict(
            {
                dim: CEFRClassificationHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    pooling=pooling,
                )
                for dim in self.dimensions
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass for all dimensions.

        Args:
            hidden_states: Encoder hidden states [B, T, D]
            attention_mask: Optional mask [B, T]

        Returns:
            Dict mapping dimension name to prediction dict
        """
        outputs = {}
        for dim_name, head in self.heads.items():
            outputs[dim_name] = head(hidden_states, attention_mask)
        return outputs

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss across all dimensions.

        Args:
            hidden_states: Encoder hidden states [B, T, D]
            labels: Dict mapping dimension name to labels [B]
            attention_mask: Optional mask [B, T]
            dimension_weights: Optional weights for each dimension

        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        dimension_weights = dimension_weights or {dim: 1.0 for dim in self.dimensions}

        for dim_name, head in self.heads.items():
            if dim_name in labels:
                loss = head.compute_loss(
                    hidden_states, labels[dim_name], attention_mask
                )
                weight = dimension_weights.get(dim_name, 1.0)
                total_loss = total_loss + weight * loss

        return total_loss

    def add_dimension(
        self,
        dimension: str,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        """Add a new dimension head (for Phase 2/3 expansion)."""
        if dimension not in self.dimensions:
            self.dimensions.append(dimension)
            self.heads[dimension] = CEFRClassificationHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                pooling=pooling,
            )


def create_cefr_heads(
    phase: Literal[1, 2, 3] = 1,
    input_dim: int = 1024,
    **kwargs,
) -> CEFRMultiDimensionHeads:
    """
    Factory function to create CEFR heads for a specific phase.

    Args:
        phase: Implementation phase (1, 2, or 3)
        input_dim: Input dimension from encoder
        **kwargs: Additional arguments for heads

    Returns:
        CEFRMultiDimensionHeads configured for the phase
    """
    dimensions_by_phase = {
        1: ["fluency", "range", "overall"],
        2: ["fluency", "range", "accuracy", "phonology", "overall"],
        3: ["fluency", "range", "accuracy", "phonology", "coherence", "overall"],
    }

    dimensions = dimensions_by_phase.get(phase, dimensions_by_phase[1])

    return CEFRMultiDimensionHeads(
        input_dim=input_dim,
        dimensions=dimensions,
        **kwargs,
    )


if __name__ == "__main__":
    # Demo
    print("CEFR Classification Heads Demo")
    print("=" * 50)

    # Create dummy hidden states
    batch_size = 2
    seq_length = 100
    hidden_dim = 1024

    hidden_states = torch.randn(batch_size, seq_length, hidden_dim)

    # Single head
    print("\nSingle Head (Overall):")
    head = CEFRClassificationHead(input_dim=hidden_dim)
    output = head(hidden_states)
    print(f"  Predicted levels: {output['predicted_level']}")
    print(f"  Confidence: {output['confidence'].tolist()}")
    print(f"  Probabilities shape: {output['probs'].shape}")

    # Multi-dimension heads (Phase 1)
    print("\nMulti-Dimension Heads (Phase 1):")
    heads = create_cefr_heads(phase=1, input_dim=hidden_dim)
    outputs = heads(hidden_states)
    for dim, out in outputs.items():
        print(f"  {dim}: {out['predicted_level']} (conf: {out['confidence'].tolist()})")

    # Multi-dimension heads (Phase 3)
    print("\nMulti-Dimension Heads (Phase 3):")
    heads = create_cefr_heads(phase=3, input_dim=hidden_dim)
    outputs = heads(hidden_states)
    for dim, out in outputs.items():
        print(f"  {dim}: {out['predicted_level']} (conf: {out['confidence'].tolist()})")

    # Test loss computation
    print("\nLoss Computation:")
    labels = {
        "fluency": torch.tensor([1, 2]),  # A2, B1
        "range": torch.tensor([1, 2]),
        "overall": torch.tensor([1, 2]),
    }
    heads = create_cefr_heads(phase=1, input_dim=hidden_dim)
    loss = heads.compute_loss(hidden_states, labels)
    print(f"  Combined loss: {loss.item():.4f}")

    # Parameter count
    total_params = sum(p.numel() for p in heads.parameters())
    print(f"\nTotal parameters (Phase 1): {total_params:,} ({total_params / 1e6:.2f}M)")
