"""
AOXLM Models

Components:
- SemanticEncoder: WavLM/MMS/XLS-R audio encoders
- CTCHead: CTC decoding for transcription
- CEFRHeads: Classification heads for CEFR level prediction
"""

from .encoder import SemanticEncoder, ENCODER_CONFIGS
from .ctc_head import CTCHead, CTCDecoder
from .cefr_heads import (
    CEFRClassificationHead,
    CEFRMultiDimensionHeads,
    PoolingLayer,
    create_cefr_heads,
    CEFR_LEVELS,
    CEFR_DIMENSIONS,
    CEFR_LEVEL_TO_IDX,
    CEFR_IDX_TO_LEVEL,
)

__all__ = [
    # Encoder
    "SemanticEncoder",
    "ENCODER_CONFIGS",
    # CTC
    "CTCHead",
    "CTCDecoder",
    # CEFR Heads
    "CEFRClassificationHead",
    "CEFRMultiDimensionHeads",
    "PoolingLayer",
    "create_cefr_heads",
    "CEFR_LEVELS",
    "CEFR_DIMENSIONS",
    "CEFR_LEVEL_TO_IDX",
    "CEFR_IDX_TO_LEVEL",
]
