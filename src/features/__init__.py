"""
Feature extraction modules for CEFR assessment.

Provides interpretable, research-based feature extraction for:
- Fluency (speech rate, pauses, hesitations, flow)
- Range (vocabulary diversity, lexical sophistication)
- Accuracy (grammar patterns) - coming soon
"""

from .fluency import (
    FluencyAssessor,
    FluencyFeatureExtractor,
    FluencyScorer,
    FluencyFeatures,
    FluencyScore,
    assess_fluency,
)

from .range import (
    RangeAssessor,
    RangeFeatureExtractor,
    RangeScorer,
    RangeFeatures,
    RangeScore,
    assess_range,
)

__all__ = [
    # Fluency
    "FluencyAssessor",
    "FluencyFeatureExtractor",
    "FluencyScorer",
    "FluencyFeatures",
    "FluencyScore",
    "assess_fluency",
    # Range
    "RangeAssessor",
    "RangeFeatureExtractor",
    "RangeScorer",
    "RangeFeatures",
    "RangeScore",
    "assess_range",
]
