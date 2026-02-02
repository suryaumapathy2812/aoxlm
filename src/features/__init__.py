"""
Feature extraction modules for CEFR assessment.

Provides interpretable, research-based feature extraction for:
- Fluency (speech rate, pauses, hesitations, flow)
- Range (vocabulary diversity, lexical sophistication)
- Accuracy (grammar patterns, error detection)
- Phonology (pronunciation quality, prosody)

Also provides standalone calculation functions in `calculations` module
that can be imported independently for testing or sharing.
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

from .accuracy import (
    AccuracyAssessor,
    AccuracyFeatureExtractor,
    AccuracyScorer,
    AccuracyFeatures,
    AccuracyScore,
    GrammarError,
    assess_accuracy,
)

from .phonology import (
    PhonologyAssessor,
    PhonologyFeatureExtractor,
    PhonologyScorer,
    PhonologyFeatures,
    PhonologyScore,
    assess_phonology,
)

# Standalone calculations (shareable with teammates)
from . import calculations

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
    # Accuracy
    "AccuracyAssessor",
    "AccuracyFeatureExtractor",
    "AccuracyScorer",
    "AccuracyFeatures",
    "AccuracyScore",
    "GrammarError",
    "assess_accuracy",
    # Phonology
    "PhonologyAssessor",
    "PhonologyFeatureExtractor",
    "PhonologyScorer",
    "PhonologyFeatures",
    "PhonologyScore",
    "assess_phonology",
    # Calculations module
    "calculations",
]
