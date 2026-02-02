"""
Coherence & Cohesion Assessment Module for CEFR Speech Assessment

Analyzes how well ideas connect and flow in spoken language.

CEFR Coherence Criteria:
- A1: Can link words with basic connectors (and, then)
- A2: Can link groups of words with simple connectors (but, because)
- B1: Can link ideas with a range of connectors (however, although, therefore)
- B2+: Uses cohesive devices effectively, clear logical flow

Features Extracted:
- Discourse marker count and variety
- Connector sophistication level
- Sentence-to-sentence coherence
- Topic consistency

Usage:
    >>> from src.features.coherence import CoherenceAssessor
    >>> assessor = CoherenceAssessor()
    >>> result = assessor.assess(text)
    >>> print(result.level)  # "B1"
    >>> print(result.score)  # 65.5
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
import re
from collections import Counter

# Import pure calculation functions
from .calculations import score_to_cefr_level


# =============================================================================
# DISCOURSE MARKER CATEGORIES
# =============================================================================

# A1-A2 Level: Basic connectors
BASIC_CONNECTORS = {
    # Addition
    "and",
    "also",
    "too",
    # Sequence
    "then",
    "next",
    "first",
    "second",
    "third",
    "finally",
    # Contrast
    "but",
    "or",
    # Cause
    "so",
    "because",
}

# B1 Level: Intermediate connectors
INTERMEDIATE_CONNECTORS = {
    # Addition
    "moreover",
    "furthermore",
    "in addition",
    "besides",
    "as well as",
    # Contrast
    "however",
    "although",
    "though",
    "even though",
    "whereas",
    "while",
    "on the other hand",
    "instead",
    "yet",
    "still",
    # Cause/Effect
    "therefore",
    "thus",
    "hence",
    "as a result",
    "consequently",
    "due to",
    "because of",
    "since",
    "as",
    # Condition
    "if",
    "unless",
    "provided that",
    "as long as",
    # Purpose
    "in order to",
    "so that",
    "so as to",
    # Example
    "for example",
    "for instance",
    "such as",
    "like",
    # Clarification
    "in other words",
    "that is",
    "i mean",
    "actually",
    # Summary
    "in conclusion",
    "to sum up",
    "overall",
    "in summary",
}

# B2+ Level: Advanced connectors
ADVANCED_CONNECTORS = {
    # Addition
    "additionally",
    "not only",
    "what is more",
    "equally important",
    # Contrast
    "nevertheless",
    "nonetheless",
    "on the contrary",
    "conversely",
    "in contrast",
    "despite",
    "in spite of",
    "notwithstanding",
    # Cause/Effect
    "accordingly",
    "thereby",
    "as a consequence",
    "it follows that",
    # Concession
    "admittedly",
    "granted",
    "albeit",
    "even so",
    # Emphasis
    "indeed",
    "in fact",
    "certainly",
    "undoubtedly",
    "clearly",
    # Qualification
    "to some extent",
    "in a sense",
    "in general",
    "broadly speaking",
    # Transition
    "regarding",
    "with respect to",
    "as for",
    "concerning",
    "turning to",
    "as regards",
    # Summary/Conclusion
    "in essence",
    "ultimately",
    "all in all",
    "taking everything into account",
}

# All connectors combined with their levels
CONNECTOR_LEVELS = {}
for c in BASIC_CONNECTORS:
    CONNECTOR_LEVELS[c] = "basic"
for c in INTERMEDIATE_CONNECTORS:
    CONNECTOR_LEVELS[c] = "intermediate"
for c in ADVANCED_CONNECTORS:
    CONNECTOR_LEVELS[c] = "advanced"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CoherenceFeatures:
    """Features extracted for coherence assessment."""

    # Discourse marker counts
    total_connectors: int = 0
    basic_connectors: int = 0
    intermediate_connectors: int = 0
    advanced_connectors: int = 0

    # Variety metrics
    unique_connectors: int = 0
    connector_variety_ratio: float = 0.0  # unique / total

    # Density metrics
    connectors_per_sentence: float = 0.0
    connectors_per_100_words: float = 0.0

    # Sophistication
    sophistication_score: float = 0.0  # Weighted by level

    # Lists for feedback
    connectors_used: List[str] = field(default_factory=list)
    basic_used: List[str] = field(default_factory=list)
    intermediate_used: List[str] = field(default_factory=list)
    advanced_used: List[str] = field(default_factory=list)

    # Text stats
    sentence_count: int = 0
    word_count: int = 0

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Limit lists in output
        result["connectors_used"] = result["connectors_used"][:20]
        result["basic_used"] = result["basic_used"][:10]
        result["intermediate_used"] = result["intermediate_used"][:10]
        result["advanced_used"] = result["advanced_used"][:10]
        return result


@dataclass
class CoherenceScore:
    """Coherence assessment result."""

    score: float  # 0-100
    level: str  # CEFR level
    confidence: float  # 0-1
    features: CoherenceFeatures = field(default_factory=CoherenceFeatures)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "confidence": round(self.confidence, 3),
            "features": self.features.to_dict(),
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "feedback": self.feedback,
        }


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================


class CoherenceFeatureExtractor:
    """
    Extract coherence features from transcription text.

    Analyzes:
    1. Discourse marker usage (count, variety, sophistication)
    2. Connector distribution across CEFR levels
    3. Cohesion density (connectors per sentence/words)

    Example:
        >>> extractor = CoherenceFeatureExtractor()
        >>> features = extractor.extract("I like coffee. However, I don't drink it often.")
        >>> features.intermediate_connectors
        1
    """

    def __init__(self):
        """Initialize extractor with connector patterns."""
        # Build regex patterns for multi-word connectors (check these first)
        self.multiword_patterns = []
        for connector in sorted(CONNECTOR_LEVELS.keys(), key=len, reverse=True):
            if " " in connector:
                pattern = re.compile(
                    r"\b" + re.escape(connector) + r"\b", re.IGNORECASE
                )
                self.multiword_patterns.append((connector, pattern))

    def extract(self, text: str) -> CoherenceFeatures:
        """
        Extract coherence features from text.

        Args:
            text: Transcription text

        Returns:
            CoherenceFeatures
        """
        if not text or not text.strip():
            return CoherenceFeatures()

        # Clean text
        text = text.strip()

        # Count sentences (approximate)
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        # Count words
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return CoherenceFeatures()

        # Find all connectors
        connectors_found = self._find_connectors(text)

        # Categorize by level
        basic_used = []
        intermediate_used = []
        advanced_used = []

        for connector in connectors_found:
            level = CONNECTOR_LEVELS.get(connector.lower(), "basic")
            if level == "basic":
                basic_used.append(connector)
            elif level == "intermediate":
                intermediate_used.append(connector)
            else:
                advanced_used.append(connector)

        # Calculate metrics
        total = len(connectors_found)
        unique = len(set(c.lower() for c in connectors_found))

        # Variety ratio
        variety_ratio = unique / total if total > 0 else 0.0

        # Density
        per_sentence = total / sentence_count if sentence_count > 0 else 0.0
        per_100_words = (total / word_count) * 100 if word_count > 0 else 0.0

        # Sophistication score (weighted average)
        # Basic = 1, Intermediate = 2, Advanced = 3
        if total > 0:
            weighted_sum = (
                len(basic_used) * 1
                + len(intermediate_used) * 2
                + len(advanced_used) * 3
            )
            sophistication = weighted_sum / total
        else:
            sophistication = 0.0

        return CoherenceFeatures(
            total_connectors=total,
            basic_connectors=len(basic_used),
            intermediate_connectors=len(intermediate_used),
            advanced_connectors=len(advanced_used),
            unique_connectors=unique,
            connector_variety_ratio=variety_ratio,
            connectors_per_sentence=per_sentence,
            connectors_per_100_words=per_100_words,
            sophistication_score=sophistication,
            connectors_used=connectors_found,
            basic_used=list(set(basic_used)),
            intermediate_used=list(set(intermediate_used)),
            advanced_used=list(set(advanced_used)),
            sentence_count=sentence_count,
            word_count=word_count,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        # Handle common abbreviations
        text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|etc)\.\s", r"\1<DOT> ", text)

        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)

        # Restore dots
        sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

        # Filter empty
        sentences = [s for s in sentences if s]

        return sentences if sentences else [""]

    def _find_connectors(self, text: str) -> List[str]:
        """Find all discourse markers/connectors in text."""
        found = []
        text_lower = text.lower()

        # Track positions already matched (for multi-word connectors)
        matched_positions = set()

        # First, find multi-word connectors
        for connector, pattern in self.multiword_patterns:
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                # Check if this position overlaps with already matched
                if not any(
                    start < pos < end or pos == start for pos in matched_positions
                ):
                    found.append(connector)
                    # Mark all positions in this match
                    for pos in range(start, end):
                        matched_positions.add(pos)

        # Then find single-word connectors
        words = re.findall(r"\b\w+\b", text_lower)
        word_positions = [
            (m.start(), m.group()) for m in re.finditer(r"\b\w+\b", text_lower)
        ]

        for pos, word in word_positions:
            if pos not in matched_positions and word in CONNECTOR_LEVELS:
                # Skip if it's part of a multi-word connector
                if (
                    word not in INTERMEDIATE_CONNECTORS
                    and word not in ADVANCED_CONNECTORS
                ):
                    # Only count basic single-word connectors
                    if word in BASIC_CONNECTORS:
                        found.append(word)

        return found


# =============================================================================
# SCORER
# =============================================================================


class CoherenceScorer:
    """
    Score coherence features to produce CEFR level.

    Scoring dimensions:
    1. Connector variety (30%) - Range of different connectors
    2. Sophistication (40%) - Level of connectors used
    3. Density (30%) - Appropriate use frequency

    CEFR Benchmarks:
    - A1-A2: Basic connectors only, low variety
    - B1: Mix of basic and intermediate, moderate variety
    - B2+: Advanced connectors, high variety, appropriate density
    """

    def __init__(self):
        """Initialize scorer with default weights."""
        self.weights = {
            "variety": 0.30,
            "sophistication": 0.40,
            "density": 0.30,
        }

    def score(self, features: CoherenceFeatures) -> CoherenceScore:
        """
        Score coherence features.

        Args:
            features: Extracted coherence features

        Returns:
            CoherenceScore with level, score, and feedback
        """
        if features.word_count == 0:
            return CoherenceScore(
                score=50.0,
                level="A2",
                confidence=0.0,
                features=features,
                sub_scores={},
                feedback=["Insufficient text for coherence assessment."],
            )

        # Calculate sub-scores
        sub_scores = {
            "variety": self._score_variety(features),
            "sophistication": self._score_sophistication(features),
            "density": self._score_density(features),
        }

        # Weighted average
        total_score = sum(sub_scores[k] * self.weights[k] for k in self.weights.keys())

        # Determine level
        level = score_to_cefr_level(total_score)

        # Calculate confidence
        confidence = self._calculate_confidence(features)

        # Generate feedback
        feedback = self._generate_feedback(features, sub_scores)

        return CoherenceScore(
            score=total_score,
            level=level,
            confidence=confidence,
            features=features,
            sub_scores=sub_scores,
            feedback=feedback,
        )

    def _score_variety(self, features: CoherenceFeatures) -> float:
        """
        Score connector variety (0-100).

        CEFR Benchmarks:
        - A1-A2: 1-3 unique connectors
        - B1: 4-7 unique connectors
        - B2+: 8+ unique connectors
        """
        unique = features.unique_connectors

        if unique >= 12:
            return 95
        elif unique >= 8:
            return 80 + (unique - 8) * 3.75  # 80-95
        elif unique >= 5:
            return 60 + (unique - 5) * 6.67  # 60-80
        elif unique >= 3:
            return 40 + (unique - 3) * 10  # 40-60
        elif unique >= 1:
            return 20 + (unique - 1) * 10  # 20-40
        else:
            return 20  # No connectors

    def _score_sophistication(self, features: CoherenceFeatures) -> float:
        """
        Score connector sophistication (0-100).

        Based on weighted average:
        - Basic = 1, Intermediate = 2, Advanced = 3
        - Max score = 3.0

        CEFR Mapping:
        - A1-A2: score ~1.0 (all basic)
        - B1: score ~1.5-2.0 (mix)
        - B2+: score ~2.0-3.0 (intermediate/advanced)
        """
        soph = features.sophistication_score

        if soph == 0:
            return 30  # No connectors

        if soph >= 2.5:
            return 95
        elif soph >= 2.0:
            return 80 + (soph - 2.0) * 30  # 80-95
        elif soph >= 1.5:
            return 60 + (soph - 1.5) * 40  # 60-80
        elif soph >= 1.2:
            return 40 + (soph - 1.2) * 66.67  # 40-60
        else:
            return max(20, soph * 33.33)  # 20-40

    def _score_density(self, features: CoherenceFeatures) -> float:
        """
        Score connector density (0-100).

        Optimal density: 3-6 connectors per 100 words
        Too low: Ideas not connected
        Too high: Overuse, unnatural

        CEFR Benchmarks:
        - A1-A2: <2 per 100 words (underuse)
        - B1: 2-4 per 100 words
        - B2+: 4-7 per 100 words (optimal)
        """
        density = features.connectors_per_100_words

        if density == 0:
            return 25  # No connectors
        elif density < 1:
            return 30 + density * 20  # 30-50
        elif density < 2:
            return 50 + (density - 1) * 15  # 50-65
        elif density < 4:
            return 65 + (density - 2) * 10  # 65-85
        elif density <= 7:
            return 85 + (density - 4) * 3.33  # 85-95
        elif density <= 10:
            return 85 - (density - 7) * 5  # 85-70 (slight penalty for overuse)
        else:
            return max(50, 70 - (density - 10) * 2)  # Penalty for heavy overuse

    def _calculate_confidence(self, features: CoherenceFeatures) -> float:
        """Calculate confidence in the assessment."""
        confidence = 0.4  # Base

        # More words = more confidence
        if features.word_count >= 200:
            confidence += 0.3
        elif features.word_count >= 100:
            confidence += 0.2
        elif features.word_count >= 50:
            confidence += 0.1

        # More sentences = more confidence
        if features.sentence_count >= 10:
            confidence += 0.2
        elif features.sentence_count >= 5:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_feedback(
        self,
        features: CoherenceFeatures,
        sub_scores: Dict[str, float],
    ) -> List[str]:
        """Generate actionable feedback."""
        feedback = []

        # Variety feedback
        if features.unique_connectors == 0:
            feedback.append(
                "No linking words detected. Use connectors like 'and', 'but', 'so' "
                "to connect your ideas."
            )
        elif features.unique_connectors < 3:
            feedback.append(
                "Limited connector variety. Try using more linking words like "
                "'however', 'therefore', 'for example' to improve flow."
            )
        elif features.unique_connectors >= 8:
            feedback.append("Good variety of linking words - ideas flow well together.")

        # Sophistication feedback
        if features.advanced_connectors > 0:
            advanced_examples = ", ".join(features.advanced_used[:3])
            feedback.append(f"Good use of advanced connectors ({advanced_examples}).")
        elif features.intermediate_connectors > 0:
            if features.intermediate_connectors < 3:
                feedback.append(
                    "Try using more intermediate connectors like 'however', "
                    "'therefore', 'although' to show complex relationships."
                )
            else:
                feedback.append(
                    "Good use of intermediate connectors for linking ideas."
                )
        else:
            if features.total_connectors > 0:
                feedback.append(
                    "Using only basic connectors (and, but, so). "
                    "Practice using 'however', 'therefore', 'for example' for clearer connections."
                )

        # Density feedback
        density = features.connectors_per_100_words
        if density < 2:
            feedback.append(
                "Ideas could be better connected. Add more linking words "
                "between sentences to improve coherence."
            )
        elif density > 10:
            feedback.append(
                "Many connectors used - ensure each adds meaning. "
                "Sometimes simpler connections work better."
            )

        return feedback


# =============================================================================
# COMBINED ASSESSOR
# =============================================================================


class CoherenceAssessor:
    """
    Complete coherence assessment pipeline.

    Combines feature extraction and scoring.

    Example:
        >>> assessor = CoherenceAssessor()
        >>> result = assessor.assess("I like coffee. However, I prefer tea.")
        >>> print(f"Level: {result.level}, Score: {result.score:.1f}")
    """

    def __init__(self):
        """Initialize assessor."""
        self.extractor = CoherenceFeatureExtractor()
        self.scorer = CoherenceScorer()

    def assess(self, text: str) -> CoherenceScore:
        """
        Assess coherence from transcription text.

        Args:
            text: Transcription text

        Returns:
            CoherenceScore with level, score, features, and feedback
        """
        features = self.extractor.extract(text)
        return self.scorer.score(features)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def assess_coherence(text: str) -> CoherenceScore:
    """
    Quick coherence assessment.

    Args:
        text: Transcription text

    Returns:
        CoherenceScore
    """
    assessor = CoherenceAssessor()
    return assessor.assess(text)


# =============================================================================
# MAIN (for testing)
# =============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        # Example text for testing
        text = """
        I want to learn from my job. First, I want to learn how office work is going.
        And I want to learn easily with seniors and colleagues. Also, how to do work 
        in teamwork. That also I want to learn. And I want to perfect communication 
        and more skills, computer knowledge. However, I need to improve my English.
        Therefore, I am taking classes. For example, I practice speaking every day.
        In conclusion, I believe I can improve with practice.
        """

    print("=" * 60)
    print("COHERENCE ASSESSMENT")
    print("=" * 60)
    print(f"Text: {text[:200]}...")
    print()

    assessor = CoherenceAssessor()
    result = assessor.assess(text)

    print(f"Score: {result.score:.1f}/100")
    print(f"Level: {result.level}")
    print(f"Confidence: {result.confidence:.1%}")
    print()
    print("Sub-scores:")
    for name, score in result.sub_scores.items():
        print(f"  {name:15}: {score:.1f}")
    print()
    print("Features:")
    print(f"  Total connectors:      {result.features.total_connectors}")
    print(f"  Unique connectors:     {result.features.unique_connectors}")
    print(f"  Basic:                 {result.features.basic_connectors}")
    print(f"  Intermediate:          {result.features.intermediate_connectors}")
    print(f"  Advanced:              {result.features.advanced_connectors}")
    print(f"  Per 100 words:         {result.features.connectors_per_100_words:.1f}")
    print(f"  Sophistication:        {result.features.sophistication_score:.2f}")
    print()
    print("Connectors used:")
    if result.features.basic_used:
        print(f"  Basic:        {', '.join(result.features.basic_used)}")
    if result.features.intermediate_used:
        print(f"  Intermediate: {', '.join(result.features.intermediate_used)}")
    if result.features.advanced_used:
        print(f"  Advanced:     {', '.join(result.features.advanced_used)}")
    print()
    if result.feedback:
        print("Feedback:")
        for fb in result.feedback:
            print(f"  - {fb}")
