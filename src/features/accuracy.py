"""
Accuracy Assessment Module for CEFR Speech Assessment

Analyzes grammatical accuracy in transcribed speech using LanguageTool.

CEFR Accuracy Criteria:
- A1: Very limited control of simple structures, frequent errors
- A2: Uses simple structures correctly but still makes basic errors
- B1: Reasonable accuracy in familiar contexts, some errors
- B2+: Good grammatical control, few errors, self-correction

Features Extracted:
- Total grammar errors
- Error density (errors per 100 words)
- Error types (grammar, spelling, style, etc.)
- Common ESL error patterns (articles, prepositions, subject-verb agreement)

Usage:
    >>> from src.features.accuracy import AccuracyAssessor
    >>> assessor = AccuracyAssessor()
    >>> result = assessor.assess("I goes to school yesterday")
    >>> print(result.level)  # "A1"
    >>> print(result.score)  # 35.5
    >>> print(result.feedback)  # ["Subject-verb agreement errors detected..."]

Example:
    >>> extractor = AccuracyFeatureExtractor()
    >>> features, errors = extractor.extract("She go to school everyday")
    >>> print(features.total_errors)  # 2
    >>> print(features.grammar_errors)  # 1 (go -> goes)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import re

# Import pure calculation functions
from .calculations import score_to_cefr_level


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GrammarError:
    """Single grammar error detected."""

    message: str
    category: str  # GRAMMAR, SPELLING, TYPOS, PUNCTUATION, STYLE, etc.
    rule_id: str
    context: str
    offset: int
    length: int
    replacements: List[str]
    sentence: str = ""


@dataclass
class AccuracyFeatures:
    """Features extracted for accuracy assessment."""

    # Error counts by category
    total_errors: int = 0
    grammar_errors: int = 0
    spelling_errors: int = 0
    punctuation_errors: int = 0
    style_errors: int = 0
    other_errors: int = 0

    # ESL-specific error patterns
    article_errors: int = 0  # a/an/the
    preposition_errors: int = 0
    subject_verb_errors: int = 0
    tense_errors: int = 0
    word_order_errors: int = 0
    plural_errors: int = 0

    # Density metrics
    error_density: float = 0.0  # errors per 100 words
    grammar_density: float = 0.0  # grammar errors per 100 words

    # Text stats
    word_count: int = 0
    sentence_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AccuracyScore:
    """Accuracy assessment result."""

    score: float  # 0-100
    level: str  # CEFR level
    confidence: float  # 0-1
    features: AccuracyFeatures = field(default_factory=AccuracyFeatures)
    errors: List[GrammarError] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "level": self.level,
            "confidence": self.confidence,
            "features": self.features.to_dict(),
            "errors": [asdict(e) for e in self.errors[:10]],  # Limit for output
            "sub_scores": self.sub_scores,
            "feedback": self.feedback,
        }


# =============================================================================
# ESL ERROR PATTERNS
# =============================================================================

# Rule IDs that indicate specific ESL error types
ARTICLE_RULES = {
    "EN_A_VS_AN",
    "A_PLURAL",
    "THE_SUPERLATIVE",
    "MISSING_DETERMINER",
    "DETERMINER_NOUN",
    "A_UNCOUNTABLE",
}

PREPOSITION_RULES = {
    "IN_ON_AT",
    "PREPOSITION_AT",
    "PREPOSITION_IN",
    "PREPOSITION_ON",
    "PREPOSITION_TO",
    "MISSING_PREPOSITION",
    "UNNECESSARY_PREPOSITION",
}

SUBJECT_VERB_RULES = {
    "SUBJECT_VERB_AGREEMENT",
    "HE_VERB_AGR",
    "THEY_VERB_AGR",
    "SINGULAR_VERB_AFTER_EACH",
    "SINGULAR_AGREEMENT",
    "AGREEMENT_SENT_START",
}

TENSE_RULES = {
    "PAST_TENSE",
    "PRESENT_PERFECT",
    "FUTURE_TENSE",
    "DID_BASEFORM",
    "GONNA_WILL",
    "WILL_MUST",
    "HAVE_PART_AGREEMENT",
}

WORD_ORDER_RULES = {
    "WORD_ORDER",
    "ADJECTIVE_ORDER",
    "ADVERB_POSITION",
    "QUESTION_WORD_ORDER",
}

PLURAL_RULES = {
    "PLURAL_NOUN",
    "SINGULAR_NOUN",
    "COUNTABLE_PLURAL",
    "MANY_MUCH",
    "FEWER_LESS",
}


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================


class AccuracyFeatureExtractor:
    """
    Extract accuracy features using LanguageTool.

    Analyzes grammar, spelling, and style errors in text.
    Categorizes errors by type for ESL-specific feedback.

    Example:
        >>> extractor = AccuracyFeatureExtractor()
        >>> features, errors = extractor.extract("I goes to school everyday")
        >>> features.grammar_errors
        1
        >>> features.subject_verb_errors
        1
    """

    def __init__(self, language: str = "en-US"):
        """
        Initialize the extractor.

        Args:
            language: Language code for LanguageTool (default: en-US)
        """
        self.language = language
        self._tool = None  # Lazy load

    @property
    def tool(self):
        """Lazy load LanguageTool (it's slow to initialize)."""
        if self._tool is None:
            import language_tool_python

            self._tool = language_tool_python.LanguageTool(self.language)
        return self._tool

    def extract(self, text: str) -> Tuple[AccuracyFeatures, List[GrammarError]]:
        """
        Extract accuracy features from text.

        Args:
            text: Transcription text to analyze

        Returns:
            Tuple of (AccuracyFeatures, List[GrammarError])
        """
        if not text or not text.strip():
            return AccuracyFeatures(), []

        # Clean text (speech often lacks punctuation)
        text = self._preprocess_text(text)

        # Get word and sentence counts
        words = text.split()
        word_count = len(words)
        sentence_count = max(1, len(re.findall(r"[.!?]+", text)) or 1)

        if word_count == 0:
            return AccuracyFeatures(), []

        # Check with LanguageTool
        matches = self.tool.check(text)

        # Convert to our error format and categorize
        errors = []
        category_counts = {
            "GRAMMAR": 0,
            "SPELLING": 0,
            "PUNCTUATION": 0,
            "STYLE": 0,
            "OTHER": 0,
        }

        esl_counts = {
            "article": 0,
            "preposition": 0,
            "subject_verb": 0,
            "tense": 0,
            "word_order": 0,
            "plural": 0,
        }

        for match in matches:
            # Skip certain rule types that aren't relevant for speech
            if self._should_skip_rule(match):
                continue

            # Get rule_id (handle both camelCase and snake_case)
            rule_id = (
                getattr(match, "ruleId", None) or getattr(match, "rule_id", None) or ""
            )
            error_length = getattr(match, "errorLength", None) or getattr(
                match, "error_length", 0
            )

            error = GrammarError(
                message=match.message,
                category=match.category or "OTHER",
                rule_id=rule_id,
                context=match.context,
                offset=match.offset,
                length=error_length,
                replacements=match.replacements[:3] if match.replacements else [],
                sentence=match.sentence if hasattr(match, "sentence") else "",
            )
            errors.append(error)

            # Count by category
            cat = self._map_category(match.category)
            category_counts[cat] += 1

            # Count ESL-specific patterns (rule_id already set above)
            if rule_id in ARTICLE_RULES or "ARTICLE" in rule_id:
                esl_counts["article"] += 1
            if rule_id in PREPOSITION_RULES or "PREPOSITION" in rule_id:
                esl_counts["preposition"] += 1
            if rule_id in SUBJECT_VERB_RULES or "AGREEMENT" in rule_id:
                esl_counts["subject_verb"] += 1
            if rule_id in TENSE_RULES or "TENSE" in rule_id:
                esl_counts["tense"] += 1
            if rule_id in WORD_ORDER_RULES or "ORDER" in rule_id:
                esl_counts["word_order"] += 1
            if rule_id in PLURAL_RULES or "PLURAL" in rule_id:
                esl_counts["plural"] += 1

        total_errors = len(errors)

        features = AccuracyFeatures(
            # Error counts by category
            total_errors=total_errors,
            grammar_errors=category_counts["GRAMMAR"],
            spelling_errors=category_counts["SPELLING"],
            punctuation_errors=category_counts["PUNCTUATION"],
            style_errors=category_counts["STYLE"],
            other_errors=category_counts["OTHER"],
            # ESL-specific
            article_errors=esl_counts["article"],
            preposition_errors=esl_counts["preposition"],
            subject_verb_errors=esl_counts["subject_verb"],
            tense_errors=esl_counts["tense"],
            word_order_errors=esl_counts["word_order"],
            plural_errors=esl_counts["plural"],
            # Density
            error_density=(total_errors / word_count) * 100 if word_count > 0 else 0,
            grammar_density=(category_counts["GRAMMAR"] / word_count) * 100
            if word_count > 0
            else 0,
            # Text stats
            word_count=word_count,
            sentence_count=sentence_count,
        )

        return features, errors

    def _preprocess_text(self, text: str) -> str:
        """Preprocess transcription text for grammar checking."""
        # Basic cleanup
        text = text.strip()

        # Ensure text ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        # Capitalize first letter of sentences
        sentences = re.split(r"([.!?]+\s*)", text)
        result = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Sentence content
                part = part.strip()
                if part:
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
            result.append(part)

        return "".join(result)

    def _should_skip_rule(self, match) -> bool:
        """Check if rule should be skipped for speech analysis."""
        rule_id = (
            getattr(match, "ruleId", None) or getattr(match, "rule_id", None) or ""
        )

        # Skip whitespace and formatting rules
        skip_patterns = [
            "WHITESPACE",
            "COMMA_PARENTHESIS",
            "UPPERCASE_SENTENCE_START",
            "DOUBLE_PUNCTUATION",
            "MORFOLOGIK_RULE",  # Often triggers on names/places
        ]

        for pattern in skip_patterns:
            if pattern in rule_id:
                return True

        return False

    def _map_category(self, category: str) -> str:
        """Map LanguageTool category to our categories."""
        if not category:
            return "OTHER"

        category = category.upper()

        if "GRAMMAR" in category:
            return "GRAMMAR"
        elif "SPELL" in category or "TYPO" in category:
            return "SPELLING"
        elif "PUNCT" in category:
            return "PUNCTUATION"
        elif "STYLE" in category or "REDUNDANCY" in category:
            return "STYLE"
        else:
            return "OTHER"


# =============================================================================
# SCORER
# =============================================================================


class AccuracyScorer:
    """
    Score accuracy features to produce CEFR level.

    Scoring is based on error density and error types.
    Lower error density = higher score.

    CEFR Benchmarks (errors per 100 words):
    - A1: > 10 errors/100 words
    - A2: 5-10 errors/100 words
    - B1: 2-5 errors/100 words
    - B2+: < 2 errors/100 words
    """

    def __init__(self):
        """Initialize scorer with default weights."""
        # Sub-score weights
        self.weights = {
            "error_density": 0.40,  # Overall error rate
            "grammar": 0.30,  # Grammar-specific errors
            "esl_patterns": 0.20,  # Common ESL mistakes
            "consistency": 0.10,  # Error distribution
        }

    def score(
        self,
        features: AccuracyFeatures,
        errors: List[GrammarError],
    ) -> AccuracyScore:
        """
        Score accuracy features.

        Args:
            features: Extracted accuracy features
            errors: List of grammar errors

        Returns:
            AccuracyScore with level, score, and feedback
        """
        if features.word_count == 0:
            return AccuracyScore(
                score=50.0,
                level="A2",
                confidence=0.0,
                features=features,
                errors=[],
                sub_scores={},
                feedback=["Insufficient text for accuracy assessment."],
            )

        # Calculate sub-scores
        sub_scores = {
            "error_density": self._score_error_density(features.error_density),
            "grammar": self._score_grammar(features),
            "esl_patterns": self._score_esl_patterns(features),
            "consistency": self._score_consistency(features, errors),
        }

        # Weighted average
        total_score = sum(sub_scores[k] * self.weights[k] for k in self.weights.keys())

        # Determine level
        level = score_to_cefr_level(total_score)

        # Calculate confidence
        confidence = self._calculate_confidence(features, errors)

        # Generate feedback
        feedback = self._generate_feedback(features, errors, sub_scores, level)

        return AccuracyScore(
            score=total_score,
            level=level,
            confidence=confidence,
            features=features,
            errors=errors,
            sub_scores=sub_scores,
            feedback=feedback,
        )

    def _score_error_density(self, density: float) -> float:
        """
        Score based on error density (errors per 100 words).

        Benchmarks:
        - 0-1 errors/100 words: 90-100 (B2+)
        - 1-3 errors/100 words: 70-90 (B1-B2)
        - 3-6 errors/100 words: 50-70 (A2-B1)
        - 6-10 errors/100 words: 30-50 (A1-A2)
        - >10 errors/100 words: 0-30 (A1)
        """
        if density <= 1:
            return 90 + (1 - density) * 10
        elif density <= 3:
            return 70 + (3 - density) / 2 * 20
        elif density <= 6:
            return 50 + (6 - density) / 3 * 20
        elif density <= 10:
            return 30 + (10 - density) / 4 * 20
        else:
            return max(0, 30 - (density - 10) * 2)

    def _score_grammar(self, features: AccuracyFeatures) -> float:
        """Score grammar-specific accuracy."""
        if features.word_count == 0:
            return 50.0

        grammar_density = features.grammar_density

        if grammar_density <= 0.5:
            return 90 + (0.5 - grammar_density) * 20
        elif grammar_density <= 2:
            return 70 + (2 - grammar_density) / 1.5 * 20
        elif grammar_density <= 5:
            return 50 + (5 - grammar_density) / 3 * 20
        else:
            return max(0, 50 - grammar_density * 5)

    def _score_esl_patterns(self, features: AccuracyFeatures) -> float:
        """Score based on common ESL error patterns."""
        if features.word_count == 0:
            return 50.0

        # Total ESL-specific errors
        esl_total = (
            features.article_errors
            + features.preposition_errors
            + features.subject_verb_errors
            + features.tense_errors
            + features.word_order_errors
            + features.plural_errors
        )

        esl_density = (esl_total / features.word_count) * 100

        # Score based on ESL error density
        if esl_density <= 0.5:
            return 95
        elif esl_density <= 1:
            return 80
        elif esl_density <= 2:
            return 65
        elif esl_density <= 4:
            return 50
        else:
            return max(20, 50 - esl_density * 5)

    def _score_consistency(
        self,
        features: AccuracyFeatures,
        errors: List[GrammarError],
    ) -> float:
        """
        Score error consistency (distribution of error types).

        Concentrated errors in one area suggest specific weakness.
        Scattered errors suggest general proficiency issues.
        """
        if features.total_errors == 0:
            return 100.0

        # Check how concentrated errors are
        counts = [
            features.grammar_errors,
            features.spelling_errors,
            features.punctuation_errors,
            features.style_errors,
        ]

        max_count = max(counts)
        concentration = (
            max_count / features.total_errors if features.total_errors else 0
        )

        # High concentration = specific weakness (better than scattered)
        if concentration > 0.7:
            return 70  # Specific weakness, can be targeted
        elif concentration > 0.5:
            return 60
        else:
            return 50  # Scattered errors, general issues

    def _calculate_confidence(
        self,
        features: AccuracyFeatures,
        errors: List[GrammarError],
    ) -> float:
        """Calculate confidence in the assessment."""
        confidence = 0.5  # Base

        # More words = more confidence
        if features.word_count >= 100:
            confidence += 0.2
        elif features.word_count >= 50:
            confidence += 0.1

        # Multiple sentences = more confidence
        if features.sentence_count >= 5:
            confidence += 0.15
        elif features.sentence_count >= 3:
            confidence += 0.1

        # Some errors found = tool is working
        if features.total_errors > 0:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_feedback(
        self,
        features: AccuracyFeatures,
        errors: List[GrammarError],
        sub_scores: Dict[str, float],
        level: str,
    ) -> List[str]:
        """Generate actionable feedback."""
        feedback = []

        # Error density feedback
        if features.error_density > 5:
            feedback.append(
                f"High error rate ({features.error_density:.1f} per 100 words). "
                "Focus on basic grammar structures."
            )
        elif features.error_density > 2:
            feedback.append(
                f"Moderate error rate ({features.error_density:.1f} per 100 words). "
                "Review common grammar rules."
            )

        # ESL-specific feedback
        if features.article_errors >= 2:
            feedback.append(
                f"Article usage needs work ({features.article_errors} errors). "
                "Practice when to use 'a', 'an', and 'the'."
            )

        if features.subject_verb_errors >= 2:
            feedback.append(
                f"Subject-verb agreement errors ({features.subject_verb_errors}). "
                "Make sure verbs match their subjects (he goes, they go)."
            )

        if features.tense_errors >= 2:
            feedback.append(
                f"Tense consistency issues ({features.tense_errors}). "
                "Stay consistent with past, present, or future tense."
            )

        if features.preposition_errors >= 2:
            feedback.append(
                f"Preposition errors ({features.preposition_errors}). "
                "Review common preposition usage (in/on/at, to/for)."
            )

        # Positive feedback
        if features.error_density < 2:
            feedback.append("Good grammatical accuracy - few errors detected.")

        if features.grammar_errors == 0 and features.word_count > 20:
            feedback.append("No major grammar errors - well structured speech.")

        return feedback


# =============================================================================
# COMBINED ASSESSOR
# =============================================================================


class AccuracyAssessor:
    """
    Complete accuracy assessment pipeline.

    Combines feature extraction and scoring.

    Example:
        >>> assessor = AccuracyAssessor()
        >>> result = assessor.assess("I goes to school everyday")
        >>> print(f"Level: {result.level}, Score: {result.score:.1f}")
    """

    def __init__(self, language: str = "en-US"):
        """Initialize with language setting."""
        self.extractor = AccuracyFeatureExtractor(language=language)
        self.scorer = AccuracyScorer()

    def assess(self, text: str) -> AccuracyScore:
        """
        Assess grammatical accuracy of text.

        Args:
            text: Transcription text

        Returns:
            AccuracyScore with level, score, features, and feedback
        """
        features, errors = self.extractor.extract(text)
        return self.scorer.score(features, errors)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def assess_accuracy(text: str, language: str = "en-US") -> AccuracyScore:
    """
    Quick accuracy assessment.

    Args:
        text: Text to assess
        language: Language code (default: en-US)

    Returns:
        AccuracyScore
    """
    assessor = AccuracyAssessor(language=language)
    return assessor.assess(text)
