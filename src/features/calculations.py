"""
Pure Calculation Functions for Speech Assessment

This module contains standalone calculation functions with NO external dependencies
(except Python stdlib and numpy). These can be shared, tested, and used independently.

All formulas are documented with references for transparency.

Usage:
    from src.features.calculations import (
        calculate_wpm,
        calculate_ttr,
        detect_pauses,
        count_syllables,
    )

    # Calculate words per minute
    wpm = calculate_wpm(word_count=100, duration_seconds=60)

    # Calculate Type-Token Ratio
    ttr = calculate_ttr(total_words=100, unique_words=60)

Author: AOXLM Team
License: MIT
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import math
import re


# =============================================================================
# SPEECH RATE CALCULATIONS
# =============================================================================


def calculate_wpm(word_count: int, duration_seconds: float) -> float:
    """
    Calculate Words Per Minute (WPM).

    Formula: WPM = (word_count / duration_seconds) * 60

    CEFR Benchmarks:
    - A1: 40-80 WPM
    - A2: 80-110 WPM
    - B1: 110-140 WPM
    - B2+: 140-180 WPM

    Args:
        word_count: Total number of words spoken
        duration_seconds: Total duration in seconds

    Returns:
        Words per minute (float)

    Example:
        >>> calculate_wpm(word_count=150, duration_seconds=120)
        75.0
    """
    if duration_seconds <= 0:
        return 0.0
    return (word_count / duration_seconds) * 60


def calculate_articulation_rate(
    word_count: int,
    speech_time_seconds: float,
) -> float:
    """
    Calculate Articulation Rate (WPM excluding pauses).

    Formula: AR = (word_count / speech_time) * 60

    This measures "pure" speaking speed without pauses.
    Higher than WPM indicates significant pause time.

    Args:
        word_count: Total number of words spoken
        speech_time_seconds: Time spent actually speaking (excluding pauses)

    Returns:
        Articulation rate in words per minute

    Example:
        >>> calculate_articulation_rate(word_count=150, speech_time_seconds=100)
        90.0
    """
    if speech_time_seconds <= 0:
        return 0.0
    return (word_count / speech_time_seconds) * 60


def calculate_syllables_per_second(
    syllable_count: int,
    duration_seconds: float,
) -> float:
    """
    Calculate syllables per second.

    Formula: SPS = syllable_count / duration_seconds

    Native speakers typically produce 4-6 syllables/second.

    Args:
        syllable_count: Total number of syllables
        duration_seconds: Total duration in seconds

    Returns:
        Syllables per second
    """
    if duration_seconds <= 0:
        return 0.0
    return syllable_count / duration_seconds


# =============================================================================
# PAUSE DETECTION & ANALYSIS
# =============================================================================


def detect_pauses(
    words: List[Dict],
    threshold_seconds: float = 0.25,
) -> List[Dict]:
    """
    Detect pauses between words based on timestamps.

    A pause is defined as a gap between word_end and next_word_start
    that exceeds the threshold.

    Threshold Guidelines:
    - 0.15s: Micro-pause (not considered a real pause)
    - 0.25s: Short pause (default threshold)
    - 0.50s: Medium pause
    - 1.00s: Long pause (disfluency indicator)
    - 2.00s: Very long pause (major hesitation)

    Args:
        words: List of dicts with "word", "start", "end" keys
        threshold_seconds: Minimum gap to count as pause (default: 0.25s)

    Returns:
        List of pause dicts with "start", "end", "duration", "position"

    Example:
        >>> words = [
        ...     {"word": "I", "start": 0.0, "end": 0.2},
        ...     {"word": "think", "start": 0.8, "end": 1.2},  # 0.6s gap = pause
        ... ]
        >>> pauses = detect_pauses(words, threshold_seconds=0.25)
        >>> len(pauses)
        1
        >>> pauses[0]["duration"]
        0.6
    """
    if not words or len(words) < 2:
        return []

    pauses = []
    for i in range(1, len(words)):
        prev_end = words[i - 1].get("end", 0)
        curr_start = words[i].get("start", 0)
        gap = curr_start - prev_end

        if gap >= threshold_seconds:
            pauses.append(
                {
                    "start": prev_end,
                    "end": curr_start,
                    "duration": gap,
                    "position": i,  # Index of word after the pause
                }
            )

    return pauses


def calculate_pause_metrics(
    pauses: List[Dict],
    total_duration: float,
    long_pause_threshold: float = 1.0,
) -> Dict:
    """
    Calculate comprehensive pause metrics.

    Args:
        pauses: List of pause dicts from detect_pauses()
        total_duration: Total audio duration in seconds
        long_pause_threshold: Threshold for "long" pause (default: 1.0s)

    Returns:
        Dict with:
        - count: Total number of pauses
        - total_time: Total pause time in seconds
        - mean_duration: Average pause length
        - pause_ratio: Pause time / total time
        - pause_rate: Pauses per minute
        - long_pause_count: Number of pauses > threshold

    Example:
        >>> pauses = [{"duration": 0.5}, {"duration": 1.2}, {"duration": 0.3}]
        >>> metrics = calculate_pause_metrics(pauses, total_duration=60.0)
        >>> metrics["count"]
        3
        >>> metrics["total_time"]
        2.0
    """
    if not pauses:
        return {
            "count": 0,
            "total_time": 0.0,
            "mean_duration": 0.0,
            "pause_ratio": 0.0,
            "pause_rate": 0.0,
            "long_pause_count": 0,
        }

    durations = [p["duration"] for p in pauses]
    total_pause_time = sum(durations)

    return {
        "count": len(pauses),
        "total_time": total_pause_time,
        "mean_duration": total_pause_time / len(pauses),
        "pause_ratio": total_pause_time / total_duration if total_duration > 0 else 0,
        "pause_rate": (len(pauses) / total_duration) * 60 if total_duration > 0 else 0,
        "long_pause_count": sum(1 for d in durations if d >= long_pause_threshold),
    }


def calculate_speech_time(
    total_duration: float,
    total_pause_time: float,
) -> float:
    """
    Calculate actual speech time (excluding pauses).

    Formula: speech_time = total_duration - total_pause_time

    Args:
        total_duration: Total audio duration in seconds
        total_pause_time: Total time spent pausing

    Returns:
        Speech time in seconds
    """
    return max(0, total_duration - total_pause_time)


# =============================================================================
# HESITATION MARKERS
# =============================================================================

# Common English filler words
FILLER_WORDS_EN = {
    "um",
    "uh",
    "er",
    "ah",
    "eh",
    "mm",
    "hmm",
    "like",
    "you know",
    "i mean",
    "basically",
    "actually",
    "sort of",
    "kind of",
    "well",
    "so",
    "right",
    "okay",
}


def detect_fillers(
    words: List[Dict],
    filler_set: Optional[Set[str]] = None,
) -> Tuple[int, List[str]]:
    """
    Detect filler words in speech.

    Filler words indicate hesitation and are common at lower proficiency levels.

    Args:
        words: List of dicts with "word" key
        filler_set: Set of filler words (default: English fillers)

    Returns:
        Tuple of (count, list of fillers found)

    Example:
        >>> words = [{"word": "I"}, {"word": "um"}, {"word": "think"}]
        >>> count, fillers = detect_fillers(words)
        >>> count
        1
        >>> fillers
        ["um"]
    """
    if filler_set is None:
        filler_set = FILLER_WORDS_EN

    count = 0
    found = []

    for w in words:
        word = w.get("word", "").lower().strip()
        if word in filler_set:
            count += 1
            found.append(word)

    return count, found


def calculate_filler_rate(
    filler_count: int,
    duration_seconds: float,
) -> float:
    """
    Calculate filler words per minute.

    Formula: filler_rate = (filler_count / duration_seconds) * 60

    Guidelines:
    - < 2/min: Excellent (minimal hesitation)
    - 2-5/min: Good (normal hesitation)
    - 5-10/min: Moderate (noticeable hesitation)
    - > 10/min: High (significant hesitation)

    Args:
        filler_count: Number of filler words
        duration_seconds: Total duration in seconds

    Returns:
        Fillers per minute
    """
    if duration_seconds <= 0:
        return 0.0
    return (filler_count / duration_seconds) * 60


def detect_repetitions(
    words: List[Dict],
    filler_set: Optional[Set[str]] = None,
) -> int:
    """
    Detect immediate word repetitions (stuttering/hesitation).

    Counts cases where the same word is repeated consecutively.
    Excludes filler words from repetition count.

    Args:
        words: List of dicts with "word" key
        filler_set: Set of filler words to exclude

    Returns:
        Number of repetitions

    Example:
        >>> words = [{"word": "I"}, {"word": "I"}, {"word": "think"}]
        >>> detect_repetitions(words)
        1
    """
    if filler_set is None:
        filler_set = FILLER_WORDS_EN

    count = 0
    for i in range(1, len(words)):
        prev_word = words[i - 1].get("word", "").lower().strip()
        curr_word = words[i].get("word", "").lower().strip()

        # Skip fillers
        if prev_word in filler_set or curr_word in filler_set:
            continue

        # Count if same word repeated
        if prev_word == curr_word and len(prev_word) > 1:
            count += 1

    return count


def detect_false_starts(
    words: List[Dict],
    long_pause_threshold: float = 1.0,
) -> int:
    """
    Detect false starts (abandoned words/phrases).

    Heuristic: A very short word followed by a long pause
    indicates an abandoned utterance.

    Args:
        words: List of dicts with "word", "start", "end" keys
        long_pause_threshold: Pause length indicating abandonment

    Returns:
        Number of detected false starts
    """
    count = 0
    for i in range(len(words) - 1):
        word = words[i].get("word", "")
        curr_end = words[i].get("end", 0)
        next_start = words[i + 1].get("start", 0)
        gap = next_start - curr_end

        # Short word + long pause = possible false start
        if len(word) <= 2 and gap >= long_pause_threshold:
            count += 1

    return count


# =============================================================================
# FLOW & RHYTHM
# =============================================================================


def calculate_speech_rate_variability(
    words: List[Dict],
    window_seconds: float = 5.0,
) -> float:
    """
    Calculate speech rate variability (standard deviation of local rates).

    Lower variability = more consistent, fluent speech.
    Higher variability = inconsistent pace (possible disfluency).

    Args:
        words: List of dicts with "start" key
        window_seconds: Window size for calculating local rates

    Returns:
        Standard deviation of speech rates across windows

    Example:
        >>> # Consistent speaker has low variability (~10-20)
        >>> # Inconsistent speaker has high variability (~30-50)
    """
    if len(words) < 5:
        return 0.0

    start_time = words[0].get("start", 0)
    end_time = words[-1].get("end", 0)

    if end_time - start_time < window_seconds:
        return 0.0

    rates = []
    current = start_time

    while current + window_seconds <= end_time:
        window_words = [
            w for w in words if current <= w.get("start", 0) < current + window_seconds
        ]
        if window_words:
            rate = (len(window_words) / window_seconds) * 60
            rates.append(rate)
        current += window_seconds / 2  # 50% overlap

    if len(rates) < 2:
        return 0.0

    # Calculate standard deviation
    mean = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / len(rates)
    return math.sqrt(variance)


def calculate_mean_run_length(
    words: List[Dict],
    pause_threshold: float = 0.25,
) -> float:
    """
    Calculate mean length of runs (words between pauses).

    Longer runs indicate more fluent speech.

    CEFR Guidelines:
    - A1-A2: 3-5 words per run
    - B1: 6-8 words per run
    - B2+: 8+ words per run

    Args:
        words: List of dicts with "start", "end" keys
        pause_threshold: Gap threshold to count as pause

    Returns:
        Mean number of words between pauses
    """
    if not words:
        return 0.0

    # Find pause positions
    pause_indices = []
    for i in range(1, len(words)):
        prev_end = words[i - 1].get("end", 0)
        curr_start = words[i].get("start", 0)
        gap = curr_start - prev_end

        if gap >= pause_threshold:
            pause_indices.append(i)

    if not pause_indices:
        return float(len(words))

    # Calculate run lengths
    runs = []
    prev_idx = 0
    for idx in pause_indices:
        runs.append(idx - prev_idx)
        prev_idx = idx
    runs.append(len(words) - prev_idx)

    return sum(runs) / len(runs)


# =============================================================================
# LEXICAL DIVERSITY CALCULATIONS
# =============================================================================


def calculate_ttr(total_words: int, unique_words: int) -> float:
    """
    Calculate Type-Token Ratio (TTR).

    Formula: TTR = unique_words / total_words

    TTR measures vocabulary diversity. Higher = more diverse.
    Note: TTR decreases with text length, so use with caution.

    Args:
        total_words: Total word count (tokens)
        unique_words: Unique word count (types)

    Returns:
        TTR value between 0 and 1

    Example:
        >>> calculate_ttr(total_words=100, unique_words=60)
        0.6
    """
    if total_words <= 0:
        return 0.0
    return unique_words / total_words


def calculate_root_ttr(total_words: int, unique_words: int) -> float:
    """
    Calculate Root TTR (Guiraud's R).

    Formula: RTTR = unique_words / sqrt(total_words)

    More stable than TTR across different text lengths.
    Also known as Guiraud's Index.

    Reference: Guiraud (1960)

    Args:
        total_words: Total word count
        unique_words: Unique word count

    Returns:
        Root TTR value
    """
    if total_words <= 0:
        return 0.0
    return unique_words / math.sqrt(total_words)


def calculate_corrected_ttr(total_words: int, unique_words: int) -> float:
    """
    Calculate Corrected TTR (CTTR).

    Formula: CTTR = unique_words / sqrt(2 * total_words)

    Further correction for text length effects.
    Also known as Carroll's Corrected TTR.

    Reference: Carroll (1964)

    Args:
        total_words: Total word count
        unique_words: Unique word count

    Returns:
        Corrected TTR value
    """
    if total_words <= 0:
        return 0.0
    return unique_words / math.sqrt(2 * total_words)


def calculate_hapax_ratio(word_frequencies: Counter) -> Tuple[int, float]:
    """
    Calculate hapax legomena count and ratio.

    Hapax legomena = words appearing exactly once.
    Higher hapax ratio indicates more diverse vocabulary.

    Args:
        word_frequencies: Counter of word frequencies

    Returns:
        Tuple of (hapax_count, hapax_ratio)

    Example:
        >>> from collections import Counter
        >>> freq = Counter(["the", "the", "a", "cat", "dog"])
        >>> count, ratio = calculate_hapax_ratio(freq)
        >>> count  # "a", "cat", "dog" appear once
        3
    """
    if not word_frequencies:
        return 0, 0.0

    hapax_count = sum(1 for word, count in word_frequencies.items() if count == 1)
    unique_count = len(word_frequencies)

    return hapax_count, hapax_count / unique_count if unique_count > 0 else 0.0


def calculate_lexical_density(
    content_words: int,
    total_words: int,
) -> float:
    """
    Calculate lexical density.

    Formula: LD = content_words / total_words

    Content words = nouns, verbs, adjectives, adverbs
    Function words = articles, prepositions, pronouns, etc.

    Higher density = more informative speech.

    Args:
        content_words: Number of content words
        total_words: Total word count

    Returns:
        Lexical density between 0 and 1
    """
    if total_words <= 0:
        return 0.0
    return content_words / total_words


# =============================================================================
# SYLLABLE & WORD ANALYSIS
# =============================================================================


def count_syllables(word: str) -> int:
    """
    Estimate syllable count for a word.

    Uses vowel-group heuristic with adjustments for common patterns.

    Algorithm:
    1. Count vowel groups (consecutive vowels = 1 syllable)
    2. Adjust for silent 'e' at end
    3. Ensure minimum 1 syllable

    Args:
        word: Single word to analyze

    Returns:
        Estimated syllable count

    Example:
        >>> count_syllables("hello")
        2
        >>> count_syllables("beautiful")
        3
        >>> count_syllables("a")
        1
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjust for silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1

    return max(1, count)


def calculate_avg_word_length(words: List[str]) -> float:
    """
    Calculate average word length in characters.

    Args:
        words: List of words

    Returns:
        Average word length
    """
    if not words:
        return 0.0

    total_chars = sum(len(w) for w in words)
    return total_chars / len(words)


def calculate_long_word_ratio(
    words: List[str],
    threshold: int = 6,
) -> float:
    """
    Calculate ratio of long words.

    Args:
        words: List of words
        threshold: Minimum length to count as "long" (default: 6)

    Returns:
        Ratio of words longer than threshold
    """
    if not words:
        return 0.0

    long_count = sum(1 for w in words if len(w) > threshold)
    return long_count / len(words)


# =============================================================================
# TEXT TOKENIZATION
# =============================================================================


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words.

    Removes punctuation, converts to lowercase, filters empty strings.

    Args:
        text: Input text

    Returns:
        List of lowercase word tokens

    Example:
        >>> tokenize("Hello, World! How are you?")
        ["hello", "world", "how", "are", "you"]
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    words = text.split()
    words = [w.strip("'") for w in words if w and not w.isdigit()]
    return words


def get_word_frequencies(words: List[str]) -> Counter:
    """
    Get word frequency distribution.

    Args:
        words: List of word tokens

    Returns:
        Counter with word frequencies
    """
    return Counter(words)


# =============================================================================
# CEFR LEVEL MAPPING
# =============================================================================


def score_to_cefr_level(score: float) -> str:
    """
    Map numeric score (0-100) to CEFR level.

    Score Ranges:
    - 0-34: A1 (Beginner)
    - 35-54: A2 (Elementary)
    - 55-74: B1 (Intermediate)
    - 75-100: B2+ (Upper Intermediate and above)

    Args:
        score: Numeric score between 0 and 100

    Returns:
        CEFR level string

    Example:
        >>> score_to_cefr_level(65)
        "B1"
    """
    if score < 35:
        return "A1"
    elif score < 55:
        return "A2"
    elif score < 75:
        return "B1"
    else:
        return "B2+"


def wpm_to_cefr_level(wpm: float) -> str:
    """
    Map WPM to approximate CEFR level.

    Based on research benchmarks:
    - A1: 40-80 WPM
    - A2: 80-110 WPM
    - B1: 110-140 WPM
    - B2+: 140-180 WPM

    Args:
        wpm: Words per minute

    Returns:
        Approximate CEFR level
    """
    if wpm < 80:
        return "A1"
    elif wpm < 110:
        return "A2"
    elif wpm < 140:
        return "B1"
    else:
        return "B2+"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_speech_rate(
    words: List[Dict],
    duration_seconds: float,
) -> Dict:
    """
    Complete speech rate analysis.

    Args:
        words: List of dicts with "word", "start", "end" keys
        duration_seconds: Total duration

    Returns:
        Dict with WPM, articulation rate, syllables/sec
    """
    word_count = len(words)

    # Estimate syllables
    syllables = sum(count_syllables(w.get("word", "")) for w in words)

    # Detect pauses and calculate speech time
    pauses = detect_pauses(words)
    pause_metrics = calculate_pause_metrics(pauses, duration_seconds)
    speech_time = calculate_speech_time(duration_seconds, pause_metrics["total_time"])

    return {
        "word_count": word_count,
        "wpm": calculate_wpm(word_count, duration_seconds),
        "articulation_rate": calculate_articulation_rate(word_count, speech_time),
        "syllables_per_second": calculate_syllables_per_second(
            syllables, duration_seconds
        ),
        "syllable_count": syllables,
        "speech_time": speech_time,
    }


def analyze_lexical_diversity(text: str) -> Dict:
    """
    Complete lexical diversity analysis.

    Args:
        text: Input text

    Returns:
        Dict with TTR, RTTR, CTTR, hapax metrics
    """
    words = tokenize(text)
    frequencies = get_word_frequencies(words)

    total = len(words)
    unique = len(frequencies)
    hapax_count, hapax_ratio = calculate_hapax_ratio(frequencies)

    return {
        "total_words": total,
        "unique_words": unique,
        "ttr": calculate_ttr(total, unique),
        "root_ttr": calculate_root_ttr(total, unique),
        "corrected_ttr": calculate_corrected_ttr(total, unique),
        "hapax_count": hapax_count,
        "hapax_ratio": hapax_ratio,
    }


# =============================================================================
# ACCURACY CALCULATIONS
# =============================================================================


def calculate_error_density(
    error_count: int,
    word_count: int,
    per: int = 100,
) -> float:
    """
    Calculate error density (errors per N words).

    Formula: density = (error_count / word_count) * per

    CEFR Benchmarks (errors per 100 words):
    - A1: > 10 errors
    - A2: 5-10 errors
    - B1: 2-5 errors
    - B2+: < 2 errors

    Args:
        error_count: Number of errors
        word_count: Total word count
        per: Normalization factor (default: 100)

    Returns:
        Errors per N words

    Example:
        >>> calculate_error_density(5, 200)
        2.5
    """
    if word_count <= 0:
        return 0.0
    return (error_count / word_count) * per


def calculate_grammar_score(error_density: float) -> float:
    """
    Convert error density to a 0-100 score.

    Lower error density = higher score.

    Args:
        error_density: Errors per 100 words

    Returns:
        Score from 0-100
    """
    if error_density <= 1:
        return 90 + (1 - error_density) * 10
    elif error_density <= 3:
        return 70 + (3 - error_density) / 2 * 20
    elif error_density <= 6:
        return 50 + (6 - error_density) / 3 * 20
    elif error_density <= 10:
        return 30 + (10 - error_density) / 4 * 20
    else:
        return max(0, 30 - (error_density - 10) * 2)


# =============================================================================
# PHONOLOGY CALCULATIONS
# =============================================================================


def calculate_mean_confidence(confidences: List[float]) -> float:
    """
    Calculate mean confidence score from word confidences.

    Args:
        confidences: List of confidence values (0-1)

    Returns:
        Mean confidence

    Example:
        >>> calculate_mean_confidence([0.9, 0.8, 0.85])
        0.85
    """
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def calculate_confidence_std(confidences: List[float]) -> float:
    """
    Calculate standard deviation of confidence scores.

    Lower std = more consistent pronunciation.

    Args:
        confidences: List of confidence values

    Returns:
        Standard deviation
    """
    if len(confidences) < 2:
        return 0.0

    mean = sum(confidences) / len(confidences)
    variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
    return math.sqrt(variance)


def calculate_low_confidence_ratio(
    confidences: List[float],
    threshold: float = 0.7,
) -> float:
    """
    Calculate ratio of words with low confidence.

    Args:
        confidences: List of confidence values
        threshold: Confidence below this = low (default: 0.7)

    Returns:
        Ratio of low-confidence words (0-1)
    """
    if not confidences:
        return 0.0
    low_count = sum(1 for c in confidences if c < threshold)
    return low_count / len(confidences)


def calculate_pitch_variation(
    pitch_values: List[float],
) -> Dict[str, float]:
    """
    Calculate pitch variation metrics.

    Natural speech has moderate variation (0.15-0.35 relative).
    Monotone speech has low variation (< 0.10).

    Args:
        pitch_values: List of pitch (F0) values in Hz

    Returns:
        Dict with mean, std, range, and variation_ratio
    """
    if not pitch_values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "variation_ratio": 0.0,
        }

    # Filter out zeros/nans
    valid = [p for p in pitch_values if p > 0]
    if not valid:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "variation_ratio": 0.0,
        }

    mean_pitch = sum(valid) / len(valid)
    variance = sum((p - mean_pitch) ** 2 for p in valid) / len(valid)
    std_pitch = math.sqrt(variance)
    range_pitch = max(valid) - min(valid)

    return {
        "mean": mean_pitch,
        "std": std_pitch,
        "range": range_pitch,
        "variation_ratio": std_pitch / mean_pitch if mean_pitch > 0 else 0.0,
    }


def confidence_to_score(mean_confidence: float) -> float:
    """
    Convert mean confidence to a 0-100 score.

    Higher confidence = clearer pronunciation = higher score.

    CEFR Benchmarks:
    - A1: < 0.65 confidence
    - A2: 0.65-0.75 confidence
    - B1: 0.75-0.85 confidence
    - B2+: > 0.85 confidence

    Args:
        mean_confidence: Average word confidence (0-1)

    Returns:
        Score from 0-100
    """
    if mean_confidence >= 0.90:
        return 95
    elif mean_confidence >= 0.85:
        return 85 + (mean_confidence - 0.85) * 200
    elif mean_confidence >= 0.75:
        return 70 + (mean_confidence - 0.75) * 150
    elif mean_confidence >= 0.65:
        return 50 + (mean_confidence - 0.65) * 200
    elif mean_confidence >= 0.50:
        return 30 + (mean_confidence - 0.50) * 133
    else:
        return max(0, mean_confidence * 60)


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    # Speech Rate
    "calculate_wpm",
    "calculate_articulation_rate",
    "calculate_syllables_per_second",
    # Pause Analysis
    "detect_pauses",
    "calculate_pause_metrics",
    "calculate_speech_time",
    # Hesitation Markers
    "detect_fillers",
    "calculate_filler_rate",
    "detect_repetitions",
    "detect_false_starts",
    "FILLER_WORDS_EN",
    # Flow & Rhythm
    "calculate_speech_rate_variability",
    "calculate_mean_run_length",
    # Lexical Diversity
    "calculate_ttr",
    "calculate_root_ttr",
    "calculate_corrected_ttr",
    "calculate_hapax_ratio",
    "calculate_lexical_density",
    # Syllable & Word Analysis
    "count_syllables",
    "calculate_avg_word_length",
    "calculate_long_word_ratio",
    # Tokenization
    "tokenize",
    "get_word_frequencies",
    # CEFR Mapping
    "score_to_cefr_level",
    "wpm_to_cefr_level",
    # Convenience
    "analyze_speech_rate",
    "analyze_lexical_diversity",
    # Accuracy Calculations
    "calculate_error_density",
    "calculate_grammar_score",
    # Phonology Calculations
    "calculate_mean_confidence",
    "calculate_confidence_std",
    "calculate_low_confidence_ratio",
    "calculate_pitch_variation",
    "confidence_to_score",
]
