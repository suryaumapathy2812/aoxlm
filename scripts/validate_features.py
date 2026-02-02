#!/usr/bin/env python3
"""
Validate Feature Extraction Against Established Libraries

Compares our feature calculations to:
- textstat: Readability and text statistics
- lexicalrichness: Lexical diversity measures (TTR, MTLD, etc.)

Usage:
    # Validate with sample texts
    uv run python scripts/validate_features.py

    # Validate with custom text
    uv run python scripts/validate_features.py --text "Your custom text here"

    # Validate from audio file (transcribes first)
    uv run python scripts/validate_features.py --audio data/1769858336827.mp3
    uv run python scripts/validate_features.py --audio data/1769858336827.mp3 --device cuda

    # Validate from audio URL
    uv run python scripts/validate_features.py --url "https://s3.amazonaws.com/..."
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

import textstat
from lexicalrichness import LexicalRichness

from src.features.range import RangeFeatureExtractor, RangeAssessor
from src.features.fluency import FluencyAssessor


def download_audio(url: str) -> str:
    """Download audio from URL."""
    suffix = ".mp3" if ".mp3" in url else ".wav"
    output_path = tempfile.mktemp(suffix=suffix)
    print(f"Downloading audio from: {url[:80]}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")
    return output_path


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Transcribe audio using faster-whisper."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\nLoading faster-whisper model: {model_size} (device: {device})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing...")
    segments, info = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    # Convert to result format
    result_segments = []
    full_text = []
    total_words = 0

    for segment in segments:
        seg_words = []
        if segment.words:
            for word in segment.words:
                seg_words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )
                total_words += 1

        result_segments.append(
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": seg_words,
            }
        )
        full_text.append(segment.text)

    print(f"  Duration: {info.duration:.1f}s")
    print(f"  Transcribed {total_words} words in {len(result_segments)} segments")

    return {
        "text": "".join(full_text),
        "segments": result_segments,
        "duration": info.duration,
    }


def validate_fluency_features(whisper_result: Dict[str, Any], duration: float):
    """Validate fluency features (WPM, pauses, etc.)."""

    print("\n" + "=" * 70)
    print("FLUENCY FEATURE VALIDATION")
    print("=" * 70)

    # Extract words from whisper result
    words = []
    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append(
                {
                    "word": word_info.get("word", ""),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                }
            )

    if not words:
        print("No words with timestamps found!")
        return

    # Our fluency assessment
    assessor = FluencyAssessor()
    result = assessor.assess(words, duration)
    features = result.features

    print(f"\nDuration: {duration:.1f}s")
    print(f"Word Count: {features.word_count}")
    print()

    print("-" * 70)
    print("SPEECH RATE VALIDATION")
    print("-" * 70)

    # Manual WPM calculation for validation
    manual_wpm = (features.word_count / duration) * 60
    print(f"{'Metric':<35} {'Our Value':>15} {'Manual Calc':>15}")
    print("-" * 70)
    print(f"{'Words Per Minute (WPM)':<35} {features.wpm:>15.1f} {manual_wpm:>15.1f}")

    # Speech time calculation
    actual_speech_time = sum(w["end"] - w["start"] for w in words)
    print(
        f"{'Speech Time (excl. pauses)':<35} {features.speech_time:>15.1f}s {actual_speech_time:>15.1f}s"
    )

    # Articulation rate
    manual_art_rate = (
        (features.word_count / features.speech_time) * 60
        if features.speech_time > 0
        else 0
    )
    print(
        f"{'Articulation Rate':<35} {features.articulation_rate:>15.1f} {manual_art_rate:>15.1f}"
    )

    print()
    print("-" * 70)
    print("PAUSE ANALYSIS VALIDATION")
    print("-" * 70)

    # Manual pause calculation
    manual_pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= 0.25:  # Same threshold as our extractor
            manual_pauses.append(gap)

    manual_pause_count = len(manual_pauses)
    manual_total_pause = sum(manual_pauses)
    manual_pause_ratio = manual_total_pause / duration if duration > 0 else 0
    manual_long_pauses = sum(1 for p in manual_pauses if p >= 1.0)

    print(f"{'Metric':<35} {'Our Value':>15} {'Manual Calc':>15}")
    print("-" * 70)
    print(
        f"{'Number of Pauses (>250ms)':<35} {features.num_pauses:>15} {manual_pause_count:>15}"
    )
    print(
        f"{'Long Pauses (>1s)':<35} {features.num_long_pauses:>15} {manual_long_pauses:>15}"
    )
    print(
        f"{'Total Pause Time':<35} {features.total_pause_time:>15.1f}s {manual_total_pause:>15.1f}s"
    )
    print(
        f"{'Pause Ratio':<35} {features.pause_ratio:>15.1%} {manual_pause_ratio:>15.1%}"
    )

    if manual_pauses:
        manual_mean_pause = sum(manual_pauses) / len(manual_pauses)
        print(
            f"{'Mean Pause Duration':<35} {features.mean_pause_duration:>15.2f}s {manual_mean_pause:>15.2f}s"
        )

    print()
    print("-" * 70)
    print("HESITATION MARKERS")
    print("-" * 70)
    print(f"{'Filler Words':<35} {features.filler_count:>15}")
    print(f"{'Filler Rate (per minute)':<35} {features.filler_rate:>15.1f}")
    print(f"{'Repetitions':<35} {features.repetition_count:>15}")
    print(f"{'False Starts':<35} {features.false_start_count:>15}")

    print()
    print("-" * 70)
    print("FLOW METRICS")
    print("-" * 70)
    print(f"{'Speech Rate Variability':<35} {features.speech_rate_variability:>15.1f}")
    print(f"{'Mean Run Length (words)':<35} {features.mean_run_length:>15.1f}")

    print()
    print("-" * 70)
    print("OUR CEFR FLUENCY ASSESSMENT")
    print("-" * 70)
    print(f"Level: {result.level} (Score: {result.score:.1f}/100)")
    print(f"Confidence: {result.confidence:.1%}")
    print("\nSub-scores:")
    for name, score in result.sub_scores.items():
        print(f"  {name:<20}: {score:.1f}/100")

    return result


def compare_range_features(text: str):
    """Compare our Range features to established libraries."""

    print("=" * 70)
    print("RANGE FEATURE VALIDATION")
    print("=" * 70)
    print(f"\nText ({len(text)} chars):")
    print(f"  '{text[:100]}{'...' if len(text) > 100 else ''}'")
    print()

    # Our implementation
    extractor = RangeFeatureExtractor()
    our_features, academic, rare = extractor.extract(text)

    # Established libraries
    lex = LexicalRichness(text)

    print("-" * 70)
    print("LEXICAL DIVERSITY COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<30} {'Ours':>15} {'Library':>15} {'Diff':>10}")
    print("-" * 70)

    # Word count
    our_words = our_features.total_words
    lib_words = lex.words
    diff = our_words - lib_words
    print(f"{'Word Count':<30} {our_words:>15} {lib_words:>15} {diff:>+10}")

    # Unique words (types)
    our_unique = our_features.unique_words
    lib_unique = lex.terms
    diff = our_unique - lib_unique
    print(f"{'Unique Words (Types)':<30} {our_unique:>15} {lib_unique:>15} {diff:>+10}")

    # Type-Token Ratio
    our_ttr = our_features.type_token_ratio
    lib_ttr = lex.ttr
    diff = our_ttr - lib_ttr
    print(
        f"{'Type-Token Ratio (TTR)':<30} {our_ttr:>15.4f} {lib_ttr:>15.4f} {diff:>+10.4f}"
    )

    # Root TTR (Guiraud's R)
    our_rttr = our_features.root_ttr
    lib_rttr = lex.rttr
    diff = our_rttr - lib_rttr
    print(
        f"{'Root TTR (Guiraud R)':<30} {our_rttr:>15.4f} {lib_rttr:>15.4f} {diff:>+10.4f}"
    )

    # Corrected TTR
    our_cttr = our_features.corrected_ttr
    lib_cttr = lex.cttr
    diff = our_cttr - lib_cttr
    print(
        f"{'Corrected TTR (CTTR)':<30} {our_cttr:>15.4f} {lib_cttr:>15.4f} {diff:>+10.4f}"
    )

    print()
    print("-" * 70)
    print("TEXT STATISTICS COMPARISON (textstat)")
    print("-" * 70)

    # Syllable count
    our_syllables = sum(
        extractor._count_syllables(w) for w in extractor._tokenize(text)
    )
    lib_syllables = textstat.syllable_count(text)
    diff = our_syllables - lib_syllables
    print(f"{'Syllable Count':<30} {our_syllables:>15} {lib_syllables:>15} {diff:>+10}")

    # Average syllables per word
    our_avg_syl = our_features.avg_syllables
    lib_avg_syl = lib_syllables / lib_words if lib_words > 0 else 0
    diff = our_avg_syl - lib_avg_syl
    print(
        f"{'Avg Syllables/Word':<30} {our_avg_syl:>15.3f} {lib_avg_syl:>15.3f} {diff:>+10.3f}"
    )

    # Lexicon count (content words) - textstat
    lib_lexicon = textstat.lexicon_count(text, removepunct=True)
    print(
        f"{'Lexicon Count (textstat)':<30} {our_words:>15} {lib_lexicon:>15} {our_words - lib_lexicon:>+10}"
    )

    # Average word length
    our_avg_len = our_features.avg_word_length
    lib_avg_len = textstat.avg_letter_per_word(text)
    diff = our_avg_len - lib_avg_len
    print(
        f"{'Avg Word Length':<30} {our_avg_len:>15.3f} {lib_avg_len:>15.3f} {diff:>+10.3f}"
    )

    print()
    print("-" * 70)
    print("ADDITIONAL METRICS FROM LIBRARIES")
    print("-" * 70)

    # MTLD (Measure of Textual Lexical Diversity) - more robust than TTR
    try:
        mtld = lex.mtld(threshold=0.72)
        print(f"{'MTLD (lexicalrichness)':<30} {mtld:>15.2f}")
    except Exception as e:
        print(f"{'MTLD (lexicalrichness)':<30} {'N/A (text too short)':>15}")

    # Flesch Reading Ease
    flesch = textstat.flesch_reading_ease(text)
    print(f"{'Flesch Reading Ease':<30} {flesch:>15.2f}")

    # Flesch-Kincaid Grade
    fk_grade = textstat.flesch_kincaid_grade(text)
    print(f"{'Flesch-Kincaid Grade':<30} {fk_grade:>15.2f}")

    # Dale-Chall Readability
    dale_chall = textstat.dale_chall_readability_score(text)
    print(f"{'Dale-Chall Score':<30} {dale_chall:>15.2f}")

    # Difficult words
    difficult = textstat.difficult_words(text)
    print(f"{'Difficult Words (textstat)':<30} {difficult:>15}")

    print()
    print("-" * 70)
    print("OUR UNIQUE FEATURES (not in standard libraries)")
    print("-" * 70)
    print(f"{'Academic Word Ratio':<30} {our_features.academic_word_ratio:>15.4f}")
    print(f"{'Academic Words Found':<30} {our_features.academic_word_count:>15}")
    print(f"{'Rare Word Ratio':<30} {our_features.rare_word_ratio:>15.4f}")
    print(f"{'Content Word Ratio':<30} {our_features.content_word_ratio:>15.4f}")
    print(f"{'Lexical Density':<30} {our_features.lexical_density:>15.4f}")
    print(f"{'Long Word Ratio (>6 chars)':<30} {our_features.long_word_ratio:>15.4f}")
    print(f"{'Hapax Legomena':<30} {our_features.hapax_legomena:>15}")
    print(f"{'Hapax Ratio':<30} {our_features.hapax_ratio:>15.4f}")

    if academic:
        print(f"\nAcademic words: {', '.join(academic[:10])}")
    if rare:
        print(f"Rare words: {', '.join(rare[:10])}")

    return our_features, lex


def validate_with_samples():
    """Run validation on multiple sample texts."""

    samples = {
        "A1 (Basic)": """
            I like my house. It is big. I have a dog. My dog is nice.
            I go to school. School is good. I like my teacher.
        """,
        "B1 (Intermediate)": """
            I believe that learning English is very important for my career
            because many international companies require employees to communicate
            effectively in English. Additionally, being able to speak English
            allows me to access more information and connect with people from
            different countries and cultures.
        """,
        "B2+ (Advanced)": """
            The implementation of sustainable development strategies requires
            a comprehensive understanding of environmental, economic, and social
            factors. Research indicates that organizations which prioritize
            sustainability tend to demonstrate superior long-term performance.
            Furthermore, the integration of renewable energy sources has become
            increasingly significant in addressing climate change challenges.
        """,
        "Real Student Sample": """
            I am interested in making ganesha and alka. It is feeling good for me 
            to make it and craft it. So it is enjoyable for me. Most of the time 
            I am interested in traveling because I have to explore more places. 
            So I am interested in traveling. So I am trying whenever I get time, 
            I am traveling. I travel a lot. So it is giving me to learn more from society.
        """,
    }

    for name, text in samples.items():
        print("\n" + "=" * 70)
        print(f"SAMPLE: {name}")
        print("=" * 70)

        compare_range_features(text.strip())

        # Also show our CEFR assessment
        assessor = RangeAssessor()
        result = assessor.assess(text.strip())
        print(
            f"\n{'OUR CEFR ASSESSMENT:':<30} {result.level} (score: {result.score:.1f}/100)"
        )
        print()


def validate_from_audio(
    audio_path: str,
    device: str = "cuda",
    model: str = "large-v3",
):
    """Full validation from audio file."""

    print("=" * 70)
    print("AUDIO FEATURE VALIDATION")
    print("=" * 70)
    print(f"Audio: {audio_path}")
    print(f"Model: {model}")
    print(f"Device: {device}")

    # 1. Transcribe
    whisper_result = transcribe_audio(audio_path, model_size=model, device=device)
    duration = whisper_result.get("duration", 0)
    text = whisper_result.get("text", "")

    print("\n" + "-" * 70)
    print("TRANSCRIPTION:")
    print("-" * 70)
    print(text[:500])
    if len(text) > 500:
        print("...")

    # 2. Validate Fluency Features
    validate_fluency_features(whisper_result, duration)

    # 3. Validate Range Features
    compare_range_features(text)

    return whisper_result, text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate feature extraction against established libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", help="Custom text to analyze")
    parser.add_argument("--audio", help="Audio file to transcribe and validate")
    parser.add_argument("--url", help="URL to download audio from")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for transcription (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Run validation on sample texts",
    )

    args = parser.parse_args()

    if args.url:
        # Download and validate audio from URL
        audio_path = download_audio(args.url)
        validate_from_audio(audio_path, device=args.device, model=args.model)
    elif args.audio:
        # Validate from local audio file
        if not Path(args.audio).exists():
            print(f"Error: Audio file not found: {args.audio}")
            sys.exit(1)
        validate_from_audio(args.audio, device=args.device, model=args.model)
    elif args.text:
        # Validate custom text
        compare_range_features(args.text)
    elif args.samples:
        # Run sample validation
        validate_with_samples()
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 70)
        print("QUICK START")
        print("=" * 70)
        print("""
# Validate from audio file:
uv run python scripts/validate_features.py --audio data/1769858336827.mp3 --device cuda

# Validate custom text:
uv run python scripts/validate_features.py --text "I am interested in traveling..."

# Run sample validation:
uv run python scripts/validate_features.py --samples
        """)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. Word count, TTR, RTTR, CTTR should match lexicalrichness closely
2. Syllable counting may differ slightly (heuristic vs dictionary-based)
3. Fluency metrics (WPM, pauses) are calculated from word timestamps
4. Our unique features (academic words, rare words) add CEFR-specific value

If differences are large (>10%), our implementation may need adjustment.
If differences are small (<5%), our implementation is reliable.
    """)


if __name__ == "__main__":
    main()
