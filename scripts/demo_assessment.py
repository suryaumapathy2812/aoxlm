#!/usr/bin/env python3
"""
Demo: CEFR Speech Assessment Pipeline

This script demonstrates the complete assessment pipeline:
1. Load audio file
2. Transcribe with Whisper (word timestamps)
3. Assess fluency using feature-based scoring
4. Assess range (vocabulary diversity)
5. Output combined CEFR assessment

Usage:
    uv run python scripts/demo_assessment.py data/1769858336827.mp3
    uv run python scripts/demo_assessment.py --url "https://s3.amazonaws.com/..."
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
import urllib.request

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_audio(url: str, output_path: Optional[str] = None) -> str:
    """Download audio from URL."""
    if output_path is None:
        suffix = ".mp3" if ".mp3" in url else ".wav"
        output_path = tempfile.mktemp(suffix=suffix)

    print(f"Downloading audio from: {url[:80]}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")
    return output_path


def load_audio(path: str, max_duration: float = 60.0):
    """Load audio file using librosa."""
    import librosa

    print(f"Loading audio: {path}")
    audio, sr = librosa.load(path, sr=16000, mono=True, duration=max_duration)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    return audio, sr, duration


def transcribe_whisper(audio, device: str = "cpu") -> Dict:
    """Transcribe audio using Whisper with word timestamps."""
    import whisper

    print(f"Loading Whisper model (device: {device})...")
    model = whisper.load_model("base", device=device)

    print("Transcribing...")
    result = model.transcribe(
        audio,
        word_timestamps=True,
        language="en",
        verbose=False,
    )

    # Count words
    word_count = sum(len(seg.get("words", [])) for seg in result.get("segments", []))
    print(f"  Transcribed {word_count} words")

    return result


def assess_fluency_score(whisper_result: Dict, duration: float):
    """Assess fluency from Whisper output."""
    from src.features.fluency import FluencyAssessor

    print("\n" + "=" * 60)
    print("FLUENCY ASSESSMENT")
    print("=" * 60)

    assessor = FluencyAssessor()
    result = assessor.assess_from_whisper(whisper_result, duration=duration)

    print(result.summary())
    return result


def assess_range_score(whisper_result: Dict):
    """Assess vocabulary range from transcription."""
    from src.features.range import RangeAssessor

    print("\n" + "=" * 60)
    print("RANGE ASSESSMENT (Vocabulary)")
    print("=" * 60)

    # Extract text from Whisper result
    text = whisper_result.get("text", "")

    assessor = RangeAssessor()
    result = assessor.assess(text)

    print(result.summary())
    return result


def combine_assessments(fluency_result, range_result) -> Dict:
    """Combine fluency and range into overall assessment."""
    print("\n" + "=" * 60)
    print("COMBINED CEFR ASSESSMENT")
    print("=" * 60)

    # Weight: Fluency 60%, Range 40% (for spoken assessment)
    fluency_weight = 0.6
    range_weight = 0.4

    combined_score = (
        fluency_result.score * fluency_weight + range_result.score * range_weight
    )

    # Map to CEFR level
    if combined_score < 35:
        level = "A1"
    elif combined_score < 55:
        level = "A2"
    elif combined_score < 75:
        level = "B1"
    else:
        level = "B2+"

    # Combined confidence
    combined_confidence = (
        fluency_result.confidence * fluency_weight
        + range_result.confidence * range_weight
    )

    print(f"\nOverall CEFR Level: {level}")
    print(f"Combined Score: {combined_score:.1f}/100")
    print(f"Confidence: {combined_confidence:.1%}")
    print(f"\nBreakdown:")
    print(
        f"  Fluency ({fluency_weight:.0%}): {fluency_result.level} ({fluency_result.score:.1f})"
    )
    print(
        f"  Range ({range_weight:.0%}):   {range_result.level} ({range_result.score:.1f})"
    )

    return {
        "level": level,
        "score": combined_score,
        "confidence": combined_confidence,
        "components": {
            "fluency": fluency_result.to_dict(),
            "range": range_result.to_dict(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="CEFR Speech Assessment Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to audio file",
    )
    parser.add_argument(
        "--url",
        help="URL to download audio from",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper (default: cpu)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="Maximum audio duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--output-json",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--fluency-only",
        action="store_true",
        help="Only run fluency assessment",
    )

    args = parser.parse_args()

    # Get audio path
    if args.url:
        audio_path = download_audio(args.url)
    elif args.audio_path:
        audio_path = args.audio_path
    else:
        # Default to sample file
        audio_path = "data/1769858336827.mp3"
        if not Path(audio_path).exists():
            parser.error("No audio file specified. Use --url or provide a path.")

    # Check file exists
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("CEFR SPEECH ASSESSMENT DEMO")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Device: {args.device}")
    print()

    # 1. Load audio
    audio, sr, duration = load_audio(audio_path, max_duration=args.max_duration)

    # 2. Transcribe
    whisper_result = transcribe_whisper(audio, device=args.device)

    print("\n" + "-" * 60)
    print("TRANSCRIPTION:")
    print("-" * 60)
    print(whisper_result.get("text", "")[:500])
    if len(whisper_result.get("text", "")) > 500:
        print("...")

    # 3. Assess fluency
    fluency_result = assess_fluency_score(whisper_result, duration)

    # 4. Assess range (if not fluency-only)
    if not args.fluency_only:
        try:
            range_result = assess_range_score(whisper_result)

            # 5. Combine assessments
            combined = combine_assessments(fluency_result, range_result)
        except ImportError:
            print("\nNote: Range assessment not available yet.")
            combined = {
                "level": fluency_result.level,
                "score": fluency_result.score,
                "confidence": fluency_result.confidence,
                "components": {
                    "fluency": fluency_result.to_dict(),
                },
            }
    else:
        combined = {
            "level": fluency_result.level,
            "score": fluency_result.score,
            "confidence": fluency_result.confidence,
            "components": {
                "fluency": fluency_result.to_dict(),
            },
        }

    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    print("\n" + "=" * 60)
    print("Assessment complete!")
    print("=" * 60)

    return combined


if __name__ == "__main__":
    main()
