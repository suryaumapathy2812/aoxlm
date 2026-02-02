#!/usr/bin/env python3
"""
Demo: CEFR Speech Assessment Pipeline

Simple demonstration of the modular assessment pipeline.

Usage:
    uv run python scripts/demo_assessment.py data/audio.mp3
    uv run python scripts/demo_assessment.py --device cuda data/audio.mp3
    uv run python scripts/demo_assessment.py --device cuda --enable-wavlm data/audio.mp3
    uv run python scripts/demo_assessment.py "https://example.com/audio.mp3"
    uv run python scripts/demo_assessment.py --device cuda "https://s3.amazonaws.com/..."
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
import urllib.request

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AssessmentPipeline, assess_audio


def download_audio(url: str) -> str:
    """Download audio from URL."""
    suffix = ".mp3" if ".mp3" in url else ".wav"
    output_path = tempfile.mktemp(suffix=suffix)
    print(f"Downloading audio from: {url[:80]}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="CEFR Speech Assessment Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to audio file or URL (auto-detected)",
    )
    parser.add_argument(
        "--url",
        help="URL to download audio from (alternative to passing URL as audio_path)",
    )
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
        "--output-json",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--fluency-weight",
        type=float,
        default=0.35,
        help="Weight for fluency in overall score (default: 0.35)",
    )
    parser.add_argument(
        "--range-weight",
        type=float,
        default=0.25,
        help="Weight for range in overall score (default: 0.25)",
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Disable accuracy (grammar) assessment",
    )
    parser.add_argument(
        "--no-phonology",
        action="store_true",
        help="Disable phonology (pronunciation) assessment",
    )
    parser.add_argument(
        "--enable-wavlm",
        action="store_true",
        help="Enable WavLM-based pronunciation analysis (adds smoothness metric)",
    )
    parser.add_argument(
        "--wavlm-model",
        default="wavlm-base",
        choices=["wavlm-base", "wavlm-large"],
        help="WavLM model variant (default: wavlm-base, faster; wavlm-large more accurate)",
    )

    args = parser.parse_args()

    # Get audio path (supports local files and URLs)
    if args.url:
        audio_path = download_audio(args.url)
    elif args.audio_path:
        # Auto-detect if audio_path is a URL
        if args.audio_path.startswith(("http://", "https://")):
            audio_path = download_audio(args.audio_path)
        else:
            audio_path = args.audio_path
    else:
        # Default to sample file
        audio_path = "data/1769858336827.mp3"
        if not Path(audio_path).exists():
            parser.error("No audio file specified. Provide a path or URL.")

    # Check file exists (skip check if we just downloaded)
    if not args.url and not (
        args.audio_path and args.audio_path.startswith(("http://", "https://"))
    ):
        if not Path(audio_path).exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

    print("=" * 60)
    print("CEFR SPEECH ASSESSMENT")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.enable_wavlm:
        print(f"WavLM: {args.wavlm_model} (enabled)")
    print()

    # Create pipeline and assess
    pipeline = AssessmentPipeline(
        transcriber="faster-whisper",
        model_size=args.model,
        device=args.device,
        fluency_weight=args.fluency_weight,
        range_weight=args.range_weight,
        enable_accuracy=not args.no_accuracy,
        enable_phonology=not args.no_phonology,
        enable_wavlm=args.enable_wavlm,
        wavlm_model=args.wavlm_model,
    )

    print("Transcribing and assessing...")
    result = pipeline.assess(audio_path)

    # Show transcription
    print("\n" + "-" * 60)
    print("TRANSCRIPTION:")
    print("-" * 60)
    text = result.transcription.text if result.transcription else ""
    print(text[:500])
    if len(text) > 500:
        print("...")

    # Show result
    print(result.summary())

    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return result


if __name__ == "__main__":
    main()
