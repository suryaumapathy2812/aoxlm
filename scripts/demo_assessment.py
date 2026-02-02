#!/usr/bin/env python3
"""
Demo: CEFR Speech Assessment Pipeline

This script demonstrates the complete assessment pipeline:
1. Load audio file
2. Transcribe with faster-whisper (word timestamps)
3. Assess fluency using feature-based scoring
4. Assess range (vocabulary diversity)
5. Output combined CEFR assessment

Usage:
    uv run python scripts/demo_assessment.py data/1769858336827.mp3
    uv run python scripts/demo_assessment.py data/1769858336827.mp3 --device cuda
    uv run python scripts/demo_assessment.py --url "https://s3.amazonaws.com/..."
    uv run python scripts/demo_assessment.py data/audio.mp3 --model medium --device cuda
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    initial_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio using faster-whisper with word timestamps.

    Args:
        audio_path: Path to audio file
        model_size: Model size (tiny, base, small, medium, large-v2, large-v3)
        device: Device to use (cuda, cpu)
        compute_type: Compute type (float16, int8, int8_float16)
        language: Language code
        initial_prompt: Optional prompt to guide transcription

    Returns:
        Dict with 'text' and 'segments' (Whisper-compatible format)
    """
    from faster_whisper import WhisperModel

    # Adjust compute type for CPU
    if device == "cpu":
        compute_type = "int8"

    print(
        f"Loading faster-whisper model: {model_size} (device: {device}, compute: {compute_type})..."
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        vad_filter=True,  # Filter out silence
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Convert to Whisper-compatible format
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

    print(f"  Language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"  Duration: {info.duration:.1f}s")
    print(f"  Transcribed {total_words} words in {len(result_segments)} segments")

    return {
        "text": "".join(full_text),
        "segments": result_segments,
        "language": info.language,
        "duration": info.duration,
    }


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
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for Whisper (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8", "int8_float16"],
        help="Compute type for faster-whisper (default: float16)",
    )
    parser.add_argument(
        "--prompt",
        default="Indian English speaker discussing hobbies, interests, travel, making Ganesha idols, cooking, sports.",
        help="Initial prompt to guide transcription (helps with domain-specific words)",
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
    print(f"Model: faster-whisper {args.model}")
    print(f"Device: {args.device}")
    print()

    # 1. Transcribe with faster-whisper
    whisper_result = transcribe_faster_whisper(
        audio_path,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        initial_prompt=args.prompt,
    )

    duration = whisper_result.get("duration", 60.0)

    print("\n" + "-" * 60)
    print("TRANSCRIPTION:")
    print("-" * 60)
    text = whisper_result.get("text", "")
    print(text[:500])
    if len(text) > 500:
        print("...")

    # 2. Assess fluency
    fluency_result = assess_fluency_score(whisper_result, duration)

    # 3. Assess range (if not fluency-only)
    if not args.fluency_only:
        try:
            range_result = assess_range_score(whisper_result)

            # 4. Combine assessments
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
