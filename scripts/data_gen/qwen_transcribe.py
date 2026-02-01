#!/usr/bin/env python3
"""
Qwen3-ASR Transcription with Timestamps

Uses qwen-asr package for transcription + forced alignment.
Outputs word-level timestamps with 42.9ms accuracy.

Usage:
    pip install qwen-asr
    python scripts/data_gen/qwen_transcribe.py audio.mp3
    python scripts/data_gen/qwen_transcribe.py audio.mp3 --output result.json

    # From URL
    python scripts/data_gen/qwen_transcribe.py https://example.com/audio.mp3

    # Without timestamps (faster)
    python scripts/data_gen/qwen_transcribe.py audio.mp3 --no-timestamps
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import torch
    from qwen_asr import Qwen3ASRModel
except ImportError:
    print("Error: qwen-asr not installed")
    print("Run: pip install qwen-asr")
    print("For faster inference: pip install 'qwen-asr[vllm]'")
    sys.exit(1)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    with_aligner: bool = True,
    device: Optional[str] = None,
):
    """
    Load Qwen3-ASR model with optional forced aligner.

    Args:
        model_name: Model to use (Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B)
        with_aligner: Whether to load ForcedAligner for timestamps
        device: Device to use (auto-detected if None)

    Returns:
        Loaded model
    """
    device = device or get_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading {model_name} on {device}...")

    kwargs = {
        "dtype": dtype,
        "device_map": device,
        "max_inference_batch_size": 8,
        "max_new_tokens": 2048,  # For long audio
    }

    if with_aligner:
        kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
        kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": device,
        }

    model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)

    return model


def transcribe(
    audio: str,
    model=None,
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    with_timestamps: bool = True,
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe audio using Qwen3-ASR.

    Args:
        audio: Audio file path or URL
        model: Pre-loaded model (optional, will load if None)
        model_name: Model to use if loading
        with_timestamps: Whether to include word-level timestamps
        language: Force language (None for auto-detection)

    Returns:
        Transcription result dict
    """
    # Load model if not provided
    if model is None:
        model = load_model(model_name, with_aligner=with_timestamps)

    start_time = time.time()

    print(f"Transcribing: {audio}")

    # Transcribe
    results = model.transcribe(
        audio=audio,
        language=language,
        return_time_stamps=with_timestamps,
    )

    elapsed = time.time() - start_time
    result = results[0]  # Single audio input

    # Build output dict
    output = {
        "text": result.text,
        "language": result.language,
        "processing_time": round(elapsed, 2),
    }

    # Add timestamps if available
    if with_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
        words = []
        for ts in result.time_stamps:
            word_info = {
                "w": ts.text,
                "s": round(ts.start_time, 3),
                "e": round(ts.end_time, 3),
            }
            words.append(word_info)
        output["words"] = words

    return output


def print_result(result: dict, verbose: bool = False):
    """Pretty print transcription result."""
    print("\n" + "=" * 70)
    print("QWEN3-ASR TRANSCRIPTION RESULT")
    print("=" * 70)

    print(f"\nLanguage: {result.get('language', 'unknown')}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")

    print(f"\nText:")
    print(f"  {result.get('text', 'N/A')}")

    words = result.get("words", [])
    if words:
        print(f"\nWords ({len(words)}):")
        print("-" * 70)

        if verbose:
            for w in words:
                word = w.get("w", "")
                start = w.get("s", 0)
                end = w.get("e", 0)
                print(f"  [{start:6.3f} - {end:6.3f}] {word}")
        else:
            # Show first and last few words
            for w in words[:5]:
                print(f"  [{w['s']:6.3f} - {w['e']:6.3f}] {w['w']}")
            if len(words) > 10:
                print(f"  ... {len(words) - 10} more words ...")
            for w in words[-5:]:
                print(f"  [{w['s']:6.3f} - {w['e']:6.3f}] {w['w']}")

            print(f"\n  (use --verbose to see all {len(words)} words)")

        # Calculate duration from timestamps
        if words:
            duration = words[-1]["e"]
            print(f"\nAudio duration: ~{duration:.1f}s")
            rtf = result.get("processing_time", 0) / duration if duration > 0 else 0
            print(f"Real-time factor: {rtf:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Qwen3-ASR")
    parser.add_argument("audio", type=str, help="Audio file path or URL")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        choices=["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"],
        help="Model to use",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Skip timestamp extraction (faster)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language (e.g., English, Chinese). Auto-detect if not set.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all words with timestamps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda:0, mps, cpu). Auto-detect if not set.",
    )

    args = parser.parse_args()

    # Check if audio exists (if local file)
    if not args.audio.startswith(("http://", "https://")):
        if not Path(args.audio).exists():
            print(f"Error: File not found: {args.audio}")
            sys.exit(1)

    try:
        # Load model
        model = load_model(
            model_name=args.model,
            with_aligner=not args.no_timestamps,
            device=args.device,
        )

        # Transcribe
        result = transcribe(
            audio=args.audio,
            model=model,
            with_timestamps=not args.no_timestamps,
            language=args.language,
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")

    # Print result
    print_result(result, verbose=args.verbose)

    # Print raw JSON if verbose
    if args.verbose:
        print("\n" + "=" * 70)
        print("RAW JSON")
        print("=" * 70)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
