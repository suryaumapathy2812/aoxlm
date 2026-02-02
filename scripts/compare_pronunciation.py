#!/usr/bin/env python3
"""
Compare pronunciation assessment across different encoders.

Available encoders:
- wavlm-large: General semantic understanding, noise robustness (315M)
- wavlm-base: Smaller/faster option (94M)
- mms-300m: Code-switching, 1000+ languages
- xls-r-300m: Cross-lingual, accents, 128 languages
- unispeech-sat: Speaker variation, different speaking styles
- hubert-large: Baseline semantic encoder

Usage:
    python scripts/compare_pronunciation.py <audio_path> [--device cuda]
    python scripts/compare_pronunciation.py <audio_path> --models wavlm-large mms-300m
"""

import argparse
import sys
import time
from typing import List, Optional

# Available encoder models (excluding 1B models due to memory)
AVAILABLE_MODELS = [
    "wavlm-large",
    "wavlm-base",
    "mms-300m",
    "xls-r-300m",
    "unispeech-sat",
    "hubert-large",
]


def run_comparison(
    audio_path: str,
    models: List[str],
    device: Optional[str] = None,
):
    """Run pronunciation assessment with multiple encoders."""
    from src.features.wavlm_features import (
        WavLMFeatureExtractor,
        WavLMPronunciationScorer,
    )
    import torch

    results = {}
    scorer = WavLMPronunciationScorer()

    # Pre-load audio once
    import librosa

    print(f"Loading audio: {audio_path}")
    audio_array, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio_array) / sr
    print(f"Duration: {duration:.1f}s ({len(audio_array)} samples)")
    print()

    for model_name in models:
        print(f"{'=' * 60}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 60}")

        try:
            start_time = time.time()

            # Create extractor with specific model
            extractor = WavLMFeatureExtractor(
                model_name=model_name,
                device=device,
            )

            # Extract features
            features, _ = extractor.extract(audio_array=audio_array)

            # Score
            result = scorer.score(features)

            elapsed = time.time() - start_time

            results[model_name] = {
                "score": result.score,
                "level": result.level,
                "consistency": result.sub_scores.get("consistency", 0),
                "smoothness": result.sub_scores.get("smoothness", 0),
                "articulation": result.sub_scores.get("articulation", 0),
                "time": elapsed,
            }

            print(f"Score: {result.score:.1f}/100 ({result.level})")
            print(f"  Consistency:  {result.sub_scores.get('consistency', 0):.1f}")
            print(f"  Smoothness:   {result.sub_scores.get('smoothness', 0):.1f}")
            print(f"  Articulation: {result.sub_scores.get('articulation', 0):.1f}")
            print(f"Time: {elapsed:.1f}s")
            print()

            # Clear GPU memory
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: {e}")
            results[model_name] = {"error": str(e)}
            print()

    # Summary table
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<20} {'Score':>8} {'Level':>6} {'Consist':>8} {'Smooth':>8} {'Artic':>8} {'Time':>8}"
    )
    print("-" * 70)

    for model_name in models:
        r = results.get(model_name, {})
        if "error" in r:
            print(f"{model_name:<20} {'ERROR':>8}")
        else:
            print(
                f"{model_name:<20} "
                f"{r['score']:>7.1f} "
                f"{r['level']:>6} "
                f"{r['consistency']:>7.1f} "
                f"{r['smoothness']:>7.1f} "
                f"{r['articulation']:>7.1f} "
                f"{r['time']:>7.1f}s"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare pronunciation assessment across encoders"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["wavlm-large", "wavlm-base", "mms-300m", "xls-r-300m"],
        help=f"Models to compare. Available: {AVAILABLE_MODELS}",
    )

    args = parser.parse_args()

    # Validate models
    for m in args.models:
        if m not in AVAILABLE_MODELS:
            print(f"Unknown model: {m}")
            print(f"Available: {AVAILABLE_MODELS}")
            sys.exit(1)

    run_comparison(
        audio_path=args.audio_path,
        models=args.models,
        device=args.device,
    )


if __name__ == "__main__":
    main()
