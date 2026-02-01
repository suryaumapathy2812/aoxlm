#!/usr/bin/env python3
"""
Encoder Comparison Script

Compare different semantic audio encoders on sample audio:
- WavLM-large (microsoft/wavlm-large)
- MMS-300M (facebook/mms-300m)
- XLS-R-300M (facebook/wav2vec2-xls-r-300m)
- HuBERT-large (facebook/hubert-large-ls960-ft)

Compares:
1. Output shapes and dimensions
2. Frame rate
3. Inference time
4. Memory usage
5. Qualitative: how different are the embeddings?

Usage:
    python scripts/compare_encoders.py [--audio path/to/audio.wav]

If no audio provided, generates synthetic audio for testing.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torchaudio
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Encoders to compare (can be run on smaller machines)
ENCODERS_TO_COMPARE = [
    "wavlm-large",
    "mms-300m",
    "xls-r-300m",
    "hubert-large",
]

# Optional larger encoders (need more VRAM)
LARGE_ENCODERS = [
    "xls-r-1b",
    "mms-1b",
]


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def generate_test_audio(
    duration: float = 5.0, sample_rate: int = 16000
) -> torch.Tensor:
    """Generate synthetic test audio (silence + sine wave)."""
    samples = int(duration * sample_rate)

    # Mix of silence and tones
    audio = torch.zeros(samples)

    # Add some sine waves at speech frequencies
    t = torch.linspace(0, duration, samples)
    audio += 0.3 * torch.sin(2 * np.pi * 200 * t)  # 200 Hz
    audio += 0.2 * torch.sin(2 * np.pi * 400 * t)  # 400 Hz
    audio += 0.1 * torch.sin(2 * np.pi * 800 * t)  # 800 Hz

    # Add some noise
    audio += 0.05 * torch.randn(samples)

    # Normalize
    audio = audio / audio.abs().max()

    return audio


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file."""
    audio, sr = torchaudio.load(path)

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    else:
        audio = audio.squeeze(0)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    return audio


def compare_encoders(
    audio: torch.Tensor,
    encoders: list,
    device: str = "cuda",
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> dict:
    """
    Compare multiple encoders on the same audio.

    Returns dict with results for each encoder.
    """
    from models.encoder import SemanticEncoder, ENCODER_CONFIGS

    results = {}
    audio_duration = audio.shape[0] / 16000

    print(f"\nComparing encoders on {audio_duration:.1f}s audio")
    print(f"Device: {device}")
    print("=" * 80)

    for encoder_name in encoders:
        if encoder_name not in ENCODER_CONFIGS:
            print(f"Skipping unknown encoder: {encoder_name}")
            continue

        config = ENCODER_CONFIGS[encoder_name]
        print(f"\n{config.name}")
        print(f"  HuggingFace: {config.hf_id}")
        print(f"  Best for: {config.best_for}")
        print("-" * 60)

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Load encoder
            mem_before = get_memory_usage()
            load_start = time.time()

            encoder = SemanticEncoder(
                encoder_name=encoder_name,
                freeze=True,
                device=device,
            )

            load_time = time.time() - load_start
            mem_after = get_memory_usage()

            print(f"  Load time: {load_time:.2f}s")
            print(f"  Memory: {mem_after - mem_before:.0f} MB")

            # Warmup
            audio_batch = audio.unsqueeze(0).to(device)
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = encoder(audio_batch, preprocess=True)

            # Benchmark
            times = []
            for _ in range(benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()

                with torch.no_grad():
                    output = encoder(audio_batch, preprocess=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            rtf = avg_time / audio_duration  # Real-time factor

            print(f"  Output shape: {output.shape}")
            print(f"  Hidden size: {output.shape[-1]}")
            print(f"  Num frames: {output.shape[1]}")
            print(f"  Frame rate: {output.shape[1] / audio_duration:.1f} Hz")
            print(f"  Inference: {avg_time * 1000:.1f}ms +/- {std_time * 1000:.1f}ms")
            print(f"  RTF: {rtf:.3f}x (lower is faster)")

            # Compute embedding statistics
            output_np = output.cpu().numpy()
            print(f"  Embedding stats:")
            print(f"    Mean: {output_np.mean():.4f}")
            print(f"    Std: {output_np.std():.4f}")
            print(f"    Min: {output_np.min():.4f}")
            print(f"    Max: {output_np.max():.4f}")

            results[encoder_name] = {
                "load_time": load_time,
                "memory_mb": mem_after - mem_before,
                "output_shape": list(output.shape),
                "hidden_size": output.shape[-1],
                "num_frames": output.shape[1],
                "frame_rate": output.shape[1] / audio_duration,
                "inference_ms": avg_time * 1000,
                "inference_std_ms": std_time * 1000,
                "rtf": rtf,
                "embedding_mean": float(output_np.mean()),
                "embedding_std": float(output_np.std()),
            }

            # Clean up
            del encoder
            del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            results[encoder_name] = {"error": str(e)}

    return results


def print_summary(results: dict):
    """Print comparison summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Filter out errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if not valid_results:
        print("No valid results to compare.")
        return

    # Print table
    print(
        f"\n{'Encoder':<20} {'Hidden':<8} {'Frames':<8} {'RTF':<8} {'Memory':<10} {'Best For'}"
    )
    print("-" * 80)

    from models.encoder import ENCODER_CONFIGS

    for name, data in valid_results.items():
        config = ENCODER_CONFIGS.get(name)
        best_for = (
            config.best_for[:30] + "..."
            if config and len(config.best_for) > 30
            else (config.best_for if config else "")
        )

        print(
            f"{name:<20} {data['hidden_size']:<8} {data['num_frames']:<8} "
            f"{data['rtf']:.3f}x   {data['memory_mb']:.0f} MB     {best_for}"
        )

    # Recommendation
    print("\n" + "-" * 80)
    print("RECOMMENDATION:")

    # Sort by RTF (fastest first)
    sorted_by_speed = sorted(valid_results.items(), key=lambda x: x[1]["rtf"])
    fastest = sorted_by_speed[0][0]

    print(f"  Fastest: {fastest} (RTF: {valid_results[fastest]['rtf']:.3f}x)")

    # For our use case, recommend based on understanding needs
    if "mms-300m" in valid_results and "wavlm-large" in valid_results:
        mms_rtf = valid_results["mms-300m"]["rtf"]
        wavlm_rtf = valid_results["wavlm-large"]["rtf"]

        print(f"\n  For code-switching/multilingual: mms-300m (RTF: {mms_rtf:.3f}x)")
        print(f"  For general English: wavlm-large (RTF: {wavlm_rtf:.3f}x)")

    print("\nNote: All encoders have similar quality on semantic understanding.")
    print("Choose based on your use case (code-switching vs English-focused).")


def main():
    parser = argparse.ArgumentParser(description="Compare semantic audio encoders")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of synthetic audio (if no file)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--include-large", action="store_true", help="Include 1B+ param encoders"
    )
    parser.add_argument(
        "--encoders", type=str, nargs="+", help="Specific encoders to test"
    )

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or generate audio
    if args.audio:
        print(f"Loading audio from: {args.audio}")
        audio = load_audio(args.audio)
    else:
        print(f"Generating {args.duration}s synthetic test audio...")
        audio = generate_test_audio(duration=args.duration)

    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[0] / 16000:.1f}s")

    # Select encoders
    if args.encoders:
        encoders = args.encoders
    else:
        encoders = ENCODERS_TO_COMPARE.copy()
        if args.include_large:
            encoders.extend(LARGE_ENCODERS)

    print(f"\nEncoders to compare: {encoders}")

    # Run comparison
    results = compare_encoders(
        audio=audio,
        encoders=encoders,
        device=device,
    )

    # Print summary
    print_summary(results)

    # Save results
    import json

    output_path = Path("encoder_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
