#!/usr/bin/env python3
"""
Pipeline Validation Script

Tests each component of the CEFR assessment pipeline:
1. Audio loading (torchaudio)
2. Semantic encoder (WavLM)
3. CEFR classification heads

Usage:
    python scripts/validate_pipeline.py data/audio.mp3
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_audio_loading(audio_path: str):
    """Test audio loading with torchaudio."""
    print("\n" + "=" * 60)
    print("[1/3] Testing Audio Loading")
    print("=" * 60)

    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"  File: {audio_path}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {waveform.shape[0]}")
    print(f"  Samples: {waveform.shape[1]:,}")
    print(f"  Duration: {waveform.shape[1] / sample_rate:.2f} seconds")

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"  Converted to mono")

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        print(f"  Resampled to 16000 Hz")

    print("  ✅ Audio loading: PASSED")
    return waveform.squeeze(0)  # [samples]


def test_encoder(waveform):
    """Test semantic encoder (WavLM)."""
    print("\n" + "=" * 60)
    print("[2/3] Testing Semantic Encoder (WavLM)")
    print("=" * 60)

    import torch
    from src.models.encoder import SemanticEncoder

    # Use CPU for testing
    device = "cpu"

    print(f"  Loading WavLM-large...")
    encoder = SemanticEncoder(
        encoder_name="wavlm-large",
        freeze=True,
        device=device,
    )

    print(f"  Encoder: {encoder.config.name}")
    print(f"  Hidden size: {encoder.hidden_size}")
    print(f"  Frame rate: {encoder.frame_rate} Hz")

    # Encode audio
    waveform = waveform.unsqueeze(0).to(device)  # [1, samples]
    print(f"  Input shape: {waveform.shape}")

    with torch.no_grad():
        hidden_states = encoder(waveform, preprocess=True)

    print(f"  Output shape: {hidden_states.shape}")
    print(f"  Expected: [1, ~{waveform.shape[1] // 320}, {encoder.hidden_size}]")

    # Validate output
    assert hidden_states.dim() == 3, "Expected 3D output [B, T, D]"
    assert hidden_states.shape[0] == 1, "Batch size should be 1"
    assert hidden_states.shape[2] == encoder.hidden_size, f"Hidden size mismatch"

    print("  ✅ Semantic encoder: PASSED")
    return hidden_states


def test_cefr_heads(hidden_states):
    """Test CEFR classification heads."""
    print("\n" + "=" * 60)
    print("[3/3] Testing CEFR Classification Heads")
    print("=" * 60)

    import torch
    from src.models.cefr_heads import create_cefr_heads, CEFR_LEVELS

    input_dim = hidden_states.shape[2]
    print(f"  Input dimension: {input_dim}")

    # Test Phase 1 heads
    print(f"\n  Phase 1 heads (fluency, range, overall):")
    heads = create_cefr_heads(phase=1, input_dim=input_dim)
    heads.eval()

    print(f"  Dimensions: {heads.dimensions}")

    with torch.no_grad():
        outputs = heads(hidden_states)

    for dim_name, output in outputs.items():
        level = output["predicted_level"][0]
        conf = output["confidence"][0].item()
        probs = output["probs"][0].tolist()
        prob_str = " | ".join([f"{l}:{p:.2f}" for l, p in zip(CEFR_LEVELS, probs)])
        print(f"    {dim_name:10}: {level} (conf: {conf:.2%}) [{prob_str}]")

    # Validate output
    for dim_name, output in outputs.items():
        assert output["logits"].shape == (1, 4), f"Logits shape mismatch for {dim_name}"
        assert output["probs"].shape == (1, 4), f"Probs shape mismatch for {dim_name}"
        assert len(output["predicted_level"]) == 1, f"Predicted level length mismatch"
        assert output["predicted_level"][0] in CEFR_LEVELS, (
            f"Invalid level for {dim_name}"
        )

    # Count parameters
    total_params = sum(p.numel() for p in heads.parameters())
    print(f"\n  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    print("  ✅ CEFR heads: PASSED")
    return outputs


def test_whisperx():
    """Test WhisperX import (may fail due to pyannote compatibility)."""
    print("\n" + "=" * 60)
    print("[Bonus] Testing WhisperX Import")
    print("=" * 60)

    try:
        import whisperx

        print("  ✅ WhisperX import: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ WhisperX import: FAILED")
        print(f"     Error: {e}")
        print(
            f"     Note: This is a known compatibility issue with Python 3.13 + torchaudio 2.10"
        )
        print(
            f"     The encoder and heads still work - just need to fix WhisperX separately."
        )
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_pipeline.py <audio_file>")
        print("\nExample:")
        print("  python scripts/validate_pipeline.py data/1769076152896.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("\n" + "#" * 60)
    print("# CEFR Assessment Pipeline Validation")
    print("#" * 60)
    print(f"\nAudio file: {audio_path}")

    try:
        # Test 1: Audio loading
        waveform = test_audio_loading(audio_path)

        # Test 2: Encoder
        hidden_states = test_encoder(waveform)

        # Test 3: CEFR heads
        outputs = test_cefr_heads(hidden_states)

        # Bonus: WhisperX (may fail)
        whisperx_ok = test_whisperx()

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print("  ✅ Audio loading: PASSED")
        print("  ✅ Semantic encoder (WavLM): PASSED")
        print("  ✅ CEFR classification heads: PASSED")
        if whisperx_ok:
            print("  ✅ WhisperX: PASSED")
        else:
            print("  ⚠️  WhisperX: FAILED (compatibility issue)")

        print("\n" + "=" * 60)
        if whisperx_ok:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("🎉 CORE PIPELINE WORKS!")
            print("   (WhisperX needs Python 3.12 or compatibility fix)")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
