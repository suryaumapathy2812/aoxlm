#!/usr/bin/env python3
"""
Explore Qwen2-Audio architecture and capabilities.

This script helps us understand:
1. How to load and use Qwen2-Audio
2. What outputs are available (hidden states, attention, etc.)
3. Where to tap in for alignment module
4. Test instruction-following capabilities

Usage:
    python scripts/explore_qwen2_audio.py --audio path/to/audio.wav
    python scripts/explore_qwen2_audio.py --audio path/to/audio.wav --model Qwen/Qwen2-Audio-7B-Instruct
"""

import argparse
import torch
import soundfile as sf
from pathlib import Path


def load_model(model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"):
    """Load Qwen2-Audio model and processor."""
    print(f"Loading model: {model_name}")

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model loaded on: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, processor


def explore_architecture(model):
    """Print model architecture to understand structure."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    # Print top-level modules
    print("\nTop-level modules:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")

    # Look for audio encoder
    print("\nLooking for audio encoder...")
    if hasattr(model, "audio_encoder"):
        print(f"  Found: model.audio_encoder")
        print(f"  Type: {type(model.audio_encoder).__name__}")
    elif hasattr(model, "audio_tower"):
        print(f"  Found: model.audio_tower")
        print(f"  Type: {type(model.audio_tower).__name__}")
    else:
        print("  Not found at top level, searching deeper...")
        for name, module in model.named_modules():
            if "audio" in name.lower() or "whisper" in name.lower():
                print(f"  Found: {name} ({type(module).__name__})")

    # Look for projection layer
    print("\nLooking for projection/adapter layer...")
    for name, module in model.named_modules():
        if "proj" in name.lower() or "adapter" in name.lower():
            print(f"  Found: {name} ({type(module).__name__})")

    # Print model config
    print("\nModel config:")
    config = model.config
    for key in ["hidden_size", "num_hidden_layers", "vocab_size", "audio_token_index"]:
        if hasattr(config, key):
            print(f"  {key}: {getattr(config, key)}")


def test_basic_transcription(model, processor, audio_path: str):
    """Test basic transcription capability."""
    print("\n" + "=" * 60)
    print("BASIC TRANSCRIPTION TEST")
    print("=" * 60)

    # Load audio
    audio, sr = sf.read(audio_path)
    print(f"Audio: {len(audio) / sr:.2f}s at {sr}Hz")

    # Prepare input
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = [audio]

    inputs = processor(
        text=text, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    print("\nGenerating transcription...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nResponse:\n{response}")

    return response


def test_with_hidden_states(model, processor, audio_path: str):
    """Test generation with hidden states output."""
    print("\n" + "=" * 60)
    print("HIDDEN STATES EXTRACTION TEST")
    print("=" * 60)

    # Load audio
    audio, sr = sf.read(audio_path)

    # Prepare input
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = [audio]

    inputs = processor(
        text=text, audios=audios, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate with hidden states
    print("\nGenerating with output_hidden_states=True...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    print(f"\nOutput keys: {outputs.keys()}")

    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        print(f"Number of generation steps: {len(outputs.hidden_states)}")
        print(f"Hidden states per step: {len(outputs.hidden_states[0])} layers")
        print(f"First layer shape: {outputs.hidden_states[0][0].shape}")

    if hasattr(outputs, "scores") and outputs.scores:
        print(f"Number of score tensors: {len(outputs.scores)}")
        print(f"Score shape: {outputs.scores[0].shape}")

    # Decode
    response = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"\nGenerated text:\n{response}")

    return outputs


def test_instruction_following(model, processor, audio_path: str):
    """Test various instructions."""
    print("\n" + "=" * 60)
    print("INSTRUCTION FOLLOWING TEST")
    print("=" * 60)

    # Load audio
    audio, sr = sf.read(audio_path)

    instructions = [
        "Transcribe this audio.",
        "Transcribe this audio in Hindi.",
        "What is the speaker talking about?",
        "Summarize this audio in one sentence.",
        "What emotion is the speaker expressing?",
    ]

    for instruction in instructions:
        print(f"\n--- Instruction: {instruction} ---")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=text,
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        print(f"Response: {response[:200]}...")


def explore_audio_embeddings(model, processor, audio_path: str):
    """Try to extract audio embeddings for alignment."""
    print("\n" + "=" * 60)
    print("AUDIO EMBEDDINGS EXPLORATION")
    print("=" * 60)

    # Load audio
    audio, sr = sf.read(audio_path)

    # Try to find and use audio encoder directly
    print("\nAttempting to extract audio embeddings...")

    # Method 1: Check for audio_tower (common in multimodal models)
    if hasattr(model, "audio_tower"):
        print("Found audio_tower, attempting extraction...")
        # Would need to process audio through feature extractor first

    # Method 2: Check model internals during forward pass
    print("\nChecking model structure for audio processing...")

    # List all modules with their shapes
    for name, param in model.named_parameters():
        if "audio" in name.lower() or "whisper" in name.lower():
            print(f"  {name}: {param.shape}")


def main():
    parser = argparse.ArgumentParser(description="Explore Qwen2-Audio architecture")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "arch", "transcribe", "hidden", "instruct", "embed"],
        help="Which test to run",
    )

    args = parser.parse_args()

    # Check audio file exists
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return

    # Load model
    model, processor = load_model(args.model)

    # Run tests
    if args.test in ["all", "arch"]:
        explore_architecture(model)

    if args.test in ["all", "transcribe"]:
        test_basic_transcription(model, processor, args.audio)

    if args.test in ["all", "hidden"]:
        test_with_hidden_states(model, processor, args.audio)

    if args.test in ["all", "instruct"]:
        test_instruction_following(model, processor, args.audio)

    if args.test in ["all", "embed"]:
        explore_audio_embeddings(model, processor, args.audio)

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
