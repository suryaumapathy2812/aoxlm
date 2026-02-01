#!/usr/bin/env python3
"""
Gemini Transcription with Structured Output

Uses google-genai SDK with response_schema for guaranteed JSON structure.
Gets verbatim + corrected transcription pairs from Gemini.

Usage:
    pip install google-genai
    export GEMINI_API_KEY="your-api-key"
    python scripts/data_gen/gemini_transcribe.py audio.wav
    python scripts/data_gen/gemini_transcribe.py audio.wav --output result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai not installed")
    print("Run: pip install google-genai")
    sys.exit(1)


# Output schema for structured response
TRANSCRIPTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["words", "verbatim", "text", "language"],
    properties={
        "words": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["w", "s", "e", "c"],
                properties={
                    "w": types.Schema(
                        type=types.Type.STRING,
                        description="Word (cleaned/corrected version)",
                    ),
                    "s": types.Schema(
                        type=types.Type.NUMBER,
                        description="Start time in seconds",
                    ),
                    "e": types.Schema(
                        type=types.Type.NUMBER,
                        description="End time in seconds",
                    ),
                    "c": types.Schema(
                        type=types.Type.NUMBER,
                        description="Confidence score 0.0-1.0",
                    ),
                    "t": types.Schema(
                        type=types.Type.STRING,
                        description="Type: filler, partial, rep, foreign (optional)",
                        nullable=True,
                    ),
                    "spoken": types.Schema(
                        type=types.Type.STRING,
                        description="What was actually spoken if different from w (optional)",
                        nullable=True,
                    ),
                    "cor": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="True if word was corrected (optional)",
                        nullable=True,
                    ),
                },
            ),
        ),
        "verbatim": types.Schema(
            type=types.Type.STRING,
            description="Full verbatim transcription (exactly what was said)",
        ),
        "text": types.Schema(
            type=types.Type.STRING,
            description="Clean transcription (corrected, no fillers)",
        ),
        "language": types.Schema(
            type=types.Type.STRING,
            description="Primary language code (e.g., en, hi, ta)",
        ),
        "duration": types.Schema(
            type=types.Type.NUMBER,
            description="Audio duration in seconds",
            nullable=True,
        ),
    },
)


SYSTEM_PROMPT = """You are an expert speech transcription system for language learners.

Your task is to transcribe audio and provide BOTH:
1. **Verbatim**: Exactly what was spoken (mispronunciations, grammar errors, fillers)
2. **Corrected**: What the speaker intended/meant (fixed grammar, pronunciation, no fillers)

For each word, provide:
- `w`: The corrected word (what they meant)
- `s`: Start time in seconds
- `e`: End time in seconds  
- `c`: Confidence score (0.0-1.0, lower if uncertain or corrected)
- `spoken`: What was actually said (only if different from `w`)
- `cor`: true if the word was corrected
- `t`: Type marker for special words:
  - "filler": um, uh, like, you know, basically
  - "partial": incomplete/trailed off words
  - "rep": repeated words (stuttering)
  - "foreign": foreign word in otherwise English speech

Examples:
- Mispronunciation: {"w": "library", "spoken": "libary", "cor": true, "c": 0.95}
- Grammar error: {"w": "goes", "spoken": "go", "cor": true, "c": 0.92}
- Filler: {"w": "um", "t": "filler", "c": 0.94}
- Repetition: {"w": "more", "t": "rep", "c": 0.91} (for "more more")
- Contraction: {"w": "want to", "spoken": "wanna", "cor": true, "c": 0.97}

Important:
- Estimate timestamps based on speech rhythm (approximate is fine)
- Lower confidence for corrections and uncertain words
- Keep proper nouns/names as-is (don't "correct" names)
- For fillers, `w` is the filler word itself, no `spoken` needed"""


def transcribe(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    Transcribe audio using Gemini with structured output.

    Args:
        audio_path: Path to audio file
        api_key: Gemini API key (or use GEMINI_API_KEY env var)
        model: Model to use (gemini-2.0-flash, gemini-2.5-pro, etc.)

    Returns:
        Structured transcription dict
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Export it or pass --api-key")

    client = genai.Client(api_key=api_key)

    # Read audio file
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Reading: {audio_path}")
    audio_bytes = audio_path.read_bytes()

    # Determine mime type
    ext = audio_path.suffix.lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }
    mime_type = mime_types.get(ext, "audio/wav")

    # Build content with audio
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(
                    text="Transcribe this audio following the system instructions."
                ),
            ],
        ),
    ]

    # Configure generation
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=TRANSCRIPTION_SCHEMA,
        temperature=0.1,  # Low temperature for consistent output
    )

    print(f"Transcribing with {model}...")

    # Generate (non-streaming for structured output)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # Parse response
    result = json.loads(response.text)

    return result


def transcribe_streaming(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash",
):
    """
    Transcribe with streaming output (for progress indication).
    Note: Structured output may not work well with streaming.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    audio_path = Path(audio_path)
    audio_bytes = audio_path.read_bytes()

    ext = audio_path.suffix.lower()
    mime_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
    mime_type = mime_types.get(ext, "audio/wav")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(
                    text="Transcribe this audio following the system instructions."
                ),
            ],
        ),
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=TRANSCRIPTION_SCHEMA,
    )

    print(f"Streaming transcription with {model}...")

    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            full_response += chunk.text
            print(".", end="", flush=True)

    print()  # newline

    return json.loads(full_response)


def print_result(result: dict, verbose: bool = False):
    """Pretty print transcription result."""
    print("\n" + "=" * 70)
    print("TRANSCRIPTION RESULT")
    print("=" * 70)

    print(f"\nLanguage: {result.get('language', 'unknown')}")
    if result.get("duration"):
        print(f"Duration: {result['duration']:.1f}s")

    print(f"\nVerbatim:")
    print(f"  {result.get('verbatim', 'N/A')}")

    print(f"\nCorrected:")
    print(f"  {result.get('text', 'N/A')}")

    words = result.get("words", [])
    print(f"\nWords ({len(words)}):")
    print("-" * 70)

    corrections = []
    fillers = []

    for w in words:
        word = w.get("w", "")
        spoken = w.get("spoken")
        word_type = w.get("t")
        start = w.get("s", 0)
        end = w.get("e", 0)
        conf = w.get("c", 1.0)

        # Track corrections and fillers
        if spoken and spoken != word:
            corrections.append((spoken, word))
        if word_type == "filler":
            fillers.append(word)

        if verbose:
            line = f"  [{start:5.2f}-{end:5.2f}] "
            if spoken and spoken != word:
                line += f"'{spoken}' -> '{word}'"
            elif word_type:
                line += f"'{word}' [{word_type}]"
            else:
                line += f"'{word}'"
            line += f" (conf: {conf:.2f})"
            print(line)

    if not verbose and words:
        # Show summary
        print(f"  (use --verbose to see all {len(words)} words)")

    # Show corrections summary
    if corrections:
        print(f"\nCorrections ({len(corrections)}):")
        for spoken, corrected in corrections[:10]:
            print(f"  '{spoken}' -> '{corrected}'")
        if len(corrections) > 10:
            print(f"  ... and {len(corrections) - 10} more")

    if fillers:
        print(f"\nFillers ({len(fillers)}): {', '.join(fillers[:10])}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Gemini (structured output)"
    )
    parser.add_argument("audio", type=str, help="Audio file path")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model (gemini-2.0-flash, gemini-2.5-pro, etc.)",
    )
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all words with timestamps"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming (shows progress)"
    )

    args = parser.parse_args()

    try:
        if args.stream:
            result = transcribe_streaming(args.audio, args.api_key, args.model)
        else:
            result = transcribe(args.audio, args.api_key, args.model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")

    # Print result
    print_result(result, verbose=args.verbose)

    # Also print raw JSON if verbose
    if args.verbose:
        print("\n" + "=" * 70)
        print("RAW JSON")
        print("=" * 70)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
