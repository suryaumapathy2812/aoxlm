#!/usr/bin/env python3
"""
AOXLM Training Data Generation Pipeline

Combines WhisperX (timestamps) + Gemini (corrections) to create training data.

Pipeline:
1. Audio -> WhisperX -> Verbatim transcription with word-level timestamps
2. Audio -> Gemini -> Verbatim + Corrected transcription pairs
3. Merge -> Aligned training data with timestamps + corrections

Usage:
    python scripts/data_gen/pipeline.py --audio path/to/audio.wav --output data/
    python scripts/data_gen/pipeline.py --audio-dir path/to/audios/ --output data/

Requirements:
    pip install whisperx google-genai
    export GEMINI_API_KEY="your-api-key"
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# Check dependencies
try:
    import whisperx

    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: whisperx not installed. Run: pip install whisperx")

try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Run: pip install google-genai")


@dataclass
class WordAlignment:
    """A word with timing and correction information."""

    word: str  # Corrected/clean word
    spoken: str  # What was actually said (verbatim)
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float  # Confidence score (0-1)
    word_type: Optional[str] = None  # "filler", "partial", "rep", None
    corrected: bool = False  # Was this word corrected?
    speaker: Optional[int] = None  # Speaker ID (if diarization)

    def to_jsonl(self) -> dict:
        """Convert to JSONL output format."""
        d = {
            "w": self.word,
            "s": round(self.start, 2),
            "e": round(self.end, 2),
            "c": round(self.confidence, 2),
        }
        if self.word_type:
            d["t"] = self.word_type
        if self.corrected and self.spoken != self.word:
            d["spoken"] = self.spoken
            d["cor"] = True
        if self.speaker is not None:
            d["spk"] = self.speaker
        return d


@dataclass
class TrainingSample:
    """A complete training sample."""

    audio_path: str
    words: List[WordAlignment]
    duration: float
    language: str = "en"

    def to_dict(self) -> dict:
        """Convert to training format."""
        return {
            "audio": self.audio_path,
            "duration": self.duration,
            "language": self.language,
            "words": [w.to_jsonl() for w in self.words],
            "verbatim": " ".join(w.spoken for w in self.words),
            "text": " ".join(w.word for w in self.words if w.word_type != "filler"),
        }


class WhisperXProcessor:
    """Get word-level timestamps using WhisperX."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        if not WHISPERX_AVAILABLE:
            raise ImportError("whisperx not installed")

        self.device = device
        self.compute_type = compute_type

        print(f"Loading WhisperX model: {model_size}")
        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        # Load alignment model (for word-level timestamps)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
        )

        print("WhisperX ready")

    def transcribe(self, audio_path: str) -> List[dict]:
        """
        Transcribe audio and get word-level timestamps.

        Returns list of words with timing:
        [{"word": "hello", "start": 0.0, "end": 0.45}, ...]
        """
        # Transcribe
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=16)

        # Align for word-level timestamps
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Extract words
        words = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                words.append(
                    {
                        "word": word_info["word"].strip(),
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "score": word_info.get("score", 0.9),
                    }
                )

        return words


class GeminiProcessor:
    """Get verbatim + corrected transcriptions using Gemini with structured output."""

    SYSTEM_PROMPT = """You are an expert speech transcription system for language learners.

Your task is to transcribe audio and provide BOTH:
1. **Verbatim**: Exactly what was spoken (mispronunciations, grammar errors, fillers)
2. **Corrected**: What the speaker intended/meant (fixed grammar, pronunciation, no fillers)

For each word, provide:
- `w`: The corrected word (what they meant)
- `spoken`: What was actually said (only if different from `w`)
- `cor`: true if the word was corrected
- `c`: Confidence score (0.0-1.0, lower if uncertain or corrected)
- `t`: Type marker for special words:
  - "filler": um, uh, like, you know, basically
  - "partial": incomplete/trailed off words
  - "rep": repeated words (stuttering)
  - "foreign": foreign word in otherwise English speech

Important:
- Do NOT include timestamps (we get those from WhisperX)
- Lower confidence for corrections and uncertain words
- Keep proper nouns/names as-is"""

    # Schema for structured output
    SCHEMA = types.Schema(
        type=types.Type.OBJECT,
        required=["words", "verbatim", "text", "language"],
        properties={
            "words": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["w", "c"],
                    properties={
                        "w": types.Schema(type=types.Type.STRING),
                        "c": types.Schema(type=types.Type.NUMBER),
                        "t": types.Schema(type=types.Type.STRING, nullable=True),
                        "spoken": types.Schema(type=types.Type.STRING, nullable=True),
                        "cor": types.Schema(type=types.Type.BOOLEAN, nullable=True),
                    },
                ),
            ),
            "verbatim": types.Schema(type=types.Type.STRING),
            "text": types.Schema(type=types.Type.STRING),
            "language": types.Schema(type=types.Type.STRING),
        },
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed")

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"Gemini ready ({model})")

    def transcribe(self, audio_path: str) -> dict:
        """
        Get verbatim + corrected transcription from Gemini.

        Returns dict with words array and full transcriptions.
        """
        audio_path = Path(audio_path)
        audio_bytes = audio_path.read_bytes()

        # Determine mime type
        ext = audio_path.suffix.lower()
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
        }
        mime_type = mime_types.get(ext, "audio/wav")

        # Build content
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
            system_instruction=self.SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=self.SCHEMA,
            temperature=0.1,
        )

        # Generate
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return json.loads(response.text)


class DataMerger:
    """Merge WhisperX timestamps with Gemini corrections."""

    # Common filler words
    FILLERS = {
        "um",
        "uh",
        "ah",
        "er",
        "like",
        "you know",
        "i mean",
        "so",
        "well",
        "basically",
    }

    def merge(
        self,
        whisperx_words: List[dict],
        gemini_result: dict,
    ) -> List[WordAlignment]:
        """
        Merge WhisperX timestamps with Gemini corrections.

        Strategy:
        1. Use WhisperX for timestamps (ground truth)
        2. Use Gemini for verbatim/corrected pairs
        3. Align based on word position and text matching
        """
        gemini_words = gemini_result.get("words", [])
        alignments = []

        gemini_idx = 0

        for wx_word in whisperx_words:
            wx_text = wx_word["word"].lower().strip(".,!?")

            # Find matching Gemini word
            correction = None
            spoken = None
            word_type = None
            confidence = wx_word.get("score", 0.9)

            if gemini_idx < len(gemini_words):
                gw = gemini_words[gemini_idx]
                gw_word = gw.get("w", "").lower().strip(".,!?")
                gw_spoken = (
                    gw.get("spoken", "").lower().strip(".,!?")
                    if gw.get("spoken")
                    else gw_word
                )

                # Check for match (either corrected or spoken version)
                if self._words_match(wx_text, gw_word) or self._words_match(
                    wx_text, gw_spoken
                ):
                    correction = gw.get("w")
                    spoken = gw.get("spoken")
                    word_type = gw.get("t")
                    confidence = gw.get("c", confidence)
                    gemini_idx += 1
                else:
                    # Try lookahead
                    for lookahead in range(1, 4):
                        if gemini_idx + lookahead < len(gemini_words):
                            gw = gemini_words[gemini_idx + lookahead]
                            gw_word = gw.get("w", "").lower().strip(".,!?")
                            gw_spoken = (
                                gw.get("spoken", "").lower().strip(".,!?")
                                if gw.get("spoken")
                                else gw_word
                            )

                            if self._words_match(wx_text, gw_word) or self._words_match(
                                wx_text, gw_spoken
                            ):
                                correction = gw.get("w")
                                spoken = gw.get("spoken")
                                word_type = gw.get("t")
                                confidence = gw.get("c", confidence)
                                gemini_idx += lookahead + 1
                                break

            # Check if filler (if not already marked)
            if word_type is None and wx_text in self.FILLERS:
                word_type = "filler"

            # Determine final word and spoken
            final_word = correction if correction else wx_word["word"]
            final_spoken = spoken if spoken else wx_word["word"]
            is_corrected = spoken is not None and spoken.lower() != final_word.lower()

            alignment = WordAlignment(
                word=final_word,
                spoken=final_spoken,
                start=wx_word["start"],
                end=wx_word["end"],
                confidence=confidence,
                word_type=word_type,
                corrected=is_corrected,
            )
            alignments.append(alignment)

        return alignments

    def _words_match(self, w1: str, w2: str) -> bool:
        """Check if two words match (fuzzy)."""
        w1 = w1.lower().strip(".,!?'\"")
        w2 = w2.lower().strip(".,!?'\"")

        if w1 == w2:
            return True

        # Handle contractions
        if w1.replace("'", "") == w2.replace("'", ""):
            return True

        # Handle partial matches
        if len(w1) > 2 and len(w2) > 2:
            if w1 in w2 or w2 in w1:
                return True

        return False


class DataGenerationPipeline:
    """Main pipeline for generating training data."""

    def __init__(
        self,
        whisperx_model: str = "large-v3",
        device: str = "cuda",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
    ):
        print("Initializing data generation pipeline...")

        # Initialize processors
        if WHISPERX_AVAILABLE:
            self.whisperx = WhisperXProcessor(
                model_size=whisperx_model,
                device=device,
            )
        else:
            self.whisperx = None
            print("WhisperX not available - will use Gemini timestamps (less accurate)")

        if GEMINI_AVAILABLE:
            self.gemini = GeminiProcessor(
                api_key=gemini_api_key,
                model=gemini_model,
            )
        else:
            self.gemini = None
            print("Gemini not available - will skip corrections")

        self.merger = DataMerger()

        print("Pipeline ready!")

    def process_audio(self, audio_path: str) -> TrainingSample:
        """
        Process a single audio file.

        Returns TrainingSample with merged timestamps + corrections.
        """
        audio_path = str(Path(audio_path).absolute())
        print(f"\nProcessing: {audio_path}")

        # Step 1: WhisperX for timestamps
        wx_words = []
        if self.whisperx:
            print("  Running WhisperX...")
            wx_words = self.whisperx.transcribe(audio_path)
            print(f"  WhisperX: {len(wx_words)} words")

        # Step 2: Gemini for corrections
        gm_result = {}
        if self.gemini:
            print("  Running Gemini...")
            gm_result = self.gemini.transcribe(audio_path)
            print(f"  Gemini: {len(gm_result.get('words', []))} words")

        # Step 3: Merge
        print("  Merging...")
        if wx_words and gm_result:
            alignments = self.merger.merge(wx_words, gm_result)
        elif wx_words:
            # WhisperX only - no corrections
            alignments = [
                WordAlignment(
                    word=w["word"],
                    spoken=w["word"],
                    start=w["start"],
                    end=w["end"],
                    confidence=w.get("score", 0.9),
                )
                for w in wx_words
            ]
        elif gm_result:
            # Gemini only - no accurate timestamps (use indices)
            alignments = []
            for i, w in enumerate(gm_result.get("words", [])):
                alignments.append(
                    WordAlignment(
                        word=w.get("w", ""),
                        spoken=w.get("spoken", w.get("w", "")),
                        start=i * 0.3,  # Fake timestamps
                        end=(i + 1) * 0.3,
                        confidence=w.get("c", 0.9),
                        word_type=w.get("t"),
                        corrected=w.get("cor", False),
                    )
                )
        else:
            alignments = []

        # Calculate duration
        duration = alignments[-1].end if alignments else 0.0
        language = gm_result.get("language", "en") if gm_result else "en"

        return TrainingSample(
            audio_path=audio_path,
            words=alignments,
            duration=duration,
            language=language,
        )

    def process_directory(
        self,
        audio_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a"),
    ) -> List[str]:
        """
        Process all audio files in a directory.

        Returns list of output file paths.
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"**/*{ext}"))

        print(f"Found {len(audio_files)} audio files")

        # Process each file
        output_files = []
        for i, audio_path in enumerate(audio_files):
            print(f"\n[{i + 1}/{len(audio_files)}]")

            try:
                sample = self.process_audio(str(audio_path))

                # Save
                output_path = output_dir / f"{audio_path.stem}.json"
                with open(output_path, "w") as f:
                    json.dump(sample.to_dict(), f, indent=2, ensure_ascii=False)

                output_files.append(str(output_path))
                print(f"  Saved: {output_path}")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Create manifest
        manifest_path = output_dir / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for path in output_files:
                f.write(json.dumps({"path": path}) + "\n")

        print(f"\nManifest: {manifest_path}")
        print(f"Total: {len(output_files)} samples")

        return output_files


def main():
    parser = argparse.ArgumentParser(description="Generate AOXLM training data")
    parser.add_argument("--audio", type=str, help="Single audio file to process")
    parser.add_argument("--audio-dir", type=str, help="Directory of audio files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--whisperx-model", type=str, default="large-v3", help="WhisperX model size"
    )
    parser.add_argument(
        "--gemini-model", type=str, default="gemini-2.0-flash", help="Gemini model"
    )

    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        parser.error("Must specify --audio or --audio-dir")

    # Initialize pipeline
    pipeline = DataGenerationPipeline(
        whisperx_model=args.whisperx_model,
        device=args.device,
        gemini_model=args.gemini_model,
    )

    # Process
    if args.audio:
        sample = pipeline.process_audio(args.audio)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{Path(args.audio).stem}.json"
        with open(output_path, "w") as f:
            json.dump(sample.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {output_path}")
        print(f"\nSample output:")
        print(json.dumps(sample.to_dict(), indent=2, ensure_ascii=False)[:2000])

    else:
        pipeline.process_directory(args.audio_dir, args.output)


if __name__ == "__main__":
    main()
