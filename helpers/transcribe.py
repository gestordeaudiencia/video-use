"""Transcribe a video with local Whisper (openai-whisper).

Extracts mono 16kHz audio via ffmpeg, runs Whisper with word-level
timestamps, writes a Scribe-compatible JSON to <edit_dir>/transcripts/<stem>.json.

Fork of the original ElevenLabs Scribe transcriber. Diarization and audio
events (`(laughter)` etc.) are NOT produced — Whisper doesn't support them.
All words get `speaker_id: "0"` so downstream helpers (pack_transcripts.py,
render.py) keep working.

Whisper model size controlled by WHISPER_MODEL env var (default "base").
Options: tiny, base, small, medium, large-v2, large-v3. Quality vs speed
tradeoff — `large-v3` is best for multilingual/Portuguese, `base` for
fast iteration.

Cached: if the output file already exists, the transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language pt
    python helpers/transcribe.py <video_path> --model large-v3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_api_key() -> str:
    """Backwards-compat shim — Whisper is local, no key needed.

    Retained so `transcribe_batch.py` can import it without breaking.
    """
    return ""


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_whisper(
    audio_path: Path,
    language: str | None = None,
    model_size: str | None = None,
) -> dict:
    """Run openai-whisper locally and return a Scribe-compatible payload.

    Shape returned (matches what pack_transcripts.py and render.py expect):
        {
          "language_code": "pt",
          "text": "full transcript string",
          "words": [
            {"text": "hello", "type": "word", "start": 0.0, "end": 0.3, "speaker_id": "0"},
            ...
          ]
        }
    """
    # Import lazily — loading whisper is slow (~1s); defer until needed.
    import whisper  # type: ignore

    size = model_size or os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(size)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
    )

    words: list[dict] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "type": "word",
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "speaker_id": "0",
            })

    return {
        "language_code": result.get("language", "unknown"),
        "text": result.get("text", "").strip(),
        "words": words,
    }


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str = "",          # ignored; kept for compat with transcribe_batch.py
    language: str | None = None,
    num_speakers: int | None = None,  # ignored; Whisper doesn't diarize
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    del api_key, num_speakers  # unused — Whisper is local and mono-speaker

    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(
                f"  whisper transcribing {video.stem}.wav ({size_mb:.1f} MB)",
                flush=True,
            )
        payload = call_whisper(audio, language=language)

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with local Whisper")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'pt', 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model size (tiny/base/small/medium/large-v3). "
             "Overrides WHISPER_MODEL env var.",
    )
    # --num-speakers kept but ignored (Whisper doesn't diarize)
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Ignored — Whisper doesn't diarize. Kept for CLI compat.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    if args.model:
        os.environ["WHISPER_MODEL"] = args.model

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        language=args.language,
    )


if __name__ == "__main__":
    main()
