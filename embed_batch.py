"""
embed_batch.py — Bulk audio-to-embedding importer

Walks a directory of labeled audio samples, generates ECAPA-TDNN embeddings,
and stores them in Supabase (via VazamDB).

Expected directory layout
--------------------------
    samples/
    └── <Actor Name>/
        ├── natural/          ← natural speaking voice (interviews, panels)
        │   ├── clip_01.wav
        │   └── clip_02.mp3
        └── <Character Name>/ ← one sub-dir per character voice
            ├── ep01_scene3.wav
            └── ep07_scene1.wav

Any WAV or MP3 file found under <Actor Name>/<Voice Label>/ is processed.
The actor is created (or upserted) in the DB; the voice label comes from
the subdirectory name.

Usage
-----
    # Dry run — show what would be imported without writing to DB
    python embed_batch.py samples/ --dry-run

    # Import all samples; isolate vocals with Demucs first
    python embed_batch.py samples/ --isolate

    # Import only verified-quality samples into the verified bucket
    python embed_batch.py samples/ --verified

Environment variables
---------------------
    SUPABASE_URL   https://<project-ref>.supabase.co
    SUPABASE_KEY   <service_role_key>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _iter_samples(root: Path):
    """Yield (actor_name, voice_label, audio_path) triples from a samples tree."""
    for actor_dir in sorted(root.iterdir()):
        if not actor_dir.is_dir() or actor_dir.name.startswith("."):
            continue
        actor_name = actor_dir.name

        for label_dir in sorted(actor_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name.startswith("."):
                continue
            # Normalise the voice label: "natural" → "Natural Voice"
            raw_label = label_dir.name
            voice_label = (
                "Natural Voice"
                if raw_label.lower() in ("natural", "natural_voice", "interview", "self")
                else raw_label.replace("_", " ").title()
            )

            for audio_file in sorted(label_dir.iterdir()):
                if audio_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield actor_name, voice_label, audio_file


def run_batch(
    root: Path,
    isolate: bool = False,
    verified: bool = False,
    dry_run: bool = False,
) -> None:
    samples = list(_iter_samples(root))

    if not samples:
        print(f"No audio files found under {root}. Check the directory structure.")
        sys.exit(1)

    print(f"Found {len(samples)} audio file(s) across {root}\n")

    if dry_run:
        print("DRY RUN — no DB writes\n")
        for actor_name, voice_label, path in samples:
            print(f"  [{actor_name}] [{voice_label}] {path.name}")
        return

    # Late imports so the script can do --dry-run without loading PyTorch
    from db import VazamDB
    from pipeline import VazamPipeline, isolate_vocals

    db       = VazamDB()
    pipeline = VazamPipeline(db=db, hf_token="", use_vad=False, use_diarization=False)

    stats = {"ok": 0, "skipped": 0, "error": 0}
    actor_cache: dict[str, int] = {}   # actor_name → db id

    for i, (actor_name, voice_label, audio_path) in enumerate(samples, 1):
        label = f"[{i}/{len(samples)}] {actor_name} / {voice_label} / {audio_path.name}"
        print(label)

        try:
            # Resolve actor ID (create if new)
            if actor_name not in actor_cache:
                actor_id = db.add_actor(actor_name)
                actor_cache[actor_name] = actor_id
            actor_id = actor_cache[actor_name]

            # Generate embedding
            path_str = str(audio_path)
            if isolate:
                path_str = isolate_vocals(path_str)

            embedding = pipeline.embed_file(path_str, isolate=False)

            # Store in Supabase
            db.add_embedding(
                actor_id=actor_id,
                embedding=embedding,
                voice_label=voice_label,
                audio_source=str(audio_path.relative_to(root)),
                verified=verified,
            )
            print(f"           ✓ embedded (dim={embedding.shape[0]}, "
                  f"norm={float(np.linalg.norm(embedding)):.4f})")
            stats["ok"] += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:
            print(f"           ✗ error: {exc}")
            stats["error"] += 1

    db.close()

    print(f"\n── Summary ───────────────────────────────")
    print(f"  Processed : {stats['ok']}")
    print(f"  Errors    : {stats['error']}")
    print(f"  Skipped   : {stats['skipped']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-generate speaker embeddings from labeled audio samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing <Actor>/<VoiceLabel>/<audio> structure",
    )
    parser.add_argument("--isolate", action="store_true",
                        help="Run Demucs vocal isolation on each file before embedding")
    parser.add_argument("--verified", action="store_true",
                        help="Mark imported embeddings as verified (higher trust)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be imported without writing to DB")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} does not exist.")
        sys.exit(1)

    run_batch(
        root=args.root,
        isolate=args.isolate,
        verified=args.verified,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
