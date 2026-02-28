"""
scrape_audio.py — Automated voice sample collector using yt-dlp

Queries Supabase for voice actors that have no embeddings yet, then searches
YouTube for each actor's interviews / demo reels, downloads audio, and
generates + stores embeddings automatically.

Usage
-----
    # Process all actors with zero embeddings (default limit: 50)
    python scrape_audio.py

    # Single actor by name
    python scrape_audio.py --actor "Steve Blum"

    # Cap how many actors to process in one run
    python scrape_audio.py --limit 20

    # Preview what would be downloaded without fetching anything
    python scrape_audio.py --dry-run

    # Also scrape per-character clips (slower, more accurate)
    python scrape_audio.py --include-characters

Environment variables
---------------------
    SUPABASE_URL   https://<project-ref>.supabase.co
    SUPABASE_KEY   <service_role_key>
    HF_TOKEN       HuggingFace token (optional — enables VAD trimming)
    DEVICE         cuda or cpu (auto-detected if unset)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_LIMIT = 50
SCRAPE_DELAY  = 2.5   # seconds between downloads (be polite)
MAX_DURATION  = 600   # skip videos longer than 10 minutes


# ── YouTube download ──────────────────────────────────────────────────────────

def _search_and_download(query: str, output_path: str) -> Optional[str]:
    """Search YouTube for `query`, download the first result as MP3.

    Returns the path to the downloaded file, or None on failure.
    Uses yt-dlp's ytsearch1: prefix which returns a single result.
    """
    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",          # ~128 kbps — enough for speaker ID
        "--match-filter", f"duration < {MAX_DURATION}",
        "--output", output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and Path(output_path).exists():
            return output_path
        # yt-dlp appends the extension automatically — check for .mp3
        mp3_path = output_path if output_path.endswith(".mp3") else output_path + ".mp3"
        if Path(mp3_path).exists():
            return mp3_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _build_query(actor_name: str, character_name: Optional[str] = None, show_title: Optional[str] = None) -> str:
    """Build a YouTube search query for a voice actor."""
    if character_name and show_title:
        return f'"{character_name}" "{show_title}" english dub voice clip'
    return f'"{actor_name}" voice actor interview'


# ── Core scraping logic ────────────────────────────────────────────────────────

def scrape_actors(
    actor_names: Optional[list[str]] = None,
    limit: int = DEFAULT_LIMIT,
    include_characters: bool = False,
    dry_run: bool = False,
) -> None:
    """Main entry point: fetch actors without embeddings, download + embed audio."""
    from db import VazamDB
    from pipeline import VazamPipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    device   = os.environ.get("DEVICE", "") or None

    db = VazamDB()
    pipeline = VazamPipeline(
        db=db,
        hf_token=hf_token,
        device=device,
        use_vad=bool(hf_token),
        use_diarization=False,  # not needed for embedding collection
    )

    # Determine which actors to process
    if actor_names:
        # Look up by name
        all_actors = db.list_actors(limit=10000)
        targets = [
            a for a in all_actors
            if any(name.lower() in a["name"].lower() for name in actor_names)
        ]
        if not targets:
            print(f"No actors found matching: {actor_names}")
            return
    else:
        # Find actors with zero embeddings
        all_actors = db.list_actors(limit=10000)
        targets = []
        for actor in all_actors:
            if len(targets) >= limit:
                break
            # Check if this actor has any embeddings
            count_result = (
                db._client.table("vazam_embeddings")
                .select("id", count="exact")
                .eq("actor_id", actor["id"])
                .execute()
            )
            if (count_result.count or 0) == 0:
                targets.append(actor)

    print(f"Found {len(targets)} actor(s) to process\n")
    if dry_run:
        print("DRY RUN — showing queries only\n")

    stats = {"ok": 0, "skipped": 0, "error": 0}

    for i, actor in enumerate(targets, 1):
        actor_id   = actor["id"]
        actor_name = actor["name"]

        print(f"[{i}/{len(targets)}] {actor_name}")

        jobs: list[tuple[str, Optional[str], Optional[str], str]] = []

        # Natural voice: interview / panel clip
        jobs.append((
            _build_query(actor_name),
            None,
            None,
            "Natural Voice",
        ))

        # Per-character clips (optional)
        if include_characters:
            filmography = db.get_actor_filmography(actor_id)
            for role in filmography[:3]:  # cap at 3 characters per actor
                char  = role["character_name"]
                show  = role["show_title"]
                label = char or "Natural Voice"
                jobs.append((_build_query(actor_name, char, show), char, show, label))

        for query, char_name, show_title, voice_label in jobs:
            print(f"  Query : {query}")
            if dry_run:
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                out_tmpl = os.path.join(tmpdir, "audio.%(ext)s")
                downloaded = _search_and_download(query, out_tmpl)

                if not downloaded:
                    # yt-dlp writes <audio.mp3> even when template is <audio.%(ext)s>
                    candidate = os.path.join(tmpdir, "audio.mp3")
                    downloaded = candidate if Path(candidate).exists() else None

                if not downloaded:
                    print(f"  ✗ download failed")
                    stats["skipped"] += 1
                    continue

                try:
                    embedding = pipeline.embed_file(downloaded, isolate=False)
                    emb_id = db.add_embedding(
                        actor_id=actor_id,
                        embedding=embedding,
                        voice_label=voice_label,
                        audio_source=query,
                        verified=False,
                    )
                    import numpy as np
                    print(f"  ✓ stored embedding #{emb_id} "
                          f"(dim={embedding.shape[0]}, norm={float(np.linalg.norm(embedding)):.4f})")
                    stats["ok"] += 1
                except Exception as exc:
                    print(f"  ✗ embedding error: {exc}")
                    stats["error"] += 1

        if not dry_run and i < len(targets):
            time.sleep(SCRAPE_DELAY)

    db.close()

    if not dry_run:
        print(f"\n── Summary ───────────────────────────────")
        print(f"  Embedded  : {stats['ok']}")
        print(f"  Skipped   : {stats['skipped']}")
        print(f"  Errors    : {stats['error']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-download YouTube voice clips and generate embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--actor", type=str, default="", metavar="NAME",
        help="Process a specific actor by name (partial match, can be repeated)",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Max number of actors to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--include-characters", action="store_true",
        help="Also scrape per-character clips in addition to the natural voice interview",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print search queries without downloading or storing anything",
    )
    args = parser.parse_args()

    actor_names = [args.actor] if args.actor else None

    scrape_actors(
        actor_names=actor_names,
        limit=args.limit,
        include_characters=args.include_characters,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
