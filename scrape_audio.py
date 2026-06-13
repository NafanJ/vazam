"""
scrape_audio.py — Consensus voice sample collector using yt-dlp

Implements the cross-video consensus strategy from docs/data-acquisition-plan.md.
For each actor with no embeddings:

  1. Search YouTube with several independent query templates (interview /
     panel / podcast) and pick 2–3 *distinct* videos.
  2. Download only the first few minutes of audio from each.
  3. Diarize each video and embed every speaker found (with HF_TOKEN);
     without a token, fall back to one whole-clip embedding per video.
  4. The actor is the voice cluster that recurs across videos — interviewers
     and co-panelists differ per video. Store the consensus centroid as the
     "Natural Voice" embedding with a quality score; if no cluster recurs,
     store nothing (never fall back to a blind, unvalidated embedding).
  5. Re-embed the winning voice under simulated query conditions (reverb,
     noise, TV-speaker band-limiting — see augment.py) and fold those into
     the stored centroid, so references match the phone-mic-of-a-TV channel
     they will be searched against. Disable with --no-augment.

Per the acquisition plan, Demucs is intentionally skipped here (interviews
have no music bed) and per-character clip scraping is no longer done by this
script — character embeddings are added lazily via show-aware search instead.

Usage
-----
    python scrape_audio.py                      # all actors with zero embeddings
    python scrape_audio.py --actor "Steve Blum" # single actor by name
    python scrape_audio.py --limit 20           # cap actors per run
    python scrape_audio.py --force              # re-scrape even if embeddings exist
    python scrape_audio.py --dry-run            # print queries only, no network

Environment variables
---------------------
    SUPABASE_URL   https://<project-ref>.supabase.co
    SUPABASE_KEY   <service_role_key>
    HF_TOKEN       HuggingFace token — strongly recommended; enables per-speaker
                   diarization, which makes consensus far more reliable
    DEVICE         cuda or cpu (auto-detected if unset)

Requires the migrations/001_embedding_quality.sql migration to be applied.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from consensus import VideoSpeaker, build_consensus

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_LIMIT    = 50
# Invoke yt-dlp as a module via the current interpreter, not the bare "yt-dlp"
# console script — that depends on the venv's bin being on PATH, which it isn't
# when the script is run as `.venv/bin/python scrape_audio.py`.
YT_DLP = [sys.executable, "-m", "yt_dlp"]

SCRAPE_DELAY     = 2.5    # seconds between downloads (be polite)
VIDEOS_PER_ACTOR = 3      # independent videos to seek consensus across
SEARCH_RESULTS   = 5      # candidates fetched per search query

MIN_VIDEO_SECONDS   = 120   # too short → likely a clip compilation / short
MAX_VIDEO_SECONDS   = 3600  # too long → skip
MAX_PROCESS_SECONDS = 600   # only download/process the first 10 minutes

MIN_SPEAKER_SECONDS = 8.0   # diarized speakers with less speech are ignored
MAX_EMBED_SECONDS   = 60.0  # cap audio per speaker embedding (compute bound)

QUERY_TEMPLATES = [
    '"{name}" voice actor interview',
    '"{name}" voice actor convention panel',
    '"{name}" voice actor podcast',
]


@dataclass
class VideoCandidate:
    video_id: str
    duration: float
    title: str

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


# ── YouTube search & download ─────────────────────────────────────────────────

def parse_candidates(yt_dlp_output: str) -> list[VideoCandidate]:
    """Parse `yt-dlp --print "%(id)s\\t%(duration)s\\t%(title)s"` output."""
    candidates = []
    for line in yt_dlp_output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        vid, duration_str, title = parts
        try:
            duration = float(duration_str)
        except ValueError:
            continue  # live streams / premieres report "NA"
        candidates.append(VideoCandidate(vid.strip(), duration, title.strip()))
    return candidates


def select_videos(
    candidates_per_query: list[list[VideoCandidate]],
    n_videos: int = VIDEOS_PER_ACTOR,
    min_seconds: float = MIN_VIDEO_SECONDS,
    max_seconds: float = MAX_VIDEO_SECONDS,
) -> list[VideoCandidate]:
    """Pick up to n_videos distinct, duration-acceptable videos.

    Round-robins across queries so the chosen videos come from independent
    searches where possible — independence is what makes consensus work.
    """
    selected: list[VideoCandidate] = []
    seen_ids: set[str] = set()
    queues = [list(c) for c in candidates_per_query]

    while len(selected) < n_videos and any(queues):
        for queue in queues:
            while queue:
                cand = queue.pop(0)
                if cand.video_id in seen_ids:
                    continue
                if not (min_seconds <= cand.duration <= max_seconds):
                    continue
                selected.append(cand)
                seen_ids.add(cand.video_id)
                break
            if len(selected) >= n_videos:
                break

    return selected


def _search_candidates(query: str) -> list[VideoCandidate]:
    """Search YouTube and return candidate metadata without downloading."""
    cmd = [
        *YT_DLP,
        f"ytsearch{SEARCH_RESULTS}:{query}",
        "--skip-download",
        "--no-playlist",
        "--print", "%(id)s\t%(duration)s\t%(title)s",
        "--quiet",
        "--no-warnings",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("  ✗ yt-dlp not found — install it with: pip install yt-dlp", file=sys.stderr)
        return []
    except subprocess.TimeoutExpired:
        return []
    return parse_candidates(proc.stdout)


def _download_audio(candidate: VideoCandidate, output_dir: str) -> Optional[str]:
    """Download the first MAX_PROCESS_SECONDS of a video as MP3."""
    out_path = os.path.join(output_dir, f"{candidate.video_id}.mp3")
    cmd = [
        *YT_DLP,
        candidate.url,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",          # ~128 kbps — enough for speaker ID
        "--download-sections", f"*0-{MAX_PROCESS_SECONDS}",
        "--output", os.path.join(output_dir, f"{candidate.video_id}.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out_path if Path(out_path).exists() else None


# ── Per-video speaker extraction ──────────────────────────────────────────────

def _load_mono_16k(audio_path: str):
    """Load audio as a (1, N) mono float tensor at 16 kHz."""
    import torchaudio

    signal, fs = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
    return signal


def collect_video_speakers(
    audio_path: str,
    video_id: str,
    hf_token: str,
    device: str,
) -> tuple[list[VideoSpeaker], dict[tuple[str, str], "torch.Tensor"]]:
    """Embed every speaker found in one video.

    With HF_TOKEN: diarize, then embed each speaker's concatenated speech
    (capped at MAX_EMBED_SECONDS). Without a token: one whole-clip embedding
    labelled "FULL" — consensus still works but is contaminated by whoever
    else talks in the clip, so diarization is strongly preferred.

    Also returns each speaker's speech tensor keyed by (video_id, label) so
    the consensus winner can be re-embedded under channel augmentation.
    """
    import torch

    from pipeline import diarize, get_embedding

    speakers: list[VideoSpeaker] = []
    speech_by_key: dict[tuple[str, str], torch.Tensor] = {}

    if hf_token:
        segments = diarize(audio_path, hf_token)
        if not segments:
            return [], {}

        signal = _load_mono_16k(audio_path)

        by_label: dict[str, list] = {}
        for seg in segments:
            by_label.setdefault(seg.speaker_label, []).append(seg)

        for label, segs in by_label.items():
            total = sum(s.duration for s in segs)
            if total < MIN_SPEAKER_SECONDS:
                continue
            chunks = [
                signal[:, int(s.start * 16000): int(s.end * 16000)]
                for s in segs
            ]
            speech = torch.cat(chunks, dim=1)[:, : int(MAX_EMBED_SECONDS * 16000)]
            emb = get_embedding(speech, device=device)
            speakers.append(VideoSpeaker(
                video_id=video_id,
                speaker_label=label,
                embedding=emb,
                speech_seconds=min(total, MAX_EMBED_SECONDS),
            ))
            speech_by_key[(video_id, label)] = speech
    else:
        signal = _load_mono_16k(audio_path)[:, : int(MAX_EMBED_SECONDS * 16000)]
        seconds = signal.shape[1] / 16000
        if seconds < MIN_SPEAKER_SECONDS:
            return [], {}
        emb = get_embedding(signal, device=device)
        speakers.append(VideoSpeaker(
            video_id=video_id,
            speaker_label="FULL",
            embedding=emb,
            speech_seconds=seconds,
        ))
        speech_by_key[(video_id, "FULL")] = signal

    return speakers, speech_by_key


def augmented_embedding(result, speech_by_key, device: str):
    """Fold channel-augmented embeddings of the winning cluster into its centroid.

    Each member's speech is re-embedded under simulated query conditions
    (reverb, noise, TV-speaker band-limiting — see augment.py), pulling the
    stored reference toward the phone-mic-of-a-TV channel it will actually be
    matched against. Only the consensus winner is augmented; the quality
    score stays computed on clean embeddings.
    """
    from augment import augment_speech
    from consensus import centroid
    from pipeline import get_embedding

    embeddings = [s.embedding for s in result.members]
    for member in result.members:
        speech = speech_by_key.get((member.video_id, member.speaker_label))
        if speech is None:
            continue
        for variant in augment_speech(speech):
            embeddings.append(get_embedding(variant, device=device))

    return centroid(embeddings)


# ── Core scraping logic ───────────────────────────────────────────────────────

def scrape_actor(
    db,
    actor_id: int,
    actor_name: str,
    hf_token: str,
    device: str,
    dry_run: bool = False,
    augment: bool = True,
) -> str:
    """Run the full consensus pipeline for one actor.

    Returns one of: "ok", "no_videos", "no_consensus", "dry_run".
    """
    queries = [t.format(name=actor_name) for t in QUERY_TEMPLATES]
    for q in queries:
        print(f"  Query : {q}")
    if dry_run:
        return "dry_run"

    candidates_per_query = [_search_candidates(q) for q in queries]
    videos = select_videos(candidates_per_query)

    if len(videos) < 2:
        print(f"  ✗ only {len(videos)} usable video(s) found — need ≥ 2 for consensus")
        return "no_videos"

    all_speakers: list[VideoSpeaker] = []
    speech_by_key: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for cand in videos:
            print(f"  ↓ {cand.title[:70]}  ({cand.duration:.0f}s)")
            audio_path = _download_audio(cand, tmpdir)
            if not audio_path:
                print("    ✗ download failed")
                continue
            try:
                found, tensors = collect_video_speakers(
                    audio_path, cand.video_id, hf_token, device,
                )
                print(f"    {len(found)} speaker(s) embedded")
                all_speakers.extend(found)
                speech_by_key.update(tensors)
            except Exception as exc:
                print(f"    ✗ embedding error: {exc}")
            time.sleep(SCRAPE_DELAY)

    result = build_consensus(all_speakers)
    if result is None:
        print("  ✗ no voice recurred across videos — skipping (nothing stored)")
        return "no_consensus"

    embedding = result.embedding
    if augment:
        embedding = augmented_embedding(result, speech_by_key, device)
        print("  + channel augmentation folded into centroid")

    id_to_url = {c.video_id: c.url for c in videos}
    source_urls = ",".join(id_to_url[v] for v in result.video_ids)

    emb_id = db.add_embedding(
        actor_id=actor_id,
        embedding=embedding,
        voice_label="Natural Voice",
        audio_source="consensus_scrape",
        verified=False,
        source_url=source_urls,
        duration_s=result.speech_seconds,
        quality_score=result.quality_score,
    )
    print(f"  ✓ stored embedding #{emb_id} "
          f"(quality={result.quality_score:.3f}, "
          f"speech={result.speech_seconds:.0f}s, "
          f"videos={result.n_videos})")
    return "ok"


def scrape_actors(
    actor_names: Optional[list[str]] = None,
    limit: int = DEFAULT_LIMIT,
    force: bool = False,
    dry_run: bool = False,
    augment: bool = True,
) -> None:
    """Main entry point: fetch target actors, run consensus scrape for each."""
    from db import VazamDB

    hf_token = os.environ.get("HF_TOKEN", "")
    device   = os.environ.get("DEVICE", "") or None

    if not hf_token:
        print("⚠ HF_TOKEN not set — falling back to whole-clip embeddings.\n"
              "  Consensus is much more reliable with diarization enabled.\n")

    db = VazamDB()
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine which actors to process
    if actor_names:
        all_actors = db.list_actors(limit=10000)
        targets = [
            a for a in all_actors
            if any(name.lower() in a["name"].lower() for name in actor_names)
        ]
        if not targets:
            print(f"No actors found matching: {actor_names}")
            return
        targets = targets[:limit]
    else:
        all_actors = db.list_actors(limit=10000)
        targets = []
        for actor in all_actors:
            if len(targets) >= limit:
                break
            if not force:
                count_result = (
                    db._client.table("vazam_embeddings")
                    .select("id", count="exact")
                    .eq("actor_id", actor["id"])
                    .execute()
                )
                if (count_result.count or 0) > 0:
                    continue
            targets.append(actor)

    print(f"Found {len(targets)} actor(s) to process\n")
    if dry_run:
        print("DRY RUN — showing queries only\n")

    stats = {"ok": 0, "no_videos": 0, "no_consensus": 0, "dry_run": 0}

    for i, actor in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {actor['name']}")
        outcome = scrape_actor(
            db, actor["id"], actor["name"], hf_token, device,
            dry_run=dry_run, augment=augment,
        )
        stats[outcome] += 1
        if not dry_run and i < len(targets):
            time.sleep(SCRAPE_DELAY)

    db.close()

    if not dry_run:
        print(f"\n── Summary ───────────────────────────────")
        print(f"  Embedded     : {stats['ok']}")
        print(f"  No videos    : {stats['no_videos']}")
        print(f"  No consensus : {stats['no_consensus']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consensus-scrape YouTube voice clips and store validated embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--actor", type=str, default="", metavar="NAME",
        help="Process a specific actor by name (partial match)",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Max number of actors to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scrape actors even if they already have embeddings "
             "(use to replace old blind-scraped embeddings)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print search queries without downloading or storing anything",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Skip channel augmentation (reverb/noise/band-limit) of the "
             "consensus voice before storing",
    )
    args = parser.parse_args()

    actor_names = [args.actor] if args.actor else None

    scrape_actors(
        actor_names=actor_names,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
