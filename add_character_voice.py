"""
add_character_voice.py — Lazy per-character voice embedding ingestion

Stores a *character-voice* embedding (e.g. Yuuki Kaji **as Eren Yeager**) for a
single marquee role, from one or more curated YouTube clips or local files.

Why this exists
---------------
The consensus scraper (scrape_audio.py) stores one "Natural Voice" embedding per
actor from interviews/panels. But heavily-altered character voices sit far from an
actor's natural voice in embedding space — live validation measured ~0.88 cosine on
matched-condition interview audio vs only ~0.52 for an in-character voice against the
same "Natural Voice" reference. For top-billed roles where that gap costs matches, we
store a *separate* embedding under the character's ``voice_label`` (the schema already
supports multiple embeddings per actor).

This stays deliberately **lazy and selective** — per the acquisition-plan anti-goal,
character clips are never bulk-scraped. You point this tool at one good source
(ideally a character voice-line compilation, which is dominated by a single speaker)
for a single named role.

How a source is turned into an embedding
----------------------------------------
1. Download the first few minutes of each source (yt-dlp), or load a local file.
2. Diarize and select a speaker. Two policies (--select):
   - **dominant** (default) — the speaker with the most speech. In a single-character
     compilation that is the character; co-speakers and uploader intros get dropped.
   - **nearest-natural** — the speaker closest to the actor's stored Natural Voice.
     Use this for *ensemble* characters who never carry a scene alone (e.g. Connie):
     their only clips are group scenes where someone else dominates, so "dominant"
     picks the wrong voice. Requires the actor to already have a Natural Voice, and
     only works when the character voice still ranks against it.
   (Without HF_TOKEN there is no diarization — the whole clip is embedded.)
3. Embed that speaker's speech (ECAPA-TDNN, capped at MAX_EMBED_SECONDS).
4. With ≥2 sources, require the per-source dominant-speaker embeddings to *agree*
   (mean pairwise cosine ≥ MIN_CLIP_AGREEMENT) — if the "dominant speaker" isn't the
   same voice across clips, the sources are inconsistent and nothing is stored. The
   stored embedding is their centroid; agreement becomes the quality score.
5. Optionally fold channel-augmented variants into the centroid (same reverb / noise /
   band-limit treatment the scraper applies), so the reference matches phone-mic-of-TV
   query conditions. Disable with --no-augment.

Usage
-----
    # One compilation clip for Eren (Yuuki Kaji), auto-linked to the AoT character row
    python add_character_voice.py \\
        --actor "Yuuki Kaji" --character "Eren Yeager" \\
        --url https://www.youtube.com/watch?v=XXXX

    # Two sources (cross-clip agreement enforced) + explicit show
    python add_character_voice.py --actor "Hiroshi Kamiya" --character "Levi" \\
        --show "Attack on Titan" --url <url1> --url <url2>

    # Local file, no augmentation, preview without writing
    python add_character_voice.py --actor "Steve Blum" --character "Spike Spiegel" \\
        --file spike.wav --no-augment --dry-run

Environment variables
---------------------
    SUPABASE_URL / SUPABASE_KEY   Supabase project (service role for writes)
    HF_TOKEN                      HuggingFace token — enables diarization, which is
                                  what isolates the character from co-speakers.
    DEVICE                        cuda or cpu (auto-detected if unset)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Reuse the scraper's download/audio/augmentation machinery — same pipeline,
# different selection policy (dominant speaker instead of cross-video consensus).
from consensus import mean_pairwise_cosine
from scrape_audio import (
    MAX_EMBED_SECONDS,
    MIN_SPEAKER_SECONDS,
    VideoCandidate,
    _download_audio,
    _load_mono_16k,
)

load_dotenv()

# A character voice is the *same* across compilations of that character, so when
# multiple sources are given their dominant-speaker embeddings must be consistent.
# Below this mean-pairwise cosine the sources disagree (the "dominant speaker"
# wasn't the same voice) and we refuse to store rather than blend two voices.
MIN_CLIP_AGREEMENT = 0.55

_YT_ID_RE = re.compile(
    r"(?:v=|/shorts/|youtu\.be/|/embed/)([0-9A-Za-z_-]{11})"
)


@dataclass
class SourceEmbedding:
    """One source's dominant-speaker embedding and the speech behind it."""

    source: str          # url or file path, for provenance
    embedding: "np.ndarray"   # noqa: F821 — numpy imported lazily by callers
    speech: "torch.Tensor"    # noqa: F821
    seconds: float


def video_id_from_url(url: str) -> Optional[str]:
    """Extract a YouTube video id from common URL shapes, else None."""
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


def _all_speaker_embeddings(audio_path: str, hf_token: str, device: str):
    """Embed every speaker with ≥ MIN_SPEAKER_SECONDS of speech in a clip.

    Returns a list of (label, embedding, speech_tensor, seconds). With a token we
    diarize and embed each speaker's concatenated speech (capped at
    MAX_EMBED_SECONDS); without one, a single whole-clip "FULL" entry (no speaker
    separation). Shared by the selection policies below.
    """
    import torch

    from pipeline import diarize, get_embedding

    signal = _load_mono_16k(audio_path)
    out: list[tuple] = []

    if hf_token:
        segments = diarize(audio_path, hf_token)
        if not segments:
            return []
        by_label: dict[str, list] = {}
        for seg in segments:
            by_label.setdefault(seg.speaker_label, []).append(seg)
        for label, segs in by_label.items():
            total = sum(s.duration for s in segs)
            if total < MIN_SPEAKER_SECONDS:
                continue
            chunks = [
                signal[:, int(s.start * 16000): int(s.end * 16000)] for s in segs
            ]
            speech = torch.cat(chunks, dim=1)[:, : int(MAX_EMBED_SECONDS * 16000)]
            out.append((label, get_embedding(speech, device=device), speech,
                        min(total, MAX_EMBED_SECONDS)))
    else:
        speech = signal[:, : int(MAX_EMBED_SECONDS * 16000)]
        seconds = speech.shape[1] / 16000
        if seconds >= MIN_SPEAKER_SECONDS:
            out.append(("FULL", get_embedding(speech, device=device), speech, seconds))
    return out


def dominant_speaker(audio_path: str, hf_token: str, device: str):
    """Embed the speaker with the most speech (the character, in a single-character
    compilation).

    Returns (embedding, speech_tensor, seconds), or None if no speaker clears
    MIN_SPEAKER_SECONDS.
    """
    cands = _all_speaker_embeddings(audio_path, hf_token, device)
    if not cands:
        return None
    _, emb, speech, seconds = max(cands, key=lambda c: c[3])
    return emb, speech, seconds


def nearest_natural_speaker(audio_path: str, hf_token: str, device: str, natural_ref):
    """Embed the speaker whose voice is closest to the actor's Natural Voice.

    For ensemble characters who never carry a scene alone — so the *dominant*
    speaker in their clips is usually someone else — this picks the right speaker
    out of a group clip, provided the character voice still ranks against the
    actor's natural-voice reference (it won't help a voice so altered the natural
    reference doesn't rank at all, but for those dominant-speaker already works).
    Returns (embedding, speech_tensor, seconds), or None.
    """
    import numpy as np

    cands = _all_speaker_embeddings(audio_path, hf_token, device)
    if not cands:
        return None
    _, emb, speech, seconds = max(
        cands, key=lambda c: float(np.dot(c[1], natural_ref))
    )
    return emb, speech, seconds


def resolve_character_id(
    db, actor_id: int, character: str, show_id: Optional[int]
) -> Optional[int]:
    """Find the vazam_characters row for this actor/character (optionally show-scoped).

    Returns the character id, or None if no row matches (the embedding is still
    storable — character_id is nullable — but linking is preferred for filmography).
    """
    q = (
        db._client.table("vazam_characters")
        .select("id, name, show_id")
        .eq("actor_id", actor_id)
        .ilike("name", f"%{character}%")
    )
    rows = q.execute().data or []
    if show_id is not None:
        rows = [r for r in rows if r.get("show_id") == show_id] or rows
    return rows[0]["id"] if rows else None


def _parse_embedding(raw):
    """Parse a stored pgvector value (a list, or a '[..]' string) into a unit float32 array."""
    import numpy as np

    if isinstance(raw, str):
        raw = json.loads(raw)
    v = np.asarray(raw, dtype="float32")
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def actor_natural_embedding(db, actor_id: int):
    """Return the actor's stored 'Natural Voice' embedding (unit float32), or None.

    The reference the `nearest-natural` selection policy matches speakers against.
    """
    rows = (
        db._client.table("vazam_embeddings")
        .select("embedding, voice_label")
        .eq("actor_id", actor_id)
        .eq("voice_label", "Natural Voice")
        .execute()
        .data
        or []
    )
    return _parse_embedding(rows[0]["embedding"]) if rows else None


def resolve_actor_id(db, actor_name: str) -> Optional[int]:
    """Resolve an actor name (partial, case-insensitive) to a single id."""
    rows = (
        db._client.table("vazam_actors")
        .select("id, name")
        .ilike("name", f"%{actor_name}%")
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    # Prefer an exact (case-insensitive) name match if present.
    for r in rows:
        if r["name"].lower() == actor_name.lower():
            return r["id"]
    return rows[0]["id"]


def add_character_voice(
    actor_name: str,
    character: str,
    sources: list[str],
    show: str = "",
    show_id: Optional[int] = None,
    augment: bool = True,
    isolate: bool = False,
    select: str = "dominant",
    dry_run: bool = False,
) -> str:
    """Ingest a character-voice embedding for one role from one or more sources.

    ``isolate`` runs Demucs vocal isolation on each source before diarization —
    recommended for anime/show clips (which carry a music/SFX bed) and unnecessary
    for clean voice-line rips. ``select`` chooses the per-source speaker policy:
    ``"dominant"`` (most speech — the character in a single-character compilation)
    or ``"nearest-natural"`` (the speaker closest to the actor's stored Natural
    Voice — for ensemble characters who never speak solo; requires that Natural
    Voice to exist). Returns: "ok", "no_actor", "no_natural", "no_audio",
    "no_speaker", "inconsistent", or "dry_run".
    """
    from db import VazamDB

    hf_token = os.environ.get("HF_TOKEN", "")
    device = os.environ.get("DEVICE", "") or None
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hf_token:
        print("⚠ HF_TOKEN not set — embedding whole clips without diarization.\n"
              "  A character compilation may then be contaminated by co-speakers.\n")

    db = VazamDB()

    actor_id = resolve_actor_id(db, actor_name)
    if actor_id is None:
        print(f"✗ no actor matching '{actor_name}' — seed metadata first (seed.py)")
        return "no_actor"

    if show and show_id is None:
        matches = db.search_show(show)
        if matches:
            show_id = matches[0]["id"]
    character_id = resolve_character_id(db, actor_id, character, show_id)
    link = f"character_id={character_id}" if character_id else "unlinked (no character row)"
    print(f"Actor #{actor_id} '{actor_name}' as '{character}' — {link}")

    # nearest-natural needs the actor's Natural Voice to match speakers against.
    natural_ref = None
    if select == "nearest-natural":
        natural_ref = actor_natural_embedding(db, actor_id)
        if natural_ref is None:
            print(f"✗ --select nearest-natural needs a stored 'Natural Voice' for "
                  f"'{actor_name}' to match against — run scrape_audio.py first")
            return "no_natural"
        print("  selection: speaker nearest the actor's Natural Voice")
    else:
        print("  selection: dominant speaker")
    for s in sources:
        print(f"  source: {s}")
    if dry_run:
        return "dry_run"

    # ── Embed the selected speaker of each source ────────────────────────────
    per_source: list[SourceEmbedding] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for src in sources:
            audio_path = _resolve_source(src, tmpdir)
            if not audio_path:
                print(f"  ✗ could not obtain audio: {src}")
                continue
            if isolate:
                from pipeline import isolate_vocals

                audio_path = isolate_vocals(audio_path, output_dir=tmpdir)
                print("    · vocals isolated (Demucs)")
            try:
                if select == "nearest-natural":
                    got = nearest_natural_speaker(audio_path, hf_token, device, natural_ref)
                else:
                    got = dominant_speaker(audio_path, hf_token, device)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  ✗ embedding error ({src}): {exc}")
                continue
            if got is None:
                print(f"  ✗ no speaker with ≥{MIN_SPEAKER_SECONDS:.0f}s speech: {src}")
                continue
            emb, speech, seconds = got
            per_source.append(SourceEmbedding(src, emb, speech, seconds))
            print(f"  ✓ speaker embedded ({seconds:.0f}s)")

        if not per_source:
            print("✗ no usable audio across sources — nothing stored")
            return "no_audio" if len(per_source) == 0 else "no_speaker"

        # ── Require cross-source agreement when ≥2 sources ──────────────────
        agreement = mean_pairwise_cosine([s.embedding for s in per_source])
        if len(per_source) >= 2:
            print(f"  cross-source agreement: {agreement:.3f} "
                  f"(floor {MIN_CLIP_AGREEMENT:.2f})")
            if agreement < MIN_CLIP_AGREEMENT:
                print("✗ sources disagree — dominant speaker differs between clips; "
                      "nothing stored. Use cleaner single-character compilations.")
                return "inconsistent"

        # ── Centroid (+ optional channel augmentation) ──────────────────────
        embedding = _build_embedding(per_source, augment, device)

    quality = round(agreement, 4) if len(per_source) >= 2 else None
    total_speech = sum(s.seconds for s in per_source)
    source_urls = ",".join(s.source for s in per_source)

    emb_id = db.add_embedding(
        actor_id=actor_id,
        embedding=embedding,
        character_id=character_id,
        voice_label=character,
        audio_source="character_clip",
        verified=False,
        source_url=source_urls,
        duration_s=total_speech,
        quality_score=quality,
    )
    q_str = f"{quality:.3f}" if quality is not None else "n/a (single source)"
    print(f"✓ stored character embedding #{emb_id} for '{character}' "
          f"(voice_label='{character}', quality={q_str}, "
          f"speech={total_speech:.0f}s, sources={len(per_source)})")
    return "ok"


def _resolve_source(src: str, tmpdir: str) -> Optional[str]:
    """Return a local audio path for a source that is a URL or a file path."""
    if os.path.exists(src):
        return src
    vid = video_id_from_url(src)
    if vid is None:
        return None
    return _download_audio(VideoCandidate(vid, duration=0.0, title=""), tmpdir)


def _build_embedding(per_source: list, augment: bool, device: str):
    """Centroid of per-source embeddings, optionally with augmented variants folded in."""
    from consensus import centroid

    embeddings = [s.embedding for s in per_source]
    if augment:
        from augment import augment_speech
        from pipeline import get_embedding

        for s in per_source:
            for variant in augment_speech(s.speech):
                embeddings.append(get_embedding(variant, device=device))
        print("  + channel augmentation folded into centroid")
    return centroid(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a lazy per-character voice embedding for a marquee role",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--actor", required=True, metavar="NAME",
                        help="Voice actor name (partial, case-insensitive match)")
    parser.add_argument("--character", required=True, metavar="NAME",
                        help="Character / role name — stored as the voice_label")
    parser.add_argument("--show", default="", metavar="TITLE",
                        help="Show title, to disambiguate the character row (optional)")
    parser.add_argument("--show-id", type=int, default=None,
                        help="Show id, if known (overrides --show lookup)")
    parser.add_argument("--url", action="append", default=[], metavar="URL",
                        help="YouTube URL of a character clip (repeatable)")
    parser.add_argument("--file", action="append", default=[], metavar="PATH",
                        help="Local audio file of a character clip (repeatable)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip channel augmentation of the character voice")
    parser.add_argument("--isolate", action="store_true",
                        help="Run Demucs vocal isolation before diarization "
                             "(recommended for anime/show clips with a music bed)")
    parser.add_argument("--select", choices=["dominant", "nearest-natural"],
                        default="dominant",
                        help="Speaker-selection policy: 'dominant' (most speech — the "
                             "character in a single-character compilation; default) or "
                             "'nearest-natural' (speaker closest to the actor's stored "
                             "Natural Voice — use for ensemble characters who never "
                             "speak solo, e.g. Connie; requires that Natural Voice)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve actor/character and print sources without writing")
    args = parser.parse_args()

    sources = [*args.url, *args.file]
    if not sources:
        parser.error("provide at least one --url or --file")

    outcome = add_character_voice(
        actor_name=args.actor,
        character=args.character,
        sources=sources,
        show=args.show,
        show_id=args.show_id,
        augment=not args.no_augment,
        isolate=args.isolate,
        select=args.select,
        dry_run=args.dry_run,
    )
    sys.exit(0 if outcome in ("ok", "dry_run") else 1)


if __name__ == "__main__":
    main()
