"""
test_add_character_voice.py — lazy per-character voice ingestion

Network, yt-dlp, and ML models are mocked. `dominant_speaker` is exercised with a
faked `diarize` + `get_embedding`; the higher-level `add_character_voice` flow is
driven through the in-memory fake Supabase (the `db` fixture) with the per-source
embedding step stubbed, so the selection / agreement / storage logic is covered
without audio.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import add_character_voice as acv
from pipeline import SpeakerSegment


# ── Synthetic voice helpers (mirrors test_consensus) ──────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def _voice(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.standard_normal(192))


def _appearance(voice: np.ndarray, rng: np.random.Generator, noise: float = 0.4) -> np.ndarray:
    return _unit(voice + noise * _unit(rng.standard_normal(192)))


def _src(source: str, emb: np.ndarray, seconds: float = 20.0):
    """Build a SourceEmbedding with a throwaway speech tensor."""
    import torch

    return acv.SourceEmbedding(source, emb, torch.zeros(1, 16000), seconds)


# ── URL parsing ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("not a url", None),
    ("https://example.com/video", None),
])
def test_video_id_from_url(url, expected):
    assert acv.video_id_from_url(url) == expected


# ── dominant_speaker ──────────────────────────────────────────────────────────

def test_dominant_speaker_picks_longest_speaker(rng):
    """The speaker with the most total speech is the one embedded."""
    import torch

    target = _voice(rng)
    segs = [
        SpeakerSegment("SPEAKER_00", 0.0, 3.0),     # 3s minor speaker
        SpeakerSegment("SPEAKER_01", 3.0, 23.0),    # 20s dominant speaker
        SpeakerSegment("SPEAKER_01", 25.0, 35.0),   # +10s -> 30s total
    ]
    captured = {}

    def fake_embed(speech, device="cpu"):
        captured["nsamples"] = speech.shape[1]
        return target

    with (
        patch.object(acv, "_load_mono_16k", return_value=torch.zeros(1, 16000 * 40)),
        patch("pipeline.diarize", return_value=segs),
        patch("pipeline.get_embedding", side_effect=fake_embed),
    ):
        emb, speech, seconds = acv.dominant_speaker("x.wav", "hf", "cpu")

    assert np.allclose(emb, target)
    assert seconds == pytest.approx(30.0)       # 20 + 10, the SPEAKER_01 total
    # Embedded audio is the dominant speaker's two chunks (20s + 10s), not the 3s one
    assert captured["nsamples"] == int(30.0 * 16000)


def test_dominant_speaker_rejects_too_short(rng):
    import torch

    segs = [SpeakerSegment("SPEAKER_00", 0.0, 2.0)]  # below MIN_SPEAKER_SECONDS
    with (
        patch.object(acv, "_load_mono_16k", return_value=torch.zeros(1, 16000 * 5)),
        patch("pipeline.diarize", return_value=segs),
        patch("pipeline.get_embedding", return_value=_voice(rng)),
    ):
        assert acv.dominant_speaker("x.wav", "hf", "cpu") is None


def test_dominant_speaker_no_token_embeds_whole_clip(rng):
    import torch

    with (
        patch.object(acv, "_load_mono_16k", return_value=torch.zeros(1, 16000 * 20)),
        patch("pipeline.get_embedding", return_value=_voice(rng)) as ge,
    ):
        got = acv.dominant_speaker("x.wav", "", "cpu")  # no hf_token
    assert got is not None
    ge.assert_called_once()


# ── actor / character resolution against the fake DB ──────────────────────────

def test_resolve_actor_prefers_exact_match(db):
    db.add_actor("Yuuki Kaji", anilist_id=1)
    db.add_actor("Yuuki Kajikawa", anilist_id=2)   # partial-match decoy
    aid = acv.resolve_actor_id(db, "Yuuki Kaji")
    actor = db.get_actor(aid)
    assert actor["name"] == "Yuuki Kaji"


def test_resolve_actor_none_when_absent(db):
    assert acv.resolve_actor_id(db, "Nobody") is None


def test_resolve_character_id_scoped_to_show(db):
    actor_id = db.add_actor("Yuuki Kaji", anilist_id=1)
    show_a = db.add_show("Attack on Titan", anilist_id=10)
    show_b = db.add_show("Other Show", anilist_id=11)
    db.add_character("Eren Yeager", show_b, actor_id, anilist_id=100)
    target = db.add_character("Eren Yeager", show_a, actor_id, anilist_id=101)
    cid = acv.resolve_character_id(db, actor_id, "Eren", show_a)
    assert cid == target


def test_resolve_character_id_none_when_no_row(db):
    actor_id = db.add_actor("Yuuki Kaji", anilist_id=1)
    assert acv.resolve_character_id(db, actor_id, "Eren", None) is None


# ── add_character_voice end to end (embedding step stubbed) ───────────────────

def _patch_db(db):
    """Patch VazamDB() inside add_character_voice to return the fixture db."""
    return patch("db.VazamDB", return_value=db)


def test_add_character_voice_stores_linked_embedding(db, rng):
    actor_id = db.add_actor("Yuuki Kaji", anilist_id=1)
    show_id = db.add_show("Attack on Titan", anilist_id=10)
    char_id = db.add_character("Eren Yeager", show_id, actor_id, anilist_id=100)
    voice = _voice(rng)

    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf", "DEVICE": "cpu"}),
        patch.object(acv, "dominant_speaker", return_value=(voice, None, 20.0)),
        patch.object(acv, "_resolve_source", return_value="clip.wav"),
        patch.object(acv, "_build_embedding", return_value=voice),
    ):
        outcome = acv.add_character_voice(
            "Yuuki Kaji", "Eren Yeager",
            sources=["https://youtu.be/aaaaaaaaaaa"],
            show="Attack on Titan", augment=False,
        )

    assert outcome == "ok"
    rows = db._client.table("vazam_embeddings").select("*").execute().data
    assert len(rows) == 1
    row = rows[0]
    assert row["voice_label"] == "Eren Yeager"
    assert row["character_id"] == char_id
    assert row["actor_id"] == actor_id
    assert row["quality_score"] is None          # single source → no agreement score


def test_add_character_voice_rejects_inconsistent_sources(db, rng):
    db.add_actor("Yuuki Kaji", anilist_id=1)
    voice_a = _voice(rng)
    voice_b = _voice(rng)                          # a different speaker entirely
    embeds = iter([(voice_a, None, 20.0), (voice_b, None, 20.0)])

    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf", "DEVICE": "cpu"}),
        patch.object(acv, "dominant_speaker", side_effect=lambda *a, **k: next(embeds)),
        patch.object(acv, "_resolve_source", side_effect=lambda s, d: s),
    ):
        outcome = acv.add_character_voice(
            "Yuuki Kaji", "Eren Yeager",
            sources=["a.wav", "b.wav"], augment=False,
        )

    assert outcome == "inconsistent"
    assert db._client.table("vazam_embeddings").select("*").execute().data == []


def test_add_character_voice_consistent_sources_store_quality(db, rng):
    db.add_actor("Yuuki Kaji", anilist_id=1)
    base = _voice(rng)
    a, b = _appearance(base, rng), _appearance(base, rng)   # same voice, two clips
    embeds = iter([(a, None, 20.0), (b, None, 18.0)])

    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf", "DEVICE": "cpu"}),
        patch.object(acv, "dominant_speaker", side_effect=lambda *a, **k: next(embeds)),
        patch.object(acv, "_resolve_source", side_effect=lambda s, d: s),
        patch.object(acv, "_build_embedding", return_value=base),
    ):
        outcome = acv.add_character_voice(
            "Yuuki Kaji", "Eren Yeager",
            sources=["a.wav", "b.wav"], augment=False,
        )

    assert outcome == "ok"
    row = db._client.table("vazam_embeddings").select("*").execute().data[0]
    assert row["quality_score"] is not None and row["quality_score"] > acv.MIN_CLIP_AGREEMENT
    assert row["duration_s"] == pytest.approx(38.0)        # 20 + 18


def test_add_character_voice_no_actor(db):
    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf"}),
    ):
        outcome = acv.add_character_voice("Ghost", "X", sources=["a.wav"])
    assert outcome == "no_actor"


def test_add_character_voice_dry_run_writes_nothing(db):
    db.add_actor("Yuuki Kaji", anilist_id=1)
    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf"}),
    ):
        outcome = acv.add_character_voice(
            "Yuuki Kaji", "Eren Yeager", sources=["a.wav"], dry_run=True,
        )
    assert outcome == "dry_run"
    assert db._client.table("vazam_embeddings").select("*").execute().data == []


def test_add_character_voice_unlinked_when_no_character_row(db, rng):
    db.add_actor("Steve Blum", anilist_id=1)       # no character seeded
    voice = _voice(rng)
    with (
        _patch_db(db),
        patch.dict("os.environ", {"HF_TOKEN": "hf", "DEVICE": "cpu"}),
        patch.object(acv, "dominant_speaker", return_value=(voice, None, 20.0)),
        patch.object(acv, "_resolve_source", return_value="clip.wav"),
        patch.object(acv, "_build_embedding", return_value=voice),
    ):
        outcome = acv.add_character_voice(
            "Steve Blum", "Spike Spiegel", sources=["a.wav"], augment=False,
        )
    assert outcome == "ok"
    row = db._client.table("vazam_embeddings").select("*").execute().data[0]
    assert row["character_id"] is None
    assert row["voice_label"] == "Spike Spiegel"


# ── _resolve_source ───────────────────────────────────────────────────────────

def test_resolve_source_local_file(tmp_wav):
    assert acv._resolve_source(tmp_wav, "/tmp") == tmp_wav


def test_resolve_source_downloads_url():
    with patch.object(acv, "_download_audio", return_value="/tmp/x.mp3") as dl:
        out = acv._resolve_source("https://youtu.be/dQw4w9WgXcQ", "/tmp")
    assert out == "/tmp/x.mp3"
    dl.assert_called_once()


def test_resolve_source_bad_url_returns_none():
    assert acv._resolve_source("https://example.com/nope", "/tmp") is None
