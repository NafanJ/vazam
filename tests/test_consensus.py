"""
test_consensus.py — cross-video consensus clustering and scraper helpers

Pure numpy tests: no network, no ML models. Synthetic voices are random unit
vectors; appearances of the same voice are the base vector plus small noise,
so same-voice cosine is high and cross-voice cosine is near zero (192-dim).
"""

from __future__ import annotations

import numpy as np
import pytest

from consensus import (
    CLUSTER_LINK_THRESHOLD,
    VideoSpeaker,
    build_consensus,
    centroid,
    cluster_speakers,
    mean_pairwise_cosine,
)
from scrape_audio import VideoCandidate, parse_candidates, select_videos


# ── Synthetic voice helpers ───────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def _voice(rng: np.random.Generator) -> np.ndarray:
    """A new distinct voice (random unit vector)."""
    return _unit(rng.standard_normal(192))


def _appearance(voice: np.ndarray, rng: np.random.Generator, noise: float = 0.4) -> np.ndarray:
    """One noisy observation of a voice (same speaker, different recording).

    The perturbation has total norm `noise` (not per-component), giving
    same-voice cosine ≈ 0.9 and cross-voice cosine ≈ 0 in 192 dims.
    """
    return _unit(voice + noise * _unit(rng.standard_normal(192)))


# ── centroid / quality helpers ────────────────────────────────────────────────

def test_centroid_is_normalized(rng):
    embs = [_voice(rng) for _ in range(3)]
    c = centroid(embs)
    assert c.dtype == np.float32
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-5)


def test_mean_pairwise_cosine_identical_vectors(rng):
    v = _voice(rng)
    assert mean_pairwise_cosine([v, v, v]) == pytest.approx(1.0, abs=1e-5)


def test_mean_pairwise_cosine_single_vector(rng):
    assert mean_pairwise_cosine([_voice(rng)]) == 1.0


# ── Clustering ────────────────────────────────────────────────────────────────

def test_same_voice_across_videos_clusters_together(rng):
    voice = _voice(rng)
    speakers = [
        VideoSpeaker("v1", "SPEAKER_00", _appearance(voice, rng), 30.0),
        VideoSpeaker("v2", "SPEAKER_01", _appearance(voice, rng), 25.0),
        VideoSpeaker("v3", "SPEAKER_00", _appearance(voice, rng), 40.0),
    ]
    clusters = cluster_speakers(speakers)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_distinct_voices_stay_separate(rng):
    speakers = [
        VideoSpeaker(f"v{i}", "SPEAKER_00", _voice(rng), 30.0)
        for i in range(3)
    ]
    clusters = cluster_speakers(speakers)
    assert len(clusters) == 3


# ── Consensus selection ───────────────────────────────────────────────────────

def test_recurring_actor_beats_per_video_interviewers(rng):
    """The canonical case: actor in all 3 videos, a different interviewer in each."""
    actor = _voice(rng)
    speakers = []
    for i in range(3):
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_00", _appearance(actor, rng), 45.0))
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_01", _voice(rng), 60.0))

    result = build_consensus(speakers)
    assert result is not None
    assert result.n_videos == 3
    assert sorted(result.video_ids) == ["v0", "v1", "v2"]
    # The centroid must be the actor's voice, not any interviewer's
    assert float(np.dot(result.embedding, actor)) > 0.8
    assert result.quality_score > 0.7
    assert result.speech_seconds == pytest.approx(135.0)
    # members drive channel augmentation downstream — must be the cluster itself
    assert len(result.members) == 3
    assert all(s.speaker_label == "SPEAKER_00" for s in result.members)


def test_actor_in_more_videos_beats_recurring_interviewer(rng):
    """An interviewer recurring in 2 videos loses to the actor present in 3."""
    actor, host = _voice(rng), _voice(rng)
    speakers = []
    for i in range(3):
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_00", _appearance(actor, rng), 20.0))
    for i in range(2):
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_01", _appearance(host, rng), 90.0))

    result = build_consensus(speakers)
    assert result is not None
    assert float(np.dot(result.embedding, actor)) > 0.8


def test_video_count_tie_breaks_on_speech_time(rng):
    """Same video coverage → the cluster with more speech wins (the guest
    usually talks most in their own interview)."""
    actor, host = _voice(rng), _voice(rng)
    speakers = []
    for i in range(2):
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_00", _appearance(actor, rng), 80.0))
        speakers.append(VideoSpeaker(f"v{i}", "SPEAKER_01", _appearance(host, rng), 30.0))

    result = build_consensus(speakers)
    assert result is not None
    assert float(np.dot(result.embedding, actor)) > 0.8


def test_no_consensus_when_nothing_recurs(rng):
    """All speakers distinct → nothing recurs → None, never a blind embedding."""
    speakers = [
        VideoSpeaker(f"v{i}", "SPEAKER_00", _voice(rng), 30.0)
        for i in range(3)
    ]
    assert build_consensus(speakers) is None


def test_no_consensus_for_single_video(rng):
    voice = _voice(rng)
    speakers = [
        VideoSpeaker("v0", "SPEAKER_00", _appearance(voice, rng), 30.0),
        VideoSpeaker("v0", "SPEAKER_01", _appearance(voice, rng), 30.0),
    ]
    assert build_consensus(speakers) is None


def test_empty_input():
    assert build_consensus([]) is None


# ── scrape_audio helpers (pure, no network) ───────────────────────────────────

def test_parse_candidates():
    output = (
        "abc123\t245.0\tSteve Blum Interview\n"
        "def456\tNA\tLive stream now\n"          # live → no duration → skipped
        "ghi789\t1800\tAnime Expo Panel\n"
        "malformed line\n"
    )
    cands = parse_candidates(output)
    assert [c.video_id for c in cands] == ["abc123", "ghi789"]
    assert cands[0].duration == 245.0
    assert cands[1].title == "Anime Expo Panel"
    assert cands[0].url.endswith("abc123")


def test_select_videos_dedupes_and_filters():
    q1 = [
        VideoCandidate("dup", 300, "Interview"),
        VideoCandidate("a", 400, "Interview 2"),
    ]
    q2 = [
        VideoCandidate("dup", 300, "Same video, other query"),
        VideoCandidate("short", 30, "Too short"),
        VideoCandidate("long", 7200, "Too long"),
        VideoCandidate("b", 500, "Panel"),
    ]
    selected = select_videos([q1, q2], n_videos=3)
    assert [c.video_id for c in selected] == ["dup", "b", "a"]


def test_select_videos_round_robins_across_queries():
    """First pick should come from each query before second picks — videos
    from independent searches give consensus its power."""
    q1 = [VideoCandidate("a1", 300, ""), VideoCandidate("a2", 300, "")]
    q2 = [VideoCandidate("b1", 300, ""), VideoCandidate("b2", 300, "")]
    q3 = [VideoCandidate("c1", 300, "")]
    selected = select_videos([q1, q2, q3], n_videos=3)
    assert [c.video_id for c in selected] == ["a1", "b1", "c1"]


def test_select_videos_handles_empty_queries():
    assert select_videos([[], [], []]) == []
