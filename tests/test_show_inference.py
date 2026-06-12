"""
test_show_inference.py — cast-graph show inference

Covers the vote_shows logic (pure), VazamPipeline.identify_show orchestration
(diarization and embeddings mocked), and db.get_shows_for_actors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from db import VazamDB
from pipeline import (
    POSSIBLE_THRESHOLD,
    IdentificationResult,
    VazamPipeline,
    vote_shows,
)


# ── vote_shows ────────────────────────────────────────────────────────────────

class TestVoteShows:
    def test_two_speakers_in_one_cast_win(self):
        candidates = {
            "SPEAKER_00": [(1, 0.80), (9, 0.75)],
            "SPEAKER_01": [(2, 0.78)],
        }
        actor_shows = {1: {100}, 2: {100}, 9: {200}}

        votes = vote_shows(candidates, actor_shows)

        assert votes[0].show_id == 100
        assert votes[0].n_speakers == 2
        assert votes[0].score == pytest.approx(0.80 + 0.78)
        # Show 200 explains only one speaker → ranked below
        assert votes[1].show_id == 200
        assert votes[1].n_speakers == 1

    def test_weak_candidates_do_not_vote(self):
        candidates = {
            "SPEAKER_00": [(1, POSSIBLE_THRESHOLD - 0.01)],
            "SPEAKER_01": [(2, 0.90)],
        }
        actor_shows = {1: {100}, 2: {100}}

        votes = vote_shows(candidates, actor_shows)
        assert votes[0].n_speakers == 1   # only the strong candidate voted

    def test_one_speaker_votes_once_per_show(self):
        """Two castmates matching the same speaker must not double-count it."""
        candidates = {"SPEAKER_00": [(1, 0.80), (2, 0.70)]}
        actor_shows = {1: {100}, 2: {100}}

        votes = vote_shows(candidates, actor_shows)
        assert votes[0].n_speakers == 1
        assert votes[0].score == pytest.approx(0.80)   # best similarity only

    def test_speaker_count_beats_score(self):
        candidates = {
            "SPEAKER_00": [(1, 0.55), (3, 0.99)],
            "SPEAKER_01": [(2, 0.55)],
        }
        # Show 100: two weak speakers; show 200: one very strong speaker
        actor_shows = {1: {100}, 2: {100}, 3: {200}}

        votes = vote_shows(candidates, actor_shows)
        assert votes[0].show_id == 100

    def test_actor_without_shows_is_ignored(self):
        candidates = {"SPEAKER_00": [(1, 0.90)]}
        assert vote_shows(candidates, {}) == []

    def test_empty(self):
        assert vote_shows({}, {}) == []


# ── VazamPipeline.identify_show ───────────────────────────────────────────────

def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


def _row(actor_id: int, similarity: float) -> dict:
    return {
        "actor_id":    actor_id,
        "actor_name":  f"Actor {actor_id}",
        "voice_label": "Natural Voice",
        "similarity":  similarity,
    }


class TestIdentifyShow:

    def _make_pipeline(self, mock_db) -> VazamPipeline:
        # hf_token set so use_diarization is True; diarization itself is mocked
        return VazamPipeline(db=mock_db, hf_token="fake", use_vad=False)

    def test_show_inferred_and_speakers_reranked(self, tmp_wav):
        mock_db = MagicMock()
        # Global searches: speaker A → actor 1, speaker B → actor 2
        # Closed-set searches (with show_id): same winners, higher similarity
        mock_db.search_embeddings.side_effect = [
            [_row(1, 0.72), _row(9, 0.60)],   # SPEAKER_00 global
            [_row(2, 0.68)],                  # SPEAKER_01 global
            [_row(1, 0.72)],                  # SPEAKER_00 within show
            [_row(2, 0.68)],                  # SPEAKER_01 within show
        ]
        mock_db.get_shows_for_actors.return_value = {1: {100}, 2: {100}, 9: {200}}
        mock_db.get_show.return_value = {"id": 100, "title": "Cowboy Bebop"}

        p = self._make_pipeline(mock_db)
        speaker_embs = {"SPEAKER_00": _emb(0), "SPEAKER_01": _emb(1)}

        with (
            patch.object(p, "_speaker_embeddings", return_value=speaker_embs),
            patch("pipeline.isolate_vocals", side_effect=lambda path, **kw: path),
        ):
            inference, per_speaker = p.identify_show(tmp_wav)

        assert inference is not None
        assert inference.show_id == 100
        assert inference.show_title == "Cowboy Bebop"
        assert inference.speakers_matched == 2
        assert inference.speakers_total == 2

        assert set(per_speaker) == {"SPEAKER_00", "SPEAKER_01"}
        assert per_speaker["SPEAKER_00"][0].actor_id == 1

        # Re-rank calls must be restricted to the inferred show
        rerank_calls = mock_db.search_embeddings.call_args_list[2:]
        assert all(c.kwargs.get("show_id") == 100 for c in rerank_calls)

    def test_no_consensus_returns_global_results(self, tmp_wav):
        """Speakers from different casts → no show inferred, global results kept."""
        mock_db = MagicMock()
        mock_db.search_embeddings.side_effect = [
            [_row(1, 0.72)],   # SPEAKER_00 global
            [_row(2, 0.68)],   # SPEAKER_01 global
        ]
        mock_db.get_shows_for_actors.return_value = {1: {100}, 2: {200}}

        p = self._make_pipeline(mock_db)
        speaker_embs = {"SPEAKER_00": _emb(0), "SPEAKER_01": _emb(1)}

        with (
            patch.object(p, "_speaker_embeddings", return_value=speaker_embs),
            patch("pipeline.isolate_vocals", side_effect=lambda path, **kw: path),
        ):
            inference, per_speaker = p.identify_show(tmp_wav)

        assert inference is None
        assert per_speaker["SPEAKER_00"][0].actor_id == 1
        assert mock_db.search_embeddings.call_count == 2   # no re-rank calls

    def test_single_speaker_cannot_infer_show(self, tmp_wav):
        mock_db = MagicMock()
        mock_db.search_embeddings.side_effect = [[_row(1, 0.90)]]
        mock_db.get_shows_for_actors.return_value = {1: {100}}

        p = self._make_pipeline(mock_db)

        with (
            patch.object(p, "_speaker_embeddings", return_value={"SPEAKER_00": _emb(0)}),
            patch("pipeline.isolate_vocals", side_effect=lambda path, **kw: path),
        ):
            inference, per_speaker = p.identify_show(tmp_wav)

        assert inference is None
        assert set(per_speaker) == {"SPEAKER_00"}

    def test_diarization_unavailable_falls_back_to_identify(self, tmp_wav):
        mock_db = MagicMock()
        p = VazamPipeline(db=mock_db, hf_token="", use_vad=False, use_diarization=False)

        with patch.object(p, "identify", return_value=[
            IdentificationResult(1, "Actor 1", "Natural Voice", 0.9)
        ]) as mock_identify:
            inference, per_speaker = p.identify_show(tmp_wav, isolate=False)

        assert inference is None
        assert per_speaker["SPEAKER_00"][0].actor_id == 1
        mock_identify.assert_called_once()


# ── db.get_shows_for_actors ───────────────────────────────────────────────────

def test_get_shows_for_actors(db: VazamDB):
    a1 = db.add_actor("Steve Blum")
    a2 = db.add_actor("Wendee Lee")
    bebop = db.add_show("Cowboy Bebop")
    other = db.add_show("Digimon")

    db.add_character("Spike Spiegel", bebop, a1)
    db.add_character("Faye Valentine", bebop, a2)
    db.add_character("TK", other, a2)

    result = db.get_shows_for_actors([a1, a2])
    assert result == {a1: {bebop}, a2: {bebop, other}}


def test_get_shows_for_actors_empty(db: VazamDB):
    assert db.get_shows_for_actors([]) == {}
