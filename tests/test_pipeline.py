"""
test_pipeline.py — unit tests for pipeline.py

All ML model calls are mocked. Tests cover:
  - IdentificationResult data class
  - SpeakerSegment / merge_speaker_segments
  - extract_speech_audio
  - VazamPipeline.identify / identify_multi (using a mocked VazamDB)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pipeline import (
    CONFIDENT_THRESHOLD,
    EMBEDDING_DIM,
    IdentificationResult,
    SpeakerSegment,
    VazamPipeline,
    extract_speech_audio,
    merge_speaker_segments,
    split_windows,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rand_emb(seed: int = 0, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _norm(rng.standard_normal(dim).astype("float32"))


def _make_db_rows(*actor_names: str, similarity: float = 0.95) -> list[dict]:
    """Build fake search_embeddings rows for the given actor names."""
    return [
        {
            "actor_id":   i + 1,
            "actor_name": name,
            "voice_label": "Natural Voice",
            "similarity": similarity,
        }
        for i, name in enumerate(actor_names)
    ]


# ── IdentificationResult ──────────────────────────────────────────────────────

class TestIdentificationResult:
    def test_confident_flag(self):
        r = IdentificationResult(1, "A", "B", CONFIDENT_THRESHOLD)
        assert r.confident is True

    def test_not_confident(self):
        r = IdentificationResult(1, "A", "B", CONFIDENT_THRESHOLD - 0.01)
        assert r.confident is False

    def test_possible_flag(self):
        r = IdentificationResult(1, "A", "B", 0.55)
        assert r.possible is True
        assert r.confident is False

    def test_to_dict_keys(self):
        r = IdentificationResult(1, "Alice", "Raven", 0.82)
        d = r.to_dict()
        assert d["actor_id"]       == 1
        assert d["actor_name"]     == "Alice"
        assert d["character_name"] == "Raven"
        assert d["confidence"]     == round(0.82, 4)
        assert d["match_level"]    == "confident"

    def test_to_dict_match_level_possible(self):
        r = IdentificationResult(1, "A", "B", 0.60)
        assert r.to_dict()["match_level"] == "possible"

    def test_to_dict_match_level_none(self):
        r = IdentificationResult(1, "A", "B", 0.30)
        assert r.to_dict()["match_level"] == "none"

    def test_failed_window_verification_demotes_to_possible(self):
        r = IdentificationResult(1, "A", "B", 0.95, window_agreement=1 / 3)
        assert r.confident is False
        assert r.possible is True
        assert r.to_dict()["match_level"] == "possible"

    def test_majority_window_agreement_stays_confident(self):
        r = IdentificationResult(1, "A", "B", 0.95, window_agreement=2 / 3)
        assert r.confident is True
        assert r.to_dict()["match_level"] == "confident"

    def test_no_window_data_behaves_as_before(self):
        r = IdentificationResult(1, "A", "B", 0.95)
        assert r.confident is True
        assert r.to_dict()["window_agreement"] is None


# ── SpeakerSegment ────────────────────────────────────────────────────────────

class TestSpeakerSegment:
    def test_duration(self):
        seg = SpeakerSegment("SPEAKER_00", 1.0, 4.5)
        assert seg.duration == pytest.approx(3.5)


# ── merge_speaker_segments ────────────────────────────────────────────────────

class TestMergeSpeakerSegments:
    def test_empty(self):
        assert merge_speaker_segments([]) == []

    def test_single(self):
        seg    = SpeakerSegment("A", 0.0, 2.0)
        merged = merge_speaker_segments([seg])
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end   == 2.0

    def test_merge_same_speaker(self):
        segs = [
            SpeakerSegment("A", 0.0, 2.0),
            SpeakerSegment("A", 2.5, 5.0),
        ]
        merged = merge_speaker_segments(segs)
        assert len(merged) == 1
        assert merged[0].end == 5.0

    def test_no_merge_different_speakers(self):
        segs = [
            SpeakerSegment("A", 0.0, 2.0),
            SpeakerSegment("B", 2.0, 4.0),
        ]
        merged = merge_speaker_segments(segs)
        assert len(merged) == 2

    def test_alternating_speakers(self):
        segs = [
            SpeakerSegment("A", 0.0, 1.0),
            SpeakerSegment("B", 1.0, 2.0),
            SpeakerSegment("A", 2.0, 3.0),
        ]
        # A-B-A → three segments (A is not re-merged across B)
        assert len(merge_speaker_segments(segs)) == 3


# ── extract_speech_audio ──────────────────────────────────────────────────────

class TestExtractSpeechAudio:
    def _fake_load(self, path):
        return torch.zeros(1, 32000), 16000   # 2s of silence at 16 kHz

    def test_extracts_specified_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            result = extract_speech_audio(tmp_wav, [(0.0, 1.0)])
            assert result.shape == (1, 16000)

    def test_falls_back_to_full_audio_when_no_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            result = extract_speech_audio(tmp_wav, [])
            assert result.shape[1] == 32000   # full 2s

    def test_concatenates_multiple_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            result = extract_speech_audio(tmp_wav, [(0.0, 0.5), (1.0, 1.5)])
            assert result.shape == (1, 16000)   # 0.5s + 0.5s = 1s × 16000


# ── split_windows ─────────────────────────────────────────────────────────────

class TestSplitWindows:
    def test_three_overlapping_windows_cover_signal(self):
        signal  = torch.zeros(1, 160000)   # 10s at 16 kHz
        windows = split_windows(signal)
        assert len(windows) == 3
        assert all(w.shape == (1, 80000) for w in windows)   # half-length each

    def test_windows_span_start_to_end(self):
        # Distinct values let us verify window positions
        signal  = torch.arange(160000, dtype=torch.float32).reshape(1, -1)
        windows = split_windows(signal)
        assert windows[0][0, 0]   == 0.0        # first window starts at 0
        assert windows[-1][0, -1] == 159999.0   # last window ends at the end

    def test_too_short_returns_empty(self):
        signal = torch.zeros(1, 48000)   # 3s → 1.5s windows < 2s minimum
        assert split_windows(signal) == []

    def test_minimum_length_clip_windows(self):
        signal = torch.zeros(1, 64000)   # 4s → 2s windows, exactly the minimum
        assert len(split_windows(signal)) == 3


# ── VazamPipeline ─────────────────────────────────────────────────────────────

class TestVazamPipeline:

    def _make_pipeline(self, search_results: list[dict]) -> VazamPipeline:
        """Build a pipeline with a mocked db.search_embeddings."""
        mock_db = MagicMock()
        mock_db.search_embeddings.return_value = search_results
        return VazamPipeline(db=mock_db, hf_token="", use_vad=False, use_diarization=False)

    def _mock_audio(self, p: VazamPipeline, seconds: float = 1.0):
        """Patch audio loading + embedding. 1s of speech is too short to
        window, so verification is skipped unless a longer clip is given."""
        speech = torch.zeros(1, int(seconds * 16000))
        return (
            patch.object(p, "_speech_tensor", return_value=speech),
            patch("pipeline.get_embedding", return_value=_rand_emb(0)),
        )

    def test_identify_returns_correct_actor(self, tmp_wav):
        rows = _make_db_rows("Steve Blum")
        p    = self._make_pipeline(rows)

        audio, emb = self._mock_audio(p)
        with audio, emb:
            results = p.identify(tmp_wav, top_k=1)

        assert len(results) == 1
        assert results[0].actor_name  == "Steve Blum"
        assert results[0].confidence  == pytest.approx(0.95)

    def test_identify_empty_returns_empty(self, tmp_wav):
        p = self._make_pipeline([])
        audio, emb = self._mock_audio(p)
        with audio, emb:
            results = p.identify(tmp_wav)
        assert results == []

    def test_identify_passes_show_id_to_db(self, tmp_wav):
        p = self._make_pipeline([])

        audio, emb = self._mock_audio(p)
        with audio, emb:
            p.identify(tmp_wav, show_id=42)

        p.db.search_embeddings.assert_called_once()
        _, kwargs = p.db.search_embeddings.call_args
        assert kwargs.get("show_id") == 42

    def test_identify_multi_falls_back_to_single(self, tmp_wav):
        """Without HF_TOKEN, identify_multi uses the single-speaker path."""
        rows = _make_db_rows("Actor 1")
        p    = self._make_pipeline(rows)

        with patch.object(p, "identify", return_value=[
            IdentificationResult(1, "Actor 1", "Natural Voice", 0.95)
        ]) as mock_identify:
            result = p.identify_multi(tmp_wav, isolate=False)

        assert "SPEAKER_00" in result
        mock_identify.assert_called_once()

    def test_identify_result_fields(self, tmp_wav):
        rows = [{"actor_id": 7, "actor_name": "Tara Strong", "voice_label": "Raven", "similarity": 0.88}]
        p    = self._make_pipeline(rows)

        audio, emb = self._mock_audio(p)
        with audio, emb:
            results = p.identify(tmp_wav)

        assert results[0].actor_id      == 7
        assert results[0].character_name == "Raven"
        assert results[0].confidence    == pytest.approx(0.88)


# ── Multi-window verification in identify() ──────────────────────────────────

class TestWindowVerification:

    def _make_verifying_pipeline(self, per_call_rows: list[list[dict]]) -> VazamPipeline:
        """Pipeline whose db returns per_call_rows[i] for the i-th search call
        (first call = main ranking, subsequent calls = one per window)."""
        mock_db = MagicMock()
        mock_db.search_embeddings.side_effect = per_call_rows
        return VazamPipeline(db=mock_db, hf_token="", use_vad=False, use_diarization=False)

    def _identify(self, p: VazamPipeline, tmp_wav: str, **kwargs):
        speech = torch.zeros(1, 128000)   # 8s → windowable
        with (
            patch.object(p, "_speech_tensor", return_value=speech),
            patch("pipeline.get_embedding", return_value=_rand_emb(0)),
        ):
            return p.identify(tmp_wav, **kwargs)

    def test_consistent_winner_stays_confident(self, tmp_wav):
        main = _make_db_rows("Steve Blum")            # actor_id 1, sim 0.95
        win  = [{"actor_id": 1}]
        p    = self._make_verifying_pipeline([main, win, win, win])

        results = self._identify(p, tmp_wav)

        assert results[0].window_agreement == pytest.approx(1.0)
        assert results[0].confident is True
        assert p.db.search_embeddings.call_count == 4   # main + 3 windows

    def test_majority_agreement_is_enough(self, tmp_wav):
        main  = _make_db_rows("Steve Blum")
        mine  = [{"actor_id": 1}]
        other = [{"actor_id": 2}]
        p     = self._make_verifying_pipeline([main, mine, mine, other])

        results = self._identify(p, tmp_wav)

        assert results[0].window_agreement == pytest.approx(2 / 3)
        assert results[0].to_dict()["match_level"] == "confident"

    def test_failed_verification_demotes(self, tmp_wav):
        """High similarity but another actor wins every window → 'possible'."""
        main  = _make_db_rows("Steve Blum")
        other = [{"actor_id": 2}]
        p     = self._make_verifying_pipeline([main, other, other, other])

        results = self._identify(p, tmp_wav)

        assert results[0].confidence == pytest.approx(0.95)
        assert results[0].window_agreement == 0.0
        assert results[0].confident is False
        assert results[0].to_dict()["match_level"] == "possible"

    def test_verify_false_skips_windows(self, tmp_wav):
        main = _make_db_rows("Steve Blum")
        p    = self._make_verifying_pipeline([main])

        results = self._identify(p, tmp_wav, verify=False)

        assert results[0].window_agreement is None
        assert p.db.search_embeddings.call_count == 1

    def test_short_clip_skips_verification(self, tmp_wav):
        main = _make_db_rows("Steve Blum")
        p    = self._make_verifying_pipeline([main])

        speech = torch.zeros(1, 16000)   # 1s — too short to window
        with (
            patch.object(p, "_speech_tensor", return_value=speech),
            patch("pipeline.get_embedding", return_value=_rand_emb(0)),
        ):
            results = p.identify(tmp_wav)

        assert results[0].window_agreement is None
        assert results[0].confident is True
        assert p.db.search_embeddings.call_count == 1

    def test_window_searches_respect_show_filter(self, tmp_wav):
        main = _make_db_rows("Steve Blum")
        win  = [{"actor_id": 1}]
        p    = self._make_verifying_pipeline([main, win, win, win])

        self._identify(p, tmp_wav, show_id=42)

        for call in p.db.search_embeddings.call_args_list:
            assert call.kwargs.get("show_id") == 42
