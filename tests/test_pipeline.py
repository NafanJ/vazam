"""
test_pipeline.py — unit tests for pipeline.py

All ML model calls are mocked. Tests focus on the logic in:
  - EmbeddingIndex (FAISS wrapper)
  - SpeakerSegment / merge_speaker_segments
  - extract_speech_audio
  - VazamPipeline.identify / identify_multi
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pipeline import (
    CONFIDENT_THRESHOLD,
    EMBEDDING_DIM,
    EmbeddingIndex,
    IdentificationResult,
    SpeakerSegment,
    VazamPipeline,
    extract_speech_audio,
    merge_speaker_segments,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rand_emb(seed: int = 0, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _norm(rng.standard_normal(dim).astype("float32"))


# ── EmbeddingIndex ────────────────────────────────────────────────────────────

class TestEmbeddingIndex:
    def test_empty_index_search_returns_empty(self):
        idx = EmbeddingIndex()
        assert idx.search(_rand_emb()) == []

    def test_size_zero_on_init(self):
        assert EmbeddingIndex().size == 0

    def test_add_single_entry(self):
        idx = EmbeddingIndex()
        emb = _rand_emb(0)
        idx.add(1, "Steve Blum", "Spike Spiegel", emb)
        assert idx.size == 1

    def test_search_returns_correct_actor(self):
        idx = EmbeddingIndex()
        emb = _rand_emb(0)
        idx.add(1, "Steve Blum", "Spike Spiegel", emb)

        # Searching with the same vector should return similarity ~1.0
        results = idx.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0].actor_name == "Steve Blum"
        assert results[0].confidence > 0.99

    def test_search_top_k(self):
        idx = EmbeddingIndex()
        for i in range(5):
            idx.add(i, f"Actor {i}", "Role", _rand_emb(i))

        results = idx.search(_rand_emb(0), top_k=3)
        assert len(results) == 3

    def test_search_top_k_capped_at_index_size(self):
        idx = EmbeddingIndex()
        idx.add(1, "Only Actor", "Role", _rand_emb(0))
        results = idx.search(_rand_emb(0), top_k=10)
        assert len(results) == 1

    def test_build_from_list(self):
        entries = [
            (i, f"Actor {i}", "Role", _rand_emb(i))
            for i in range(4)
        ]
        idx = EmbeddingIndex()
        idx.build_from_list(entries)
        assert idx.size == 4

    def test_build_from_empty_list(self):
        idx = EmbeddingIndex()
        idx.build_from_list([])
        assert idx.size == 0
        assert idx.search(_rand_emb()) == []

    def test_closest_vector_is_ranked_first(self):
        """The vector most similar to the query should come back first."""
        target = _rand_emb(99)
        noise  = _rand_emb(1)

        idx = EmbeddingIndex()
        idx.add(1, "Wrong Actor", "Role", noise)
        idx.add(2, "Right Actor", "Role", target)

        results = idx.search(target, top_k=2)
        assert results[0].actor_name == "Right Actor"

    def test_save_and_load(self, tmp_path):
        idx = EmbeddingIndex()
        emb = _rand_emb(0)
        idx.add(1, "Steve Blum", "Spike Spiegel", emb)

        save_path = str(tmp_path / "index.faiss")
        idx.save(save_path)

        idx2 = EmbeddingIndex()
        idx2.load(save_path, [(1, "Steve Blum", "Spike Spiegel")])
        assert idx2.size == 1
        results = idx2.search(emb, top_k=1)
        assert results[0].actor_name == "Steve Blum"


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
        assert d["actor_id"] == 1
        assert d["actor_name"] == "Alice"
        assert d["character_name"] == "Raven"
        assert d["confidence"] == round(0.82, 4)
        assert d["match_level"] == "confident"

    def test_to_dict_match_level_possible(self):
        r = IdentificationResult(1, "A", "B", 0.60)
        assert r.to_dict()["match_level"] == "possible"

    def test_to_dict_match_level_none(self):
        r = IdentificationResult(1, "A", "B", 0.30)
        assert r.to_dict()["match_level"] == "none"


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
        seg = SpeakerSegment("A", 0.0, 2.0)
        merged = merge_speaker_segments([seg])
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0

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
        merged = merge_speaker_segments(segs)
        # A-B-A → three segments (A is not re-merged across B)
        assert len(merged) == 3


# ── extract_speech_audio ──────────────────────────────────────────────────────

class TestExtractSpeechAudio:
    def _fake_load(self, path):
        return torch.zeros(1, 32000), 16000   # 2s of silence at 16 kHz

    def test_extracts_specified_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            segments = [(0.0, 1.0)]  # first 1 second
            result = extract_speech_audio(tmp_wav, segments)
            assert result.shape == (1, 16000)

    def test_falls_back_to_full_audio_when_no_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            result = extract_speech_audio(tmp_wav, [])
            assert result.shape[1] == 32000   # full 2s

    def test_concatenates_multiple_segments(self, tmp_wav):
        with patch("torchaudio.load", side_effect=self._fake_load):
            segments = [(0.0, 0.5), (1.0, 1.5)]   # 0.5s + 0.5s = 1s total
            result = extract_speech_audio(tmp_wav, segments)
            assert result.shape == (1, 16000)   # 1s × 16000 = 16000 samples


# ── VazamPipeline ─────────────────────────────────────────────────────────────

class TestVazamPipeline:
    def _make_pipeline(self, embeddings: dict[int, np.ndarray]) -> VazamPipeline:
        """Build a pipeline with a pre-loaded index and mocked embed_file."""
        p = VazamPipeline(hf_token="", use_vad=False, use_diarization=False)
        entries = [
            (actor_id, f"Actor {actor_id}", "Natural Voice", emb)
            for actor_id, emb in embeddings.items()
        ]
        p.load_index(entries)
        return p

    def test_identify_returns_correct_actor(self, tmp_wav):
        target_emb = _rand_emb(0)
        p = self._make_pipeline({1: target_emb, 2: _rand_emb(1)})

        with patch.object(p, "embed_file", return_value=target_emb):
            results = p.identify(tmp_wav, top_k=2)

        assert results[0].actor_id == 1
        assert results[0].confidence > 0.99

    def test_identify_with_actor_id_filter(self, tmp_wav):
        target_emb = _rand_emb(0)
        p = self._make_pipeline({1: target_emb, 2: _rand_emb(1)})

        with patch.object(p, "embed_file", return_value=target_emb):
            # Restrict to actor_id=2 only — actor 1 should be excluded
            results = p.identify(tmp_wav, actor_ids=[2])

        assert all(r.actor_id == 2 for r in results)

    def test_identify_empty_index_returns_empty(self, tmp_wav):
        p = VazamPipeline(hf_token="", use_vad=False, use_diarization=False)
        with patch.object(p, "embed_file", return_value=_rand_emb(0)):
            results = p.identify(tmp_wav)
        assert results == []

    def test_identify_multi_falls_back_to_single(self, tmp_wav):
        """Without a HF_TOKEN, identify_multi should use single-speaker path."""
        target_emb = _rand_emb(0)
        p = self._make_pipeline({1: target_emb})

        with patch.object(p, "identify", return_value=[
            IdentificationResult(1, "Actor 1", "Natural Voice", 0.95)
        ]) as mock_identify:
            result = p.identify_multi(tmp_wav, isolate=False)

        assert "SPEAKER_00" in result
        mock_identify.assert_called_once()

    def test_add_entry_increments_index(self):
        p = VazamPipeline(hf_token="", use_vad=False, use_diarization=False)
        assert p.index.size == 0
        p.add_entry(1, "Actor", "Voice", _rand_emb(0))
        assert p.index.size == 1
