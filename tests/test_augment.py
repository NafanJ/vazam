"""
test_augment.py — channel augmentation (augment.py) and its use in the scraper

The DSP functions run on real tensors (no models, no network); the scraper's
centroid-folding is tested with the embedding model mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from augment import NOISE_SNR_DB, add_noise, augment_speech, band_limit, reverb
from consensus import ConsensusResult, VideoSpeaker
from scrape_audio import augmented_embedding


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def speech() -> torch.Tensor:
    """1s of 440 Hz tone at 16 kHz — a stand-in for voiced speech."""
    t = torch.arange(16000, dtype=torch.float32) / 16000
    return (0.5 * torch.sin(2 * torch.pi * 440 * t)).reshape(1, -1)


# ── DSP properties ────────────────────────────────────────────────────────────

def test_augmentations_preserve_shape(speech):
    for variant in augment_speech(speech):
        assert variant.shape == speech.shape


def test_augmentations_change_the_signal(speech):
    for variant in augment_speech(speech):
        assert not torch.allclose(variant, speech)


def test_augmentations_are_deterministic(speech):
    a = augment_speech(speech)
    b = augment_speech(speech)
    for va, vb in zip(a, b):
        assert torch.equal(va, vb)


def test_add_noise_hits_target_snr(speech):
    noisy = add_noise(speech)
    noise = noisy - speech
    snr_db = 10 * torch.log10(speech.pow(2).mean() / noise.pow(2).mean())
    assert snr_db.item() == pytest.approx(NOISE_SNR_DB, abs=0.5)


def test_add_noise_on_silence_stays_silent():
    silent = torch.zeros(1, 16000)
    assert torch.allclose(add_noise(silent), silent, atol=1e-5)


def test_band_limit_attenuates_out_of_band(speech):
    # A 50 Hz tone is below the 200 Hz highpass → should lose most energy
    t = torch.arange(16000, dtype=torch.float32) / 16000
    low_tone = (0.5 * torch.sin(2 * torch.pi * 50 * t)).reshape(1, -1)

    in_band  = band_limit(speech)     # 440 Hz, inside the band
    out_band = band_limit(low_tone)

    assert in_band.pow(2).mean() > 0.5 * speech.pow(2).mean()
    assert out_band.pow(2).mean() < 0.2 * low_tone.pow(2).mean()


def test_reverb_smears_energy(speech):
    """An impulse followed by silence should have energy spread into the
    silent tail after reverb."""
    impulse = torch.zeros(1, 16000)
    impulse[0, 0] = 1.0
    wet = reverb(impulse)
    tail_energy = wet[:, 1000:].pow(2).sum()
    assert tail_energy > 0


# ── augmented_embedding (scraper integration) ─────────────────────────────────

def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


def test_augmented_embedding_folds_variants_into_centroid(speech):
    clean = _unit(1)
    member = VideoSpeaker("v1", "SPEAKER_00", clean, 30.0)
    result = ConsensusResult(
        embedding=clean, quality_score=1.0, speech_seconds=30.0,
        n_videos=2, video_ids=["v1"], members=[member],
    )
    speech_by_key = {("v1", "SPEAKER_00"): speech}

    aug_emb = _unit(2)
    with patch("pipeline.get_embedding", return_value=aug_emb) as mock_embed:
        out = augmented_embedding(result, speech_by_key, device="cpu")

    # 3 augmented variants embedded for the single member
    assert mock_embed.call_count == 3
    # Centroid of clean + 3×aug, re-normalized
    expected = clean + 3 * aug_emb
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_augmented_embedding_skips_members_without_audio():
    clean = _unit(1)
    member = VideoSpeaker("v1", "SPEAKER_00", clean, 30.0)
    result = ConsensusResult(
        embedding=clean, quality_score=1.0, speech_seconds=30.0,
        n_videos=2, video_ids=["v1"], members=[member],
    )

    with patch("pipeline.get_embedding") as mock_embed:
        out = augmented_embedding(result, {}, device="cpu")

    mock_embed.assert_not_called()
    np.testing.assert_allclose(out, clean, atol=1e-5)
