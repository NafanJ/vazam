"""
conftest.py — shared fixtures for the Vazam test suite

Heavy ML dependencies (SpeechBrain, Demucs, pyannote) are mocked at the
module level so tests run fast without GPU/model-download requirements.
"""

from __future__ import annotations

import io
import struct
import tempfile
import wave
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from db import VazamDB


# ── Audio helpers ─────────────────────────────────────────────────────────────

def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV file in memory."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def write_tmp_wav(duration_s: float = 1.0) -> str:
    """Write a temporary WAV file and return its path."""
    data = make_wav_bytes(duration_s)
    fd, path = tempfile.mkstemp(suffix=".wav")
    import os
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_wav(tmp_path) -> str:
    """A temporary silent WAV file."""
    path = tmp_path / "test.wav"
    path.write_bytes(make_wav_bytes())
    return str(path)


@pytest.fixture
def random_embedding() -> np.ndarray:
    """A random L2-normalized 192-dim embedding."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def db(tmp_path) -> Generator[VazamDB, None, None]:
    """Fresh in-memory-equivalent SQLite DB (temp file, auto-deleted)."""
    db_path = tmp_path / "test.db"
    instance = VazamDB(path=str(db_path))
    yield instance
    instance.close()


# ── API test client with mocked pipeline ─────────────────────────────────────

def _make_fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def api_client(tmp_path) -> Generator[TestClient, None, None]:
    """
    TestClient for api.py with all ML dependencies mocked.

    Patches:
      - pipeline.isolate_vocals   → returns input path unchanged
      - pipeline._load_embedding_model → fake encoder
      - pipeline._load_vad        → not called (HF_TOKEN not set)
      - pipeline._load_diarizer   → not called (HF_TOKEN not set)
      - torchaudio.load           → returns a (1, 16000) zero tensor
    """
    import torch
    import torchaudio

    fake_signal = (torch.zeros(1, 16000), 16000)

    # Build a fake EncoderClassifier
    fake_encoder = MagicMock()
    fake_encoder.encode_batch.return_value = torch.tensor(
        _make_fake_embedding(0)
    ).reshape(1, 1, -1)

    db_path = str(tmp_path / "api_test.db")

    with (
        patch("pipeline.isolate_vocals", side_effect=lambda p, **kw: p),
        patch("pipeline._load_embedding_model", return_value=fake_encoder),
        patch("torchaudio.load", return_value=fake_signal),
        patch.dict("os.environ", {"DB_PATH": db_path, "HF_TOKEN": ""}),
    ):
        # Import api *after* patches are applied so lifespan picks up mocks
        import importlib
        import api as api_module
        importlib.reload(api_module)

        with TestClient(api_module.app) as client:
            yield client
