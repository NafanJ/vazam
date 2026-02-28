"""
conftest.py — shared fixtures for the Vazam test suite

Heavy ML dependencies (SpeechBrain, Demucs, pyannote) are mocked so tests run
fast without GPU/model-download requirements.  The Supabase client is replaced
by an in-memory fake (_FakeSupabase) so no network connection is needed.
"""

from __future__ import annotations

import io
import wave
from collections import defaultdict
from typing import Any, Generator
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


# ── In-memory fake Supabase client ────────────────────────────────────────────

class _FakeQueryResult:
    def __init__(self, data: list, count: int | None = None) -> None:
        self.data  = data
        self.count = count


class _FakeStore:
    """Shared in-memory storage for a fake Supabase session."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)

    def next_id(self, table: str) -> int:
        self._counters[table] += 1
        return self._counters[table]


class _FakeQuery:
    """Minimal Supabase query-builder fake — supports insert/upsert/select."""

    def __init__(self, table: str, store: _FakeStore) -> None:
        self._table         = table
        self._store         = store
        self._op            = "select"
        self._payload: Any  = None
        self._on_conflict: str | None = None
        self._eq_filters:    list     = []
        self._ilike_filters: list     = []
        self._order_col: str | None   = None
        self._range: tuple | None     = None
        self._count_mode: str | None  = None

    def select(self, *_args: Any, count: str | None = None) -> "_FakeQuery":
        self._op         = "select"
        self._count_mode = count
        return self

    def insert(self, data: dict) -> "_FakeQuery":
        self._op      = "insert"
        self._payload = data
        return self

    def upsert(self, data: dict, on_conflict: str | None = None) -> "_FakeQuery":
        self._op          = "upsert"
        self._payload     = data
        self._on_conflict = on_conflict
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._eq_filters.append((col, val))
        return self

    def ilike(self, col: str, pattern: str) -> "_FakeQuery":
        self._ilike_filters.append((col, pattern))
        return self

    def order(self, col: str) -> "_FakeQuery":
        self._order_col = col
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        self._range = (start, end)
        return self

    def execute(self) -> _FakeQueryResult:
        rows = self._store.tables[self._table]

        if self._op == "insert":
            row = dict(self._payload)
            row.setdefault("id", self._store.next_id(self._table))
            rows.append(row)
            return _FakeQueryResult([row])

        if self._op == "upsert":
            row          = dict(self._payload)
            conflict_col = self._on_conflict
            existing     = None
            if conflict_col and row.get(conflict_col) is not None:
                for r in rows:
                    if r.get(conflict_col) == row[conflict_col]:
                        existing = r
                        break
            if existing:
                existing.update(row)
                return _FakeQueryResult([existing])
            row.setdefault("id", self._store.next_id(self._table))
            rows.append(row)
            return _FakeQueryResult([row])

        # select
        filtered = list(rows)
        for col, val in self._eq_filters:
            filtered = [r for r in filtered if r.get(col) == val]
        for col, pat in self._ilike_filters:
            p        = pat.lower().replace("%", "")
            filtered = [r for r in filtered if p in str(r.get(col, "")).lower()]
        if self._order_col:
            filtered = sorted(filtered, key=lambda r: r.get(self._order_col) or "")
        if self._range:
            s, e     = self._range
            filtered = filtered[s: e + 1]
        count = len(filtered) if self._count_mode else None
        return _FakeQueryResult(filtered, count=count)


class _FakeRpcQuery:
    def __init__(self, results: list) -> None:
        self._results = results

    def execute(self) -> _FakeQueryResult:
        return _FakeQueryResult(self._results)


class _FakeSupabase:
    """In-memory fake Supabase client.

    Supports .table(name), .rpc(func, params), and .set_rpc(func, handler).
    """

    def __init__(self) -> None:
        self._store        = _FakeStore()
        self._rpc_handlers: dict[str, Any] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(name, self._store)

    def rpc(self, func_name: str, params: dict) -> _FakeRpcQuery:
        handler = self._rpc_handlers.get(func_name)
        results = handler(params) if handler else []
        return _FakeRpcQuery(results)

    def set_rpc(self, func_name: str, handler: Any) -> None:
        self._rpc_handlers[func_name] = handler


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
    v   = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def db() -> Generator[VazamDB, None, None]:
    """VazamDB backed by an in-memory fake Supabase client."""
    fake_sb = _FakeSupabase()
    with (
        patch("db.create_client", return_value=fake_sb),
        patch.dict("os.environ", {"SUPABASE_URL": "http://test", "SUPABASE_KEY": "test"}),
    ):
        yield VazamDB()


# ── API test client with mocked pipeline ─────────────────────────────────────

def _make_fake_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(192).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """
    TestClient for api.py with all ML dependencies and Supabase mocked.

    Patches:
      - pipeline.isolate_vocals        → returns input path unchanged
      - pipeline._load_embedding_model → fake encoder returning a fixed embedding
      - torchaudio.load                → returns a (1, 16000) zero tensor
      - db.create_client               → in-memory _FakeSupabase
    """
    import torch

    fake_signal  = (torch.zeros(1, 16000), 16000)
    fake_encoder = MagicMock()
    fake_encoder.encode_batch.return_value = torch.tensor(
        _make_fake_embedding(0)
    ).reshape(1, 1, -1)

    fake_sb = _FakeSupabase()

    # Wire match_embeddings RPC to the in-memory store so /identify works
    def _match_embeddings(params: dict) -> list[dict]:
        top_k          = params.get("top_k", 5)
        show_id_filter = params.get("show_id_filter")

        embeddings = list(fake_sb._store.tables["vazam_embeddings"])
        actors     = {
            a["id"]: a["name"]
            for a in fake_sb._store.tables["vazam_actors"]
        }

        if show_id_filter is not None:
            chars      = fake_sb._store.tables["vazam_characters"]
            valid_ids  = {c["actor_id"] for c in chars if c.get("show_id") == show_id_filter}
            embeddings = [e for e in embeddings if e.get("actor_id") in valid_ids]

        results = []
        for emb_row in embeddings[:top_k]:
            aid = emb_row["actor_id"]
            results.append({
                "actor_id":   aid,
                "actor_name": actors.get(aid, "Unknown"),
                "voice_label": emb_row.get("voice_label", "Natural Voice"),
                "similarity": 0.95,
            })
        return results

    fake_sb.set_rpc("match_embeddings", _match_embeddings)

    with (
        patch("pipeline.isolate_vocals", side_effect=lambda p, **kw: p),
        patch("pipeline._load_embedding_model", return_value=fake_encoder),
        patch("torchaudio.load", return_value=fake_signal),
        patch("db.create_client", return_value=fake_sb),
        patch.dict("os.environ", {
            "SUPABASE_URL": "http://test",
            "SUPABASE_KEY": "test",
            "HF_TOKEN": "",
        }),
    ):
        import importlib
        import api as api_module
        importlib.reload(api_module)

        with TestClient(api_module.app) as client:
            yield client
