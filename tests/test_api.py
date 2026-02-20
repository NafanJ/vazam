"""
test_api.py — integration tests for the FastAPI endpoints (api.py)

Uses the api_client fixture from conftest.py, which mocks all ML dependencies
and spins up a TestClient backed by a fresh temp SQLite database.
"""

from __future__ import annotations

import io
import wave

import pytest

from tests.conftest import make_wav_bytes


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "index_size" in body
    assert "db_embeddings" in body


# ── /actors ───────────────────────────────────────────────────────────────────

def test_create_actor(api_client):
    resp = api_client.post("/actors", json={"name": "Steve Blum", "bio": "VA"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Steve Blum"
    assert "id" in body


def test_create_actor_minimal(api_client):
    resp = api_client.post("/actors", json={"name": "Test Actor"})
    assert resp.status_code == 201


def test_list_actors_empty(api_client):
    resp = api_client.get("/actors")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_actors(api_client):
    api_client.post("/actors", json={"name": "Alice"})
    api_client.post("/actors", json={"name": "Bob"})
    resp = api_client.get("/actors")
    names = [a["name"] for a in resp.json()]
    assert "Alice" in names
    assert "Bob" in names


def test_get_actor_profile(api_client):
    create_resp = api_client.post("/actors", json={"name": "Tara Strong"})
    actor_id = create_resp.json()["id"]

    resp = api_client.get(f"/actors/{actor_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Tara Strong"
    assert "filmography" in body
    assert isinstance(body["filmography"], list)


def test_get_actor_not_found(api_client):
    resp = api_client.get("/actors/99999")
    assert resp.status_code == 404


# ── /actors/{id}/embeddings ───────────────────────────────────────────────────

def _wav_upload(duration_s: float = 1.0) -> tuple[str, bytes, str]:
    return ("audio", make_wav_bytes(duration_s), "test.wav")


def test_add_embedding_updates_index(api_client):
    actor_id = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]

    initial_health = api_client.get("/health").json()
    initial_size   = initial_health["index_size"]

    resp = api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "embedding_id" in body
    assert body["index_size"] == initial_size + 1


def test_add_embedding_actor_not_found(api_client):
    resp = api_client.post(
        "/actors/99999/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )
    assert resp.status_code == 404


def test_add_multiple_embeddings_same_actor(api_client):
    actor_id = api_client.post("/actors", json={"name": "Seth MacFarlane"}).json()["id"]

    for label in ("Natural Voice", "Peter Griffin", "Stewie Griffin"):
        resp = api_client.post(
            f"/actors/{actor_id}/embeddings",
            files={"audio": _wav_upload()},
            data={"voice_label": label},
        )
        assert resp.status_code == 201

    health = api_client.get("/health").json()
    assert health["index_size"] >= 3


# ── /shows ────────────────────────────────────────────────────────────────────

def test_create_show(api_client):
    resp = api_client.post("/shows", json={
        "title": "Cowboy Bebop",
        "media_type": "anime",
        "year": 1998,
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "Cowboy Bebop"
    assert body["year"] == 1998


def test_create_show_invalid_media_type(api_client):
    resp = api_client.post("/shows", json={"title": "Test", "media_type": "podcast"})
    assert resp.status_code == 422


def test_list_shows_empty(api_client):
    resp = api_client.get("/shows")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_show_with_cast(api_client):
    actor_id  = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]
    show_resp = api_client.post("/shows", json={"title": "Cowboy Bebop", "media_type": "anime", "year": 1998})
    show_id   = show_resp.json()["id"]

    resp = api_client.get(f"/shows/{show_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Cowboy Bebop"
    assert "cast" in body


def test_get_show_not_found(api_client):
    resp = api_client.get("/shows/99999")
    assert resp.status_code == 404


def test_search_shows(api_client):
    api_client.post("/shows", json={"title": "Cowboy Bebop",   "media_type": "anime"})
    api_client.post("/shows", json={"title": "Dragon Ball Z",  "media_type": "anime"})

    resp = api_client.get("/shows/search?q=cowboy")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["title"] == "Cowboy Bebop"


# ── /identify ─────────────────────────────────────────────────────────────────

def test_identify_no_embeddings_returns_503(api_client):
    resp = api_client.post(
        "/identify",
        files={"audio": _wav_upload()},
        data={"isolate": "false"},
    )
    assert resp.status_code == 503


def test_identify_returns_results(api_client):
    # Seed one actor + embedding
    actor_id = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )

    resp = api_client.post(
        "/identify",
        files={"audio": _wav_upload()},
        data={"isolate": "false", "top_k": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["actor_name"] == "Steve Blum"
    assert "confidence" in result
    assert "match_level" in result


def test_identify_show_aware_filter(api_client):
    """show_id filter should restrict results to actors in that show."""
    actor1_id = api_client.post("/actors", json={"name": "Actor One"}).json()["id"]
    actor2_id = api_client.post("/actors", json={"name": "Actor Two"}).json()["id"]
    show_id   = api_client.post("/shows", json={"title": "Show A", "media_type": "anime"}).json()["id"]

    # Only Actor One is in Show A (via the DB — we don't add a character here,
    # so actor_ids for this show will be empty and the filter returns [])
    for aid in (actor1_id, actor2_id):
        api_client.post(
            f"/actors/{aid}/embeddings",
            files={"audio": _wav_upload()},
            data={"voice_label": "Natural Voice"},
        )

    resp = api_client.post(
        "/identify",
        files={"audio": _wav_upload()},
        data={"isolate": "false", "show_id": str(show_id)},
    )
    assert resp.status_code == 200
    # Show has no characters → empty actor_ids → empty results
    assert resp.json()["results"] == []


# ── /identify/multi ───────────────────────────────────────────────────────────

def test_identify_multi_no_embeddings_returns_503(api_client):
    resp = api_client.post(
        "/identify/multi",
        files={"audio": _wav_upload()},
        data={"isolate": "false"},
    )
    assert resp.status_code == 503


def test_identify_multi_returns_speakers_dict(api_client):
    actor_id = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )

    resp = api_client.post(
        "/identify/multi",
        files={"audio": _wav_upload()},
        data={"isolate": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "speakers" in body
    assert isinstance(body["speakers"], dict)
    # Falls back to single-speaker (no HF_TOKEN in test env)
    assert "SPEAKER_00" in body["speakers"]


# ── /index/rebuild ────────────────────────────────────────────────────────────

def test_rebuild_index_empty(api_client):
    resp = api_client.post("/index/rebuild")
    assert resp.status_code == 200
    body = resp.json()
    assert body["embeddings_loaded"] == 0


def test_rebuild_index_with_embeddings(api_client):
    actor_id = api_client.post("/actors", json={"name": "Test Actor"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )

    resp = api_client.post("/index/rebuild")
    assert resp.status_code == 200
    assert resp.json()["embeddings_loaded"] == 1
