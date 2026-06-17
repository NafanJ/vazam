"""
test_api.py — integration tests for the FastAPI endpoints (api.py)

Uses the api_client fixture from conftest.py, which mocks all ML dependencies
and Supabase via an in-memory fake client.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import pytest

from tests.conftest import make_wav_bytes


# ── / (dashboard) ─────────────────────────────────────────────────────────────

def test_dashboard_served(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    # Serves the recording dashboard HTML (or the JSON fallback if it's absent).
    if "text/html" in resp.headers.get("content-type", ""):
        assert "Vaz" in resp.text and "Identify" in resp.text
    else:
        assert resp.json()["docs"] == "/docs"


def test_identify_stream_emits_progress_and_result(api_client):
    actor_id = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": ("s.wav", make_wav_bytes(), "audio/wav")},
        data={"voice_label": "Natural Voice"},
    )
    resp = api_client.post(
        "/identify/stream",
        files={"audio": ("c.wav", make_wav_bytes(), "audio/wav")},
        data={"isolate": "false"},
    )
    assert resp.status_code == 200
    events = [json.loads(l) for l in resp.text.strip().split("\n") if l]
    assert any("stage" in e for e in events)        # at least one progress line
    assert events[-1].get("done") is True           # final event carries the result
    assert isinstance(events[-1]["results"], list)


def test_enroll_adds_sample_to_fingerprint(api_client):
    actor_id = api_client.post("/actors", json={"name": "Mayumi Tanaka"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": ("a.wav", make_wav_bytes(), "audio/wav")},
        data={"voice_label": "Luffy"},
    )
    # Enroll a second clip into the same fingerprint.
    resp = api_client.post(
        "/enroll",
        files={"audio": ("b.wav", make_wav_bytes(), "audio/wav")},
        data={"actor_id": str(actor_id), "voice_label": "Luffy", "isolate": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["voice_label"] == "Luffy"
    assert body["samples"] == 2                       # now two reference clips

    voices = api_client.get("/voices").json()
    luffy = [v for v in voices if v["voice_label"] == "Luffy"][0]
    assert luffy["samples"] == 2 and luffy["actor_id"] == actor_id


def test_enroll_unknown_actor_404(api_client):
    resp = api_client.post(
        "/enroll",
        files={"audio": ("b.wav", make_wav_bytes(), "audio/wav")},
        data={"actor_id": "99999", "voice_label": "X", "isolate": "false"},
    )
    assert resp.status_code == 404


def test_characters_list_and_update(api_client):
    import api
    actor_id = api.db.add_actor("Mayumi Tanaka", anilist_id=1)
    show_id = api.db.add_show("One Piece", anilist_id=2)
    cid = api.db.add_character("Luffy", show_id, actor_id, image_url="http://x/old.png", anilist_id=3)

    listing = api_client.get("/characters").json()
    luffy = [c for c in listing if c["id"] == cid][0]
    assert luffy["name"] == "Luffy"
    assert luffy["actor_name"] == "Mayumi Tanaka"
    assert luffy["show_title"] == "One Piece"
    assert luffy["samples"] == 0

    # Edit image + occupation
    resp = api_client.patch(
        f"/characters/{cid}",
        json={"image_url": "http://x/new.png", "occupation": "Pirate Captain"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_url"] == "http://x/new.png"
    assert body["occupation"] == "Pirate Captain"

    detail = api_client.get(f"/characters/{cid}").json()
    assert detail["occupation"] == "Pirate Captain"
    assert "embeddings" in detail


def test_character_detail_lists_embedding_sources(api_client, random_embedding):
    import api
    actor_id = api.db.add_actor("Kazuya Nakai", anilist_id=10)
    show_id = api.db.add_show("One Piece", anilist_id=11)
    cid = api.db.add_character("Zoro", show_id, actor_id, anilist_id=12)
    api.db.add_embedding(
        actor_id, random_embedding, character_id=cid, voice_label="Zoro",
        source_url="https://youtu.be/abc", duration_s=43.0,
    )
    detail = api_client.get(f"/characters/{cid}").json()
    assert len(detail["embeddings"]) == 1
    assert detail["embeddings"][0]["source_url"] == "https://youtu.be/abc"
    assert detail["embeddings"][0]["voice_label"] == "Zoro"


def test_delete_embedding_removes_from_fingerprint(api_client, random_embedding):
    import api
    actor_id = api.db.add_actor("Kappei Yamaguchi", anilist_id=20)
    show_id = api.db.add_show("One Piece", anilist_id=21)
    cid = api.db.add_character("Usopp", show_id, actor_id, anilist_id=22)
    emb_id = api.db.add_embedding(
        actor_id, random_embedding, character_id=cid, voice_label="Usopp",
    )
    assert len(api_client.get(f"/characters/{cid}").json()["embeddings"]) == 1

    resp = api_client.delete(f"/embeddings/{emb_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == emb_id

    assert api_client.get(f"/characters/{cid}").json()["embeddings"] == []
    # Deleting again is a 404
    assert api_client.delete(f"/embeddings/{emb_id}").status_code == 404


def test_enroll_stores_playable_audio(api_client, random_embedding):
    import api
    actor_id = api.db.add_actor("Mayumi Tanaka", anilist_id=30)
    show_id = api.db.add_show("One Piece", anilist_id=31)
    cid = api.db.add_character("Luffy", show_id, actor_id, anilist_id=32)
    # Seed a character-linked embedding so the enroll inherits character_id.
    api.db.add_embedding(actor_id, random_embedding, character_id=cid, voice_label="Luffy")

    resp = api_client.post(
        "/enroll",
        files={"audio": ("c.wav", make_wav_bytes(), "audio/wav")},
        data={"actor_id": str(actor_id), "voice_label": "Luffy", "isolate": "false"},
    )
    assert resp.status_code == 200

    enrolled = [e for e in api_client.get(f"/characters/{cid}").json()["embeddings"]
                if e.get("audio_path")]
    assert len(enrolled) == 1
    eid = enrolled[0]["id"]

    audio = api_client.get(f"/embeddings/{eid}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/")
    assert len(audio.content) > 0

    # Deleting the embedding removes the stored audio too.
    assert api_client.delete(f"/embeddings/{eid}").status_code == 200
    assert api_client.get(f"/embeddings/{eid}/audio").status_code == 404


def test_identify_url_downloads_then_identifies(api_client, random_embedding, monkeypatch, tmp_path):
    import api
    import ytclip
    aid = api.db.add_actor("Steve Blum", anilist_id=40)
    api.db.add_embedding(aid, random_embedding, voice_label="Spike Spiegel")
    clip = tmp_path / "clip.mp3"; clip.write_bytes(make_wav_bytes())
    monkeypatch.setattr(ytclip, "download_clip", lambda url, out, max_seconds=30: str(clip))

    resp = api_client.post("/identify/url", json={"url": "https://youtu.be/x", "isolate": False})
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_identify_url_download_failure_is_400(api_client, random_embedding, monkeypatch):
    import api
    import ytclip
    aid = api.db.add_actor("Steve Blum", anilist_id=41)
    api.db.add_embedding(aid, random_embedding, voice_label="Spike Spiegel")

    def boom(*a, **k):
        raise ytclip.DownloadError("bad url")
    monkeypatch.setattr(ytclip, "download_clip", boom)

    resp = api_client.post("/identify/url", json={"url": "not-a-real-link"})
    assert resp.status_code == 400
    assert "bad url" in resp.json()["detail"]


def test_enroll_url_adds_clip_and_stores_audio(api_client, random_embedding, monkeypatch, tmp_path):
    import api
    import ytclip
    aid = api.db.add_actor("Mayumi Tanaka", anilist_id=42)
    show_id = api.db.add_show("One Piece", anilist_id=43)
    cid = api.db.add_character("Luffy", show_id, aid, anilist_id=44)
    api.db.add_embedding(aid, random_embedding, character_id=cid, voice_label="Luffy")
    clip = tmp_path / "clip.mp3"; clip.write_bytes(make_wav_bytes())
    monkeypatch.setattr(ytclip, "download_clip", lambda url, out, max_seconds=60: str(clip))

    resp = api_client.post(
        "/enroll/url",
        json={"url": "https://youtu.be/x", "actor_id": aid, "voice_label": "Luffy"},
    )
    assert resp.status_code == 200
    assert resp.json()["source_url"] == "https://youtu.be/x"

    embs = api_client.get(f"/characters/{cid}").json()["embeddings"]
    url_emb = [e for e in embs if e["audio_source"] == "dashboard-enroll-url"][0]
    assert url_emb["source_url"] == "https://youtu.be/x"
    assert url_emb["audio_path"]
    assert api_client.get(f"/embeddings/{url_emb['id']}/audio").status_code == 200


def test_fetch_url_returns_audio_bytes(api_client, monkeypatch, tmp_path):
    import ytclip
    clip = tmp_path / "clip.mp3"; clip.write_bytes(make_wav_bytes())
    monkeypatch.setattr(ytclip, "download_clip", lambda url, out, max_seconds=240: str(clip))

    resp = api_client.post("/fetch/url", json={"url": "https://youtu.be/x"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")
    assert len(resp.content) > 0


def test_fetch_url_download_failure_is_400(api_client, monkeypatch):
    import ytclip

    def boom(*a, **k):
        raise ytclip.DownloadError("nope")
    monkeypatch.setattr(ytclip, "download_clip", boom)

    resp = api_client.post("/fetch/url", json={"url": "bad"})
    assert resp.status_code == 400


def test_character_404(api_client):
    assert api_client.get("/characters/999999").status_code == 404
    assert api_client.patch("/characters/999999", json={"occupation": "x"}).status_code == 404


def test_basic_auth_helper(monkeypatch):
    import api
    # Disabled when no credentials are configured (local dev / tests).
    monkeypatch.setattr(api, "AUTH_USER", "")
    monkeypatch.setattr(api, "AUTH_PASS", "")
    assert api._basic_auth_ok("") is True

    # Enforced when set.
    monkeypatch.setattr(api, "AUTH_USER", "joe")
    monkeypatch.setattr(api, "AUTH_PASS", "secret")
    good = "Basic " + base64.b64encode(b"joe:secret").decode()
    bad = "Basic " + base64.b64encode(b"joe:wrong").decode()
    assert api._basic_auth_ok(good) is True
    assert api._basic_auth_ok(bad) is False
    assert api._basic_auth_ok("") is False
    assert api._basic_auth_ok("Bearer x") is False


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


# ── /identify/show ────────────────────────────────────────────────────────────

def test_identify_show_no_embeddings_returns_503(api_client):
    resp = api_client.post(
        "/identify/show",
        files={"audio": _wav_upload()},
        data={"isolate": "false"},
    )
    assert resp.status_code == 503


def test_identify_show_falls_back_without_diarization(api_client):
    """No HF_TOKEN in the test env → single-speaker fallback, show is null."""
    actor_id = api_client.post("/actors", json={"name": "Steve Blum"}).json()["id"]
    api_client.post(
        f"/actors/{actor_id}/embeddings",
        files={"audio": _wav_upload()},
        data={"voice_label": "Natural Voice"},
    )

    resp = api_client.post(
        "/identify/show",
        files={"audio": _wav_upload()},
        data={"isolate": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["show"] is None
    assert "SPEAKER_00" in body["speakers"]
    assert body["speakers"]["SPEAKER_00"][0]["actor_name"] == "Steve Blum"


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
