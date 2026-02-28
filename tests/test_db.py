"""
test_db.py — unit tests for VazamDB (db.py)

Uses an in-memory _FakeSupabase client from conftest so no network connection
or real Supabase project is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from db import VazamDB


# ── Actors ────────────────────────────────────────────────────────────────────

def test_add_and_get_actor(db: VazamDB):
    actor_id = db.add_actor("Steve Blum", bio="American voice actor", image_url="http://img")
    actor = db.get_actor(actor_id)
    assert actor["name"] == "Steve Blum"
    assert actor["bio"] == "American voice actor"


def test_add_actor_upsert_on_anilist_id(db: VazamDB):
    id1 = db.add_actor("Steve Blum", anilist_id=999)
    id2 = db.add_actor("Steve Blum Updated", anilist_id=999)
    # Should update, not insert a new row
    assert id1 == id2
    actor = db.get_actor(id1)
    assert actor["name"] == "Steve Blum Updated"


def test_add_actor_no_anilist_id_inserts_new_rows(db: VazamDB):
    id1 = db.add_actor("Actor A")
    id2 = db.add_actor("Actor B")
    assert id1 != id2


def test_get_actor_not_found(db: VazamDB):
    assert db.get_actor(99999) is None


def test_list_actors_empty(db: VazamDB):
    assert db.list_actors() == []


def test_list_actors(db: VazamDB):
    db.add_actor("Alice")
    db.add_actor("Bob")
    actors = db.list_actors()
    names  = [a["name"] for a in actors]
    assert "Alice" in names
    assert "Bob" in names


# ── Shows ─────────────────────────────────────────────────────────────────────

def test_add_and_get_show(db: VazamDB):
    show_id = db.add_show("Cowboy Bebop", media_type="anime", year=1998)
    show    = db.get_show(show_id)
    assert show["title"] == "Cowboy Bebop"
    assert show["year"]  == 1998
    assert show["media_type"] == "anime"


def test_add_show_upsert_on_anilist_id(db: VazamDB):
    id1 = db.add_show("Cowboy Bebop", anilist_id=1)
    id2 = db.add_show("Cowboy Bebop (remaster)", anilist_id=1)
    assert id1 == id2


def test_get_show_not_found(db: VazamDB):
    assert db.get_show(99999) is None


def test_search_show(db: VazamDB):
    db.add_show("Cowboy Bebop", year=1998)
    db.add_show("Dragon Ball Z", year=1989)
    results = db.search_show("Cowboy")
    assert len(results) == 1
    assert results[0]["title"] == "Cowboy Bebop"


def test_search_show_case_insensitive_partial(db: VazamDB):
    db.add_show("Fullmetal Alchemist: Brotherhood", year=2009)
    results = db.search_show("alchemist")
    assert len(results) == 1


def test_search_show_no_match(db: VazamDB):
    db.add_show("Cowboy Bebop")
    assert db.search_show("Naruto") == []


# ── Characters ────────────────────────────────────────────────────────────────

def test_add_character_links_actor_and_show(db: VazamDB):
    actor_id = db.add_actor("Steve Blum")
    show_id  = db.add_show("Cowboy Bebop")
    char_id  = db.add_character("Spike Spiegel", show_id, actor_id)
    assert char_id > 0

    filmography = db.get_actor_filmography(actor_id)
    assert len(filmography) == 1
    assert filmography[0]["character_name"] == "Spike Spiegel"
    assert filmography[0]["show_title"] == "Cowboy Bebop"


def test_filmography_multiple_roles(db: VazamDB):
    actor_id = db.add_actor("Tara Strong")
    show1    = db.add_show("Teen Titans")
    show2    = db.add_show("My Little Pony")
    db.add_character("Raven",            show1, actor_id)
    db.add_character("Twilight Sparkle", show2, actor_id)

    filmography = db.get_actor_filmography(actor_id)
    names = [r["character_name"] for r in filmography]
    assert "Raven" in names
    assert "Twilight Sparkle" in names


def test_get_actors_for_show(db: VazamDB):
    actor_id = db.add_actor("Steve Blum")
    show_id  = db.add_show("Cowboy Bebop")
    db.add_character("Spike Spiegel", show_id, actor_id)

    actors = db.get_actors_for_show(show_id)
    ids    = [a["id"] for a in actors]
    assert actor_id in ids


def test_get_actors_for_show_empty(db: VazamDB):
    show_id = db.add_show("Empty Show")
    assert db.get_actors_for_show(show_id) == []


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_add_embedding_and_count(db: VazamDB, random_embedding: np.ndarray):
    actor_id = db.add_actor("Steve Blum")
    assert db.get_embedding_count() == 0

    emb_id = db.add_embedding(actor_id, random_embedding, voice_label="Spike Spiegel")
    assert emb_id > 0
    assert db.get_embedding_count() == 1


def test_add_multiple_embeddings(db: VazamDB):
    rng = np.random.default_rng(0)
    for name in ("Alice", "Bob", "Carol"):
        aid = db.add_actor(name)
        v   = rng.standard_normal(192).astype("float32")
        db.add_embedding(aid, v / np.linalg.norm(v))

    assert db.get_embedding_count() == 3


def test_search_embeddings_returns_list(db: VazamDB, random_embedding: np.ndarray):
    # Without a real match_embeddings RPC, the fake returns []
    results = db.search_embeddings(random_embedding, top_k=5)
    assert isinstance(results, list)
