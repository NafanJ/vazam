"""
db.py — Vazam metadata & embedding database (Supabase + pgvector)

Schema (Supabase tables)
------------------------
  vazam_actors      — voice actor profiles
  vazam_shows       — animated shows / games
  vazam_characters  — character roles linking actors to shows
  vazam_embeddings  — per-character (or per-voice-style) vector(192) embeddings

Similarity search is handled by the match_embeddings() PostgreSQL function
(installed via migration) which uses pgvector's cosine distance operator (<=>).

Usage
-----
    db = VazamDB()           # reads SUPABASE_URL + SUPABASE_KEY from env
    actor_id = db.add_actor("Steve Blum", bio="...", image_url="...")
    show_id   = db.add_show("Cowboy Bebop", media_type="anime", year=1998)
    char_id   = db.add_character("Spike Spiegel", show_id, actor_id)
    emb_id    = db.add_embedding(actor_id, embedding, audio_source="ep01.wav")

    # Similarity search (calls match_embeddings SQL function via RPC)
    results   = db.search_embeddings(query_vec, top_k=5)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from supabase import create_client, Client


class VazamDB:
    """Supabase-backed metadata and embedding store for Vazam."""

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        self._client: Client = create_client(url, key)

    def close(self) -> None:
        """No-op — supabase-py manages its own connection pool."""

    # ------------------------------------------------------------------
    # Actors
    # ------------------------------------------------------------------

    def add_actor(
        self,
        name: str,
        bio: str = "",
        image_url: str = "",
        anilist_id: Optional[int] = None,
    ) -> int:
        """Insert or upsert an actor. Returns the actor's row id."""
        row = self._client.table("vazam_actors").upsert(
            {"name": name, "bio": bio, "image_url": image_url, "anilist_id": anilist_id},
            on_conflict="anilist_id",
        ).execute()
        return row.data[0]["id"]

    def get_actor(self, actor_id: int) -> Optional[dict]:
        row = self._client.table("vazam_actors").select("*").eq("id", actor_id).execute()
        return row.data[0] if row.data else None

    def get_actor_filmography(self, actor_id: int) -> list[dict]:
        """Return all characters voiced by an actor, with show info."""
        chars = (
            self._client.table("vazam_characters")
            .select("name, image_url, show_id")
            .eq("actor_id", actor_id)
            .order("name")
            .execute()
            .data
        )
        result = []
        for char in chars:
            show_id = char.get("show_id")
            show: dict = {}
            if show_id is not None:
                show_data = (
                    self._client.table("vazam_shows")
                    .select("title, media_type, year")
                    .eq("id", show_id)
                    .execute()
                    .data
                )
                show = show_data[0] if show_data else {}
            result.append({
                "character_name": char["name"],
                "show_title":     show.get("title"),
                "media_type":     show.get("media_type"),
                "year":           show.get("year"),
                "image_url":      char.get("image_url"),
            })
        return result

    def list_actors(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return (
            self._client.table("vazam_actors")
            .select("id, name, image_url")
            .order("name")
            .range(offset, offset + limit - 1)
            .execute()
            .data
        )

    def get_actors_for_show(self, show_id: int) -> list[dict]:
        """Return all actors who have a character in a given show."""
        chars = (
            self._client.table("vazam_characters")
            .select("actor_id")
            .eq("show_id", show_id)
            .execute()
            .data
        )
        actor_ids = list({r["actor_id"] for r in chars})
        actors = []
        for aid in actor_ids:
            actor_data = (
                self._client.table("vazam_actors")
                .select("id, name")
                .eq("id", aid)
                .execute()
                .data
            )
            if actor_data:
                actors.append(actor_data[0])
        return actors

    # ------------------------------------------------------------------
    # Shows
    # ------------------------------------------------------------------

    def add_show(
        self,
        title: str,
        media_type: str = "anime",
        year: Optional[int] = None,
        anilist_id: Optional[int] = None,
        image_url: str = "",
    ) -> int:
        row = self._client.table("vazam_shows").upsert(
            {
                "title": title,
                "media_type": media_type,
                "year": year,
                "anilist_id": anilist_id,
                "image_url": image_url,
            },
            on_conflict="anilist_id",
        ).execute()
        return row.data[0]["id"]

    def get_show(self, show_id: int) -> Optional[dict]:
        row = self._client.table("vazam_shows").select("*").eq("id", show_id).execute()
        return row.data[0] if row.data else None

    def search_show(self, title: str) -> list[dict]:
        return (
            self._client.table("vazam_shows")
            .select("*")
            .ilike("title", f"%{title}%")
            .order("year")
            .execute()
            .data
        )

    def list_shows(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return (
            self._client.table("vazam_shows")
            .select("id, title, media_type, year, image_url")
            .order("title")
            .range(offset, offset + limit - 1)
            .execute()
            .data
        )

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def add_character(
        self,
        name: str,
        show_id: Optional[int],
        actor_id: int,
        image_url: str = "",
        anilist_id: Optional[int] = None,
    ) -> int:
        row = self._client.table("vazam_characters").upsert(
            {
                "name": name,
                "show_id": show_id,
                "actor_id": actor_id,
                "image_url": image_url,
                "anilist_id": anilist_id,
            },
            on_conflict="anilist_id",
        ).execute()
        return row.data[0]["id"]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        actor_id: int,
        embedding: np.ndarray,
        character_id: Optional[int] = None,
        voice_label: str = "Natural Voice",
        audio_source: str = "",
        verified: bool = False,
        contributor_id: Optional[int] = None,
    ) -> int:
        """Store a 192-dim L2-normalized embedding as a pgvector vector(192)."""
        row = self._client.table("vazam_embeddings").insert({
            "actor_id":       actor_id,
            "character_id":   character_id,
            "voice_label":    voice_label,
            "embedding":      embedding.tolist(),
            "audio_source":   audio_source,
            "verified":       verified,
            "contributor_id": contributor_id,
        }).execute()
        return row.data[0]["id"]

    def get_embedding_count(self) -> int:
        result = (
            self._client.table("vazam_embeddings")
            .select("id", count="exact")
            .execute()
        )
        return result.count or 0

    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        show_id: Optional[int] = None,
    ) -> list[dict]:
        """Find the top-k most similar voice actors via the match_embeddings RPC.

        Returns a list of dicts with keys:
            actor_id, actor_name, voice_label, similarity
        """
        result = self._client.rpc(
            "match_embeddings",
            {
                "query_embedding": query_embedding.tolist(),
                "top_k":           top_k,
                "show_id_filter":  show_id,
            },
        ).execute()
        return result.data or []
