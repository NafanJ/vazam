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

    def get_shows_for_actors(self, actor_ids: list[int]) -> dict[int, set[int]]:
        """Map each actor id to the set of show ids they have characters in.

        Powers cast-graph show inference: detected speakers vote for shows
        whose casts they plausibly belong to.
        """
        if not actor_ids:
            return {}
        rows = (
            self._client.table("vazam_characters")
            .select("actor_id, show_id")
            .in_("actor_id", actor_ids)
            .execute()
            .data
        )
        out: dict[int, set[int]] = {}
        for r in rows:
            if r.get("show_id") is not None:
                out.setdefault(r["actor_id"], set()).add(r["show_id"])
        return out

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

    def _all_actor_names(self) -> dict[int, str]:
        """{actor_id: name} for every actor (paged past the 1000-row API cap)."""
        names: dict[int, str] = {}
        start = 0
        while True:
            rows = (
                self._client.table("vazam_actors")
                .select("id, name")
                .range(start, start + 999)
                .execute()
                .data
                or []
            )
            for r in rows:
                names[r["id"]] = r["name"]
            if len(rows) < 1000:
                break
            start += 1000
        return names

    def list_characters(self) -> list[dict]:
        """All characters with actor, show, and voice-sample count.

        Sorted with voiced characters first (the ones that matter for ID).
        """
        chars = (
            self._client.table("vazam_characters")
            .select("id, name, occupation, image_url, actor_id, show_id")
            .execute()
            .data
            or []
        )
        actor_names = self._all_actor_names()
        show_titles = {
            s["id"]: s["title"]
            for s in (self._client.table("vazam_shows").select("id, title").execute().data or [])
        }
        counts: dict[int, int] = {}
        for e in self._client.table("vazam_embeddings").select("character_id").execute().data or []:
            cid = e.get("character_id")
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1

        out = [
            {
                "id": c["id"], "name": c["name"], "occupation": c.get("occupation"),
                "image_url": c.get("image_url"), "actor_id": c.get("actor_id"),
                "actor_name": actor_names.get(c.get("actor_id")),
                "show_id": c.get("show_id"), "show_title": show_titles.get(c.get("show_id")),
                "samples": counts.get(c["id"], 0),
            }
            for c in chars
        ]
        out.sort(key=lambda x: (-x["samples"], x["show_title"] or "", x["name"]))
        return out

    def get_character(self, character_id: int) -> Optional[dict]:
        """A character with its actor/show and the source files of its embeddings."""
        rows = (
            self._client.table("vazam_characters").select("*").eq("id", character_id).execute().data
        )
        if not rows:
            return None
        c = rows[0]
        actor = self.get_actor(c["actor_id"]) if c.get("actor_id") else None
        show = self.get_show(c["show_id"]) if c.get("show_id") else None
        embeddings = (
            self._client.table("vazam_embeddings")
            .select("id, voice_label, audio_source, source_url, duration_s, quality_score, verified")
            .eq("character_id", character_id)
            .execute()
            .data
            or []
        )
        return {
            "id": c["id"], "name": c["name"], "occupation": c.get("occupation"),
            "image_url": c.get("image_url"), "actor_id": c.get("actor_id"),
            "actor_name": actor["name"] if actor else None,
            "show_id": c.get("show_id"), "show_title": show["title"] if show else None,
            "embeddings": embeddings,
        }

    def update_character(
        self, character_id: int, image_url: Optional[str] = None,
        occupation: Optional[str] = None,
    ) -> Optional[dict]:
        """Patch a character's editable fields (image_url, occupation)."""
        patch: dict = {}
        if image_url is not None:
            patch["image_url"] = image_url
        if occupation is not None:
            patch["occupation"] = occupation
        if patch:
            self._client.table("vazam_characters").update(patch).eq("id", character_id).execute()
        return self.get_character(character_id)

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
        source_url: str = "",
        duration_s: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> int:
        """Store a 192-dim L2-normalized embedding as a pgvector vector(192).

        source_url / duration_s / quality_score require the
        migrations/001_embedding_quality.sql migration to be applied.
        """
        row = self._client.table("vazam_embeddings").insert({
            "actor_id":       actor_id,
            "character_id":   character_id,
            "voice_label":    voice_label,
            "embedding":      embedding.tolist(),
            "audio_source":   audio_source,
            "verified":       verified,
            "contributor_id": contributor_id,
            "source_url":     source_url,
            "duration_s":     duration_s,
            "quality_score":  quality_score,
        }).execute()
        return row.data[0]["id"]

    def get_embedding_count(self) -> int:
        result = (
            self._client.table("vazam_embeddings")
            .select("id", count="exact")
            .execute()
        )
        return result.count or 0

    def list_voices(self) -> list[dict]:
        """Distinct stored voices (actor + voice_label) with sample counts.

        Powers enrollment pickers — each entry is one fingerprint that new
        recordings can be added to. Returns dicts with keys:
            actor_id, actor_name, voice_label, character_id, samples
        """
        rows = (
            self._client.table("vazam_embeddings")
            .select("actor_id, voice_label, character_id")
            .execute()
            .data
            or []
        )
        agg: dict[tuple, dict] = {}
        for r in rows:
            key = (r["actor_id"], r["voice_label"])
            entry = agg.setdefault(key, {
                "actor_id": r["actor_id"], "voice_label": r["voice_label"],
                "character_id": r.get("character_id"), "samples": 0,
            })
            entry["samples"] += 1
            if entry["character_id"] is None and r.get("character_id") is not None:
                entry["character_id"] = r["character_id"]

        names: dict[int, str] = {}
        out = []
        for v in agg.values():
            aid = v["actor_id"]
            if aid not in names:
                a = self.get_actor(aid)
                names[aid] = a["name"] if a else f"#{aid}"
            v["actor_name"] = names[aid]
            out.append(v)
        out.sort(key=lambda v: (v["voice_label"] == "Natural Voice", v["actor_name"], v["voice_label"]))
        return out

    def find_voice(self, actor_id: int, voice_label: str) -> Optional[dict]:
        """Return an existing embedding row (character_id, voice_label) for this
        actor + voice label, or None — used to link new contributions correctly."""
        rows = (
            self._client.table("vazam_embeddings")
            .select("character_id, voice_label")
            .eq("actor_id", actor_id)
            .eq("voice_label", voice_label)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None

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
