"""
db.py — Vazam metadata & embedding database (SQLite)

Schema
------
  actors      — voice actor profiles
  shows       — animated shows / games
  characters  — character roles linking actors to shows
  embeddings  — per-character (or per-voice-style) 192-dim vectors

The embeddings table stores raw numpy arrays as BLOBs alongside the FAISS
ordinal (faiss_id) so the search index can be rebuilt deterministically from
the database at startup.

Usage
-----
    db = VazamDB()           # opens / creates vazam.db
    actor_id = db.add_actor("Steve Blum", bio="...", image_url="...")
    show_id   = db.add_show("Cowboy Bebop", media_type="anime", year=1998)
    char_id   = db.add_character("Spike Spiegel", show_id, actor_id)
    emb_id    = db.add_embedding(actor_id, char_id, embedding, source="ep01.wav")

    # Rebuild the FAISS index from stored embeddings
    entries = db.get_all_embeddings()   # → list of (actor_id, name, char_name, np.ndarray)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import numpy as np

DB_PATH = Path("vazam.db")

CREATE_TABLES = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS actors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    bio         TEXT,
    image_url   TEXT,
    anilist_id  INTEGER UNIQUE,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS shows (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    media_type  TEXT    CHECK(media_type IN ('anime','cartoon','game','other')),
    year        INTEGER,
    anilist_id  INTEGER UNIQUE,
    image_url   TEXT,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS characters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    show_id     INTEGER REFERENCES shows(id)  ON DELETE SET NULL,
    actor_id    INTEGER REFERENCES actors(id) ON DELETE CASCADE,
    image_url   TEXT,
    anilist_id  INTEGER UNIQUE,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id       INTEGER NOT NULL REFERENCES actors(id) ON DELETE CASCADE,
    character_id   INTEGER          REFERENCES characters(id) ON DELETE SET NULL,
    voice_label    TEXT    NOT NULL DEFAULT 'Natural Voice',
    embedding_blob BLOB    NOT NULL,
    audio_source   TEXT,
    verified       INTEGER NOT NULL DEFAULT 0,
    contributor_id INTEGER,
    faiss_id       INTEGER,
    created_at     TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_actor    ON embeddings(actor_id);
CREATE INDEX IF NOT EXISTS idx_characters_actor    ON characters(actor_id);
CREATE INDEX IF NOT EXISTS idx_characters_show     ON characters(show_id);
"""


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32").copy()


def _array_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype("float32").tobytes()


class VazamDB:
    """Thin SQLite wrapper for the Vazam metadata and embedding store."""

    def __init__(self, path: str | Path = DB_PATH) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_TABLES)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Cursor, None, None]:
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

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
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO actors (name, bio, image_url, anilist_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(anilist_id) DO UPDATE SET
                    name      = excluded.name,
                    bio       = excluded.bio,
                    image_url = excluded.image_url
                """,
                (name, bio, image_url, anilist_id),
            )
            # When the upsert resolves via DO UPDATE, cursor.lastrowid is
            # unreliable on older SQLite (pre-3.35): sqlite3_last_insert_rowid()
            # is not updated and retains the rowid of whatever was last INSERTed
            # on this connection (possibly a characters row). Always SELECT by
            # anilist_id when it is set to get the authoritative id.
            if anilist_id is not None:
                row = self._conn.execute(
                    "SELECT id FROM actors WHERE anilist_id = ?", (anilist_id,)
                ).fetchone()
                return row["id"]
            return cur.lastrowid

    def get_actor(self, actor_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM actors WHERE id = ?", (actor_id,)
        ).fetchone()

    def get_actor_filmography(self, actor_id: int) -> list[sqlite3.Row]:
        """Return all characters voiced by an actor, with show info."""
        return self._conn.execute(
            """
            SELECT c.id, c.name AS character_name,
                   s.title AS show_title, s.media_type, s.year,
                   c.image_url
            FROM   characters c
            LEFT   JOIN shows s ON s.id = c.show_id
            WHERE  c.actor_id = ?
            ORDER  BY s.year, c.name
            """,
            (actor_id,),
        ).fetchall()

    def list_actors(self, limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, name, image_url FROM actors ORDER BY name LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    def get_actors_for_show(self, show_id: int) -> list[sqlite3.Row]:
        """Return actor IDs for all actors who have a character in a given show."""
        return self._conn.execute(
            """
            SELECT DISTINCT a.id, a.name
            FROM   characters c
            JOIN   actors a ON a.id = c.actor_id
            WHERE  c.show_id = ?
            """,
            (show_id,),
        ).fetchall()

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
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO shows (title, media_type, year, anilist_id, image_url)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(anilist_id) DO UPDATE SET
                    title      = excluded.title,
                    media_type = excluded.media_type,
                    year       = excluded.year,
                    image_url  = excluded.image_url
                """,
                (title, media_type, year, anilist_id, image_url),
            )
            if anilist_id is not None:
                row = self._conn.execute(
                    "SELECT id FROM shows WHERE anilist_id = ?", (anilist_id,)
                ).fetchone()
                return row["id"]
            return cur.lastrowid

    def get_show(self, show_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM shows WHERE id = ?", (show_id,)
        ).fetchone()

    def search_show(self, title: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM shows WHERE title LIKE ? ORDER BY year",
            (f"%{title}%",),
        ).fetchall()

    def list_shows(self, limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, title, media_type, year, image_url FROM shows ORDER BY title LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

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
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO characters (name, show_id, actor_id, image_url, anilist_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(anilist_id) DO UPDATE SET
                    name     = excluded.name,
                    show_id  = excluded.show_id,
                    actor_id = excluded.actor_id
                """,
                (name, show_id, actor_id, image_url, anilist_id),
            )
            if anilist_id is not None:
                row = self._conn.execute(
                    "SELECT id FROM characters WHERE anilist_id = ?", (anilist_id,)
                ).fetchone()
                return row["id"]
            return cur.lastrowid

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
        blob = _array_to_blob(embedding)
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO embeddings
                    (actor_id, character_id, voice_label, embedding_blob,
                     audio_source, verified, contributor_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actor_id,
                    character_id,
                    voice_label,
                    blob,
                    audio_source,
                    int(verified),
                    contributor_id,
                ),
            )
            return cur.lastrowid

    def update_faiss_ids(self, id_map: dict[int, int]) -> None:
        """Update faiss_id for embeddings after an index rebuild.

        id_map: {embedding_db_id → faiss_ordinal}
        """
        with self._tx() as cur:
            for emb_id, faiss_id in id_map.items():
                cur.execute(
                    "UPDATE embeddings SET faiss_id = ? WHERE id = ?",
                    (faiss_id, emb_id),
                )

    def get_all_embeddings(
        self,
        verified_only: bool = False,
    ) -> list[tuple[int, str, str, np.ndarray]]:
        """Fetch all embeddings for FAISS index construction.

        Returns list of (actor_id, actor_name, voice_label, embedding).
        Ordered by embedding id so the FAISS ordinal is deterministic.
        """
        query = """
            SELECT e.id, e.actor_id, a.name AS actor_name, e.voice_label,
                   e.embedding_blob
            FROM   embeddings e
            JOIN   actors a ON a.id = e.actor_id
            {where}
            ORDER  BY e.id
        """
        where = "WHERE e.verified = 1" if verified_only else ""
        rows = self._conn.execute(query.format(where=where)).fetchall()
        return [
            (row["actor_id"], row["actor_name"], row["voice_label"],
             _blob_to_array(row["embedding_blob"]))
            for row in rows
        ]

    def get_embedding_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()
        return row["n"]

    def get_centroid_embeddings(
        self,
        verified_only: bool = False,
    ) -> list[tuple[int, str, str, np.ndarray]]:
        """Return one L2-normalized centroid per (actor_id, voice_label) group.

        When an actor has multiple embeddings for the same character voice,
        averaging them and re-normalizing produces a more stable representative
        vector than any individual sample. This is the recommended index-build
        strategy once you have ≥ 3 samples per voice label.

        Returns list of (actor_id, actor_name, voice_label, centroid_embedding).
        """
        where = "WHERE e.verified = 1" if verified_only else ""
        rows = self._conn.execute(
            f"""
            SELECT e.actor_id, a.name AS actor_name, e.voice_label, e.embedding_blob
            FROM   embeddings e
            JOIN   actors a ON a.id = e.actor_id
            {where}
            ORDER  BY e.actor_id, e.voice_label
            """
        ).fetchall()

        # Group by (actor_id, voice_label)
        from collections import defaultdict
        groups: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
        for row in rows:
            key = (row["actor_id"], row["actor_name"], row["voice_label"])
            groups[key].append(_blob_to_array(row["embedding_blob"]))

        centroids: list[tuple[int, str, str, np.ndarray]] = []
        for (actor_id, actor_name, voice_label), vecs in groups.items():
            centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype("float32")
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            centroids.append((actor_id, actor_name, voice_label, centroid))

        return centroids

    # ------------------------------------------------------------------
    # Show-aware search helper
    # ------------------------------------------------------------------

    def get_actor_ids_for_show(self, show_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT DISTINCT actor_id FROM characters WHERE show_id = ?",
            (show_id,),
        ).fetchall()
        return [r["actor_id"] for r in rows]
