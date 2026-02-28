"""
seed.py — Seed the Vazam Supabase database from AniList

AniList offers a free public GraphQL API (no auth needed for reads).
This script fetches popular anime titles, their characters, and voice actor
credits, then upserts everything into Supabase.

Usage
-----
    python seed.py                          # seed top 200 anime (English VAs)
    python seed.py --lang JAPANESE          # Japanese seiyuu
    python seed.py --limit 50 --delay 1    # smaller batch, slower rate
    python seed.py --show "Cowboy Bebop"   # single show by title

Supabase credentials
---------------------
Set via environment variables or CLI flags:
    SUPABASE_URL   https://<project-ref>.supabase.co
    SUPABASE_KEY   <service_role_key>   (service role required for writes)

    --supabase-url / --supabase-key override the env vars.

The Supabase project must have the vazam_actors, vazam_shows, and
vazam_characters tables (created by the create_vazam_tables migration).

The script is idempotent: re-running it will upsert existing rows (matching
on anilist_id) without creating duplicates.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import requests

# ── AniList GraphQL ──────────────────────────────────────────────────────────

ANILIST_URL = "https://graphql.anilist.co"

# Rate limit: AniList allows ~90 requests / minute; 0.7s delay keeps us safe.
DEFAULT_DELAY = 0.7


POPULAR_ANIME_QUERY = """
query PopularAnime($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title { english romaji }
      startDate { year }
      coverImage { medium }
    }
  }
}
"""

SHOW_CAST_QUERY = """
query ShowCast($mediaId: Int, $page: Int, $lang: StaffLanguage) {
  Media(id: $mediaId) {
    id
    title { english romaji }
    startDate { year }
    coverImage { medium }
    characters(page: $page, perPage: 25, sort: ROLE) {
      pageInfo { hasNextPage }
      edges {
        role
        node {
          id
          name { full }
          image { medium }
        }
        voiceActors(language: $lang) {
          id
          name { full }
          image { medium }
          description
        }
      }
    }
  }
}
"""

SEARCH_MEDIA_QUERY = """
query SearchMedia($search: String) {
  Media(search: $search, type: ANIME) {
    id
    title { english romaji }
    startDate { year }
    coverImage { medium }
  }
}
"""


def _gql(query: str, variables: dict | None = None) -> dict:
    delay = 2.0
    for attempt in range(5):
        resp = requests.post(
            ANILIST_URL,
            json={"query": query, "variables": variables or {}},
            timeout=15,
        )
        if resp.status_code == 429 and attempt < 4:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"AniList error: {data['errors']}")
        return data["data"]
    raise RuntimeError("AniList rate limit: max retries exceeded")


# ── Supabase backend ──────────────────────────────────────────────────────────

class SupabaseDB:
    """Write-only Supabase client mirroring the VazamDB seeding interface.

    Uses the PostgREST REST API to upsert into vazam_actors, vazam_shows,
    and vazam_characters tables.  Requires a service-role key so that writes
    are not blocked by RLS policies.
    """

    def __init__(self, url: str, key: str) -> None:
        self._base = url.rstrip("/") + "/rest/v1"
        self._session = requests.Session()
        self._session.headers.update({
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })

    def _upsert(self, table: str, payload: dict, on_conflict: str) -> dict:
        resp = self._session.post(
            f"{self._base}/{table}",
            params={"on_conflict": on_conflict},
            json=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else {}

    def add_actor(
        self,
        name: str,
        bio: str = "",
        image_url: str = "",
        anilist_id: Optional[int] = None,
    ) -> int:
        row = self._upsert(
            "vazam_actors",
            {"name": name, "bio": bio, "image_url": image_url, "anilist_id": anilist_id},
            on_conflict="anilist_id",
        )
        return row["id"]

    def add_show(
        self,
        title: str,
        media_type: str = "anime",
        year: Optional[int] = None,
        anilist_id: Optional[int] = None,
        image_url: str = "",
    ) -> int:
        row = self._upsert(
            "vazam_shows",
            {
                "title": title,
                "media_type": media_type,
                "year": year,
                "anilist_id": anilist_id,
                "image_url": image_url,
            },
            on_conflict="anilist_id",
        )
        return row["id"]

    def add_character(
        self,
        name: str,
        show_id: Optional[int],
        actor_id: int,
        image_url: str = "",
        anilist_id: Optional[int] = None,
    ) -> int:
        row = self._upsert(
            "vazam_characters",
            {
                "name": name,
                "show_id": show_id,
                "actor_id": actor_id,
                "image_url": image_url,
                "anilist_id": anilist_id,
            },
            on_conflict="anilist_id",
        )
        return row["id"]

    def close(self) -> None:
        self._session.close()


# ── Seeding logic ─────────────────────────────────────────────────────────────

def seed_show(db: SupabaseDB, media_id: int, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> int:
    """Fetch all character→VA mappings for one AniList media ID and store them.

    Returns the number of (character, VA) pairs inserted.
    """
    page = 1
    inserted = 0
    show_db_id: Optional[int] = None

    while True:
        data = _gql(SHOW_CAST_QUERY, {"mediaId": media_id, "page": page, "lang": lang})
        media = data["Media"]

        # Create/upsert the show on first page
        if show_db_id is None:
            title = media["title"]["english"] or media["title"]["romaji"]
            year  = (media["startDate"] or {}).get("year")
            img   = (media["coverImage"] or {}).get("medium", "")
            show_db_id = db.add_show(
                title=title,
                media_type="anime",
                year=year,
                anilist_id=media["id"],
                image_url=img,
            )
            print(f"  Show: {title} ({year}) → db id {show_db_id}")

        chars_page = media["characters"]
        for edge in chars_page["edges"]:
            char_node = edge["node"]
            vas = edge.get("voiceActors") or []

            for va in vas:
                # Upsert actor
                actor_name = va["name"]["full"]
                actor_id = db.add_actor(
                    name=actor_name,
                    bio=va.get("description") or "",
                    image_url=(va.get("image") or {}).get("medium", ""),
                    anilist_id=va["id"],
                )

                # Upsert character
                char_name = char_node["name"]["full"]
                db.add_character(
                    name=char_name,
                    show_id=show_db_id,
                    actor_id=actor_id,
                    image_url=(char_node.get("image") or {}).get("medium", ""),
                    anilist_id=char_node["id"],
                )
                inserted += 1

        if not chars_page["pageInfo"]["hasNextPage"]:
            break
        page += 1
        time.sleep(delay)

    return inserted


def seed_popular(db: SupabaseDB, limit: int = 200, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> None:
    """Seed the database with the top-N most popular anime on AniList."""
    per_page = 50
    total_pages = (limit + per_page - 1) // per_page
    all_media: list[dict] = []

    # 1. Collect media IDs
    for page in range(1, total_pages + 1):
        data = _gql(POPULAR_ANIME_QUERY, {"page": page, "perPage": per_page})
        media_list = data["Page"]["media"]
        all_media.extend(media_list)
        if not data["Page"]["pageInfo"]["hasNextPage"]:
            break
        time.sleep(delay)

    all_media = all_media[:limit]
    print(f"Seeding {len(all_media)} shows ({lang} voice actors)…\n")

    # 2. Seed each show
    for i, media in enumerate(all_media, 1):
        title = media["title"]["english"] or media["title"]["romaji"]
        print(f"[{i}/{len(all_media)}] {title}")
        try:
            n = seed_show(db, media["id"], lang=lang, delay=delay)
            print(f"         → {n} character/VA pairs\n")
        except Exception as exc:
            print(f"         ✗ skipped: {exc}\n")
        time.sleep(delay)


def seed_by_title(db: SupabaseDB, title: str, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> None:
    """Seed a single show looked up by title."""
    print(f"Searching AniList for: {title!r}")
    data = _gql(SEARCH_MEDIA_QUERY, {"search": title})
    media = data["Media"]
    found_title = media["title"]["english"] or media["title"]["romaji"]
    print(f"Found: {found_title} (anilist id {media['id']})")
    n = seed_show(db, media["id"], lang=lang, delay=delay)
    print(f"Done — {n} character/VA pairs inserted.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Vazam Supabase DB from AniList")
    parser.add_argument("--show",  type=str,   default="",      help="Seed a single show by title")
    parser.add_argument("--limit", type=int,   default=200,     help="Number of popular shows to seed")
    parser.add_argument("--lang",  type=str,   default="ENGLISH", choices=["ENGLISH", "JAPANESE"],
                        help="Voice actor language")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="Seconds to sleep between API requests")
    parser.add_argument("--supabase-url", type=str, default="",
                        help="Supabase project URL (overrides SUPABASE_URL env var)")
    parser.add_argument("--supabase-key", type=str, default="",
                        help="Supabase service-role key (overrides SUPABASE_KEY env var)")
    args = parser.parse_args()

    url = args.supabase_url or os.environ.get("SUPABASE_URL", "")
    key = args.supabase_key or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise SystemExit(
            "SUPABASE_URL and SUPABASE_KEY are required "
            "(env vars or --supabase-url / --supabase-key)."
        )

    db = SupabaseDB(url, key)
    print(f"Supabase target: {url}")

    try:
        if args.show:
            seed_by_title(db, args.show, lang=args.lang, delay=args.delay)
        else:
            seed_popular(db, limit=args.limit, lang=args.lang, delay=args.delay)
    finally:
        db.close()

    print("\nSeeding complete.")


if __name__ == "__main__":
    main()
