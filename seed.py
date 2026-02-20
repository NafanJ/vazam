"""
seed.py — Seed the Vazam metadata database from AniList

AniList offers a free public GraphQL API (no auth needed for reads).
This script fetches popular anime titles, their characters, and voice actor
credits, then writes everything into the local SQLite database.

Usage
-----
    python seed.py                        # seed top 200 anime (English VAs)
    python seed.py --lang JAPANESE        # Japanese seiyuu
    python seed.py --limit 50 --delay 1  # smaller batch, slower rate
    python seed.py --show "Cowboy Bebop" # single show by title

The script is idempotent: re-running it will upsert existing rows (matching
on anilist_id) without creating duplicates.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import requests

from db import VazamDB

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
query ShowCast($mediaId: Int, $page: Int, $lang: VoiceActorLanguage) {
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
    resp = requests.post(
        ANILIST_URL,
        json={"query": query, "variables": variables or {}},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"AniList error: {data['errors']}")
    return data["data"]


# ── Seeding logic ─────────────────────────────────────────────────────────────

def seed_show(db: VazamDB, media_id: int, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> int:
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


def seed_popular(db: VazamDB, limit: int = 200, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> None:
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


def seed_by_title(db: VazamDB, title: str, lang: str = "ENGLISH", delay: float = DEFAULT_DELAY) -> None:
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
    parser = argparse.ArgumentParser(description="Seed Vazam DB from AniList")
    parser.add_argument("--show",  type=str,   default="",      help="Seed a single show by title")
    parser.add_argument("--limit", type=int,   default=200,     help="Number of popular shows to seed")
    parser.add_argument("--lang",  type=str,   default="ENGLISH", choices=["ENGLISH", "JAPANESE"],
                        help="Voice actor language")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="Seconds to sleep between API requests")
    parser.add_argument("--db",    type=str,   default="vazam.db", help="Path to SQLite database")
    args = parser.parse_args()

    db = VazamDB(args.db)

    try:
        if args.show:
            seed_by_title(db, args.show, lang=args.lang, delay=args.delay)
        else:
            seed_popular(db, limit=args.limit, lang=args.lang, delay=args.delay)
    finally:
        db.close()

    print("\nSeeding complete.")
    print("  Rebuild the FAISS index: POST /index/rebuild")
    print("  Or restart the API server (index rebuilds automatically at startup).")


if __name__ == "__main__":
    main()
