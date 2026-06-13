-- 000_base_schema.sql
--
-- Base Vazam schema: pgvector, the vazam_* tables, and the match_embeddings()
-- similarity-search function. This was originally applied to Supabase by hand
-- and never captured in the repo; reconstructed here from db.py / pipeline.py
-- so the database is reproducible from scratch.
--
-- Apply this FIRST, then 001_embedding_quality.sql, via the Supabase SQL editor
-- or `supabase db push`.

-- pgvector — provides the vector type and cosine-distance operator (<=>)
create extension if not exists vector;

-- ── Actors ───────────────────────────────────────────────────────────────────
create table if not exists vazam_actors (
    id         bigint generated always as identity primary key,
    name       text not null,
    bio        text,
    image_url  text,
    anilist_id bigint unique          -- upsert key; NULL allowed (no conflict)
);

-- ── Shows ────────────────────────────────────────────────────────────────────
create table if not exists vazam_shows (
    id         bigint generated always as identity primary key,
    title      text not null,
    media_type text default 'anime' check (media_type in ('anime', 'cartoon', 'game', 'other')),
    year       int,
    image_url  text,
    anilist_id bigint unique
);

-- ── Characters (actor ↔ show roles) ──────────────────────────────────────────
create table if not exists vazam_characters (
    id         bigint generated always as identity primary key,
    name       text not null,
    show_id    bigint references vazam_shows (id) on delete set null,
    actor_id   bigint references vazam_actors (id) on delete cascade,
    image_url  text,
    anilist_id bigint unique
);

create index if not exists vazam_characters_actor_idx on vazam_characters (actor_id);
create index if not exists vazam_characters_show_idx  on vazam_characters (show_id);

-- ── Embeddings (192-dim ECAPA speaker vectors) ───────────────────────────────
-- source_url / duration_s / quality_score are added by 001_embedding_quality.sql.
create table if not exists vazam_embeddings (
    id             bigint generated always as identity primary key,
    actor_id       bigint not null references vazam_actors (id) on delete cascade,
    character_id   bigint references vazam_characters (id) on delete set null,
    voice_label    text default 'Natural Voice',
    embedding      vector(192) not null,
    audio_source   text,
    verified       boolean default false,
    contributor_id bigint
);

create index if not exists vazam_embeddings_actor_idx on vazam_embeddings (actor_id);

-- Approximate-nearest-neighbour index for cosine similarity. Embeddings are
-- L2-normalized, so cosine distance ranking == dot-product ranking.
create index if not exists vazam_embeddings_hnsw_idx
    on vazam_embeddings using hnsw (embedding vector_cosine_ops);

-- ── Similarity search ────────────────────────────────────────────────────────
-- Returns the top-k most similar stored voices to a query embedding.
-- show_id_filter (when not null) restricts the search to that show's cast,
-- turning open-set recognition into closed-set — the show-aware search path.
create or replace function match_embeddings(
    query_embedding vector(192),
    top_k           int default 5,
    show_id_filter  int default null
)
returns table (
    actor_id    bigint,
    actor_name  text,
    voice_label text,
    similarity  float
)
language sql
stable
as $$
    select
        e.actor_id,
        a.name as actor_name,
        e.voice_label,
        1 - (e.embedding <=> query_embedding) as similarity
    from vazam_embeddings e
    join vazam_actors a on a.id = e.actor_id
    where show_id_filter is null
       or e.actor_id in (
           select c.actor_id
           from vazam_characters c
           where c.show_id = show_id_filter
       )
    order by e.embedding <=> query_embedding
    limit top_k;
$$;
