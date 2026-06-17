-- 003_match_embeddings_character_art.sql
--
-- Extend match_embeddings() to also return the character_id, the character's
-- image_url, and the show title, so identification results can render
-- character art + the show chip without a second round-trip.
--
-- Changing the OUT-parameter list is a return-type change, which CREATE OR
-- REPLACE cannot do — drop and recreate.
--
-- Apply via the Supabase SQL editor (or `supabase db push`).

drop function if exists public.match_embeddings(vector, integer, integer);

create function public.match_embeddings(
    query_embedding vector,
    top_k integer default 5,
    show_id_filter integer default null
)
returns table(
    actor_id bigint,
    actor_name text,
    voice_label text,
    similarity double precision,
    character_id bigint,
    image_url text,
    show_title text
)
language sql
stable
as $function$
    select
        e.actor_id,
        a.name as actor_name,
        e.voice_label,
        1 - (e.embedding <=> query_embedding) as similarity,
        e.character_id,
        ch.image_url,
        sh.title as show_title
    from vazam_embeddings e
    join vazam_actors a on a.id = e.actor_id
    left join vazam_characters ch on ch.id = e.character_id
    left join vazam_shows sh on sh.id = ch.show_id
    where show_id_filter is null
       or e.actor_id in (
           select c.actor_id
           from vazam_characters c
           where c.show_id = show_id_filter
       )
    order by e.embedding <=> query_embedding
    limit top_k;
$function$;
