-- 004_embedding_audio_path.sql
--
-- Track the stored audio file for an enrolled clip so the dashboard can play it
-- back. The audio itself lives on the server (a Docker volume, VAZAM_AUDIO_DIR),
-- not in the database — this column just holds the filename (e.g. "<id>.wav").
--
-- Apply via the Supabase SQL editor (or `supabase db push`).

alter table vazam_embeddings add column if not exists audio_path text;
