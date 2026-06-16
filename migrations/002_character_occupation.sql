-- 002_character_occupation.sql
-- Adds a free-text `occupation` field to characters (e.g. "Pirate Captain",
-- "Survey Corps Commander"), editable from the character admin page.
--
-- Apply via the Supabase SQL editor or `supabase db push`. Idempotent.

alter table vazam_characters add column if not exists occupation text;
