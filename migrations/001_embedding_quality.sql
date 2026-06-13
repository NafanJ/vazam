-- 001_embedding_quality.sql
--
-- Adds provenance and quality metadata to vazam_embeddings, required by the
-- cross-video consensus scraper (scrape_audio.py / consensus.py).
--
--   source_url     comma-separated source video URLs (provenance + takedown path)
--   duration_s     seconds of clean speech behind this embedding
--   quality_score  intra-cluster cosine consistency (0..1) from consensus scraping
--
-- Apply after 000_base_schema.sql, via the Supabase SQL editor or
-- `supabase db push`.

alter table vazam_embeddings
  add column if not exists source_url    text,
  add column if not exists duration_s    real,
  add column if not exists quality_score real;
