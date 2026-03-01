# CLAUDE.md — Vazam Codebase Guide

Vazam is "Shazam for Voice Actors": point a phone at any animated show, anime, or video game and instantly identify which voice actor is speaking. This file documents the codebase structure, development conventions, and workflows for AI assistants working on this project.

## Project Overview

**Version:** 0.2.0
**Stack:** Python 3 (FastAPI backend) + TypeScript/React Native (mobile frontend)
**Database:** Supabase (PostgreSQL + pgvector)
**Vector search:** pgvector `vector(192)` with cosine similarity via `match_embeddings` RPC

## Architecture

Audio processing pipeline (backend):

```
Audio clip
  → Demucs v4 htdemucs_ft   (vocal isolation — strips music & SFX)
  → pyannote VAD             (trim silence, keep only speech)
  → pyannote diarization-3.1 (split multi-speaker clips per speaker)
  → SpeechBrain ECAPA-TDNN  (192-dim speaker embedding)
  → Supabase match_embeddings RPC  (pgvector cosine similarity search)
  → Supabase vazam_actors/characters/shows  (metadata lookup)
  → "Steve Blum as Spike Spiegel (Cowboy Bebop) — 94% confidence"
```

## Directory Structure

```
vazam/
├── api.py             FastAPI HTTP backend (main entry point for the server)
├── pipeline.py        Full pipeline: isolation → VAD → diarize → embed → search
├── db.py              Supabase wrapper: vazam_actors, vazam_shows, vazam_characters, vazam_embeddings tables
├── seed.py            AniList GraphQL seeder → Supabase (metadata only, no audio)
├── scrape_audio.py    Automated YouTube audio downloader + embedder via yt-dlp
├── embed_batch.py     Bulk audio-to-embedding importer (CLI tool)
├── main.py            Standalone demo script (single-file quick test)
├── requirements.txt   Python dependencies
├── *.mp3 / *.wav      Test audio samples
├── tests/
│   ├── __init__.py
│   ├── conftest.py    Shared fixtures; mocks all ML deps + Supabase with in-memory fakes
│   ├── test_api.py    FastAPI endpoint integration tests
│   ├── test_db.py     Database CRUD and embedding serialization tests
│   └── test_pipeline.py  Pipeline logic tests
└── app/               React Native mobile application
    ├── package.json
    ├── App.tsx         Root navigator
    └── src/
        ├── api/vazam.ts        Typed Axios API client
        ├── types/index.ts      Shared TypeScript types
        ├── screens/            HomeScreen, ResultsScreen, ActorProfileScreen, ShowSearchScreen
        ├── components/         RecordButton, ResultCard
        └── hooks/useRecorder.ts  Audio recording logic
```

## Environment Variables

| Variable        | Default    | Description                                                                          |
|-----------------|------------|--------------------------------------------------------------------------------------|
| `SUPABASE_URL`  | _(required)_ | Supabase project URL, e.g. `https://<ref>.supabase.co`                             |
| `SUPABASE_KEY`  | _(required)_ | Supabase service-role key (needed for writes)                                       |
| `HF_TOKEN`      | _(empty)_  | HuggingFace token — required for VAD + diarization (pyannote models). Accept terms at `hf.co/pyannote/voice-activity-detection` and `hf.co/pyannote/speaker-diarization-3.1` |
| `DEVICE`        | auto       | `cuda` or `cpu` — if unset, auto-detects CUDA                                       |

`python-dotenv` is installed; place credentials in a `.env` file at the project root and they are loaded automatically on startup. Without `HF_TOKEN`, VAD and diarization are disabled — the pipeline falls back to embedding the full audio file and single-speaker identification still works.

## Development Setup

### Backend

```bash
pip install -r requirements.txt

# Optional: install test dependencies (commented out in requirements.txt)
pip install pytest httpx

# Create a .env file with your credentials
cat > .env <<EOF
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_KEY=<service_role_key>
HF_TOKEN=hf_...   # optional
EOF

# Run the API server
uvicorn api:app --reload
# Interactive docs: http://localhost:8000/docs
```

### Mobile App

```bash
cd app
npm install

# iOS
npm run ios

# Android
npm run android

# Metro bundler only
npm run start
```

## Running Tests

Tests mock all heavy ML dependencies (SpeechBrain, Demucs, pyannote) **and** replace the Supabase client with an in-memory fake (`_FakeSupabase` in `conftest.py`), so they run without GPU, model downloads, or a real Supabase project.

```bash
# Run all tests
pytest

# Run a specific file
pytest tests/test_api.py -v

# Run a specific test
pytest tests/test_api.py::test_identify_returns_results -v
```

Mobile app tests:

```bash
cd app
npm test
npm run lint   # ESLint on src/**/*.{ts,tsx}
```

## Key Modules

### `api.py` — FastAPI Backend

- Entry point: `uvicorn api:app --reload`
- Global state: `db: VazamDB` and `pipeline: VazamPipeline` initialized in the `lifespan` context manager
- CORS is open (`allow_origins=["*"]`) — restrict in production
- No FAISS rebuild on startup; pgvector's HNSW index is managed automatically by Supabase
- `POST /index/rebuild` is kept for backward compatibility but is a **no-op** — it just returns the current embedding count
- Adding an embedding via `POST /actors/{id}/embeddings` calls `db.add_embedding()` which inserts into Supabase; the pgvector index updates automatically

### `pipeline.py` — Audio Processing

Key classes and functions:

| Symbol | Description |
|--------|-------------|
| `VazamPipeline` | Orchestrator — calls isolation/VAD/diarize/embed and delegates search to `db.search_embeddings()` |
| `isolate_vocals()` | Runs Demucs via subprocess; falls back to original file on failure |
| `get_speech_segments()` | Returns `[(start, end)]` tuples of speech from pyannote VAD |
| `extract_speech_audio()` | Loads audio and concatenates VAD-selected segments into a single `(1, N)` tensor |
| `get_embedding()` | Accepts a file path or a `(1, N)` tensor; returns a 192-dim L2-normalized `float32` array |
| `get_embedding_for_segment()` | Generates an embedding for a time-bounded slice of an audio file |
| `diarize()` | Returns `list[SpeakerSegment]` from pyannote speaker-diarization-3.1 |
| `merge_speaker_segments()` | Merges consecutive same-speaker segments for better embedding quality |

Constants (in `pipeline.py`):

```python
EMBEDDING_DIM       = 192
CONFIDENT_THRESHOLD = 0.70   # cosine similarity ≥ 0.70 → "confident"
POSSIBLE_THRESHOLD  = 0.50   # 0.50–0.69 → "possible"; < 0.50 → "none"
MIN_SPEECH_SECONDS  = 1.5    # segments shorter than this are dropped
```

Heavy models are module-level globals (`_classifier`, `_vad_pipeline`, `_diarize_pipeline`) loaded lazily on first call so the module imports quickly.

**torchaudio compatibility shim:** `pipeline.py` patches `torchaudio.list_audio_backends` to an empty lambda at import time, because torchaudio ≥ 2.5 removed that function from the public namespace but older SpeechBrain releases call it.

### `db.py` — Supabase Database

Connects to Supabase via the `supabase-py` client (`create_client`). All credentials are read from `SUPABASE_URL` / `SUPABASE_KEY` env vars.

Schema (Supabase tables):

```
vazam_actors      — id, name, bio, image_url, anilist_id (UNIQUE)
vazam_shows       — id, title, media_type, year, image_url, anilist_id (UNIQUE)
                    media_type ∈ {anime, cartoon, game, other}
vazam_characters  — id, name, show_id → vazam_shows, actor_id → vazam_actors,
                    image_url, anilist_id (UNIQUE)
vazam_embeddings  — id, actor_id → vazam_actors, character_id → vazam_characters,
                    voice_label (default "Natural Voice"), embedding vector(192),
                    audio_source, verified (bool), contributor_id
```

Embeddings are stored as `vector(192)` using pgvector. Serialization: `embedding.tolist()` on write; pgvector returns a native list on read.

Similarity search is handled by the `match_embeddings()` PostgreSQL function (installed via migration), called via `db._client.rpc("match_embeddings", {...})`. It accepts `query_embedding`, `top_k`, and an optional `show_id_filter` and returns `actor_id`, `actor_name`, `voice_label`, `similarity`.

All `add_actor`, `add_show`, and `add_character` calls are upserts on `anilist_id` — safe to re-run.

`VazamDB.close()` is a no-op — supabase-py manages its own connection pool.

### `seed.py` — AniList Metadata Seeder

Fetches popular anime, characters, and voice actor credits from the free AniList GraphQL API (no auth needed). Upserts into Supabase via a standalone `SupabaseDB` class that calls the PostgREST REST API directly. Does **not** download or process audio.

```bash
python seed.py                          # top 200 anime, English VAs
python seed.py --show "Cowboy Bebop"    # single show
python seed.py --lang JAPANESE          # Japanese seiyuu
python seed.py --limit 50 --delay 1    # smaller batch, slower rate

# Override credentials without .env
python seed.py --supabase-url https://... --supabase-key ...
```

Rate limit: AniList allows ~90 requests/minute; default delay is 0.7s with exponential backoff on 429 errors.

### `scrape_audio.py` — Automated YouTube Audio Scraper

Queries Supabase for actors with zero embeddings, searches YouTube for interview or demo reel clips using `yt-dlp`, downloads audio, and generates + stores embeddings automatically.

```bash
python scrape_audio.py                        # process up to 50 actors with no embeddings
python scrape_audio.py --actor "Steve Blum"   # single actor by name (partial match)
python scrape_audio.py --limit 20             # cap actors per run
python scrape_audio.py --dry-run              # preview queries without downloading
python scrape_audio.py --include-characters   # also scrape per-character clips (slower)
```

`yt-dlp` must be installed (`pip install yt-dlp`). A 2.5-second delay is applied between downloads. Videos longer than 1 hour are skipped. Up to 5 YouTube search results are tried per query so the duration filter has fallback candidates.

### `embed_batch.py` — Bulk Audio Importer

Expected sample directory layout:

```
samples/
└── Steve Blum/
    ├── natural/              → voice_label = "Natural Voice"
    │   └── interview.wav
    └── Spike Spiegel/        → voice_label = "Spike Spiegel"
        └── ep01.wav
```

```bash
python embed_batch.py samples/ --dry-run        # preview without writing
python embed_batch.py samples/ --isolate        # run Demucs first
python embed_batch.py samples/ --verified       # mark as verified embeddings
python embed_batch.py samples/ --rebuild-index http://localhost:8000
```

## API Reference

Interactive docs at `http://localhost:8000/docs` when the server is running.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/identify` | Upload audio → top-K voice actor matches |
| `POST` | `/identify/multi` | Diarize + identify each speaker separately (requires `HF_TOKEN`) |
| `POST` | `/actors` | Register a voice actor |
| `GET`  | `/actors` | List all actors (supports `limit`, `offset`) |
| `GET`  | `/actors/{id}` | Actor profile + filmography |
| `POST` | `/actors/{id}/embeddings` | Upload audio sample → generate + store embedding |
| `POST` | `/shows` | Register a show |
| `GET`  | `/shows` | List all shows |
| `GET`  | `/shows/{id}` | Show details + cast list |
| `GET`  | `/shows/search?q=` | Search shows by title (ILIKE match) |
| `POST` | `/index/rebuild` | No-op (pgvector manages indexes automatically); returns embedding count |
| `GET`  | `/health` | Liveness check + index stats |

### `/identify` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio` | required | WAV, MP3, or M4A file upload |
| `isolate` | `false` | Run Demucs vocal isolation first |
| `show_id` | `null` | Restrict results to actors in this show (show-aware search) |
| `top_k` | `5` | Number of candidates to return (1–20) |

## Testing Patterns

### ML + Supabase mocking in conftest.py

All heavy dependencies **and** the Supabase client are patched before `api.py` is imported:

```python
with (
    patch("pipeline.isolate_vocals", side_effect=lambda p, **kw: p),
    patch("pipeline._load_embedding_model", return_value=fake_encoder),
    patch("torchaudio.load", return_value=fake_signal),
    patch("db.create_client", return_value=fake_sb),        # in-memory Supabase
    patch.dict("os.environ", {
        "SUPABASE_URL": "http://test",
        "SUPABASE_KEY": "test",
        "HF_TOKEN": "",
    }),
):
    import importlib
    import api as api_module
    importlib.reload(api_module)
    with TestClient(api_module.app) as client:
        yield client
```

**Important:** `api_module` must be reloaded after patches are applied so the `lifespan` context manager picks them up. Always use `importlib.reload()` rather than a plain `import`.

### In-memory Supabase fake (`_FakeSupabase`)

`conftest.py` contains a complete in-memory Supabase fake:

- `_FakeStore` — shared dict-of-lists representing tables, with auto-incrementing IDs
- `_FakeQuery` — chainable query builder supporting `select`, `insert`, `upsert`, `eq`, `ilike`, `order`, `range`, `execute`
- `_FakeRpcQuery` — handles `.rpc()` calls; handlers are registered via `fake_sb.set_rpc(func_name, handler)`
- The `api_client` fixture wires a `_match_embeddings` handler to the `match_embeddings` RPC so `/identify` works end-to-end in tests

### Shared fixtures

| Fixture | Description |
|---------|-------------|
| `tmp_wav` | Temporary silent 16 kHz WAV file path |
| `random_embedding` | Random L2-normalized 192-dim `float32` array (seed=42) |
| `db` | `VazamDB` backed by an in-memory `_FakeSupabase`; no real network calls |
| `api_client` | `TestClient` for `api.py` with all ML and Supabase mocked |

### Writing new tests

- Import `make_wav_bytes()` from `tests.conftest` to create in-memory WAV bytes for file uploads
- Use `files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")}` in multipart requests
- Tests do not require `HF_TOKEN` — diarization/VAD are disabled when it is empty
- Tests do not require `SUPABASE_URL`/`SUPABASE_KEY` — `_FakeSupabase` is used instead
- Keep ML models mocked; never download real models in tests

## Code Conventions

### Python (backend)

- Python 3.10+ features (`match`, union types with `|`) are used
- `from __future__ import annotations` is present in all modules (for PEP 563 deferred evaluation)
- Type annotations everywhere; prefer `Optional[X]` for nullable parameters in public APIs
- Module-level docstrings document the module's purpose, public interface, and usage examples
- Use `dataclasses` for plain data containers (`SpeakerSegment`, `IdentificationResult`)
- Database access goes exclusively through `VazamDB` — no direct Supabase client calls outside `db.py`
- All embeddings are `float32` numpy arrays, L2-normalized to unit length; serialized via `.tolist()` for pgvector
- Temporary files from uploads are cleaned up in `finally` blocks (see `_save_upload` in `api.py`)
- Heavy imports (SpeechBrain, pyannote) are inside functions, not at module top-level, to keep startup fast

### TypeScript / React Native (frontend)

- TypeScript 5; strict mode via `@react-native/typescript-config`
- Navigation uses React Navigation v6 with typed stack params (defined in `src/types/index.ts`)
- All API calls go through the typed client in `src/api/vazam.ts` — never call `axios` directly from screens
- State management via Zustand; audio recording via the `useRecorder` hook
- ESLint enforced on `src/**/*.{ts,tsx}` via `npm run lint`

## Multi-Embedding Strategy

Each voice actor stores **separate embeddings per voice style**:

- **Natural Voice** — from interviews, convention panels, or demo reels
- **Per-character voices** — one or more clips per distinct character voice

This improves accuracy for actors who substantially alter their voice between roles (e.g., Seth MacFarlane as Peter Griffin vs. Stewie Griffin vs. Quagmire). The `voice_label` field on the `vazam_embeddings` table holds the label; "Natural Voice" is the default.

## Show-Aware Search

Pass `show_id` to `POST /identify` to restrict matching to actors known to appear in that show. This converts an open-set speaker recognition problem into a closed-set one, which significantly improves accuracy. The `match_embeddings` RPC accepts an optional `show_id_filter` parameter that pre-filters embeddings by actors in the given show.

## Common Workflows

### Add a new voice actor and sample

```bash
# 1. Create actor record
curl -X POST http://localhost:8000/actors \
  -H "Content-Type: application/json" \
  -d '{"name": "Steve Blum"}'
# → {"id": 1, ...}

# 2. Upload natural voice
curl -X POST http://localhost:8000/actors/1/embeddings \
  -F "audio=@blum_interview.wav" \
  -F "voice_label=Natural Voice"

# 3. Upload character voice
curl -X POST http://localhost:8000/actors/1/embeddings \
  -F "audio=@blum_spike.wav" \
  -F "voice_label=Spike Spiegel" \
  -F "isolate=true"
```

### Identify a voice

```bash
curl -X POST http://localhost:8000/identify \
  -F "audio=@test_clip.mp3" \
  -F "isolate=true" \
  -F "top_k=5"
```

### Seed metadata from AniList

```bash
python seed.py --show "Cowboy Bebop"
python seed.py --limit 100            # top 100 popular anime
```

### Auto-scrape audio from YouTube

```bash
# Scrape natural voice clips for all actors with no embeddings
python scrape_audio.py

# Scrape a specific actor + their character clips
python scrape_audio.py --actor "Steve Blum" --include-characters

# Dry run to preview queries
python scrape_audio.py --dry-run
```

### Bulk-import local audio files

```bash
python embed_batch.py samples/ --isolate --verified
```

## Supabase Setup

The Supabase project requires:

1. The `pgvector` extension enabled (`CREATE EXTENSION IF NOT EXISTS vector;`)
2. Tables: `vazam_actors`, `vazam_shows`, `vazam_characters`, `vazam_embeddings` (with `embedding vector(192)` column)
3. The `match_embeddings` SQL function installed via migration — this function performs cosine similarity search using pgvector's `<=>` operator and supports optional show-based filtering

pgvector's HNSW index on the `embedding` column is recommended in production for sub-millisecond search at scale. No manual index rebuild is needed; the index updates automatically on insert.
