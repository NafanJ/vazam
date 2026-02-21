# CLAUDE.md — Vazam Codebase Guide

Vazam is "Shazam for Voice Actors": point a phone at any animated show, anime, or video game and instantly identify which voice actor is speaking. This file documents the codebase structure, development conventions, and workflows for AI assistants working on this project.

## Project Overview

**Version:** 0.2.0
**Stack:** Python 3 (FastAPI backend) + TypeScript/React Native (mobile frontend)
**Database:** SQLite (upgradeable to PostgreSQL)
**Vector search:** FAISS IndexFlatIP (cosine similarity)

## Architecture

Audio processing pipeline (backend):

```
Audio clip
  → Demucs v4 htdemucs_ft   (vocal isolation — strips music & SFX)
  → pyannote VAD             (trim silence, keep only speech)
  → pyannote diarization-3.1 (split multi-speaker clips per speaker)
  → SpeechBrain ECAPA-TDNN  (192-dim speaker embedding)
  → FAISS IndexFlatIP        (cosine similarity search)
  → SQLite                   (actor → character → show metadata lookup)
  → "Steve Blum as Spike Spiegel (Cowboy Bebop) — 94% confidence"
```

## Directory Structure

```
vazam/
├── api.py             FastAPI HTTP backend (main entry point for the server)
├── pipeline.py        Full pipeline: isolation → VAD → diarize → embed → search
├── db.py              SQLite wrapper: actors, shows, characters, embeddings tables
├── seed.py            AniList GraphQL seeder (metadata only, no audio)
├── embed_batch.py     Bulk audio-to-embedding importer (CLI tool)
├── main.py            Standalone demo script (single-file quick test)
├── requirements.txt   Python dependencies
├── *.mp3 / *.wav      Test audio samples
├── tests/
│   ├── __init__.py
│   ├── conftest.py    Shared fixtures; mocks all ML dependencies
│   ├── test_api.py    FastAPI endpoint integration tests
│   ├── test_db.py     Database CRUD and embedding serialization tests
│   └── test_pipeline.py  Pipeline logic and FAISS index tests
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

| Variable   | Default    | Description                                                         |
|------------|------------|---------------------------------------------------------------------|
| `HF_TOKEN` | _(empty)_  | HuggingFace token — required for VAD + diarization (pyannote models). Accept terms at `hf.co/pyannote/voice-activity-detection` and `hf.co/pyannote/speaker-diarization-3.1` |
| `DB_PATH`  | `vazam.db` | SQLite database file path                                           |
| `DEVICE`   | auto       | `cuda` or `cpu` — if unset, auto-detects CUDA                      |

Without `HF_TOKEN`, VAD and diarization are disabled. The pipeline falls back to embedding the full audio file and single-speaker identification still works.

## Development Setup

### Backend

```bash
pip install -r requirements.txt

# Optional: install test dependencies (commented out in requirements.txt)
pip install pytest httpx

# Set HuggingFace token if you want VAD + diarization
export HF_TOKEN=hf_...

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

Tests mock all heavy ML dependencies (SpeechBrain, Demucs, pyannote) so they run without GPU or model downloads.

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
- FAISS index is rebuilt from the DB automatically on startup via `_rebuild_index()`
- Adding an embedding via `POST /actors/{id}/embeddings` incrementally updates the live in-memory index without requiring a full rebuild

### `pipeline.py` — Audio Processing

Key classes and functions:

| Symbol | Description |
|--------|-------------|
| `VazamPipeline` | Orchestrator — holds FAISS index, calls isolation/VAD/diarize/embed |
| `EmbeddingIndex` | Thin FAISS `IndexFlatIP` wrapper; stores `(actor_id, actor_name, character_name)` in a parallel list |
| `isolate_vocals()` | Runs Demucs via subprocess; falls back to original file on failure |
| `get_speech_segments()` | Returns `[(start, end)]` tuples of speech from pyannote VAD |
| `get_embedding()` | Accepts a file path or a `(1, N)` tensor; returns a 192-dim L2-normalized `float32` array |
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

### `db.py` — SQLite Database

Schema:

```
actors      — id, name, bio, image_url, anilist_id (UNIQUE)
shows       — id, title, media_type, year, anilist_id (UNIQUE)
              media_type ∈ {anime, cartoon, game, other}
characters  — id, name, show_id → shows, actor_id → actors, anilist_id (UNIQUE)
embeddings  — id, actor_id → actors, character_id → characters,
              voice_label (default "Natural Voice"), embedding_blob (BLOB),
              audio_source, verified (0/1), faiss_id
```

Embeddings are stored as raw `float32` bytes using `np.ndarray.tobytes()` / `np.frombuffer(..., dtype="float32")`.

All `add_actor`, `add_show`, and `add_character` calls are upserts on `anilist_id` — safe to re-run.

`VazamDB` uses WAL journal mode and enforces foreign keys. Transactions are handled via the `_tx()` context manager.

### `seed.py` — AniList Metadata Seeder

Fetches popular anime, characters, and voice actor credits from the free AniList GraphQL API (no auth needed). Inserts/upserts into the local DB. Does **not** download or process audio.

```bash
python seed.py                          # top 200 anime, English VAs
python seed.py --show "Cowboy Bebop"    # single show
python seed.py --lang JAPANESE          # Japanese seiyuu
python seed.py --limit 50 --delay 1    # smaller batch, slower rate
```

Rate limit: AniList allows ~90 requests/minute; default delay is 0.7s.

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
| `GET`  | `/shows/search?q=` | Search shows by title (LIKE match) |
| `POST` | `/index/rebuild` | Rebuild FAISS index from DB (`strategy=individual|centroid`) |
| `GET`  | `/health` | Liveness check + index stats |

### `/identify` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio` | required | WAV, MP3, or M4A file upload |
| `isolate` | `false` | Run Demucs vocal isolation first |
| `show_id` | `null` | Restrict results to actors in this show (show-aware search) |
| `top_k` | `5` | Number of candidates to return (1–20) |

### Index rebuild strategies

- `individual` (default): one FAISS entry per stored embedding row — best with 1–2 samples per voice label
- `centroid`: one entry per `(actor_id, voice_label)` group averaged and re-normalized — recommended once you have ≥ 3 samples per label; reduces index size and improves stability

## Testing Patterns

### ML mocking in conftest.py

All heavy dependencies are patched before `api.py` is imported:

```python
with (
    patch("pipeline.isolate_vocals", side_effect=lambda p, **kw: p),
    patch("pipeline._load_embedding_model", return_value=fake_encoder),
    patch("torchaudio.load", return_value=fake_signal),
    patch.dict("os.environ", {"DB_PATH": db_path, "HF_TOKEN": ""}),
):
    import importlib
    import api as api_module
    importlib.reload(api_module)
    with TestClient(api_module.app) as client:
        yield client
```

**Important:** `api_module` must be reloaded after patches are applied so the `lifespan` context manager picks them up. Always use `importlib.reload()` rather than a plain `import`.

### Shared fixtures

| Fixture | Description |
|---------|-------------|
| `tmp_wav` | Temporary silent 16 kHz WAV file path |
| `random_embedding` | Random L2-normalized 192-dim `float32` array (seed=42) |
| `db` | Fresh `VazamDB` backed by a temp SQLite file, auto-deleted |
| `api_client` | `TestClient` for `api.py` with all ML mocked |

### Writing new tests

- Import `make_wav_bytes()` from `tests.conftest` to create in-memory WAV bytes for file uploads
- Use `files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")}` in multipart requests
- Tests do not require `HF_TOKEN` — diarization/VAD are disabled when it is empty
- Keep ML models mocked; never download real models in tests

## Code Conventions

### Python (backend)

- Python 3.10+ features (`match`, union types with `|`) are used
- `from __future__ import annotations` is present in all modules (for PEP 563 deferred evaluation)
- Type annotations everywhere; prefer `Optional[X]` for nullable parameters in public APIs
- Module-level docstrings document the module's purpose, public interface, and usage examples
- Use `dataclasses` for plain data containers (`SpeakerSegment`, `IdentificationResult`)
- Database access goes exclusively through `VazamDB` — no raw SQL outside `db.py`
- All embeddings are `float32` numpy arrays, L2-normalized to unit length
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

This improves accuracy for actors who substantially alter their voice between roles (e.g., Seth MacFarlane as Peter Griffin vs. Stewie Griffin vs. Quagmire). The `voice_label` field on the `embeddings` table holds the label; "Natural Voice" is the default.

## Show-Aware Search

Pass `show_id` to `POST /identify` to restrict matching to actors known to appear in that show. This converts an open-set speaker recognition problem into a closed-set one, which significantly improves accuracy. The filter is applied post-FAISS-search by intersecting results with `db.get_actor_ids_for_show(show_id)`.

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
  -F "voice_label=Spike Spiegel"
  -F "isolate=true"
```

### Identify a voice

```bash
curl -X POST http://localhost:8000/identify \
  -F "audio=@test_clip.mp3" \
  -F "isolate=true" \
  -F "top_k=5"
```

### Rebuild FAISS index after bulk import

```bash
# Use centroid strategy once you have ≥ 3 samples per voice label
curl -X POST "http://localhost:8000/index/rebuild?strategy=centroid"
```

### Seed metadata from AniList, then import audio

```bash
python seed.py --show "Cowboy Bebop"
python embed_batch.py samples/ --isolate --rebuild-index http://localhost:8000
```

## Upgrade Path: SQLite → PostgreSQL

`db.py` uses only standard SQL compatible with PostgreSQL. To migrate:

1. Replace `sqlite3.connect(...)` with a `psycopg2` or `asyncpg` connection
2. Change `?` placeholders to `%s` (psycopg2) or `$1` (asyncpg)
3. Change `AUTOINCREMENT` → `SERIAL` / `GENERATED ALWAYS AS IDENTITY`
4. Remove `PRAGMA` statements; configure WAL at the server level

FAISS index remains in-memory regardless of the DB backend.
