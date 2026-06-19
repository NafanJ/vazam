# CLAUDE.md — Vazam Codebase Guide

Vazam is "Shazam for Voice Actors": point a phone at any animated show, anime, or video game and instantly identify which voice actor is speaking. This file documents the codebase structure, development conventions, and workflows for AI assistants working on this project.

## Project Overview

**Version:** 0.2.0
**Stack:** Python 3 (FastAPI backend) + TypeScript/React Native (mobile frontend)
**Database & vector search:** Supabase (PostgreSQL + pgvector, cosine similarity via a `match_embeddings()` SQL function)

## Architecture

Audio processing pipeline (backend):

```
Audio clip
  → Demucs v4 htdemucs_ft   (vocal isolation — strips music & SFX)
  → pyannote VAD             (trim silence, keep only speech)
  → pyannote diarization-3.1 (split multi-speaker clips per speaker)
  → SpeechBrain ECAPA-TDNN  (192-dim speaker embedding)
  → pgvector                 (cosine similarity via match_embeddings RPC)
  → multi-window verification (candidates must win overlapping sub-windows)
  → Supabase tables          (actor → character → show metadata lookup)
  → "Steve Blum as Spike Spiegel (Cowboy Bebop) — 94% confidence"
```

Two higher-level strategies sit on top:

- **Show-aware search** — `show_id` restricts matching to a known cast (closed-set, much more accurate). The filter is applied inside the `match_embeddings()` SQL function.
- **Cast-graph show inference** (`/identify/show`) — diarize the clip, match each speaker globally, and vote on shows by cast co-occurrence; when ≥ 2 distinct speakers share a cast, re-rank everyone closed-set within it. Gives show-aware accuracy without asking the user which show is playing.

## Directory Structure

```
vazam/
├── api.py             FastAPI HTTP backend (main entry point for the server)
├── pipeline.py        Full pipeline: isolation → VAD → diarize → embed → search → verify
├── db.py              Supabase wrapper: actors, shows, characters, embeddings tables
├── consensus.py       Cross-video consensus clustering for scraped voices (pure numpy)
├── augment.py         Channel augmentation: reverb / noise / band-limit (pure DSP)
├── scrape_audio.py    Consensus voice scraper (yt-dlp → diarize → consensus → store)
├── eval.py            Accuracy benchmark harness (labelled clips → top-k metrics)
├── seed.py            AniList GraphQL seeder (metadata only, no audio)
├── embed_batch.py     Bulk audio-to-embedding importer (CLI tool)
├── add_character_voice.py  Lazy per-character voice embedding ingestion (CLI tool)
├── requirements.txt   Python dependencies
├── web/               Web dashboard source — React + Vite + TS (see Web Dashboard)
│   ├── index.html / characters.html   Vite entry shells (GET / and /characters.html)
│   ├── vite.config.ts                 Two-page build → ../static/dist
│   └── src/
│       ├── shared/    api client, types, WAV 16k encode, formatting helpers
│       ├── identify/  Identify screen (App + recorder/progress hooks + components)
│       └── characters/  Character admin (App + DetailModal)
├── static/dist/       Built dashboard (gitignored; produced by `cd web && npm run build`)
├── Dockerfile         CPU image (torch CPU wheels + ffmpeg) + node web-build stage; see Deployment
├── docker-compose.yml Own stack, joins existing cloudflared network (Cloudflare tunnel)
├── migrations/        SQL migrations (apply via Supabase SQL editor / db push)
│   ├── 001_embedding_quality.sql
│   ├── 002_character_occupation.sql
│   └── 003_match_embeddings_character_art.sql
├── benchmark/         Labelled eval clips (audio gitignored; see its README
│                      for the phone-mic recording protocol)
├── docs/
│   ├── data-acquisition-plan.md   Strategy for scaling voice-actor coverage
│   └── next-steps.md              "Where were we?" status doc — read on resume
├── tests/
│   ├── conftest.py    Shared fixtures; in-memory fake Supabase + mocked ML
│   ├── test_api.py    FastAPI endpoint integration tests
│   ├── test_db.py     Database CRUD tests
│   ├── test_pipeline.py        Pipeline logic, window verification
│   ├── test_consensus.py       Consensus clustering + scraper helpers
│   ├── test_show_inference.py  Cast-graph voting + identify_show
│   ├── test_eval.py            Benchmark harness
│   └── test_augment.py         Channel augmentation DSP + centroid folding
└── app/               React Native mobile application
    ├── package.json    Scripts: test, lint, typecheck, ios/android/start
    ├── tsconfig.json / .eslintrc.js / babel.config.js / metro.config.js
    ├── App.tsx         Root navigator
    └── src/
        ├── api/vazam.ts        Typed Axios client (identify, identifyShow, …)
        ├── api/__tests__/      Jest tests with axios mocked
        ├── types/index.ts      Shared TypeScript types + navigation params
        ├── screens/            HomeScreen, ResultsScreen, ActorProfileScreen, ShowSearchScreen
        ├── components/         RecordButton, ResultCard
        └── hooks/useRecorder.ts  Audio recording logic
```

## Environment Variables

| Variable       | Default   | Description                                                       |
|----------------|-----------|-------------------------------------------------------------------|
| `SUPABASE_URL` | _(empty)_ | Supabase project URL (`https://<ref>.supabase.co`)                |
| `SUPABASE_KEY` | _(empty)_ | Supabase service-role key (writes require service role)           |
| `HF_TOKEN`     | _(empty)_ | HuggingFace token — required for VAD + diarization (pyannote). Accept terms at `hf.co/pyannote/voice-activity-detection` and `hf.co/pyannote/speaker-diarization-3.1` |
| `DEVICE`       | auto      | `cuda` or `cpu` — if unset, auto-detects CUDA                     |

A `.env` file is loaded automatically (`python-dotenv`). Without `HF_TOKEN`, VAD and diarization are disabled: single-speaker identification still works on the full audio, but `/identify/show` falls back to global results and the scraper degrades to whole-clip embeddings.

## Database (Supabase + pgvector)

Tables (all prefixed `vazam_`):

```
vazam_actors      — id, name, bio, image_url, anilist_id (UNIQUE upsert key)
vazam_shows       — id, title, media_type, year, image_url, anilist_id (UNIQUE)
                    media_type ∈ {anime, cartoon, game, other}
vazam_characters  — id, name, show_id, actor_id, image_url, anilist_id (UNIQUE)
vazam_embeddings  — id, actor_id, character_id, voice_label (default "Natural Voice"),
                    embedding vector(192), audio_source, verified, contributor_id,
                    source_url, duration_s, quality_score
```

- Similarity search goes through the `match_embeddings(query_embedding, top_k, show_id_filter)` PostgreSQL function (pgvector `<=>` cosine distance), called via Supabase RPC. It returns `actor_id, actor_name, voice_label, similarity, character_id, image_url, show_title` (the last three were added by migration `003` so results carry character art + the show name).
- Embeddings are stored as `vector(192)` (passed as Python lists, always L2-normalized float32).
- `add_actor` / `add_show` / `add_character` are upserts on `anilist_id` — safe to re-run.
- **Migrations:** the base schema and `match_embeddings()` were applied to Supabase manually, before `migrations/` existed. New schema changes live in `migrations/*.sql` and must be applied via the Supabase SQL editor (or `supabase db push`). `001_embedding_quality.sql` (source_url, duration_s, quality_score) is required — `db.add_embedding` always sends those columns. `002_character_occupation.sql` adds `vazam_characters.occupation` (edited from the dashboard). `003_match_embeddings_character_art.sql` extends `match_embeddings()` to also return `character_id`, `image_url`, and `show_title` (drop+recreate — the return signature changed).

- **Deployed DB is character-voice-only.** The "Natural Voice" consensus embeddings were deleted; every live embedding is a per-character voice (`voice_label` = character, `character_id` linked). `db.add_embedding` still defaults `voice_label="Natural Voice"`, but nothing in the live DB uses it. See `docs/next-steps.md` (17 June 2026 update).

## Development Setup

### Backend

```bash
pip install -r requirements.txt

# Optional: install test dependencies (commented out in requirements.txt)
pip install pytest httpx

# Required: Supabase project credentials
export SUPABASE_URL=https://<ref>.supabase.co
export SUPABASE_KEY=<service_role_key>

# Optional but strongly recommended: enables VAD, diarization, show inference
export HF_TOKEN=hf_...

# Run the API server
uvicorn api:app --reload
# Interactive docs: http://localhost:8000/docs
```

### Mobile App

```bash
cd app
npm install        # resolves cleanly; react-test-renderer is pinned to React 18.2

npm run ios        # or: npm run android, npm run start (Metro only)
```

## Running Tests

Backend tests mock all heavy ML dependencies (SpeechBrain, Demucs, pyannote) and replace the Supabase client with an in-memory fake — no GPU, network, or model downloads needed.

```bash
pytest                                   # all backend tests
pytest tests/test_api.py -v              # one file
pytest tests/test_api.py::test_identify_returns_results -v
```

Mobile app:

```bash
cd app
npm test           # jest (react-native preset, axios mocked)
npm run lint       # ESLint — correctness rules only; formatting rules are off
npm run typecheck  # tsc --noEmit, strict via @react-native/typescript-config
```

**Measure accuracy changes.** Any change that could affect identification accuracy (thresholds, embeddings, scraper, pipeline stages) should be evaluated before/after with `eval.py` against the labelled clips in `benchmark/` (see `benchmark/README.md` — clips must be phone-mic recordings, not clean rips).

## Key Modules

### `api.py` — FastAPI Backend

- Entry point: `uvicorn api:app --reload`
- Global state: `db: VazamDB` and `pipeline: VazamPipeline` initialized in the `lifespan` context manager
- CORS is open (`allow_origins=["*"]`) — restrict in production
- `POST /index/rebuild` is a backward-compatibility no-op: pgvector manages its index automatically

### `pipeline.py` — Audio Processing

Key symbols:

| Symbol | Description |
|--------|-------------|
| `VazamPipeline` | Orchestrator — isolation/VAD/diarize/embed/search/verify |
| `VazamPipeline.identify()` | Single-speaker ID with multi-window verification (`verify=True` default) |
| `VazamPipeline.identify_multi()` | Diarize + identify each speech turn separately |
| `VazamPipeline.identify_show()` | Cast-graph show inference → closed-set re-rank; returns `(ShowInference \| None, per_speaker_results)` |
| `vote_shows()` | Pure cast co-occurrence voting over per-speaker candidates |
| `split_windows()` | Overlapping sub-windows of a speech tensor for verification |
| `load_audio_16k()` | Load any file as (1, N) mono 16 kHz tensor |
| `get_embedding()` | File path or (1, N) tensor → 192-dim L2-normalized float32 array |
| `isolate_vocals()` | Demucs via subprocess; falls back to original file on failure |
| `diarize()` / `merge_speaker_segments()` | pyannote diarization + same-speaker merging |

Constants:

```python
EMBEDDING_DIM       = 192
CONFIDENT_THRESHOLD = 0.70   # cosine ≥ 0.70 AND window-verified → "confident"
POSSIBLE_THRESHOLD  = 0.50   # 0.50–0.69, or failed verification → "possible"
MIN_SPEECH_SECONDS  = 1.5    # segments shorter than this are dropped
VERIFY_WINDOWS      = 3      # overlapping half-length windows per query
MIN_WINDOW_SECONDS  = 2.0    # clips under ~4s of speech skip verification
WINDOW_AGREEMENT_THRESHOLD = 0.5   # majority of windows required
SHOW_INFERENCE_MIN_SPEAKERS = 2    # speakers needed to infer a show
```

**Multi-window verification:** a "confident" match must clear the similarity threshold *and* win a majority of 3 overlapping sub-windows of the query (a consistent voice wins every window; a coincidental near-neighbor wins one). High-similarity matches that fail verification demote to "possible". Results carry `window_agreement` (None when the clip was too short to window).

Heavy models are module-level globals (`_classifier`, `_vad_pipeline`, `_diarize_pipeline`) loaded lazily on first call so the module imports quickly.

### `db.py` — Supabase Wrapper

All database access goes through `VazamDB` — no raw queries elsewhere. Notable methods: `search_embeddings()` (match_embeddings RPC), `get_shows_for_actors()` (powers cast-graph voting), `add_embedding()` (with `source_url` / `duration_s` / `quality_score`).

### `consensus.py` + `scrape_audio.py` — Consensus Voice Scraper

Implements the strategy in `docs/data-acquisition-plan.md`. For each actor:

1. Search YouTube with several independent query templates (interview / panel / podcast); pick 2–3 **distinct** videos (2–60 min), download only the first 10 minutes of each.
2. Diarize each video; embed every speaker with ≥ 8s of speech (capped at 60s).
3. Cluster speakers **across** videos (`consensus.py`, greedy centroid linkage, link threshold 0.60). The actor is the cluster recurring in ≥ 2 distinct videos — interviewers differ per video. No recurrence → **nothing stored**, never a blind embedding.
4. Re-embed the winning cluster's speech under simulated query conditions (`augment.py`: synthetic-RIR reverb, white noise @ 15 dB SNR, TV-speaker band-limiting — deterministic/seeded) and fold those into the stored centroid, so references match the phone-mic-of-a-TV channel they'll be searched against. Quality score stays computed on clean embeddings. Disable with `--no-augment`.

```bash
python scrape_audio.py                      # all actors with zero embeddings
python scrape_audio.py --actor "Steve Blum" --force   # re-scrape one actor
python scrape_audio.py --dry-run            # print queries only
```

Demucs is intentionally skipped here (interviews have no music bed). Per-character clip scraping is deliberately **not** done in bulk (see anti-goal in the acquisition plan).

### `eval.py` — Accuracy Benchmark

Runs labelled clips (`benchmark/<show>/<actor>/<clip>`) through `identify()` and reports top-1/3/5 accuracy (global and show-aware), confident-claim precision, per-show breakdown, and avg latency. `--json out.json` for tracking runs over time; `--no-verify` / `--isolate` to A/B pipeline options.

### `seed.py` — AniList Metadata Seeder

Fetches popular anime, characters, and voice actor credits from the free AniList GraphQL API and upserts into Supabase. Does **not** download audio. Idempotent (upserts on `anilist_id`).

```bash
python seed.py                          # top 200 anime, English VAs
python seed.py --show "Cowboy Bebop"    # single show
python seed.py --lang JAPANESE          # Japanese seiyuu
```

Rate limit: AniList allows ~90 requests/minute; default delay is 0.7s.

### `embed_batch.py` — Bulk Audio Importer

Walks `samples/<Actor Name>/<Voice Label>/*.{wav,mp3}` and stores one embedding per file (voice label from the subdirectory name; `natural/` → "Natural Voice").

```bash
python embed_batch.py samples/ --dry-run
python embed_batch.py samples/ --isolate --verified
```

### `add_character_voice.py` — Lazy Per-Character Voice Ingestion

Stores a *character-voice* embedding (e.g. Yuuki Kaji **as Eren Yeager**) for one marquee role from one or more curated YouTube clips or local files — closing the open-set gap between an actor's "Natural Voice" reference and their in-character voice. Deliberately lazy/selective (per the acquisition-plan anti-goal, character clips are never bulk-scraped).

Per source: download (or load local) → optional Demucs isolation (`--isolate`, recommended for anime clips with a music bed; `--demucs-model` / `DEMUCS_MODEL` selects the model — default `htdemucs_ft`, or `htdemucs` for ~3.7× faster isolation at near-identical quality, measured below) → diarize → embed the selected speaker. Two selection policies (`--select`): **`dominant`** (default — the speaker with the most speech, i.e. the character in a single-character compilation) or **`nearest-natural`** (the diarized speaker closest to the actor's stored Natural Voice — for *ensemble* characters who never speak solo, so the dominant speaker in their clips is someone else; requires a stored Natural Voice or the run aborts with `no_natural`). With ≥2 sources, the per-source embeddings must agree (mean pairwise cosine ≥ `MIN_CLIP_AGREEMENT = 0.55`) or nothing is stored — this rejects clips where the selected speaker isn't consistently the character. Stored as `voice_label=<character>` with `character_id` linked (so results name the character), centroid + channel augmentation folded in (disable with `--no-augment`); cross-source agreement becomes the `quality_score`.

```bash
python add_character_voice.py --actor "Yuuki Kaji" --character "Eren Yeager" \
    --show "Attack on Titan" --isolate \
    --url <compilation1> --url <compilation2>
python add_character_voice.py --actor "Steve Blum" --character "Spike Spiegel" \
    --file spike.wav --no-augment --dry-run
```

Reuses the scraper's download/audio/augment helpers and `consensus.centroid` / `mean_pairwise_cosine`. Validated on three AoT leads (the speaker's top match on a real scene): **Eren** `Natural Voice` 0.525 → `Eren Yeager` 0.686 (margin +0.020 → +0.181, near-tie → decisive); **Levi** natural voice didn't even rank → `Levi` 0.917 (rescued a confident *mis*-ID); **Mikasa** 0.690, beating a prior wrong match *and* recovering a seiyuu the consensus scraper had missed. The `nearest-natural` policy is proven on an *ensemble* role: **Connie** (Hiro Shimono), ingested from a Sasha/Connie/Jean group clip where Connie isn't the dominant speaker, flipped a blind clip from 0.432 (correct but below the 0.50 claim line) → **0.675 verified `as Connie Springer`** (+0.243). **Annie** (Yuu Shimamura) and **Hange** (Romi Park) were added with the fast `htdemucs` model and validated on held-out scenes via diarized `identify_multi`: Annie → `Annie Leonhart` **0.796 confident** (beat Armin's seiyuu in-scene — strongest yet); Hange → `Hange` **0.543 possible** (correct, rescued a non-ranking actor). **Erwin** (Daisuke Ono, who had *zero* embeddings — a consensus miss) → `Erwin Smith` **0.639 possible, window-verified 1.00** on his held-out charge-speech scene, recovering an unidentifiable actor (same pattern as Mikasa). Note: a *taciturn/ensemble* character needs a **dialogue** scene + `identify_multi` to validate — single-speaker `identify()` locks onto whoever talks most (an Annie combat clip matched Eren). **Demucs model A/B** (Connie clip, CPU): `htdemucs_ft` 1550s vs `htdemucs` 421s (**~3.7× faster**) with the two resulting embeddings agreeing at cosine **0.972** — so `htdemucs` is the recommended model for batch ingestion; `htdemucs_ft` stays the default for one-off marquee adds. **Second show — all 10 One Piece Straw Hats** were added the same way (character-voice-only; the crew has no consensus natural voices) from **game voice-galleries** (海賊無双4 / TREASURE CRUISE 「全システムボイスセリフ集 / ボイス集」 — single-character, narrator-free; the One-Piece equivalent of AoT's trusted セリフ切り抜き uploader). Validated **cross-game** (train one game, test a different one): Chopper **0.864**, Usopp **0.793** (from 14s), Zoro **0.767**, Luffy **0.721** — all confident. Two One Piece gotchas: (1) fan 名言集 compilations carry a **narrator** that poisons the embedding (Zoro's first attempt *was* the narrator — use game galleries); (2) **combat/skill clips are bad validation targets** (battle shouts + SFX) — validate on dialogue. **Seed gotcha:** AniList mislinks long-running characters to secondary/wrong VAs (era recasts, young-version flashbacks) — 6 of 10 Straw Hats needed `vazam_characters.actor_id` re-linked to the canonical seiyuu after seeding. Always verify character→VA mapping for a long-running show before scraping.

## API Reference

Interactive docs at `http://localhost:8000/docs` when the server is running.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Web dashboard (`static/index.html`); JSON API info if the file is absent |
| `GET`  | `/characters.html` | Character admin page (`static/characters.html`) |
| `POST` | `/identify` | Upload audio → top-K matches (params: `isolate`, `show_id`, `top_k`, `verify`) |
| `POST` | `/identify/stream` | Same as `/identify` but streams NDJSON progress stages then the result. Exists for API users; the dashboard now uses plain `/identify` (survives the Cloudflare tunnel better) |
| `POST` | `/identify/multi` | Diarize + identify each speaker separately (requires `HF_TOKEN`) |
| `POST` | `/identify/show` | Infer the show from cast co-occurrence, then identify each speaker within it |
| `POST` | `/enroll` | Add an uploaded clip to a character's fingerprint (`actor_id`, `voice_label`, `isolate`) — powers the dashboard "Correct?/Incorrect?" buttons |
| `GET`  | `/voices` | Distinct stored voices (actor + `voice_label` + sample counts) for the enroll picker |
| `GET`  | `/characters` | All characters with `actor_name`, `show_title`, `image_url`, `occupation`, `samples` |
| `GET`  | `/characters/{id}` | One character + its embedding sources |
| `PATCH`| `/characters/{id}` | Edit `image_url` / `occupation` (from the admin page) |
| `POST` | `/actors` | Register a voice actor |
| `GET`  | `/actors` | List all actors (supports `limit`, `offset`) |
| `GET`  | `/actors/{id}` | Actor profile + filmography |
| `POST` | `/actors/{id}/embeddings` | Upload audio sample → generate + store embedding |
| `POST` | `/shows` | Register a show |
| `GET`  | `/shows` | List all shows |
| `GET`  | `/shows/{id}` | Show details + cast list |
| `GET`  | `/shows/search?q=` | Search shows by title (ILIKE match) |
| `POST` | `/index/rebuild` | No-op kept for compatibility (pgvector self-manages) |
| `GET`  | `/health` | Liveness check + embedding count |

Identification responses include `confidence`, `window_agreement`, `match_level` (`confident` / `possible` / `none`), and (since migration `003`) `character_id`, `image_url`, and `show_title`. `/identify/show` responses include `show` (`null` when no cast consensus) with `speakers_matched` / `speakers_total`.

## Web Dashboard (`web/` → `static/dist/`)

A **React + Vite + TypeScript** app in `web/`, built to `static/dist/` and served by `api.py` (`GET /` → `dist/index.html`, `GET /characters.html` → `dist/characters.html`, hashed bundles mounted at `/assets`). It is the primary way clips are recorded and enrolled from a phone. Two-page build (`web/vite.config.ts`); shared code (typed API client, types, WAV 16k encoder, formatters) lives in `web/src/shared/`. The class names + CSS are ported verbatim from the previous vanilla pages, so the design is unchanged.

Develop with `cd web && npm install && npm run dev` (Vite dev server; proxy or run the API alongside). Build with `npm run build` (also `npm run typecheck`). The build output is gitignored — `api.py` falls back to the JSON API-info response and the HTML routes 404 until a build exists.

- **Identify** (`web/src/identify/`, `GET /`) — record (MediaRecorder, `useRecorder`) or upload a clip → decode to 16 kHz mono 16-bit WAV client-side (`shared/wav.ts` `blobToWav16k`, 20s cap) → **plain `POST /identify`** (single voice) or `POST /identify/show` (scene mode). Pipeline stages are simulated client-side (`useProgress`, which solely owns the elapsed timer so an errored request can't leak it). A `running` state flag blocks double-submit. Result cards render character art (`image_url`) + show chip, a match-level pill, and two enroll actions: **"Correct?"** (`POST /enroll`) and **"Incorrect?"** (opens the picker to route the clip to the right character). Also has a YouTube-link path (`POST /fetch/url` → play/trim → identify/enroll the selection) and a localStorage recording history.
- **Characters** (`web/src/characters/`, `GET /characters.html`) — browse/search 650+ characters (voiced first), paginated 50/page, edit `image_url` / `occupation` (`PATCH /characters/{id}`) and delete sources (`DELETE /embeddings/{id}`) in the `DetailModal`, with inline clip playback (`/embeddings/{id}/audio`).

Failed requests surface real errors (no mock fallback). The React Native `app/` is a separate, not-yet-shipped mobile frontend — the web dashboard fills that role.

## Deployment (Docker + Cloudflare Tunnel)

`Dockerfile` is multi-stage: a `node:20-slim` **`webbuild`** stage runs `npm ci && npm run build` in `web/` (→ `/static/dist`), then the CPU Python image (torch CPU wheels + ffmpeg/libsndfile, `DEVICE=cpu`, `DEMUCS_MODEL=htdemucs`, models cached in a `/models` volume) copies that build in via `COPY --from=webbuild`. So the dashboard is built reproducibly inside the image — no Node needed on the host or in the serving path, and `static/dist` is `.dockerignore`d so the in-image build is the single source. `docker-compose.yml` runs it as its **own stack** that joins the existing `media-server_default` network so a dashboard-managed **cloudflared** container reaches it at `http://vazam:8000`; a Cloudflare Public Hostname publishes it. Every route except `/health` is gated by **HTTP Basic auth** (`VAZAM_AUTH_USER` / `VAZAM_AUTH_PASS`). Deployed copy lives at `/opt/vazam` (an rsync copy, not a git clone): sync changed files (now including `web/`), then `docker compose build && docker compose up -d` (heavy torch/pip layers cache; the web build re-runs only when `web/` changes). Secrets come from a `.env` next to the compose file (gitignored).

## Testing Patterns

### conftest.py — fake Supabase + mocked ML

The Supabase client is replaced by an in-memory fake (`_FakeSupabase`: query-builder supporting insert/upsert/select/eq/in_/ilike/order/range, plus `set_rpc()` for `match_embeddings`). Heavy ML is patched before `api.py` is imported:

```python
with (
    patch("pipeline.isolate_vocals", side_effect=lambda p, **kw: p),
    patch("pipeline._load_embedding_model", return_value=fake_encoder),
    patch("torchaudio.load", return_value=fake_signal),
    patch("db.create_client", return_value=fake_sb),
    patch.dict("os.environ", {...}),
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
| `db` | `VazamDB` backed by the in-memory fake Supabase |
| `api_client` | `TestClient` for `api.py` with ML + Supabase mocked |

### Writing new tests

- Import `make_wav_bytes()` from `tests.conftest` for in-memory WAV uploads: `files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")}`
- Tests run with `HF_TOKEN=""` — diarization/VAD are disabled; mock `diarize` or `_speaker_embeddings` to exercise multi-speaker paths
- Pipeline unit tests patch `p._speech_tensor` + `pipeline.get_embedding` (not `embed_file`)
- Keep ML models mocked; never download real models in tests
- Synthetic "same voice" vectors need noise with fixed total norm (per-component noise swamps the signal in 192 dims) — see `_appearance()` in `test_consensus.py`

## Code Conventions

### Python (backend)

- Python 3.10+ features (`match`, union types with `|`) are used
- `from __future__ import annotations` is present in all modules
- Type annotations everywhere; prefer `Optional[X]` for nullable parameters in public APIs
- Module-level docstrings document the module's purpose, public interface, and usage examples
- Use `dataclasses` for plain data containers (`SpeakerSegment`, `IdentificationResult`, `ShowVote`, `ConsensusResult`, …)
- Database access goes exclusively through `VazamDB` — no Supabase calls outside `db.py`
- All embeddings are `float32` numpy arrays, L2-normalized to unit length
- Temporary files from uploads are cleaned up in `finally` blocks (see `_save_upload` in `api.py`)
- Heavy imports (SpeechBrain, pyannote, torch in CLI tools) are inside functions, not at module top-level, to keep startup fast

### TypeScript / React Native (frontend)

- TypeScript 5; strict mode via `@react-native/typescript-config` (`npm run typecheck`)
- Navigation uses React Navigation v6 with typed stack params (defined in `src/types/index.ts`)
- All API calls go through the typed client in `src/api/vazam.ts` — never call `axios` directly from screens
- State management via Zustand; audio recording via the `useRecorder` hook
- ESLint enforced on `src/**/*.{ts,tsx}` via `npm run lint`; formatting rules (`prettier/prettier`, `quotes`) are intentionally off — the codebase uses double quotes

## Identification Strategies

### Multi-embedding per voice style

The `voice_label` column lets an actor hold **separate embeddings per voice style** — "Natural Voice" (interviews/panels, the consensus scraper's output) and/or per-character voices — which matters for actors who substantially alter their voice between roles (e.g., Seth MacFarlane as Peter vs. Stewie Griffin). Per-character embeddings are added lazily and selectively, never in bulk. **Current live DB:** all "Natural Voice" embeddings were deleted; it is now character-voice-only (every embedding has `voice_label` = character + a linked `character_id`). The dual-style mechanism still exists in code for actors who'd benefit from a natural-voice anchor.

### Show-aware search

Pass `show_id` to `POST /identify` to restrict matching to that show's cast — converts open-set recognition into closed-set, which significantly improves accuracy. Applied inside the `match_embeddings()` SQL function.

### Cast-graph show inference

`POST /identify/show` recovers show-aware accuracy with zero user input: each diarized speaker votes for shows whose casts contain a plausible candidate (similarity ≥ 0.50); shows are ranked by distinct speakers explained, then summed similarity. Multi-speaker TV audio — the hardest case for plain identification — is the *strongest* signal here. The mobile HomeScreen uses this path whenever the user hasn't picked a show filter.

## Common Workflows

### Seed metadata, scrape voices, evaluate

```bash
python seed.py --show "Cowboy Bebop"     # cast + characters from AniList
python scrape_audio.py --limit 30        # consensus-scrape actors without embeddings
python eval.py benchmark/                # measure (after recording labelled clips)
```

### Add a voice actor and sample manually

```bash
curl -X POST http://localhost:8000/actors \
  -H "Content-Type: application/json" -d '{"name": "Steve Blum"}'
# → {"id": 1, ...}

curl -X POST http://localhost:8000/actors/1/embeddings \
  -F "audio=@blum_interview.wav" -F "voice_label=Natural Voice"
```

### Identify a voice

```bash
curl -X POST http://localhost:8000/identify \
  -F "audio=@test_clip.mp3" -F "isolate=true" -F "top_k=5"

# Or infer the show automatically from a multi-speaker clip:
curl -X POST http://localhost:8000/identify/show -F "audio=@tv_clip.mp3"
```

### Apply a schema migration

Run the SQL in `migrations/*.sql` via the Supabase SQL editor (or `supabase db push`). Do this before running code that depends on the new columns.
