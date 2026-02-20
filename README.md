# Vazam — Shazam for Voice Actors

Point your phone at any animated show, anime, or video game and instantly know which voice actor is performing.

## Architecture

```
Audio clip
  → Demucs v4          (vocal isolation — strips music & SFX)
  → pyannote VAD        (trim silence, keep only speech)
  → pyannote diarize    (split multi-speaker clips per speaker)
  → ECAPA-TDNN          (192-dim speaker embedding)
  → FAISS IndexFlatIP   (cosine similarity search)
  → PostgreSQL/SQLite   (actor → character → show metadata lookup)
  → "Steve Blum as Spike Spiegel (Cowboy Bebop) — 94% confidence"
```

## File structure

```
vazam/
├── main.py        standalone demo script (single-file quick test)
├── pipeline.py    full pipeline module: isolation → VAD → diarize → embed → search
├── db.py          SQLite database: actors, shows, characters, embeddings
├── api.py         FastAPI HTTP backend
├── seed.py        AniList GraphQL seeder (metadata only — no audio)
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
# Optional: set HuggingFace token for VAD + diarization
export HF_TOKEN=hf_...

uvicorn api:app --reload
# → http://localhost:8000/docs
```

### Seed metadata from AniList (free, no auth)

```bash
# Top 200 anime, English voice actors
python seed.py

# Single show
python seed.py --show "Cowboy Bebop"

# Japanese seiyuu
python seed.py --lang JAPANESE --limit 100
```

### Add audio samples and identify voices

```bash
# 1. Create an actor record
curl -X POST http://localhost:8000/actors \
  -H "Content-Type: application/json" \
  -d '{"name": "Steve Blum"}'
# → {"id": 1, ...}

# 2. Upload a clean audio sample (interview / demo reel)
curl -X POST http://localhost:8000/actors/1/embeddings \
  -F "audio=@samples/blum_interview.wav" \
  -F "voice_label=Natural Voice"

# 3. Upload a character voice sample
curl -X POST http://localhost:8000/actors/1/embeddings \
  -F "audio=@samples/blum_spike.wav" \
  -F "voice_label=Spike Spiegel"

# 4. Identify a voice from a show clip
curl -X POST http://localhost:8000/identify \
  -F "audio=@test_clip.mp3" \
  -F "isolate=true"
```

## Key design choices

| Component | Choice | Replaces |
|-----------|--------|---------|
| Feature extraction | SpeechBrain ECAPA-TDNN (192-dim) | 13-dim MFCC mean |
| Similarity metric | FAISS `IndexFlatIP` + L2-norm (cosine) | `IndexFlatL2` (Euclidean) |
| Voice isolation | Demucs v4 `htdemucs_ft` | none |
| VAD | pyannote voice-activity-detection | none |
| Diarization | pyannote speaker-diarization-3.1 | none |
| Metadata | SQLite (upgradeable to PostgreSQL) | none |
| API | FastAPI + uvicorn | none |

### Confidence thresholds

| Score | Label | Meaning |
|-------|-------|---------|
| ≥ 0.70 | `confident` | Very likely same speaker |
| 0.50 – 0.69 | `possible` | Possible match |
| < 0.50 | `none` | No usable match |

### Multi-embedding strategy

Each voice actor is stored with **separate embeddings per voice style**:

- **Natural voice** — from interviews, convention panels, or demo reels
- **Per-character voices** — one or more clips per distinct character

This improves accuracy for actors who alter their voice heavily between roles
(e.g. Seth MacFarlane as Peter vs. Stewie vs. Quagmire).

### Show-aware search

Pass `show_id` to `/identify` to restrict the search to actors known to appear
in that show. This turns an open-set speaker recognition problem into a
closed-set one, which dramatically improves accuracy.

## API reference

Interactive docs available at `http://localhost:8000/docs` when the server is running.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/identify` | Upload audio → top-K voice actor matches |
| `POST` | `/identify/multi` | Diarize → identify each speaker separately |
| `POST` | `/actors` | Register a voice actor |
| `GET`  | `/actors` | List all actors |
| `GET`  | `/actors/{id}` | Actor profile + filmography |
| `POST` | `/actors/{id}/embeddings` | Upload audio sample → store embedding |
| `POST` | `/shows` | Register a show |
| `GET`  | `/shows` | List all shows |
| `GET`  | `/shows/{id}` | Show details + cast |
| `GET`  | `/shows/search?q=` | Search shows by title |
| `POST` | `/index/rebuild` | Rebuild FAISS index from DB |
| `GET`  | `/health` | Liveness + index stats |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(empty)_ | HuggingFace token for pyannote VAD/diarization |
| `DB_PATH` | `vazam.db` | SQLite database path |
| `DEVICE` | auto | `cuda` or `cpu` |
