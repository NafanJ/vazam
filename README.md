# Vazam — Shazam for Voice Actors

Point your phone at any animated show, anime, or video game and instantly know which voice actor is performing.

> **Status — June 2026.** The backend pipeline is validated end-to-end on real audio.
> **25 reference voices** across two shows: 8 consensus "Natural Voice" embeddings (Steve Blum +
> 7 *Attack on Titan* seiyuu) plus **17 per-character voices** — 7 *Attack on Titan*
> (Eren, Levi, Mikasa, Connie, Annie, Hange, Erwin) and the **10 *One Piece* Straw Hats**
> (Luffy, Zoro, Nami, Usopp, Sanji, Chopper, Robin, Franky, Brook, Jinbe). A held-out
> interview matched its actor at cosine **0.884**, and `/identify/show` infers the show from a
> raw multi-speaker anime scene with no hints. Full validation log + roadmap:
> [`docs/next-steps.md`](docs/next-steps.md). Architecture & conventions: [`CLAUDE.md`](CLAUDE.md).

## Architecture

```
Audio clip
  → Demucs v4 htdemucs_ft   (vocal isolation — strips music & SFX)
  → pyannote VAD            (trim silence, keep only speech)
  → pyannote diarization    (split multi-speaker clips per speaker)
  → SpeechBrain ECAPA-TDNN  (192-dim speaker embedding)
  → Supabase + pgvector     (cosine similarity via match_embeddings RPC)
  → multi-window verification (a real voice wins every sub-window; a coincidence wins one)
  → Supabase tables          (actor → character → show metadata lookup)
  → "Steve Blum as Spike Spiegel (Cowboy Bebop) — 94% confidence"
```

Two higher-level strategies sit on top:

- **Show-aware search** — pass a `show_id` to restrict matching to a known cast (closed-set, much more accurate).
- **Cast-graph show inference** (`/identify/show`) — diarize a clip, match each speaker globally, and vote on shows by cast co-occurrence; gives show-aware accuracy without asking the user which show is playing.

## File structure

```
vazam/
├── api.py             FastAPI HTTP backend (main entry point)
├── pipeline.py        Full pipeline: isolation → VAD → diarize → embed → search → verify
├── db.py              Supabase wrapper (actors, shows, characters, embeddings)
├── consensus.py       Cross-video consensus clustering for scraped voices
├── augment.py         Channel augmentation: reverb / noise / band-limit
├── scrape_audio.py    Consensus voice scraper (yt-dlp → diarize → consensus → store)
├── add_character_voice.py  Lazy per-character voice embedding ingestion (CLI)
├── embed_batch.py     Bulk audio-to-embedding importer (CLI)
├── eval.py            Accuracy benchmark harness (labelled clips → top-k metrics)
├── seed.py            AniList GraphQL metadata seeder (no audio)
├── migrations/        SQL migrations (apply via Supabase SQL editor / db push)
├── benchmark/         Labelled eval clips (audio gitignored; see its README)
├── docs/              Data-acquisition plan + status/next-steps
└── app/               React Native mobile application
```

## Quick start

```bash
pip install -r requirements.txt          # backend deps (pytest httpx for tests)

# Required: Supabase project credentials (service role for writes)
export SUPABASE_URL=https://<ref>.supabase.co
export SUPABASE_KEY=<service_role_key>
# Strongly recommended: enables VAD, diarization, show inference
export HF_TOKEN=hf_...                    # accept terms for all three pyannote gates

uvicorn api:app --reload                  # → http://localhost:8000/docs
```

### Recording dashboard

A single-page dashboard is served at **`http://localhost:8000/`** — record from the mic (laptop /
HTTPS) or upload a clip, and it returns the **character** and the **actor** who plays them. It
streams a live progress log over the (CPU-bound) isolation step via `POST /identify/stream`, has an
optional **show filter** for closed-set accuracy, and trims long clips to the first 20s client-side.
For responsive isolation on CPU, run the server with the fast Demucs model:

```bash
DEMUCS_MODEL=htdemucs uvicorn api:app --host 0.0.0.0 --port 8000
```

(Mic recording needs a secure context — works on `localhost`; from a phone use file-upload or an
HTTPS tunnel like `cloudflared tunnel --url http://localhost:8000`.)

A `.env` file in the repo root is loaded automatically. See [`CLAUDE.md`](CLAUDE.md) for the
full setup, including the three gated pyannote repos (missing the VAD gate silently tanks
single-speaker accuracy).

### Add a voice actor and their characters

The voice tools resolve actors/characters **by name**, so metadata comes first:

```bash
# 1. Seed metadata (actor + characters + show) from AniList — idempotent
python seed.py --show "Cowboy Bebop"

# 2. Natural Voice reference — consensus scrape (required baseline)
python scrape_audio.py --actor "Steve Blum"            # English actor
python scrape_audio.py --actor "Yuuki Kaji" --lang both # JP seiyuu queries

# 3. (Optional, marquee roles) per-character voice — closes the open-set gap
python add_character_voice.py --actor "Yuuki Kaji" --character "Eren Yeager" \
    --show "Attack on Titan" --isolate --url <line-cut-1> --url <line-cut-2>
```

### Identify a voice

```bash
curl -F audio=@clip.mp3 -F isolate=true -F top_k=5 http://localhost:8000/identify

# Or infer the show automatically from a multi-speaker clip:
curl -F audio=@tv_scene.mp3 http://localhost:8000/identify/show
```

## Identification strategies

### Multi-embedding per voice style

Each actor stores **separate embeddings per voice style** (`voice_label`): a "Natural Voice"
(interviews/panels, from the consensus scraper) plus optional **per-character voices**. This
matters for actors who substantially alter their voice between roles. Per-character voices are
added **lazily and selectively** (never bulk-scraped) via `add_character_voice.py`.

Measured impact (June 2026, validated on real *Attack on Titan* scenes):

| Character (actor) | Natural Voice only | + per-character embedding |
|---|---|---|
| **Eren** (Yuuki Kaji) | 0.525 — near-tie with a rival (+0.020) | **0.686**, decisive (+0.181 margin) |
| **Levi** (Hiroshi Kamiya) | natural didn't rank — would mis-ID as the wrong actor | **0.917**, confident |
| **Mikasa** (Yui Ishikawa) | unidentifiable (actor had no embedding) | **0.690**, beats the prior wrong match |
| **Connie** (Hiro Shimono) | 0.432 — correct but below the claim line | **0.675**, verified, over the line |
| **Annie** (Yuu Shimamura) | natural didn't rank in-scene | **0.796**, confident — beat the co-star's seiyuu |
| **Hange** (Romi Park) | natural didn't rank in-scene | **0.543**, possible, window-verified |
| **Erwin** (Daisuke Ono) | unidentifiable (actor had no embedding) | **0.639**, possible, window-verified 1.00 |

For Levi, Mikasa, Annie, Hange, and Erwin the per-character embedding turned a wrong-or-no answer
into a correct one. (Mikasa and Erwin also recovered seiyuu the consensus scraper had missed.) Connie — an *ensemble*
character who never carries a scene solo — was added with `--select nearest-natural`, which picks
the diarized speaker nearest the actor's Natural Voice instead of the loudest, so it works on group
clips where the character isn't the dominant speaker.

**Second show — *One Piece* Straw Hats (June 2026).** All 10 crew character voices were added from
**game voice-galleries** (海賊無双4 / TREASURE CRUISE 「全システムボイスセリフ集」) — single-character,
narrator-free audio, ingested with the fast `htdemucs` model. Validated cross-game (trained on one
game, tested on a *different* one — a held-out recording of the same seiyuu):

| Straw Hat (seiyuu) | Held-out result |
|---|---|
| **Chopper** (Ikue Ōtani) | **0.864** confident, win 1.00 |
| **Usopp** (Kappei Yamaguchi) | **0.793** confident, win 1.00 (from just 14s of speech) |
| **Zoro** (Kazuya Nakai) | **0.767** confident, win 1.00 |
| **Luffy** (Mayumi Tanaka) | **0.721** confident, win 1.00 |

Nami, Sanji, Robin, Franky, Jinbe, and Brook were added from the same clean galleries and recur as
confident/possible runners-up in each other's tests. Two sourcing lessons: (1) One Piece fan
"名言集" compilations often carry a **narrator** that poisons the embedding — game voice-galleries
avoid it; (2) AniList seeds **multiple VAs per long-running character** (era changes, young-version
flashbacks, fill-ins), so 6 of the 10 Straw Hats had to be re-linked from a secondary VA to the
canonical seiyuu before scraping.

### Show-aware search

Pass `show_id` to `/identify` to restrict the search to that show's cast — turns open-set
recognition into closed-set, which significantly improves accuracy.

### Cast-graph show inference

`/identify/show` recovers show-aware accuracy with zero user input: each diarized speaker votes
for shows whose casts contain a plausible candidate; shows rank by distinct speakers explained.
Multi-speaker TV audio — the hardest case for plain identification — is the *strongest* signal here.

## API reference

Interactive docs at `http://localhost:8000/docs` when the server is running.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/identify` | Upload audio → top-K matches (`isolate`, `show_id`, `top_k`, `verify`) |
| `POST` | `/identify/multi` | Diarize → identify each speaker separately |
| `POST` | `/identify/show` | Infer the show from cast co-occurrence, then identify within it |
| `POST` | `/actors` · `GET` `/actors` · `GET` `/actors/{id}` | Register / list / profile + filmography |
| `POST` | `/actors/{id}/embeddings` | Upload audio sample → generate + store embedding |
| `POST` | `/shows` · `GET` `/shows` · `GET` `/shows/{id}` · `GET` `/shows/search?q=` | Show CRUD + search |
| `POST` | `/index/rebuild` | No-op kept for compatibility (pgvector self-manages) |
| `GET`  | `/health` | Liveness check + embedding count |

Identification responses include `confidence`, `window_agreement`, and `match_level`
(`confident` / `possible` / `none`).

### Confidence thresholds

| Score | Label | Meaning |
|-------|-------|---------|
| ≥ 0.70 | `confident` | Very likely same speaker (and window-verified) |
| 0.50 – 0.69 | `possible` | Possible match |
| < 0.50 | `none` | No usable match |

These are uncalibrated starting values — they'll be tuned against the benchmark clips
(see [`docs/next-steps.md`](docs/next-steps.md)).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | _(empty)_ | Supabase project URL (`https://<ref>.supabase.co`) |
| `SUPABASE_KEY` | _(empty)_ | Supabase service-role key (writes require service role) |
| `HF_TOKEN` | _(empty)_ | HuggingFace token — VAD + diarization (pyannote) |
| `DEVICE` | auto | `cuda` or `cpu` (auto-detects CUDA) |

## Mobile app

`app/` is a React Native client (typed Axios client, recording hook, show-inference wiring). It
has source, tests, and configs but **no `ios/`/`android/` native folders yet**, so it can't build
onto a phone — the backend Swagger UI at `/docs` is the full test surface in the meantime.

## Testing

```bash
pytest                 # backend (ML + Supabase mocked; no GPU/network needed)
cd app && npm test     # mobile (jest, axios mocked)
```
