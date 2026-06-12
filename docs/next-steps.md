# Vazam — Status & Next Steps

*Snapshot: June 2026. Companion to [data-acquisition-plan.md](data-acquisition-plan.md).
This is the "where were we?" doc — read this first when picking the project back up.*

## Where things stand

The full accuracy roadmap from the scaling deep-dive is **built, tested, and pushed**
(127 backend tests + 4 app tests passing):

| Piece | Status | Where |
|---|---|---|
| Consensus voice scraper (replaces blind first-hit) | ✅ | `scrape_audio.py`, `consensus.py` |
| Multi-window verification on `identify()` | ✅ | `pipeline.py` (`split_windows`, `window_agreement`) |
| Accuracy benchmark harness | ✅ | `eval.py`, `benchmark/README.md` |
| Cast-graph show inference (zero-input show-aware search) | ✅ | `pipeline.identify_show`, `POST /identify/show` |
| Channel augmentation of scraped references | ✅ | `augment.py`, folded in by the scraper |
| Mobile client + UI wiring for show inference | ✅ | `app/src/api/vazam.ts`, `HomeScreen`, `ResultsScreen` |
| App toolchain (install / lint / typecheck / test) | ✅ | `app/` configs |
| CLAUDE.md matches reality | ✅ | repo root |

**What the system is:** FastAPI backend (all the intelligence) + React Native mobile app.
There is no web frontend, but the backend's Swagger UI at `http://localhost:8000/docs`
is a full test surface — the app is not needed to exercise anything.

**Mobile app caveat:** `app/` has source, tests, and configs but **no `ios/` or
`android/` native project folders** — it cannot be built onto a phone yet. Generating
the native scaffolding is its own task (see below).

## To run it (one-time setup)

1. **Apply the migration** — run `migrations/001_embedding_quality.sql` in the
   Supabase SQL editor. Required: embedding inserts fail without it.
2. **`.env` in repo root:** `SUPABASE_URL`, `SUPABASE_KEY` (service role),
   `HF_TOKEN` (free HF token; accept terms at `hf.co/pyannote/voice-activity-detection`
   and `hf.co/pyannote/speaker-diarization-3.1`). Without `HF_TOKEN` the system
   degrades: no diarization → no show inference, weaker scraping.
3. **Start + seed + scrape:**

   ```bash
   .venv/bin/uvicorn api:app --reload      # → http://localhost:8000/docs
   python seed.py --show "Cowboy Bebop"    # cast metadata from AniList
   python scrape_audio.py --actor "Steve Blum" --force   # sanity-check one actor
   python scrape_audio.py --limit 20       # rest of the cast
   ```

   First scrape downloads models; CPU diarization is minutes per actor — start small.
4. **Test in the browser:** `/docs` → `POST /identify` with any clip;
   `POST /identify/show` with a two-character scene.

## Next steps (in order)

1. **Run the setup above** and prove the loop on one show.
2. **Record benchmark clips** — the one task only a human can do. ~20 clips per show
   across 3–5 shows, **phone mic pointed at a playing TV, not clean rips**
   (protocol in `benchmark/README.md`). Then:

   ```bash
   python eval.py benchmark/ --json runs/baseline.json
   ```

   Until this exists, every accuracy claim is unmeasured. It also unlocks the A/Bs:
   `eval.py --no-verify` (what verification buys) and re-scraping with
   `--no-augment` (what channel augmentation buys).
3. **Calibrate thresholds from evidence.** `CONFIDENT_THRESHOLD = 0.70` /
   `POSSIBLE_THRESHOLD = 0.50` in `pipeline.py` are uncalibrated guesses; tune them
   against benchmark numbers. Expect them to need raising as the DB grows
   (impostor scores rise with N); consider AS-norm score normalization if precision
   degrades at scale.
4. **Data plan build order** (from `data-acquisition-plan.md`):
   entity-resolution schema change (cross-source actor IDs — do first, painful to
   retrofit) → TMDB resolver alongside AniList → `show_ingestion_status` +
   demand-driven background ingestion → overnight warm-start of top shows.
5. **Generate RN native projects** (`ios/`, `android/`) so the app actually builds;
   point the client at the dev machine's LAN IP (`setBaseUrl`) instead of localhost.
6. **Cleanup:** delete or rewrite legacy `main.py` — it still imports FAISS
   (pre-Supabase era) and won't run with current requirements.

## Things to remember

- Audio is gitignored everywhere (benchmark clips, samples) — embeddings only,
  source URLs kept for provenance/takedown.
- Per-character embeddings stay **lazy-only, top-billed-only** (plan anti-goal:
  never bulk-scrape character voices; lean on show-aware search instead).
- Any change that could affect accuracy gets an `eval.py` before/after. No vibes.
