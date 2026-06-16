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

0. **System FFmpeg** — `sudo apt-get install ffmpeg`. yt-dlp needs it to extract
   MP3 audio in the scraper, and MP3 decoding needs it too. WAV/FLAC/OGG decode
   without it (soundfile backend), so the API/eval paths work, but the scraper
   does not until FFmpeg is present. Verify with `ffmpeg -version`.
   Note: `requirements.txt` pins `torch/torchaudio < 2.6` and `pyannote.audio < 4`
   on purpose — newer releases route audio I/O through torchcodec, which breaks on
   CPU-only boxes. Don't loosen those bounds without testing audio decode.
1. **Apply migrations** — run `migrations/000_base_schema.sql` then
   `001_embedding_quality.sql` in the Supabase SQL editor (or they're already
   applied to the current project via MCP). Inserts fail without them.
2. **`.env` in repo root:** `SUPABASE_URL`, `SUPABASE_KEY` (service role),
   `HF_TOKEN` (free HF token). Accept terms for **all three** gated pyannote repos
   — they are separate gates: `hf.co/pyannote/speaker-diarization-3.1` and
   `hf.co/pyannote/segmentation-3.0` (diarization → scraping + show inference) **and**
   `hf.co/pyannote/segmentation` (the *original* — powers VAD, used by single-speaker
   `identify()` to trim to speech). Missing the last one doesn't error loudly: VAD
   silently falls back to embedding the whole clip, which tanks `identify()` accuracy
   (measured: 0.88 → 0.10 on the same clip). Without `HF_TOKEN` entirely the system
   degrades further: no diarization → no show inference, weaker scraping.
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

## Live validation (June 2026) — proven end-to-end on real data

The pipeline has now been run for real, not just unit-tested. Current Supabase
project (`rpmcsbgtsvpoczpycozr`) holds **25 embeddings** across two shows: 8 consensus
"Natural Voice" (Steve Blum + 7 of 10 Attack on Titan main-cast seiyuu — Eren/Yuuki Kaji,
Armin/Marina Inoue, Levi/Hiroshi Kamiya, Hange/Romi Park, Annie/Yuu Shimamura,
Reiner/Yoshimasa Hosoya, Connie/Hiro Shimono; Erwin/Daisuke Ono, Jean/Kishou
Taniyama, Mikasa/Yui Ishikawa missed consensus) plus **17 per-character voices**:
7 *Attack on Titan* (`Eren Yeager`, `Levi`, `Mikasa Ackerman`, `Connie Springer`,
`Annie Leonhart`, `Hange`, `Erwin Smith`) and the 10 *One Piece* Straw Hats (`Luffy`,
`Zoro`, `Nami`, `Usopp`, `Sanji`, `Chopper`, `Robin`, `Franky`, `Brook`, `Jinbe`) — see
below. Note: the Mikasa and Erwin character voices each *recovered* a seiyuu the consensus
scraper had missed (zero prior embeddings); the entire One Piece crew is character-voice-only
(no natural-voice baseline — Mayumi Tanaka's consensus scrape missed). Natural-voice quality
scores 0.57–0.72. Note: the Mikasa character voice *recovered* Yui Ishikawa, who
the consensus scraper had missed — she went from zero embeddings to identifiable.

What the validation runs showed:

- **References are sound.** A *held-out* Yuuki Kaji interview (not a source video)
  matched the stored Kaji embedding at **cosine 0.884**, confidently, with all 3
  verification windows agreeing — impostors sat at ≤0.66 and won zero windows.
- **`identify/show` works on raw anime.** A real AoT scene (Eren↔Mikasa, Japanese,
  no hints) correctly inferred **"Attack on Titan"** via cast co-occurrence
  (2/3 speakers matched). The co-occurrence vote is the robust layer — it locked
  the show even though individual character-voice scores were middling.
- **Open-set gap, quantified:** ~**0.88** on matched-condition audio (interview vs
  interview reference) vs ~**0.52** on character voices (Eren mid-scene vs the
  interview "Natural Voice" reference). This is the concrete argument for storing
  per-character embeddings on marquee roles (the `voice_label` mechanism already
  exists; see Next Steps #4).
- **Per-character embedding closes the gap (proven).** Added an `Eren Yeager`
  character embedding for Yuuki Kaji via the new `add_character_voice.py` (two
  Eren voice-line compilations, Demucs-isolated, cross-source agreement 0.781).
  Re-running the same Eren↔Mikasa scene, the Eren speaker's top match rose from
  **0.525** (`Natural Voice`) to **0.686** (`Eren Yeager`), and — the real win —
  the margin over the nearest *other* actor went **+0.020 → +0.181**, flipping a
  coin-flip into a decisive ID. Because the embedding is linked to its
  `character_id`, `identify_show` now reports "Yuuki Kaji **as Eren Yeager**"
  instead of "Natural Voice", and still infers Attack on Titan (2/3 speakers).
- **Levi (Hiroshi Kamiya) — even starker: the character embedding *rescues a wrong
  ID*.** Added a `Levi` embedding from one clean line-cut source (`Q_iIY3MVorc`;
  the second source, a rapid-cut 100連発 MAD, fragmented under diarization to no
  ≥8s speaker and was dropped — so it stored single-source, quality n/a). On the
  ep14 courtroom scene the Levi speaker matched the new `Levi` embedding at **0.917**
  (confident), while Kamiya's `Natural Voice` reference *didn't even reach the top 5*
  — with Natural Voice alone the top match was **Yuuki Kaji `[Eren Yeager]` at 0.550,
  the wrong actor**. Levi's gravelly voice is far enough from his natural voice that
  the per-character embedding is the difference between a correct ID and a confident
  *mis*-ID. (Note: `identify_show` returned no show consensus *on this clip* — the
  violent beating scene has only one cleanly-voiced speaker, Eren is grunting; cast
  voting needs ≥2. Conversation scenes like Eren↔Mikasa vote fine.)
- **Mikasa (Yui Ishikawa) — recovers a consensus miss.** Yui Ishikawa had *no*
  embedding (one of the 3 consensus misses). Added a `Mikasa Ackerman` voice from a
  single clean line-cut (`WsXnZs9ahtk`; a second source, the ep7 "speech" clip,
  diarized to a non-Mikasa dominant speaker — agreement 0.20 — and was correctly
  rejected by the guard, so it stored single-source). On the Eren↔Mikasa scene the
  Mikasa speaker now matches **`Mikasa Ackerman` at 0.690**, beating the prior
  (wrong) top match **Marina Inoue `[Natural Voice]` 0.554** — so the character
  embedding both makes a previously-unknown actor identifiable *and* fixes a mis-ID.
  The Eren speaker stayed stable at 0.685 in the same run.
- **Annie + Hange added with the fast `htdemucs` model** (both from the trusted
  `セリフ切り抜き` uploader, single clean line-cut each — `UoZdE7sH3pQ` / `G2Hh-nIt56E`).
  Validated on held-out scenes via diarized `identify_multi`: **Annie** (Yuu Shimamura)
  → `Annie Leonhart` **0.796 confident, window-verified 1.00**, beating Armin's seiyuu
  Marina Inoue `[Natural]` 0.504 in the same scene — the strongest per-character result
  so far. **Hange** (Romi Park) → `Hange` **0.543 possible, win 0.67**, correct top-1 over
  Eren 0.490, where Romi Park's Natural Voice didn't rank (the Levi/Mikasa rescue pattern).
  Method note: single-speaker `identify()` first locked onto **Eren** on an Annie *combat*
  clip (she barely speaks there) — taciturn/ensemble characters need a *dialogue* scene and
  diarized `identify_multi`, not a single-speaker pass.
- **Erwin recovered a missed seiyuu.** Daisuke Ono had **zero** embeddings (consensus miss),
  so he was unidentifiable. Ingested `Erwin Smith` from the trusted uploader's line-cut
  (`SGDAuVuWNyg`, htdemucs); on the held-out charge-speech scene (`H13ID0ymOis`) the speaker
  now matches **`Erwin Smith` 0.639, possible, window-verified 1.00**, beating Yoshimasa
  Hosoya `[Natural]` 0.489 by 0.150. Same recovery pattern as Mikasa — a character voice
  makes a previously-absent actor identifiable.
- **Second show added — One Piece Straw Hats (all 10 crew character voices).** Sourced from
  **game voice-galleries** (海賊無双4 / TREASURE CRUISE 「全システムボイスセリフ集 / ボイス集」),
  ingested with the fast `htdemucs` model. Validated **cross-game** (train on one game, test on a
  *different* one — a genuinely held-out recording of the same seiyuu): **Chopper** 0.864,
  **Usopp** 0.793 (from only 14s of speech), **Zoro** 0.767, **Luffy** 0.721 — all confident,
  window-verified 1.00. Nami/Sanji/Robin/Franky/Jinbe/Brook came from the same clean galleries
  and recur as confident/possible runners-up in each other's tests. Two hard-won sourcing lessons:
  (1) **One Piece fan 名言集 compilations carry a narrator** — Zoro's first embedding was the
  narrator (never ranked; one test even matched Erwin's deep voice), fixed by switching to a game
  gallery; game galleries are the One-Piece equivalent of AoT's trusted セリフ切り抜き uploader.
  (2) **Combat/skill clips are bad validation targets** (battle shouts + SFX, atypical vocals) —
  a Chopper 必殺技 clip matched Connie; his dialogue clip then hit 0.864. Validate on *dialogue*.
- **Seed gotcha — AniList mislinks long-running characters.** One Piece characters have had multiple
  VAs over 25+ years (era recasts, young-version flashbacks, maternity-leave fill-ins), and the
  seeder picked a secondary/wrong VA for **6 of the 10 Straw Hats** (Zoro→Megumi Urawa instead of
  Kazuya Nakai, Sanji→Ikue Ōtani [who is Chopper], etc.). All correct seiyuu already existed as
  actor rows, so the fix was re-linking `vazam_characters.actor_id` — but **always verify the
  character→VA mapping after seeding a long-running show** before scraping.
- **Blind test caught the gap in the wild.** A random Connie (Hiro Shimono) clip the
  user supplied: `identify()` ranked **Hiro Shimono #1 — correct** — but at **0.432**,
  just under the 0.50 claim threshold, because we only had his *Natural Voice* and
  Connie is a character voice. So top-1 was right, but the system wouldn't *claim* it.
  Exactly the per-character gap; a `Connie Springer` embedding would push it over.
- **Sourcing gotcha (worth remembering):** Japanese YouTube has many "声真似" clips
  that are *fans impersonating* the character, not real seiyuu audio — those would
  poison a character embedding. The tell: 声真似講座/声真似ボイス/シチュエーションボイス
  = fan performance (avoid); セリフ切り抜き = actual show line-cuts (use). The
  cross-source agreement guard would likely reject a fan clip paired with a real one,
  but the safe move is to pick `セリフ切り抜き … 声マネ練習用` line-cut sources.
- **Dominant-speaker selection had a blind spot: ensemble characters — now fixed.**
  The default policy picks the *dominant* speaker, which is the character in a
  single-character compilation — great for leads who carry scenes (Eren/Levi/Mikasa).
  But comic-relief / ensemble characters (Connie) never speak alone: their clips are
  group scenes (Sasha/Connie/Jean) where someone else dominates. Fix shipped:
  `--select nearest-natural` picks the diarized speaker closest to the actor's stored
  Natural Voice instead of the dominant one — works for any character whose voice
  still *ranks* against the natural reference (Connie did, at 0.432). **Proven:**
  ingested `Connie Springer` (Hiro Shimono) from a Sasha/Connie/Jean group clip —
  nearest-natural correctly pulled Connie (43s) out of the group, and the blind clip
  (`woO8uJkBDk0`) that previously stalled at **0.432 (no claim)** now returns
  **Hiro Shimono *as Connie Springer* 0.675, window-verified 1.00** (+0.243, over the
  0.50 claim line). Lands at "possible" not "confident" — 0.675 is just under the
  uncalibrated 0.70 bar (a threshold-tuning question, not a selection failure).

Runtime fixes landed this session (all committed/pushed): yt-dlp EJS challenge
solver + shorter download sections (`1f57695`), Japanese-language search queries
`--lang {en,ja,both}` for seiyuu (`ea120f1`), and `isolate_vocals` using
`sys.executable` with graceful fallback (`1816ada`). Dependency pins
(torch/torchaudio <2.6, pyannote <4, huggingface_hub <1) and the VAD gate are
both required and now in place.

**Why 3 AoT leads missed — it's a linking limit, not a search limit.** Re-running
the misses with `--lang both` *did* surface clean solo Japanese interviews, but
their clips diarized to one speaker each and those single embeddings didn't
cluster across videos (cross-video cosine < the 0.60 consensus link threshold —
compressed YouTube audio from different mics drifts apart). Levers: lower the link
threshold (0.60 → ~0.55, A/B for false-links) or augment *before* clustering so
cross-condition takes pull together. Deferred — 7/10 is enough for show inference.

## Next steps (in order)

1. ~~Run the setup and prove the loop on one show.~~ ✅ **Done** (see Live
   validation above — seeded, scraped, and identified end-to-end on real clips).
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
   degrades at scale. Also tune the **consensus link threshold** (0.60 in
   `consensus.py`) — it cost 3 AoT leads (see Live validation); try 0.55 and watch
   for false-links.
4. **Per-character embeddings for marquee roles.** ✅ **Tool built + proven on Eren,
   Levi, *and* Mikasa** (`add_character_voice.py`; see Live validation — Eren
   0.525→0.686, Levi 0.917 rescuing a mis-ID, Mikasa 0.690 recovering a missed actor).
   Remaining work, in rough order:
   - **Scale to more leads** — ✅ **Annie, Hange, and Erwin done** (all via the trusted
     `セリフ切り抜き … 声マネ練習用` uploader, ingested with the fast `htdemucs` model;
     validated below; Erwin recovered a zero-embedding seiyuu). The AoT main cast is now
     well covered — **branching to a second show** is the next move. Lazily, top-billed only
     (anti-goal: never bulk-scrape character clips). Each add is self-measuring — run diarized
     `identify_multi` on a *conversation*
     scene with that character (single-speaker `identify` locks onto whoever talks most,
     which for a taciturn character like Annie is the co-star). **For batch ingestion, use
     `--demucs-model htdemucs`** (or `DEMUCS_MODEL=htdemucs`): isolation is the dominant
     cost on CPU and the lighter model is ~3.7× faster (1550s → 421s on the Connie clip)
     at cosine **0.972** agreement with the `htdemucs_ft` default — measured, near-free.
   - **Speed: GPU is the real unlock for a full-cast batch.** Demucs + pyannote both honor
     `DEVICE`; on CPU a single isolate is ~7–26 min/clip. A one-off cloud-GPU run would
     crush the backlog 10–30×. Code is already GPU-ready.
   - ~~**Build `--select nearest-natural`**~~ ✅ **Done + proven on Connie** (see the
     ensemble blind-spot note in Live validation — blind clip flipped 0.432 no-claim →
     0.675 verified `as Connie Springer`). Reaches ensemble characters who never carry a
     scene solo and so defeat dominant-speaker selection.
5. **Data plan build order** (from `data-acquisition-plan.md`):
   entity-resolution schema change (cross-source actor IDs — do first, painful to
   retrofit) → TMDB resolver alongside AniList → `show_ingestion_status` +
   demand-driven background ingestion → overnight warm-start of top shows.
6. **Generate RN native projects** (`ios/`, `android/`) so the app actually builds;
   point the client at the dev machine's LAN IP (`setBaseUrl`) instead of localhost.
7. ~~**Cleanup:** delete or rewrite legacy `main.py`.~~ ✅ **Done** — removed (it
   imported FAISS from the pre-Supabase era and could not run). `README.md` is still
   broadly stale (describes FAISS/SQLite/`DB_PATH`) — a full rewrite to match
   `CLAUDE.md` is a separate, low-priority pass.

## Things to remember

- Audio is gitignored everywhere (benchmark clips, samples) — embeddings only,
  source URLs kept for provenance/takedown.
- Per-character embeddings stay **lazy-only, top-billed-only** (plan anti-goal:
  never bulk-scrape character voices; lean on show-aware search instead).
- Any change that could affect accuracy gets an `eval.py` before/after. No vibes.
