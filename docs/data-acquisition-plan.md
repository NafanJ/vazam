# Data Acquisition Plan — Voice Actor Coverage for "Any Show"

*Drafted June 2026. Goal: a user can pull up any show and test clips against its cast.
Strategy in one line: don't pre-load the world — build demand-driven, self-validating
ingestion per show, and let coverage compound.*

## The problem splits in two

- **Metadata** (who voices whom in which show): cheap, structured, basically solved.
- **Audio** (a clean voice embedding per actor): expensive, messy, all the design work.

---

## 1. Metadata sources

AniList (current seeder) is anime-only. "Any show" needs supplements:

| Source | Covers | Access |
|---|---|---|
| **AniList** (built) | Anime, EN + JP casts | Free GraphQL, ~90 req/min |
| **TMDB** | Western animation, films, TV | Free API; voice cast in credits |
| **Wikidata** | Cross-domain glue | Free SPARQL; voice-actor property (P725) |
| BTVA | Gold standard for games/western | ❌ No API; ToS-hostile to scraping — manual reference only |

### Schema prerequisite: entity resolution

One canonical `actors` table with nullable `anilist_id`, `tmdb_id`, `wikidata_qid`.
Resolve cross-source duplicates by name + show-overlap (the same actor arriving via
AniList and TMDB must merge to one row, or their embeddings fragment across rows).
**Do this schema change first — it is painful to retrofit.**

---

## 2. Audio strategy: interview-first, self-validating

### Weakness of the current scraper
`scrape_audio.py` embeds the *first* YouTube hit blind. Failure modes: wrong person,
the interviewer's voice, intro music, panel crosstalk.

### Fix: cross-video consensus
1. Download **2–3 independent videos** per actor (interview/panel/podcast queries).
2. Diarize each video; embed **every** speaker found.
3. The actor is the voice cluster that recurs **across** videos — interviewers and
   co-panelists differ per video. Keep the consensus cluster centroid as the
   "Natural Voice" embedding; discard everything else.

Zero human labelling, self-validating by construction.

### Demucs placement
Interviews/panels have no music bed → **skip Demucs for natural-voice scraping**
(big CPU saving; dev box has no GPU). Reserve isolation for character clips and
user-uploaded show audio at query time.

### Character voices — the hard 20%
Heavily-acted voices diverge from natural voice (MacFarlane: Peter vs. Stewie).
Do **not** fight this during data collection:

- Primary mitigation is the existing **show-aware closed-set search** — given the
  show, the candidate set is ~20 actors, not thousands.
- Add character-clip embeddings only **lazily, top-billed characters only**:
  query `"<character>" "<show>" voice clip` → Demucs → diarize → keep the segment
  whose embedding is nearest the actor's natural anchor → store as `voice_label=<character>`.
- Never build per-character embeddings systematically.

---

## 3. Architecture: demand-driven ingestion

```
User searches a show
  → metadata resolver (AniList → TMDB → Wikidata fallback)
    seeds show + cast rows                                   (seconds)
  → show marked "indexing"; background job queues top-billed N actors (~10–30)
  → per actor: skip if already embedded (cache forever)
               → consensus scrape → embed → store
  → show flips to "ready"                                    (minutes)
```

Only embeddings + source URLs are ever stored — never audio.

### Why this works
1. **The dub world is small.** A few hundred prolific actors cover most popular
   English dubs. After the top ~100 shows, a "new" show is typically 60–80%
   indexed already by overlap. Coverage compounds.
2. **Warm start + long tail.** Pre-seed top 100–200 shows as an overnight batch
   (ECAPA embedding is cheap on CPU; Demucs is excluded from this path).
   Everything else indexes on demand.
3. **Demo narrative.** "Type any show — if we know it, instant; if not, watch it
   index itself in two minutes." Stronger than any fixed-roster demo.

### New state to track
- `show_ingestion_status`: `not_indexed | indexing | ready` (+ progress count)
- Per-embedding: `source_url`, `voice_label`, `duration_s`, `quality_score`

---

## 4. Quality, scale, legal

- **Per-actor quality score** = intra-cluster cosine consistency + seconds of clean
  speech. Below threshold → flag `low_confidence` in API responses; don't silently degrade.
- **Ground truth for free:** scenes with known casts are labelled test cases.
  Hold out clips per show → per-show accuracy dashboard.
- **Scale:** top 200 shows ≈ 2–4k unique actors after overlap; ~768 bytes per
  embedding → storage is a non-issue. Bottleneck is polite scraping
  (2.5s delay, 2–3 videos/actor) → warm start is overnight-to-weekend, not an hour.
- **Legal hygiene:** embeddings-only (no audio retention), public figures,
  research/portfolio posture. Voice embeddings of identifiable people are arguably
  personal data under GDPR — keep provenance (source URLs) and a takedown path.
  Keep yt-dlp volume low and rate-limited.

---

## Build order

1. Entity-resolution schema change (cross-source IDs)
2. TMDB resolver alongside AniList
3. Consensus scraper (replaces blind first-hit embedding)
4. `show_ingestion_status` + background ingestion queue (the on-demand flow)
5. Overnight warm-start batch of top shows

Anti-goal: systematic per-character embeddings. Lazy-only, top-billed-only,
lean on show-aware search.
