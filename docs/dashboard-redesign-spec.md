# Vazam Dashboard — Redesign Brief

A handoff spec for modernising the Vazam web dashboard. The goal is a **visual + UX refresh**, not a feature change — every screen, state, and API call below already works and must keep working.

---

## 1. Product in one line

**Vazam is "Shazam for voice actors."** Point your phone at any animated show, anime, or game, record a few seconds, and Vazam tells you **which character is speaking and which actor voices them** — e.g. *"Spike Spiegel, voiced by Steve Blum — 94%."*

The dashboard is a **mobile-first web app** used one-handed, often in a dark room in front of a TV. It is the primary way the owner records clips on a phone (the native app isn't shipped). It's served over a private Cloudflare tunnel behind HTTP Basic auth — single trusted user, not a public product.

## 2. What we want from the redesign

Modernise the look and feel while keeping it fast, legible, and thumb-friendly.

- **Keep the dark, "instrument panel" feel** but make it more polished and less utilitarian. Think Shazam / Spotify-grade product, not a developer tool.
- **The record button is the hero.** Recording is the main action; everything else is secondary.
- **Results should feel like trading cards** — character art, actor, confidence — not log rows.
- **Confidence must be instantly readable** at a glance (confident / possible / no-match), since the owner is often glancing mid-scene.
- Smooth, reassuring **progress feedback** during the 8–45s analysis (it's slow on CPU).
- Works great on a **phone in portrait, in the dark.** Desktop is secondary but shouldn't look broken.

**Non-goals:** no new features, no backend changes, no routing/SPA framework. Don't redesign the data model or invent screens that aren't listed here.

## 3. Hard constraints (please honor these)

- **Two static HTML files**, served by FastAPI from `static/`: `index.html` (`GET /`) and `characters.html` (`GET /characters.html`). Self-contained — **inline CSS/JS, no build step, no npm, no external runtime dependencies or CDNs** (the box may not have outbound internet for assets; fonts/icons must be embedded or system). Vanilla JS only.
- **Mobile-first, portrait.** Min target width ~360px. Large tap targets (≥44px). Must remain usable one-handed.
- **Dark theme is the default and primary.** A light theme is optional/nice-to-have, not required.
- **Don't change the API contract** (section 6). The JS may be reorganized, but the same endpoints, form fields, and response fields must be consumed.
- **Audio still gets converted client-side** to 16 kHz mono 16-bit WAV before upload, capped at 20s (existing `blobToWav16k`). Keep that pipeline; it can be restyled but not removed.
- Recording uses `MediaRecorder` + `getUserMedia` (needs HTTPS — already satisfied by the tunnel). Keep the graceful fallback message when mic is unavailable → "use file upload instead."

## 4. Screens & flows

### Screen A — Identify (`index.html`, the home screen)

The single most important screen. Current structure, top to bottom:

1. **Header** — "Vazam — who's that voice?" + one-line subtitle + link to Characters page.
2. **Mode toggle** (segmented control): **Single voice** (one speaker) vs **Scene (who's in it)** (multi-speaker — diarizes and infers the show).
3. **Show filter** (dropdown, optional): "All shows (open search)" or pick one show to narrow the search to that cast (much more accurate).
4. **"Strip music/SFX" toggle** (checkbox, default on) — runs Demucs vocal isolation; recommended for TV audio.
5. **Record button** (hero) + a live elapsed timer while recording. Tap to start, tap to stop.
6. **"— or —" → file upload** (m4a/mp3/wav).
7. **Identify button** (primary CTA, disabled until a clip is staged).
8. **Enrollment row** — "add the staged clip to a character's fingerprint": a character dropdown + "Add clip" button. (Improves future accuracy by adding the recording as a reference.)
9. **Backend URL** — collapsed `<details>`, advanced/rarely used. Keep but de-emphasize.
10. **Progress panel** (appears during analysis): current stage label + elapsed time + an animated progress bar + a **streaming log** of pipeline stages (e.g. "Isolating music & SFX (Demucs)…", "Trimming silence (VAD)…", "Embedding voiceprint…", "Searching reference voices…", "Verifying across sub-windows…").
11. **Results** (newest on top).
12. **Previous recordings** — a history panel (last ~8 results, restyled cards, enrollment buttons stripped).

**Flow:** stage a clip (record or upload) → optionally set mode/show/isolate → Identify → watch progress → see result card(s) → optionally tap "Correct? Add to fingerprint" on the top result.

### Screen B — Characters (`characters.html`)

A browsable/searchable admin list of all ~650 characters.

1. Header + back-to-dashboard link + count ("657 characters · 27 with a voice fingerprint").
2. **Search box** (matches name / actor / show / occupation) + **"Only with a voice" toggle**.
3. **List of character rows**: thumbnail, name, "actor · show · occupation" subline, and a badge — green "N clips" if it has a voice fingerprint, muted "no voice" otherwise. Capped at 400 rendered; prompts to refine search beyond that. **Characters with a voice should sort first.**
4. **Detail modal** (on row tap): character art, editable **Image URL** + **Occupation** fields (PATCH to save), and a read-only list of **voice embedding source files** (voice label, type, duration, quality score, verified flag, source URLs).

## 5. Result card — the centerpiece

Each match needs, in priority order:

- **Character name** (largest, the headline).
- **"voiced by {Actor}"** (secondary).
- **Match level** — one of `confident` / `possible` / `none`, shown as both a **colored left border / accent** and a **pill badge**. This is the single most important visual signal.
- **Confidence %** + a small **confidence bar**.
- Optional **window-agreement %** (a secondary verification score; show subtly).
- **"also considered:"** — a compact list of runner-up matches (character, actor, %).
- **Enrollment CTA** under the top result: *"➕ Correct? Add this clip to {Character}'s fingerprint."*

**Scene mode** additionally shows:
- A **"Likely show" banner** when inferred (title + "N/M speakers matched").
- Results **grouped per speaker** ("Speaker 1", "Speaker 2", …), each with its own match card(s).

A character thumbnail on the card would be a great addition — `image_url` is available on characters and could be joined in; if the design wants it, note it as an optional backend tweak rather than assuming it's in the response today.

## 6. API contract (must keep consuming these)

All same-origin. Auth is handled by the browser (Basic). Endpoints the dashboard uses:

| Method | Path | Purpose | Request | Key response fields |
|--------|------|---------|---------|---------------------|
| `POST` | `/identify/stream` | Single-voice ID with **live progress** (NDJSON stream) | multipart: `audio` (wav), `isolate` (`"true"`/`"false"`), optional `show_id` | stream of `{"stage": "..."}` lines, then `{"done": true, "results": [...]}`; or `{"error": "..."}` |
| `POST` | `/identify/show` | Scene mode — diarize + infer show | multipart: `audio`, `isolate`, optional `show_id` | `{ show: {title, speakers_matched, speakers_total} \| null, speakers: { "<id>": [match...] } }` |
| `POST` | `/enroll` | Add a clip to a character's fingerprint | multipart: `audio`, `actor_id`, `voice_label`, `isolate` | `{ actor_name, voice_label, samples }` |
| `GET` | `/voices` | Populate the enrollment dropdown | — | `[{ actor_id, actor_name, voice_label, samples }]` |
| `GET` | `/shows` | Populate the show filter | — | `[{ id, title }]` |
| `GET` | `/characters` | Characters list | — | `[{ id, name, actor_name, show_title, image_url, occupation, samples }]` |
| `GET` | `/characters/{id}` | Character detail | — | `{ ...character, embeddings: [{ voice_label, audio_source, duration_s, quality_score, verified, source_url }] }` |
| `PATCH` | `/characters/{id}` | Edit image/occupation | JSON: `{ image_url, occupation }` | updated character |

**A `match` object** (in `results[...]` and `speakers[...]`):
```json
{
  "character_name": "Spike Spiegel",
  "actor_name": "Steve Blum",
  "actor_id": 1,
  "confidence": 0.94,            // 0–1 cosine similarity
  "window_agreement": 0.67,      // 0–1, or null if clip too short to window
  "match_level": "confident"     // "confident" | "possible" | "none"
}
```

## 7. States to design (don't skip these)

- **Idle** — no clip staged; Identify/enroll disabled.
- **Recording** — pulsing record button, running timer, "Stop" affordance.
- **Clip staged** — "Ready: {name} — tap Identify."
- **Converting / analyzing** — progress bar + stage label + streaming log (8–45s; design for a slow, reassuring wait).
- **Result: confident / possible / no-match** — three visually distinct treatments.
- **Scene with no show inferred** vs **show inferred**.
- **Enrolling** → **enrolled** ("✓ Added — N clips").
- **Errors** — mic denied, decode failed, server 4xx/5xx (show server message snippet), "mic needs HTTPS, use upload."
- **Empty states** — no voices enrolled yet, no character matches, history empty.

## 8. Current visual language (baseline to evolve from)

Dark GitHub-ish palette — fine as a starting point, modernize freely:

```
--bg #0d1117   --panel #161b22   --line #30363d   --txt #e6edf3   --muted #8b949e
--accent #2f81f7 (blue)   --good #2ea043 (confident/green)
--maybe #d29922 (possible/amber)   --none #6e7681 (no-match/grey)
```

System font stack, 12px rounded panels, blue primary buttons, red record button, monospace progress log. The redesign can replace all of this — these are just the semantics (green = confident, amber = possible, grey = none) worth preserving.

## 9. Deliverable

Two restyled self-contained HTML files (`index.html`, `characters.html`) that drop into `static/`, consume the same endpoints, preserve every state above, and keep the client-side WAV conversion. A shared visual system across both pages (tokens, type scale, card/badge components) is welcome — just inline it in each file (no shared asset fetched at runtime).
