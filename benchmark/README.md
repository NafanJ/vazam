# Vazam Benchmark Clips

Labelled ground-truth clips for `eval.py`. Every pipeline or data change
should be measured against this set before and after.

## Directory layout

```
benchmark/
└── Cowboy Bebop/            ← show title — must exactly match the DB show title
    ├── Steve Blum/          ← actor name — must exactly match the DB actor name
    │   ├── clip01.wav
    │   └── clip02.m4a
    └── Wendee Lee/
        └── clip01.wav
```

Matching is case-insensitive but otherwise exact. `eval.py` prints any
folder names it can't resolve against the DB.

## Recording protocol — read this before recording

**Record with a phone microphone pointed at a playing TV/laptop speaker.**
Do not use clean audio rips. The production query path is a phone mic in a
room — room reverb, speaker coloration, and background noise are exactly
what the pipeline must survive. A benchmark of clean rips will report
flattering numbers that don't transfer.

Per clip:

- **5–15 seconds**, one clearly dominant speaker (the labelled actor's
  character). Clips under ~4s of speech also skip multi-window
  verification, so include mostly longer clips.
- Normal room conditions: couch distance, phone in hand. Don't press the
  phone against the speaker.
- A mix of conditions is good: some clips with background music/SFX, some
  dialogue-only, different episodes if possible.
- Any of: `.wav` `.mp3` `.m4a` `.flac` `.ogg`

Aim for **~20 clips per show across 3–5 shows** whose casts are already
embedded in the DB.

## Running

```bash
python eval.py benchmark/
python eval.py benchmark/ --json runs/$(date +%Y%m%d).json   # track over time
```

Audio files in this directory are for local evaluation only — don't commit
them (see .gitignore).
