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
folder names it can't resolve against the DB. Folders are keyed by the
**actor (seiyuu)** name, not the character.

The scaffold for the currently-logged cast is already created (drop clips
straight in):

```
benchmark/
├── Attack on Titan/
│   ├── Yuuki Kaji/      (Eren)     ├── Hiro Shimono/  (Connie)
│   ├── Hiroshi Kamiya/  (Levi)     ├── Daisuke Ono/   (Erwin)
│   ├── Yui Ishikawa/    (Mikasa)   └── Romi Park/     (Hange)
│   └── Yuu Shimamura/   (Annie)
└── ONE PIECE/
    ├── Mayumi Tanaka/   (Luffy)    ├── Kazuya Nakai/      (Zoro)
    ├── Akemi Okamura/   (Nami)     ├── Kappei Yamaguchi/  (Usopp)
    ├── Hiroaki Hirata/  (Sanji)    ├── Ikue Ootani/       (Chopper)
    ├── Yuriko Yamaguchi/(Robin)    ├── Kazuki Yao/        (Franky)
    └── Katsuhisa Houki/ (Jinbe)    └── Choo/              (Brook)
```

Aim for ~2–4 clips per character (one each makes every number a coin-flip).
Use **held-out** scenes — don't reuse the clips/sources the fingerprints were
enrolled from. For taciturn/ensemble characters (Connie, Annie, Hange, Jinbe,
Robin) pick a *dialogue* scene where they clearly dominate the window, or
`identify()` locks onto the co-star.

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

Aim for **~20 clips per show** whose cast is already embedded in the DB.

## Negatives (specificity) — `_negatives/`

Positives measure "did we find the right actor." Negatives measure the other
half: **does the system stay quiet when no one in the DB is speaking?** That's
what the `0.70` / `0.50` thresholds protect, and they're currently uncalibrated
guesses — negatives are the data that calibrates them.

Put clips that should match **nobody** under `benchmark/_negatives/`, organized
into subfolders for a per-category breakdown (these are pre-created):

```
benchmark/_negatives/
├── noise/            ← traffic, music, room tone, TV static (no speech)
└── unknown_voices/   ← real voices NOT in the DB: other anime, a podcast
                         host, an English dub, you talking — the harder test
```

`eval.py` runs each through `identify()` and reports the **false-positive rate**
(how often a negative produced a confident/possible claim — want ~0) plus the
score gap between correct positive matches and negative top-scores. The
threshold belongs in that gap. ~5–10 clips per category is plenty. The
`_negatives/` folder is skipped by positive discovery (its `_` prefix).

## Running

```bash
python eval.py benchmark/                            # positives + negatives
python eval.py benchmark/ --json runs/$(date +%Y%m%d).json   # track over time
python eval.py benchmark/ --no-negatives             # positives only
```

Audio files in this directory are for local evaluation only — don't commit
them (see .gitignore); the folder scaffold (`.gitkeep` files) is committed so
the structure persists.
