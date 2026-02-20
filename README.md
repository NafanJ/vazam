# Vazam — Shazam for Voice Actors

Point your phone at any animated show, anime, or video game and instantly know which voice actor is performing.

## Pipeline

```
Audio clip
  → Demucs v4 (vocal isolation)
  → SpeechBrain ECAPA-TDNN (192-dim speaker embedding)
  → FAISS IndexFlatIP (cosine similarity search)
  → Voice actor + character + confidence score
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

The ECAPA-TDNN model (~80 MB) is downloaded automatically on first run from HuggingFace.

## Key design choices

| Component | Choice | Replaces |
|-----------|--------|---------|
| Feature extraction | SpeechBrain ECAPA-TDNN (192-dim) | 13-dim MFCC mean |
| Similarity metric | FAISS `IndexFlatIP` + L2-normalized vectors (cosine) | `IndexFlatL2` (Euclidean) |
| Voice isolation | Demucs v4 `htdemucs_ft` | None |

### Confidence thresholds

| Score | Meaning |
|-------|---------|
| ≥ 0.70 | Confident match |
| 0.50 – 0.69 | Possible match |
| < 0.50 | No match |

### Multi-embedding strategy

A single voice actor is stored with multiple embeddings:

- **Natural voice** — from interviews or convention panels
- **Per-character voices** — one or more clips per distinct character

Searching against individual character-voice embeddings (rather than a single
averaged vector) improves accuracy for actors who alter their voice heavily
between roles.

## Adding voice actors

```python
# Clean audio — no isolation needed
add_voice_actor("Steve Blum", "Natural Voice", "samples/blum_interview.wav")

# Mixed audio — let Demucs strip music/SFX first
add_voice_actor("Steve Blum", "Spike Spiegel", "clips/bebop_ep01.mp3",
                needs_isolation=True)

build_index()
```

## Dependencies

See `requirements.txt`. GPU is recommended for Demucs; CPU-only mode works
but is slower.
