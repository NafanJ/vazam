"""
Vazam — Voice Actor Identification Pipeline

Identifies voice actors from audio clips by:
1. Isolating vocals from background audio (Demucs v4)
2. Generating 192-dim speaker embeddings (SpeechBrain ECAPA-TDNN)
3. Searching against a voice actor database (FAISS cosine similarity)

Replaces the original MFCC + IndexFlatL2 approach with a production-grade
speaker recognition pipeline.
"""

import numpy as np
import faiss
import torch
import torchaudio
import subprocess
import os
import time

from speechbrain.inference.speaker import EncoderClassifier

# ── Config ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 192
# Cosine similarity thresholds (vectors are L2-normalized, so dot product = cosine)
CONFIDENT_THRESHOLD = 0.70   # Very likely same speaker
POSSIBLE_THRESHOLD  = 0.50   # Possible match — show with lower confidence

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading ────────────────────────────────────────────────────────────

print(f"Loading ECAPA-TDNN model on {DEVICE}...")
classifier = EncoderClassifier.from_hparams(
    source=MODEL_SOURCE,
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# ── In-memory database ───────────────────────────────────────────────────────

# Each entry: {"actor": str, "character": str, "embedding": np.ndarray}
voice_db: list[dict] = []
faiss_index: faiss.IndexFlatIP | None = None


# ── Voice isolation (Demucs) ─────────────────────────────────────────────────

def isolate_vocals(input_path: str, output_dir: str = "separated") -> str:
    """Separate vocals from background music/SFX using Demucs v4.

    Returns the path to the isolated vocals WAV file.
    """
    subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs_ft",
            "-o", output_dir,
            input_path,
        ],
        check=True,
        capture_output=True,
    )
    basename = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, "htdemucs_ft", basename, "vocals.wav")


# ── Speaker embedding (ECAPA-TDNN) ───────────────────────────────────────────

def get_embedding(audio_path: str) -> np.ndarray:
    """Generate a 192-dim L2-normalized speaker embedding from an audio file.

    The model expects 16 kHz mono audio. Resampling and channel conversion
    are applied automatically if needed.
    """
    signal, fs = torchaudio.load(audio_path)

    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    embedding = classifier.encode_batch(signal)
    embedding = embedding.squeeze().cpu().numpy().astype("float32")

    # L2-normalize so that inner product equals cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ── Database operations ──────────────────────────────────────────────────────

def add_voice_actor(
    actor_name: str,
    character_name: str,
    audio_path: str,
    needs_isolation: bool = False,
) -> None:
    """Add a voice actor sample to the in-memory database.

    Args:
        actor_name:      Full name of the voice actor.
        character_name:  Character or role label (e.g. "Natural Voice",
                         "Spike Spiegel").
        audio_path:      Path to a WAV or MP3 audio file.
        needs_isolation: Set to True when the audio contains background
                         music/SFX that must be removed first.
    """
    if needs_isolation:
        audio_path = isolate_vocals(audio_path)

    embedding = get_embedding(audio_path)
    voice_db.append({"actor": actor_name, "character": character_name, "embedding": embedding})
    print(f"  Added: {actor_name} as {character_name}")


def build_index() -> None:
    """Build the FAISS IndexFlatIP from all stored embeddings.

    Must be called after all voice actors are added and before any searches.
    IndexFlatIP computes exact inner products; with L2-normalized vectors this
    is equivalent to cosine similarity.
    """
    global faiss_index

    embeddings = np.array([entry["embedding"] for entry in voice_db], dtype="float32")
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss_index.add(embeddings)
    print(f"Built FAISS index with {faiss_index.ntotal} embeddings")


# ── Identification ───────────────────────────────────────────────────────────

def identify_voice(
    audio_path: str,
    top_k: int = 3,
    needs_isolation: bool = False,
) -> list[dict]:
    """Identify the voice actor in an audio clip.

    Args:
        audio_path:      Path to the audio file to identify.
        top_k:           Number of candidate results to return.
        needs_isolation: Set to True for mixed audio (music + dialogue).

    Returns:
        List of dicts with keys: actor, character, confidence, confident.
        Results are ordered from highest to lowest similarity.
    """
    if faiss_index is None:
        raise RuntimeError("Index not built. Call build_index() first.")

    if needs_isolation:
        audio_path = isolate_vocals(audio_path)

    query = get_embedding(audio_path).reshape(1, -1)
    similarities, indices = faiss_index.search(query, top_k)

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        entry = voice_db[idx]
        confidence = float(sim)
        results.append(
            {
                "actor":      entry["actor"],
                "character":  entry["character"],
                "confidence": confidence,
                # True = above the confident threshold; False = only a possible match
                "confident":  confidence >= CONFIDENT_THRESHOLD,
            }
        )

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    start = time.time()

    # ------------------------------------------------------------------
    # 1. Populate the voice actor database.
    #
    #    Store multiple embeddings per actor to cover both their natural
    #    speaking voice and their distinct character voices. Clean audio
    #    sources (interviews, demo reels, convention panels) work best.
    # ------------------------------------------------------------------
    print("Loading voice actor database...")

    add_voice_actor("Seth MacFarlane", "Natural Voice",  "samples/seth_interview.wav")
    add_voice_actor("Seth MacFarlane", "Peter Griffin",  "samples/seth_peter.wav")
    add_voice_actor("Seth MacFarlane", "Stewie Griffin", "samples/seth_stewie.wav")

    add_voice_actor("Steve Blum", "Natural Voice",  "samples/blum_interview.wav")
    add_voice_actor("Steve Blum", "Spike Spiegel",  "samples/blum_spike.wav")

    # ------------------------------------------------------------------
    # 2. Build the FAISS cosine-similarity index.
    # ------------------------------------------------------------------
    build_index()

    # ------------------------------------------------------------------
    # 3. Identify a voice from a real show clip.
    #
    #    Pass needs_isolation=True when the clip contains background music
    #    or SFX; Demucs will extract only the vocal stem before embedding.
    # ------------------------------------------------------------------
    print("\nIdentifying voice from test clip...")
    results = identify_voice("seth_mcfarlane_test.mp3", needs_isolation=True)

    for r in results:
        status = "MATCH   " if r["confident"] else "possible"
        print(
            f"  [{status}] {r['actor']} (as {r['character']}) "
            f"— similarity: {r['confidence']:.3f}"
        )

    print(f"\nTotal time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
