"""
pipeline.py — Full Vazam audio processing pipeline

Stages:
  1. Voice isolation  — Demucs v4 (htdemucs_ft)
  2. VAD              — pyannote voice-activity-detection
  3. Diarization      — pyannote speaker-diarization-3.1
  4. Embedding        — SpeechBrain ECAPA-TDNN (192-dim, cosine)
  5. Search           — FAISS IndexFlatIP against the embedding store
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
import torch
import torchaudio

# ── Optional heavy deps (loaded lazily so the module imports quickly) ────────

_classifier = None          # SpeechBrain EncoderClassifier
_vad_pipeline = None        # pyannote VAD
_diarize_pipeline = None    # pyannote diarization


def _load_embedding_model(device: str = "cpu"):
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    return _classifier


def _load_vad(hf_token: str):
    global _vad_pipeline
    if _vad_pipeline is None:
        from pyannote.audio import Pipeline
        _vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=hf_token,
        )
    return _vad_pipeline


def _load_diarizer(hf_token: str):
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    return _diarize_pipeline


# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 192
CONFIDENT_THRESHOLD = 0.70
POSSIBLE_THRESHOLD = 0.50

# Minimum speech duration (seconds) needed to generate a reliable embedding
MIN_SPEECH_SECONDS = 1.5


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    """A contiguous speech segment attributed to one speaker."""
    speaker_label: str   # e.g. "SPEAKER_00", or actor name when known
    start: float         # seconds
    end: float           # seconds
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class IdentificationResult:
    """Result of a single voice actor identification attempt."""
    actor_id: int
    actor_name: str
    character_name: str
    confidence: float

    @property
    def confident(self) -> bool:
        return self.confidence >= CONFIDENT_THRESHOLD

    @property
    def possible(self) -> bool:
        return POSSIBLE_THRESHOLD <= self.confidence < CONFIDENT_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "actor_id":       self.actor_id,
            "actor_name":     self.actor_name,
            "character_name": self.character_name,
            "confidence":     round(self.confidence, 4),
            "match_level":    "confident" if self.confident else
                              "possible"  if self.possible  else "none",
        }


# ── Stage 1: Voice isolation (Demucs) ────────────────────────────────────────

def isolate_vocals(input_path: str, output_dir: str = "separated") -> str:
    """Run Demucs htdemucs_ft to separate the vocal stem.

    Returns the path to the isolated vocals.wav file.
    Falls back to the original file if Demucs fails (e.g. audio is already
    clean speech with no music).
    """
    try:
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
        basename = Path(input_path).stem
        vocals_path = os.path.join(output_dir, "htdemucs_ft", basename, "vocals.wav")
        if os.path.exists(vocals_path):
            return vocals_path
    except subprocess.CalledProcessError:
        pass  # fall through to original file

    return input_path


# ── Stage 2: VAD — extract speech-only segments ──────────────────────────────

def get_speech_segments(audio_path: str, hf_token: str) -> list[tuple[float, float]]:
    """Return list of (start, end) tuples for speech regions.

    Uses pyannote voice-activity-detection. Segments shorter than
    MIN_SPEECH_SECONDS are dropped.
    """
    vad = _load_vad(hf_token)
    result = vad(audio_path)
    segments = []
    for segment in result.get_timeline().support():
        if (segment.end - segment.start) >= MIN_SPEECH_SECONDS:
            segments.append((segment.start, segment.end))
    return segments


def extract_speech_audio(audio_path: str, segments: list[tuple[float, float]]) -> torch.Tensor:
    """Load audio and concatenate only the speech segments into a single tensor.

    Returns a (1, N) float32 tensor at 16 kHz.
    """
    signal, fs = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)

    chunks = []
    for start, end in segments:
        s = int(start * 16000)
        e = int(end   * 16000)
        chunks.append(signal[:, s:e])

    if not chunks:
        return signal   # no speech found — return full audio as fallback

    return torch.cat(chunks, dim=1)


# ── Stage 3: Diarization — split multi-speaker audio ─────────────────────────

def diarize(audio_path: str, hf_token: str) -> list[SpeakerSegment]:
    """Run speaker diarization and return per-speaker segments.

    Each segment represents a contiguous speech turn by a single speaker.
    Segments shorter than MIN_SPEECH_SECONDS are dropped.
    """
    diarizer = _load_diarizer(hf_token)
    result = diarizer(audio_path)

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        duration = turn.end - turn.start
        if duration >= MIN_SPEECH_SECONDS:
            segments.append(SpeakerSegment(
                speaker_label=speaker,
                start=turn.start,
                end=turn.end,
            ))

    return segments


def merge_speaker_segments(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """Merge consecutive segments from the same speaker into longer chunks.

    This gives the embedding model more audio to work with per speaker.
    """
    if not segments:
        return []

    merged: list[SpeakerSegment] = []
    current = SpeakerSegment(
        speaker_label=segments[0].speaker_label,
        start=segments[0].start,
        end=segments[0].end,
    )

    for seg in segments[1:]:
        if seg.speaker_label == current.speaker_label:
            current.end = seg.end
        else:
            merged.append(current)
            current = SpeakerSegment(
                speaker_label=seg.speaker_label,
                start=seg.start,
                end=seg.end,
            )

    merged.append(current)
    return merged


# ── Stage 4: Speaker embedding (ECAPA-TDNN) ──────────────────────────────────

def get_embedding(audio: str | torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Generate a 192-dim L2-normalized speaker embedding.

    Accepts either a file path (str) or a pre-loaded audio tensor (1, N) at 16 kHz.
    """
    model = _load_embedding_model(device)

    if isinstance(audio, str):
        signal, fs = torchaudio.load(audio)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
    else:
        signal = audio

    embedding = model.encode_batch(signal)
    embedding = embedding.squeeze().cpu().numpy().astype("float32")

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding


def get_embedding_for_segment(
    audio_path: str,
    start: float,
    end: float,
    device: str = "cpu",
) -> np.ndarray:
    """Generate an embedding for a specific time slice of an audio file."""
    signal, fs = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)

    s = int(start * 16000)
    e = int(end   * 16000)
    chunk = signal[:, s:e]

    return get_embedding(chunk, device=device)


# ── Stage 5: FAISS vector search ─────────────────────────────────────────────

class EmbeddingIndex:
    """FAISS IndexFlatIP wrapper for cosine-similarity search.

    Works with L2-normalized 192-dim vectors. Inner product on normalized
    vectors equals cosine similarity, ranging from -1 (opposite) to 1 (identical).
    """

    def __init__(self) -> None:
        self._index: Optional[faiss.IndexFlatIP] = None
        # Parallel list mapping FAISS ordinal → (actor_id, actor_name, character_name)
        self._entries: list[tuple[int, str, str]] = []

    def add(self, actor_id: int, actor_name: str, character_name: str, embedding: np.ndarray) -> None:
        """Add a single L2-normalized embedding to the index."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        vec = embedding.reshape(1, -1).astype("float32")
        self._index.add(vec)
        self._entries.append((actor_id, actor_name, character_name))

    def build_from_list(
        self,
        entries: list[tuple[int, str, str, np.ndarray]],
    ) -> None:
        """Bulk-build the index from (actor_id, actor_name, character_name, embedding) tuples."""
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._entries = []
        if not entries:
            return
        matrix = np.array([e[3] for e in entries], dtype="float32")
        self._index.add(matrix)
        self._entries = [(e[0], e[1], e[2]) for e in entries]

    def search(self, query: np.ndarray, top_k: int = 5) -> list[IdentificationResult]:
        """Return the top-k most similar voice actors for a query embedding."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query.reshape(1, -1).astype("float32")
        sims, idxs = self._index.search(q, min(top_k, self._index.ntotal))

        results: list[IdentificationResult] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            actor_id, actor_name, character_name = self._entries[idx]
            results.append(IdentificationResult(
                actor_id=actor_id,
                actor_name=actor_name,
                character_name=character_name,
                confidence=float(sim),
            ))

        return results

    def save(self, path: str) -> None:
        """Persist the FAISS index to disk."""
        if self._index is not None:
            faiss.write_index(self._index, path)

    def load(self, path: str, entries: list[tuple[int, str, str]]) -> None:
        """Load a persisted FAISS index and restore its entry list."""
        self._index = faiss.read_index(path)
        self._entries = entries

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal


# ── Orchestrator ─────────────────────────────────────────────────────────────

class VazamPipeline:
    """End-to-end voice actor identification pipeline.

    Usage:
        pipeline = VazamPipeline(hf_token="hf_...", device="cuda")
        pipeline.load_index(entries)

        # Single-speaker clean audio
        results = pipeline.identify(audio_path)

        # Multi-speaker mixed audio (e.g. recorded from TV)
        per_speaker = pipeline.identify_multi(audio_path, isolate=True)
    """

    def __init__(
        self,
        hf_token: str = "",
        device: str | None = None,
        use_vad: bool = True,
        use_diarization: bool = True,
    ) -> None:
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vad = use_vad and bool(hf_token)
        self.use_diarization = use_diarization and bool(hf_token)
        self.index = EmbeddingIndex()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def load_index(self, entries: list[tuple[int, str, str, np.ndarray]]) -> None:
        """Build the search index from a list of (actor_id, actor_name, character_name, embedding)."""
        self.index.build_from_list(entries)

    def add_entry(self, actor_id: int, actor_name: str, character_name: str, embedding: np.ndarray) -> None:
        self.index.add(actor_id, actor_name, character_name, embedding)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_file(self, audio_path: str, isolate: bool = False) -> np.ndarray:
        """Generate an embedding from a file, with optional voice isolation.

        If use_vad is enabled, only speech segments are used for embedding.
        """
        path = isolate_vocals(audio_path) if isolate else audio_path

        if self.use_vad:
            try:
                segments = get_speech_segments(path, self.hf_token)
                if segments:
                    speech_tensor = extract_speech_audio(path, segments)
                    return get_embedding(speech_tensor, device=self.device)
            except Exception:
                pass  # fall through to full-file embedding

        return get_embedding(path, device=self.device)

    # ------------------------------------------------------------------
    # Single-speaker identification
    # ------------------------------------------------------------------

    def identify(
        self,
        audio_path: str,
        top_k: int = 5,
        isolate: bool = False,
        actor_ids: Optional[list[int]] = None,
    ) -> list[IdentificationResult]:
        """Identify the dominant voice actor in an audio clip.

        Args:
            audio_path: Path to audio file.
            top_k:      Number of candidates to return.
            isolate:    Run Demucs vocal isolation first.
            actor_ids:  If provided, restrict search to these actor IDs
                        (show-aware search — dramatically improves accuracy).
        """
        embedding = self.embed_file(audio_path, isolate=isolate)
        results = self.index.search(embedding, top_k=top_k)

        if actor_ids is not None:
            actor_id_set = set(actor_ids)
            results = [r for r in results if r.actor_id in actor_id_set]

        return results

    # ------------------------------------------------------------------
    # Multi-speaker identification
    # ------------------------------------------------------------------

    def identify_multi(
        self,
        audio_path: str,
        top_k: int = 3,
        isolate: bool = True,
    ) -> dict[str, list[IdentificationResult]]:
        """Diarize a clip and identify each detected speaker independently.

        Returns a dict keyed by speaker label (e.g. "SPEAKER_00") with
        identification results for each speaker.

        Falls back to single-speaker identification if diarization is
        unavailable or finds only one speaker.
        """
        path = isolate_vocals(audio_path) if isolate else audio_path

        # Try diarization
        if self.use_diarization:
            try:
                segments = diarize(path, self.hf_token)
                merged = merge_speaker_segments(segments)

                if len({s.speaker_label for s in merged}) > 1:
                    per_speaker: dict[str, list[IdentificationResult]] = {}
                    for seg in merged:
                        emb = get_embedding_for_segment(
                            path, seg.start, seg.end, device=self.device
                        )
                        per_speaker[seg.speaker_label] = self.index.search(emb, top_k=top_k)
                    return per_speaker
            except Exception:
                pass  # fall through to single-speaker

        # Single-speaker fallback
        results = self.identify(path, top_k=top_k, isolate=False)
        return {"SPEAKER_00": results}
