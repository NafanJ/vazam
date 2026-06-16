"""
pipeline.py — Full Vazam audio processing pipeline

Stages:
  1. Voice isolation  — Demucs v4 (htdemucs_ft)
  2. VAD              — pyannote voice-activity-detection
  3. Diarization      — pyannote speaker-diarization-3.1
  4. Embedding        — SpeechBrain ECAPA-TDNN (192-dim, cosine)
  5. Search           — pgvector cosine similarity via Supabase RPC
  6. Verification     — overlapping query sub-windows must agree on the
                        winner for a match to be reported as "confident"
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torchaudio

# torchaudio >= 2.5 removed list_audio_backends from the public namespace.
# SpeechBrain calls it at import time; shim it so older SpeechBrain releases
# don't raise AttributeError on newer torchaudio installs.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []

if TYPE_CHECKING:
    from db import VazamDB

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

# Multi-window verification: a "confident" match must also win a majority of
# overlapping sub-windows of the query — a consistent voice wins every window,
# a coincidental near-neighbor wins one. Clips with less speech than
# 2 × MIN_WINDOW_SECONDS skip verification (too short to window).
VERIFY_WINDOWS = 3
MIN_WINDOW_SECONDS = 2.0
WINDOW_AGREEMENT_THRESHOLD = 0.5

# Cast-graph show inference: one ambiguous voice is weak evidence, but several
# voices that all co-occur in one show's cast is nearly conclusive. A show must
# be supported by at least this many distinct detected speakers to be inferred.
SHOW_INFERENCE_MIN_SPEAKERS = 2


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
    """Result of a single voice actor identification attempt.

    window_agreement is the fraction of verification windows this actor won
    (None when the clip was too short to verify). A match is "confident" only
    if its similarity clears CONFIDENT_THRESHOLD *and* it won a majority of
    windows; a high-similarity match that fails verification demotes to
    "possible".
    """
    actor_id: int
    actor_name: str
    character_name: str
    confidence: float
    window_agreement: Optional[float] = None

    @property
    def confident(self) -> bool:
        if self.confidence < CONFIDENT_THRESHOLD:
            return False
        return (self.window_agreement is None
                or self.window_agreement >= WINDOW_AGREEMENT_THRESHOLD)

    @property
    def possible(self) -> bool:
        return self.confidence >= POSSIBLE_THRESHOLD and not self.confident

    def to_dict(self) -> dict:
        return {
            "actor_id":         self.actor_id,
            "actor_name":       self.actor_name,
            "character_name":   self.character_name,
            "confidence":       round(self.confidence, 4),
            "window_agreement": None if self.window_agreement is None
                                else round(self.window_agreement, 3),
            "match_level":      "confident" if self.confident else
                                "possible"  if self.possible  else "none",
        }


# ── Stage 1: Voice isolation (Demucs) ────────────────────────────────────────

# Demucs model for vocal isolation. `htdemucs_ft` (the default) is a *bag of 4*
# fine-tuned models — highest quality but ~4× slower than the single-pass
# `htdemucs`. For isolation feeding diarization (we only need speech clean
# enough to find the right speaker, not studio stems), `htdemucs` is usually a
# near-free speedup. Override per run with the DEMUCS_MODEL env var or the
# `model` argument.
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs_ft")


def isolate_vocals(
    input_path: str, output_dir: str = "separated", model: Optional[str] = None
) -> str:
    """Run Demucs to separate the vocal stem.

    ``model`` selects the Demucs model (defaults to ``DEMUCS_MODEL`` —
    ``htdemucs_ft`` unless overridden). Returns the path to the isolated
    vocals.wav file. Falls back to the original file if Demucs fails (e.g. audio
    is already clean speech with no music).
    """
    model = model or DEMUCS_MODEL
    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "-n", model,
                "-o", output_dir,
                input_path,
            ],
            check=True,
            capture_output=True,
        )
        basename = Path(input_path).stem
        vocals_path = os.path.join(output_dir, model, basename, "vocals.wav")
        if os.path.exists(vocals_path):
            return vocals_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # demucs missing or failed → fall through to original file

    return input_path


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio_16k(audio_path: str) -> torch.Tensor:
    """Load an audio file as a (1, N) mono float tensor at 16 kHz."""
    signal, fs = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
    return signal


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
    signal = load_audio_16k(audio_path)

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

    signal = load_audio_16k(audio) if isinstance(audio, str) else audio

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
    signal = load_audio_16k(audio_path)

    s = int(start * 16000)
    e = int(end   * 16000)
    chunk = signal[:, s:e]

    return get_embedding(chunk, device=device)


# ── Cast-graph show inference ─────────────────────────────────────────────────

@dataclass
class ShowVote:
    """Evidence for one show from cast co-occurrence voting."""
    show_id: int
    n_speakers: int            # distinct detected speakers with a candidate in this cast
    score: float               # sum of each matched speaker's best similarity
    speakers: list[str] = field(default_factory=list)


@dataclass
class ShowInference:
    """An inferred show plus the strength of the evidence behind it."""
    show_id: int
    show_title: str
    speakers_matched: int
    speakers_total: int
    score: float


def vote_shows(
    candidates_per_speaker: dict[str, list[tuple[int, float]]],
    actor_shows: dict[int, set[int]],
    min_similarity: float = POSSIBLE_THRESHOLD,
) -> list[ShowVote]:
    """Rank shows by how many detected speakers their casts explain.

    Each speaker contributes at most one vote per show (their best-similarity
    candidate in that cast); candidates below min_similarity are too weak to
    vote. Sorted by (distinct speakers, summed similarity), best first — the
    cast graph plays the role of Shazam's consistency check.
    """
    per_show: dict[int, dict[str, float]] = {}
    for speaker, candidates in candidates_per_speaker.items():
        for actor_id, similarity in candidates:
            if similarity < min_similarity:
                continue
            for show_id in actor_shows.get(actor_id, ()):
                best = per_show.setdefault(show_id, {})
                best[speaker] = max(best.get(speaker, 0.0), similarity)

    votes = [
        ShowVote(
            show_id=show_id,
            n_speakers=len(best),
            score=round(sum(best.values()), 4),
            speakers=sorted(best),
        )
        for show_id, best in per_show.items()
    ]
    votes.sort(key=lambda v: (v.n_speakers, v.score), reverse=True)
    return votes


# ── Multi-window verification ────────────────────────────────────────────────

def split_windows(
    signal: torch.Tensor,
    n: int = VERIFY_WINDOWS,
    fraction: float = 0.5,
    min_window_seconds: float = MIN_WINDOW_SECONDS,
    sample_rate: int = 16000,
) -> list[torch.Tensor]:
    """Split a (1, N) tensor into n overlapping windows for verification.

    Each window covers `fraction` of the signal; windows are evenly spaced so
    the first starts at 0 and the last ends at the end. Returns [] when the
    signal is too short to yield windows of at least min_window_seconds —
    callers should skip verification in that case.
    """
    total = signal.shape[1]
    window = int(total * fraction)
    if n < 2 or window < int(min_window_seconds * sample_rate):
        return []

    stride = (total - window) / (n - 1)
    return [
        signal[:, int(round(i * stride)): int(round(i * stride)) + window]
        for i in range(n)
    ]


# ── Stage 5: Orchestrator ────────────────────────────────────────────────────

class VazamPipeline:
    """End-to-end voice actor identification pipeline.

    Usage:
        pipeline = VazamPipeline(db=db, hf_token="hf_...", device="cuda")

        # Single-speaker clean audio
        results = pipeline.identify(audio_path)

        # Multi-speaker mixed audio (e.g. recorded from TV)
        per_speaker = pipeline.identify_multi(audio_path, isolate=True)
    """

    def __init__(
        self,
        db: VazamDB,
        hf_token: str = "",
        device: str | None = None,
        use_vad: bool = True,
        use_diarization: bool = True,
    ) -> None:
        self.db = db
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vad = use_vad and bool(hf_token)
        self.use_diarization = use_diarization and bool(hf_token)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _speech_tensor(self, audio_path: str, isolate: bool = False) -> torch.Tensor:
        """Load a file as a (1, N) 16 kHz speech tensor.

        Runs optional Demucs isolation, then keeps only VAD speech segments
        when use_vad is enabled (falling back to the full audio on failure).
        """
        path = isolate_vocals(audio_path) if isolate else audio_path

        if self.use_vad:
            try:
                segments = get_speech_segments(path, self.hf_token)
                if segments:
                    return extract_speech_audio(path, segments)
            except Exception:
                pass  # fall through to full-file audio

        return load_audio_16k(path)

    def embed_file(self, audio_path: str, isolate: bool = False) -> np.ndarray:
        """Generate an embedding from a file, with optional voice isolation.

        If use_vad is enabled, only speech segments are used for embedding.
        """
        speech = self._speech_tensor(audio_path, isolate=isolate)
        return get_embedding(speech, device=self.device)

    def _window_agreement(
        self,
        speech: torch.Tensor,
        show_id: Optional[int] = None,
    ) -> Optional[dict[int, float]]:
        """Per-actor fraction of verification windows won (top-1 per window).

        Returns None when the clip is too short to window — verification is
        then skipped rather than counted against the candidates.
        """
        windows = split_windows(speech)
        if len(windows) < 2:
            return None

        wins: dict[int, int] = {}
        for window in windows:
            emb = get_embedding(window, device=self.device)
            rows = self.db.search_embeddings(emb, top_k=1, show_id=show_id)
            if rows:
                aid = rows[0]["actor_id"]
                wins[aid] = wins.get(aid, 0) + 1

        if not wins:
            return None
        return {aid: count / len(windows) for aid, count in wins.items()}

    @staticmethod
    def _results_from_rows(rows: list[dict]) -> list[IdentificationResult]:
        return [
            IdentificationResult(
                actor_id=row["actor_id"],
                actor_name=row["actor_name"],
                character_name=row["voice_label"],
                confidence=float(row["similarity"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Single-speaker identification
    # ------------------------------------------------------------------

    def identify(
        self,
        audio_path: str,
        top_k: int = 5,
        isolate: bool = False,
        show_id: Optional[int] = None,
        verify: bool = True,
    ) -> list[IdentificationResult]:
        """Identify the dominant voice actor in an audio clip.

        Args:
            audio_path: Path to audio file.
            top_k:      Number of candidates to return.
            isolate:    Run Demucs vocal isolation first.
            show_id:    If provided, restrict search to actors in this show
                        (show-aware search — dramatically improves accuracy).
            verify:     Cross-check candidates against overlapping sub-windows
                        of the clip; candidates that don't win a majority of
                        windows are demoted from "confident" to "possible".
        """
        speech = self._speech_tensor(audio_path, isolate=isolate)
        embedding = get_embedding(speech, device=self.device)
        rows = self.db.search_embeddings(embedding, top_k=top_k, show_id=show_id)
        results = self._results_from_rows(rows)

        if verify and results:
            agreement = self._window_agreement(speech, show_id=show_id)
            if agreement is not None:
                for r in results:
                    r.window_agreement = agreement.get(r.actor_id, 0.0)

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
                        rows = self.db.search_embeddings(emb, top_k=top_k)
                        per_speaker[seg.speaker_label] = self._results_from_rows(rows)
                    return per_speaker
            except Exception:
                pass  # fall through to single-speaker

        # Single-speaker fallback
        results = self.identify(path, top_k=top_k, isolate=False)
        return {"SPEAKER_00": results}

    # ------------------------------------------------------------------
    # Cast-graph show inference
    # ------------------------------------------------------------------

    def _speaker_embeddings(self, audio_path: str) -> dict[str, np.ndarray]:
        """Diarize and produce one embedding per detected speaker.

        All of a speaker's segments are concatenated before embedding (more
        audio → better embedding), unlike identify_multi which embeds each
        speech turn separately.
        """
        segments = diarize(audio_path, self.hf_token)
        if not segments:
            return {}

        signal = load_audio_16k(audio_path)

        by_label: dict[str, list[SpeakerSegment]] = {}
        for seg in segments:
            by_label.setdefault(seg.speaker_label, []).append(seg)

        embeddings: dict[str, np.ndarray] = {}
        for label, segs in by_label.items():
            chunks = [
                signal[:, int(s.start * 16000): int(s.end * 16000)]
                for s in segs
            ]
            speech = torch.cat(chunks, dim=1)
            embeddings[label] = get_embedding(speech, device=self.device)
        return embeddings

    def identify_show(
        self,
        audio_path: str,
        top_k: int = 3,
        isolate: bool = True,
    ) -> tuple[Optional[ShowInference], dict[str, list[IdentificationResult]]]:
        """Infer which show is playing from cast co-occurrence, then identify
        each speaker closed-set within that show — no show_id input needed.

        Diarizes the clip, searches each speaker globally, and votes on shows:
        a show is inferred only when ≥ SHOW_INFERENCE_MIN_SPEAKERS distinct
        speakers have a plausible candidate in its cast. Multiple speakers are
        the *strongest* signal here, not a complication — one ambiguous voice
        proves little, but voices that co-occur in one cast pin the show down.

        Returns (inference, per_speaker_results). When no show can be inferred
        (one speaker, no diarization, or no cast agreement), inference is None
        and the per-speaker results are the global ones.
        """
        path = isolate_vocals(audio_path) if isolate else audio_path

        speaker_embeddings: dict[str, np.ndarray] = {}
        if self.use_diarization:
            try:
                speaker_embeddings = self._speaker_embeddings(path)
            except Exception:
                pass  # fall through to single-speaker fallback

        if not speaker_embeddings:
            return None, {"SPEAKER_00": self.identify(path, top_k=top_k, isolate=False)}

        # Global (open-set) candidates per speaker
        per_speaker_global: dict[str, list[IdentificationResult]] = {}
        candidates: dict[str, list[tuple[int, float]]] = {}
        for label, emb in speaker_embeddings.items():
            rows = self.db.search_embeddings(emb, top_k=top_k)
            per_speaker_global[label] = self._results_from_rows(rows)
            candidates[label] = [
                (row["actor_id"], float(row["similarity"])) for row in rows
            ]

        # Vote on shows via the cast graph
        actor_ids = sorted({aid for cands in candidates.values() for aid, _ in cands})
        actor_shows = self.db.get_shows_for_actors(actor_ids)
        votes = vote_shows(candidates, actor_shows)

        if not votes or votes[0].n_speakers < SHOW_INFERENCE_MIN_SPEAKERS:
            return None, per_speaker_global

        winner = votes[0]
        show = self.db.get_show(winner.show_id) or {}

        # Closed-set re-rank within the inferred show (reusing embeddings)
        per_speaker: dict[str, list[IdentificationResult]] = {}
        for label, emb in speaker_embeddings.items():
            rows = self.db.search_embeddings(emb, top_k=top_k, show_id=winner.show_id)
            per_speaker[label] = self._results_from_rows(rows)

        inference = ShowInference(
            show_id=winner.show_id,
            show_title=show.get("title", ""),
            speakers_matched=winner.n_speakers,
            speakers_total=len(speaker_embeddings),
            score=winner.score,
        )
        return inference, per_speaker
