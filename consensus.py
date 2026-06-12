"""
consensus.py — Cross-video consensus for natural-voice embeddings

Implements the self-validating scraping strategy from
docs/data-acquisition-plan.md: download several independent videos of an
actor, embed every speaker found in each, and keep the voice cluster that
recurs *across* videos. Interviewers and co-panelists differ per video, so
the recurring cluster is the actor — no human labelling required.

This module is pure numpy (no ML, no network) so it is fully unit-testable.

Usage
-----
    speakers = [VideoSpeaker(video_id, label, embedding, speech_seconds), ...]
    result = build_consensus(speakers)
    if result:
        db.add_embedding(actor_id, result.embedding,
                         quality_score=result.quality_score,
                         duration_s=result.speech_seconds)

All embeddings are assumed L2-normalized (cosine similarity == dot product).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Minimum cosine similarity for two speakers (in different videos) to be
# considered the same person. ECAPA same-speaker scores on clean interview
# audio typically land well above this; different speakers well below.
CLUSTER_LINK_THRESHOLD = 0.60

# A cluster must span at least this many distinct videos to count as consensus.
MIN_CONSENSUS_VIDEOS = 2


@dataclass
class VideoSpeaker:
    """One diarized speaker within one source video."""
    video_id: str
    speaker_label: str                       # e.g. "SPEAKER_00", or "FULL"
    embedding: np.ndarray = field(repr=False)
    speech_seconds: float = 0.0


@dataclass
class ConsensusResult:
    """The consensus voice cluster for an actor."""
    embedding: np.ndarray = field(repr=False)   # L2-normalized centroid
    quality_score: float = 0.0                  # mean pairwise cosine in cluster
    speech_seconds: float = 0.0                 # total clean speech in cluster
    n_videos: int = 0                           # distinct videos in cluster
    video_ids: list[str] = field(default_factory=list)


def centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalized mean of unit vectors."""
    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean /= norm
    return mean.astype("float32")


def mean_pairwise_cosine(embeddings: list[np.ndarray]) -> float:
    """Average cosine similarity over all pairs; 1.0 for a single vector."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    stacked = np.stack(embeddings)
    sims = stacked @ stacked.T
    upper = sims[np.triu_indices(n, k=1)]
    return float(upper.mean())


def cluster_speakers(
    speakers: list[VideoSpeaker],
    threshold: float = CLUSTER_LINK_THRESHOLD,
) -> list[list[VideoSpeaker]]:
    """Greedy centroid-linkage clustering of speakers across videos.

    Speakers are visited in descending speech-time order (most reliable
    embeddings first) and joined to the best existing cluster whose centroid
    similarity clears `threshold`, else they start a new cluster.
    """
    clusters: list[list[VideoSpeaker]] = []
    centroids: list[np.ndarray] = []

    for spk in sorted(speakers, key=lambda s: s.speech_seconds, reverse=True):
        best_idx = -1
        best_sim = threshold
        for i, c in enumerate(centroids):
            sim = float(np.dot(spk.embedding, c))
            if sim >= best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0:
            clusters[best_idx].append(spk)
            centroids[best_idx] = centroid([s.embedding for s in clusters[best_idx]])
        else:
            clusters.append([spk])
            centroids.append(spk.embedding)

    return clusters


def build_consensus(
    speakers: list[VideoSpeaker],
    min_videos: int = MIN_CONSENSUS_VIDEOS,
    threshold: float = CLUSTER_LINK_THRESHOLD,
) -> Optional[ConsensusResult]:
    """Find the actor's voice: the cluster that recurs across videos.

    Clusters are ranked by (distinct videos, total speech seconds); the
    winner must span at least `min_videos` distinct videos. Returns None
    when no cluster recurs — the caller should skip the actor rather than
    store an unvalidated embedding.
    """
    if not speakers:
        return None

    clusters = cluster_speakers(speakers, threshold=threshold)

    def rank(cluster: list[VideoSpeaker]) -> tuple[int, float]:
        return (
            len({s.video_id for s in cluster}),
            sum(s.speech_seconds for s in cluster),
        )

    best = max(clusters, key=rank)
    video_ids = sorted({s.video_id for s in best})
    if len(video_ids) < min_videos:
        return None

    embeddings = [s.embedding for s in best]
    return ConsensusResult(
        embedding=centroid(embeddings),
        quality_score=round(mean_pairwise_cosine(embeddings), 4),
        speech_seconds=round(sum(s.speech_seconds for s in best), 2),
        n_videos=len(video_ids),
        video_ids=video_ids,
    )
