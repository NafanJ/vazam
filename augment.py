"""
augment.py — channel augmentation for reference embeddings

Reference audio is clean YouTube speech, but production queries are a phone
microphone pointed at a TV in a room — reverb, background noise, and small-
speaker coloration. Speaker embeddings degrade under that mismatch, so the
consensus scraper embeds each reference voice under simulated query-channel
conditions as well and folds those embeddings into the stored centroid,
pulling references toward the channel they will actually be matched against.

Three cheap, deterministic, CPU-only simulations (no datasets, no models):

    reverb()      — convolution with a synthetic exponentially-decaying RIR
    add_noise()   — white noise at a fixed signal-to-noise ratio
    band_limit()  — high/low-pass to mimic TV-speaker frequency response

All functions take and return (1, N) float tensors at 16 kHz. Noise and RIR
generation are seeded so re-running the scraper reproduces identical
embeddings.
"""

from __future__ import annotations

import torch
import torchaudio

SAMPLE_RATE = 16000

NOISE_SNR_DB = 15.0      # typical living-room recording, not a quiet studio
RIR_SECONDS  = 0.3       # small-room decay time
BAND_LOW_HZ  = 200.0     # TV speakers roll off bass
BAND_HIGH_HZ = 6500.0    # and highs
SEED = 1234


def _generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def reverb(
    signal: torch.Tensor,
    rir_seconds: float = RIR_SECONDS,
    seed: int = SEED,
) -> torch.Tensor:
    """Convolve with a synthetic room impulse response.

    The RIR is exponentially-decaying white noise with a unit direct path —
    a standard cheap stand-in for a measured small-room response.
    """
    n = int(rir_seconds * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    decay = torch.exp(-t / (rir_seconds / 3.0))
    rir = torch.randn(n, generator=_generator(seed)) * decay
    rir[0] = 1.0
    rir = rir / rir.norm()

    wet = torchaudio.functional.fftconvolve(signal, rir.reshape(1, -1))
    return wet[:, : signal.shape[1]]


def add_noise(
    signal: torch.Tensor,
    snr_db: float = NOISE_SNR_DB,
    seed: int = SEED,
) -> torch.Tensor:
    """Add white noise scaled to the target SNR (no-op on silent input)."""
    noise = torch.randn(signal.shape, generator=_generator(seed))
    signal_power = signal.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr = 10.0 ** (snr_db / 10.0)
    scale = torch.sqrt(signal_power / (snr * noise_power + 1e-12))
    return signal + scale * noise


def band_limit(
    signal: torch.Tensor,
    low_hz: float = BAND_LOW_HZ,
    high_hz: float = BAND_HIGH_HZ,
) -> torch.Tensor:
    """Band-limit to a TV-speaker-like frequency response."""
    out = torchaudio.functional.highpass_biquad(signal, SAMPLE_RATE, low_hz)
    return torchaudio.functional.lowpass_biquad(out, SAMPLE_RATE, high_hz)


def augment_speech(signal: torch.Tensor) -> list[torch.Tensor]:
    """All channel-simulated variants of a clean speech tensor."""
    return [
        reverb(signal),
        add_noise(signal),
        band_limit(signal),
    ]
