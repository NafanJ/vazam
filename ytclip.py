"""
ytclip.py — download a short audio clip from a URL (yt-dlp) for on-demand
identify / enroll from the dashboard.

Reuses scrape_audio's yt-dlp invocation (interpreter module call + the JS-runtime
and challenge-solver network args) so URL handling behaves identically to the
batch scraper. Only the first ``max_seconds`` are fetched — a few seconds is
plenty to identify or enroll a voice, and short downloads keep the request fast
and polite.

    from ytclip import download_clip, DownloadError
    path = download_clip("https://youtu.be/…", tmpdir, max_seconds=30)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scrape_audio import YT_DLP, YT_NET_ARGS

DEFAULT_MAX_SECONDS = 30
DOWNLOAD_TIMEOUT = 300


class DownloadError(RuntimeError):
    """Raised when a URL could not be downloaded to an audio file."""


def download_clip(url: str, output_dir: str, max_seconds: int = DEFAULT_MAX_SECONDS) -> str:
    """Download the first ``max_seconds`` of ``url`` as an mp3 into ``output_dir``.

    Returns the path to the written file. Raises ``DownloadError`` if yt-dlp is
    missing, times out, or produces no audio (bad/blocked/private URL).
    """
    if not (url or "").strip():
        raise DownloadError("no URL provided")

    out_path = os.path.join(output_dir, "clip.mp3")
    cmd = [
        *YT_DLP,
        *YT_NET_ARGS,
        url,
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",          # ~128 kbps — enough for speaker ID
        "--download-sections", f"*0-{max_seconds}",
        "--output", os.path.join(output_dir, "clip.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=DOWNLOAD_TIMEOUT, check=False, text=True
        )
    except FileNotFoundError as exc:
        raise DownloadError("yt-dlp is not available on the server") from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("download timed out") from exc

    if not Path(out_path).exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1][:200] if tail else "unknown error"
        raise DownloadError(f"could not download audio from that URL: {detail}")
    return out_path
