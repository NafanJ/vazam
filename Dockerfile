# Vazam backend — CPU image (FastAPI + Demucs/pyannote/SpeechBrain).
#
# Models (~535 MB: SpeechBrain ECAPA, pyannote VAD/diarization, Demucs) download
# on first use into TORCH_HOME / HF_HOME — mount those as a volume so they
# persist across restarts (see docker-compose.yml).
FROM python:3.12-slim

# System deps: ffmpeg (mp3/m4a decode + Demucs), libsndfile (torchaudio soundfile
# backend), git (some model fetches), curl (healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of torch/torchaudio first (the PyPI default pulls a
# multi-GB CUDA build); the rest then sees them already satisfied.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache models on a mountable volume; serve responsively on CPU.
ENV TORCH_HOME=/models/torch \
    HF_HOME=/models/hf \
    DEVICE=cpu \
    DEMUCS_MODEL=htdemucs \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
