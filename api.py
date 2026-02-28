"""
api.py — Vazam FastAPI backend

Endpoints
---------
  POST /identify                 — Upload audio → get voice actor(s)
  POST /identify/multi           — Multi-speaker diarization + identification
  POST /actors                   — Create a voice actor record
  GET  /actors                   — List all actors
  GET  /actors/{id}              — Actor profile + filmography
  POST /actors/{id}/embeddings   — Add an audio sample → generate + store embedding
  POST /shows                    — Create a show record
  GET  /shows                    — List all shows
  GET  /shows/{id}               — Show details + cast
  GET  /shows/search             — Search shows by title
  POST /index/rebuild            — No-op (pgvector manages indexes automatically)
  GET  /health                   — Liveness check

Environment variables
---------------------
  HF_TOKEN      HuggingFace access token (required for VAD / diarization)
  SUPABASE_URL  Supabase project URL
  SUPABASE_KEY  Supabase service-role key
  DEVICE        torch device (default: auto-detect cuda/cpu)
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ (no-op if file absent)

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db import VazamDB
from pipeline import VazamPipeline

# ── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE   = os.getenv("DEVICE",   "")

# ── App state ────────────────────────────────────────────────────────────────

db: VazamDB
pipeline: VazamPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, pipeline

    db = VazamDB()
    pipeline = VazamPipeline(
        db=db,
        hf_token=HF_TOKEN,
        device=DEVICE or None,
        use_vad=bool(HF_TOKEN),
        use_diarization=bool(HF_TOKEN),
    )

    yield

    db.close()


# ── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vazam API",
    description="Voice actor identification — Shazam for Voice Actors",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class ActorCreate(BaseModel):
    name: str
    bio: str = ""
    image_url: str = ""
    anilist_id: Optional[int] = None


class ActorResponse(BaseModel):
    id: int
    name: str
    bio: Optional[str]
    image_url: Optional[str]


class ShowCreate(BaseModel):
    title: str
    media_type: str = Field("anime", pattern="^(anime|cartoon|game|other)$")
    year: Optional[int] = None
    anilist_id: Optional[int] = None
    image_url: str = ""


class ShowResponse(BaseModel):
    id: int
    title: str
    media_type: str
    year: Optional[int]
    image_url: Optional[str]


class IdentificationMatch(BaseModel):
    actor_id: int
    actor_name: str
    character_name: str
    confidence: float
    match_level: str   # "confident" | "possible" | "none"


class IdentifyResponse(BaseModel):
    results: list[IdentificationMatch]


class MultiIdentifyResponse(BaseModel):
    speakers: dict[str, list[IdentificationMatch]]


class IndexRebuildResponse(BaseModel):
    embeddings_loaded: int
    message: str


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temp path. Caller is responsible for cleanup."""
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        content = await file.read()
        f.write(content)
    return path


# ── Routes: identification ────────────────────────────────────────────────────

@app.post("/identify", response_model=IdentifyResponse, tags=["Identification"])
async def identify(
    audio: UploadFile = File(..., description="Audio clip (WAV, MP3, M4A)"),
    isolate: bool = Form(False, description="Run Demucs vocal isolation first"),
    show_id: Optional[int] = Form(None, description="Restrict search to actors in this show"),
    top_k: int = Form(5, ge=1, le=20),
):
    """Identify the voice actor in an uploaded audio clip.

    - Set `isolate=true` when recording contains background music or SFX.
    - Set `show_id` to restrict matching to a known cast (show-aware search).
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    path = await _save_upload(audio)
    try:
        results = pipeline.identify(path, top_k=top_k, isolate=isolate, show_id=show_id)
    finally:
        os.unlink(path)

    return IdentifyResponse(results=[IdentificationMatch(**r.to_dict()) for r in results])


@app.post("/identify/multi", response_model=MultiIdentifyResponse, tags=["Identification"])
async def identify_multi(
    audio: UploadFile = File(..., description="Audio clip with potentially multiple speakers"),
    isolate: bool = Form(True, description="Run Demucs vocal isolation first"),
    top_k: int = Form(3, ge=1, le=10),
):
    """Diarize the clip and identify each detected speaker separately.

    Requires HF_TOKEN to be set (pyannote speaker-diarization-3.1).
    Falls back to single-speaker identification if diarization is unavailable.
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    path = await _save_upload(audio)
    try:
        per_speaker = pipeline.identify_multi(path, top_k=top_k, isolate=isolate)
    finally:
        os.unlink(path)

    return MultiIdentifyResponse(
        speakers={
            speaker: [IdentificationMatch(**r.to_dict()) for r in results]
            for speaker, results in per_speaker.items()
        }
    )


# ── Routes: actors ────────────────────────────────────────────────────────────

@app.post("/actors", response_model=ActorResponse, status_code=201, tags=["Actors"])
def create_actor(body: ActorCreate):
    """Register a new voice actor."""
    actor_id = db.add_actor(
        name=body.name,
        bio=body.bio,
        image_url=body.image_url,
        anilist_id=body.anilist_id,
    )
    return ActorResponse(id=actor_id, name=body.name, bio=body.bio, image_url=body.image_url)


@app.get("/actors", response_model=list[ActorResponse], tags=["Actors"])
def list_actors(limit: int = 100, offset: int = 0):
    rows = db.list_actors(limit=limit, offset=offset)
    return [ActorResponse(id=r["id"], name=r["name"], bio=None, image_url=r.get("image_url")) for r in rows]


@app.get("/actors/{actor_id}", tags=["Actors"])
def get_actor(actor_id: int):
    """Actor profile with full filmography."""
    actor = db.get_actor(actor_id)
    if not actor:
        raise HTTPException(404, f"Actor {actor_id} not found")

    filmography = db.get_actor_filmography(actor_id)
    return {
        "id":          actor["id"],
        "name":        actor["name"],
        "bio":         actor.get("bio"),
        "image_url":   actor.get("image_url"),
        "anilist_id":  actor.get("anilist_id"),
        "filmography": [
            {
                "character_name": r["character_name"],
                "show_title":     r["show_title"],
                "media_type":     r["media_type"],
                "year":           r["year"],
                "image_url":      r["image_url"],
            }
            for r in filmography
        ],
    }


@app.post("/actors/{actor_id}/embeddings", status_code=201, tags=["Actors"])
async def add_embedding(
    actor_id: int,
    audio: UploadFile = File(..., description="Clean audio sample for this voice/character"),
    voice_label: str = Form("Natural Voice", description="Label e.g. 'Spike Spiegel' or 'Natural Voice'"),
    character_id: Optional[int] = Form(None),
    isolate: bool = Form(False, description="Run Demucs isolation on this sample"),
    audio_source: str = Form("", description="Source description e.g. 'convention_panel_2024'"),
):
    """Upload an audio sample for a voice actor, generate an embedding, and store it."""
    actor = db.get_actor(actor_id)
    if not actor:
        raise HTTPException(404, f"Actor {actor_id} not found")

    path = await _save_upload(audio)
    try:
        embedding = pipeline.embed_file(path, isolate=isolate)
    finally:
        os.unlink(path)

    emb_id = db.add_embedding(
        actor_id=actor_id,
        embedding=embedding,
        character_id=character_id,
        voice_label=voice_label,
        audio_source=audio_source,
        verified=False,
    )

    return {"embedding_id": emb_id, "index_size": db.get_embedding_count()}


# ── Routes: shows ─────────────────────────────────────────────────────────────

@app.post("/shows", response_model=ShowResponse, status_code=201, tags=["Shows"])
def create_show(body: ShowCreate):
    show_id = db.add_show(
        title=body.title,
        media_type=body.media_type,
        year=body.year,
        anilist_id=body.anilist_id,
        image_url=body.image_url,
    )
    return ShowResponse(
        id=show_id,
        title=body.title,
        media_type=body.media_type,
        year=body.year,
        image_url=body.image_url,
    )


@app.get("/shows", response_model=list[ShowResponse], tags=["Shows"])
def list_shows(limit: int = 100, offset: int = 0):
    rows = db.list_shows(limit=limit, offset=offset)
    return [ShowResponse(id=r["id"], title=r["title"], media_type=r["media_type"],
                         year=r["year"], image_url=r.get("image_url")) for r in rows]


@app.get("/shows/search", tags=["Shows"])
def search_shows(q: str):
    rows = db.search_show(q)
    return [{"id": r["id"], "title": r["title"], "year": r.get("year")} for r in rows]


@app.get("/shows/{show_id}", tags=["Shows"])
def get_show(show_id: int):
    """Show details with full cast list."""
    show = db.get_show(show_id)
    if not show:
        raise HTTPException(404, f"Show {show_id} not found")

    cast = db.get_actors_for_show(show_id)
    return {
        "id":         show["id"],
        "title":      show["title"],
        "media_type": show.get("media_type"),
        "year":       show.get("year"),
        "image_url":  show.get("image_url"),
        "cast": [{"actor_id": r["id"], "actor_name": r["name"]} for r in cast],
    }


# ── Routes: admin ─────────────────────────────────────────────────────────────

@app.post("/index/rebuild", response_model=IndexRebuildResponse, tags=["Admin"])
def rebuild_index(strategy: str = "individual"):
    """No-op endpoint kept for backward compatibility.

    pgvector manages its HNSW index automatically — no manual rebuild needed.
    Returns the current embedding count.
    """
    n = db.get_embedding_count()
    return IndexRebuildResponse(
        embeddings_loaded=n,
        message=f"pgvector index is managed automatically. {n} embeddings in store.",
    )


@app.get("/health", tags=["Admin"])
def health():
    n = db.get_embedding_count()
    return {
        "status":       "ok",
        "index_size":   n,
        "db_embeddings": n,
        "vad_enabled":  pipeline.use_vad,
        "diarization":  pipeline.use_diarization,
        "device":       pipeline.device,
    }
