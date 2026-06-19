"""
api.py — Vazam FastAPI backend

Endpoints
---------
  POST /identify                 — Upload audio → get voice actor(s)
  POST /identify/multi           — Multi-speaker diarization + identification
  POST /identify/show            — Infer the show from cast co-occurrence, then identify
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

import base64
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ (no-op if file absent)

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from db import VazamDB
from pipeline import VazamPipeline

# ── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE   = os.getenv("DEVICE",   "")

# Optional HTTP Basic auth. When both are set (e.g. for a public deployment),
# every route except /health requires these credentials; unset = open (local dev).
AUTH_USER = os.getenv("VAZAM_AUTH_USER", "")
AUTH_PASS = os.getenv("VAZAM_AUTH_PASS", "")


def _basic_auth_ok(authorization: str) -> bool:
    """True if the request may proceed (auth disabled, or credentials match)."""
    if not (AUTH_USER and AUTH_PASS):
        return True
    if not authorization.startswith("Basic "):
        return False
    try:
        user, _, pwd = base64.b64decode(authorization[6:]).decode().partition(":")
    except Exception:
        return False
    return secrets.compare_digest(user, AUTH_USER) and secrets.compare_digest(pwd, AUTH_PASS)

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


@app.middleware("http")
async def _basic_auth_middleware(request, call_next):
    """Gate every route behind HTTP Basic when VAZAM_AUTH_USER/PASS are set.

    /health is always open so container healthchecks keep working.
    """
    if request.url.path != "/health" and not _basic_auth_ok(
        request.headers.get("authorization", "")
    ):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Vazam"'})
    return await call_next(request)

# Web dashboard (recording UI + character admin) — a React/Vite app in web/,
# built into static/dist (see web/ and the Dockerfile's node build stage).
# api.py serves the two built HTML entry points and mounts their hashed assets.
_STATIC = os.path.join(os.path.dirname(__file__), "static")
_DIST = os.path.join(_STATIC, "dist")
_DASHBOARD = os.path.join(_DIST, "index.html")

# Stored enrolled-clip audio so the dashboard can play clips back. Lives on a
# server volume (VAZAM_AUDIO_DIR), keyed by embedding id — not in the DB.
_AUDIO_DIR = os.environ.get("VAZAM_AUDIO_DIR") or os.path.join(os.path.dirname(__file__), "enroll_audio")
os.makedirs(_AUDIO_DIR, exist_ok=True)
_AUDIO_MEDIA = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                ".ogg": "audio/ogg", ".flac": "audio/flac"}


def _store_audio(emb_id: int, data: bytes, ext: str) -> str:
    """Persist enrolled-clip bytes as <emb_id><ext>; returns the filename."""
    ext = ext.lower() if ext.lower() in _AUDIO_MEDIA else ".wav"
    fname = f"{emb_id}{ext}"
    with open(os.path.join(_AUDIO_DIR, fname), "wb") as f:
        f.write(data)
    return fname


def _remove_audio(fname) -> None:
    """Best-effort delete of a stored audio file."""
    if not fname:
        return
    try:
        os.remove(os.path.join(_AUDIO_DIR, os.path.basename(fname)))
    except OSError:
        pass


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the phone-recording dashboard (falls back to API info if missing)."""
    if os.path.exists(_DASHBOARD):
        return FileResponse(_DASHBOARD)
    return {"name": "Vazam API", "docs": "/docs", "health": "/health"}


@app.get("/characters.html", include_in_schema=False)
def characters_page_legacy():
    """Old URL for the character admin page — redirect to the clean /characters."""
    return RedirectResponse(url="/characters", status_code=301)


# Hashed JS/CSS bundles emitted by the Vite build. Mounted only when present so
# `uvicorn api:app` still starts in a checkout that hasn't run the web build yet
# (the HTML routes above just 404 / fall back until `npm run build` is run).
_ASSETS = os.path.join(_DIST, "assets")
if os.path.isdir(_ASSETS):
    app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")


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
    window_agreement: Optional[float] = None  # fraction of verification windows won
    match_level: str   # "confident" | "possible" | "none"


class IdentifyResponse(BaseModel):
    results: list[IdentificationMatch]


class MultiIdentifyResponse(BaseModel):
    speakers: dict[str, list[IdentificationMatch]]


class InferredShow(BaseModel):
    show_id: int
    title: str
    speakers_matched: int   # distinct speakers with a candidate in this cast
    speakers_total: int     # distinct speakers detected in the clip
    score: float


class ShowIdentifyResponse(BaseModel):
    show: Optional[InferredShow]   # None when no cast consensus was found
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
    verify: bool = Form(True, description="Multi-window consistency check on candidates"),
):
    """Identify the voice actor in an uploaded audio clip.

    - Set `isolate=true` when recording contains background music or SFX.
    - Set `show_id` to restrict matching to a known cast (show-aware search).
    - `verify` (default on) re-checks candidates against overlapping sub-windows
      of the clip; a match is only "confident" if it wins a majority of windows.
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    path = await _save_upload(audio)
    try:
        results = pipeline.identify(
            path, top_k=top_k, isolate=isolate, show_id=show_id, verify=verify,
        )
    finally:
        os.unlink(path)

    return IdentifyResponse(results=[IdentificationMatch(**r.to_dict()) for r in results])


class UrlIdentify(BaseModel):
    url: str
    isolate: bool = False
    show_id: Optional[int] = None
    top_k: int = Field(5, ge=1, le=20)


@app.post("/identify/url", response_model=IdentifyResponse, tags=["Identification"])
def identify_url(body: UrlIdentify):
    """Identify the voice actor from a YouTube (or other) URL.

    Downloads a short clip server-side (first ~30s) via yt-dlp, then runs the
    normal identification pipeline. Slower than a direct upload and subject to
    the source being reachable.
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    import tempfile

    import ytclip
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = ytclip.download_clip(body.url, tmp, max_seconds=30)
        except ytclip.DownloadError as exc:
            raise HTTPException(400, str(exc))
        results = pipeline.identify(
            path, top_k=body.top_k, isolate=body.isolate, show_id=body.show_id, verify=True,
        )
    return IdentifyResponse(results=[IdentificationMatch(**r.to_dict()) for r in results])


@app.post("/identify/stream", tags=["Identification"])
async def identify_stream(
    audio: UploadFile = File(..., description="Audio clip (WAV, MP3, M4A)"),
    isolate: bool = Form(True, description="Run Demucs vocal isolation first"),
    show_id: Optional[int] = Form(None, description="Restrict search to actors in this show"),
    top_k: int = Form(5, ge=1, le=20),
    verify: bool = Form(True),
):
    """Same as /identify, but streams newline-delimited JSON progress events.

    Each line is `{"stage": "..."}` as the pipeline advances, then a final
    `{"done": true, "results": [...]}` (or `{"error": "..."}`). Lets a UI show a
    live log instead of a blank wait through the (CPU-bound) isolation step.
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    path = await _save_upload(audio)

    def gen():
        import json
        import queue
        import threading

        q: "queue.Queue[tuple[str, object]]" = queue.Queue()

        def run():
            try:
                results = pipeline.identify(
                    path, top_k=top_k, isolate=isolate, show_id=show_id, verify=verify,
                    on_progress=lambda m: q.put(("log", m)),
                )
                q.put(("done", [r.to_dict() for r in results]))
            except Exception as exc:  # surface failures to the client log
                q.put(("error", str(exc)))
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        threading.Thread(target=run, daemon=True).start()
        yield json.dumps({"stage": "Received clip"}) + "\n"
        while True:
            kind, payload = q.get()
            if kind == "log":
                yield json.dumps({"stage": payload}) + "\n"
            elif kind == "done":
                yield json.dumps({"done": True, "results": payload}) + "\n"
                return
            else:
                yield json.dumps({"error": payload}) + "\n"
                return

    return StreamingResponse(gen(), media_type="application/x-ndjson")


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


@app.post("/identify/show", response_model=ShowIdentifyResponse, tags=["Identification"])
async def identify_show(
    audio: UploadFile = File(..., description="Multi-speaker clip recorded from a playing show"),
    isolate: bool = Form(True, description="Run Demucs vocal isolation first"),
    top_k: int = Form(3, ge=1, le=10),
):
    """Infer which show is playing from cast co-occurrence, then identify each
    speaker within that show — the no-input version of show-aware search.

    Diarizes the clip, matches each speaker globally, and votes on shows whose
    casts explain multiple speakers at once. Requires HF_TOKEN for diarization;
    without it (or with a single speaker) falls back to global identification
    with `show: null`.
    """
    if db.get_embedding_count() == 0:
        raise HTTPException(503, "No embeddings in index. Add voice samples first.")

    path = await _save_upload(audio)
    try:
        inference, per_speaker = pipeline.identify_show(path, top_k=top_k, isolate=isolate)
    finally:
        os.unlink(path)

    return ShowIdentifyResponse(
        show=None if inference is None else InferredShow(
            show_id=inference.show_id,
            title=inference.show_title,
            speakers_matched=inference.speakers_matched,
            speakers_total=inference.speakers_total,
            score=inference.score,
        ),
        speakers={
            speaker: [IdentificationMatch(**r.to_dict()) for r in results]
            for speaker, results in per_speaker.items()
        },
    )


@app.get("/voices", tags=["Identification"])
def list_voices():
    """List stored voice fingerprints (actor + voice label + sample count).

    Each entry is a fingerprint that `POST /enroll` can add recordings to.
    """
    return db.list_voices()


@app.post("/enroll", tags=["Identification"])
async def enroll(
    audio: UploadFile = File(..., description="A clip of this character speaking"),
    actor_id: int = Form(..., description="Actor who voices the character"),
    voice_label: str = Form(..., description="Character / voice label to add to"),
    isolate: bool = Form(True, description="Run Demucs vocal isolation first"),
    character_id: Optional[int] = Form(None, description="Link to this character (lets you enroll a character with no clips yet)"),
):
    """Add a recording to a character's voice fingerprint (enrollment).

    Embeds the clip and stores it as an additional reference for this
    actor+voice. Pass `character_id` to link it to a specific character
    (required to start a character that has no clips yet); otherwise it inherits
    the character of an existing voice with the same actor + label.
    Phone-mic recordings are ideal here — they match real query conditions.
    """
    actor = db.get_actor(actor_id)
    if not actor:
        raise HTTPException(404, f"Actor {actor_id} not found")

    if character_id is None:
        existing = db.find_voice(actor_id, voice_label)
        character_id = existing["character_id"] if existing else None

    path = await _save_upload(audio)
    try:
        embedding = pipeline.embed_file(path, isolate=isolate)
        with open(path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(path)

    emb_id = db.add_embedding(
        actor_id=actor_id, embedding=embedding, character_id=character_id,
        voice_label=voice_label, audio_source="dashboard-enroll", verified=False,
    )
    # Keep the clip so it can be played back from the character modal.
    ext = os.path.splitext(audio.filename or "")[1] or ".wav"
    db.set_embedding_audio(emb_id, _store_audio(emb_id, audio_bytes, ext))
    samples = next(
        (v["samples"] for v in db.list_voices()
         if v["actor_id"] == actor_id and v["voice_label"] == voice_label), 1,
    )
    return {"ok": True, "actor_name": actor["name"], "voice_label": voice_label, "samples": samples}


class UrlFetch(BaseModel):
    url: str
    max_seconds: int = Field(240, ge=5, le=600)


@app.post("/fetch/url", tags=["Identification"])
def fetch_url(body: UrlFetch):
    """Download a clip from a URL and return the audio bytes.

    Lets the dashboard play and trim the clip (client-side) before sending just
    the chosen selection to /identify or /enroll — useful when a video has
    several speakers and you want to isolate one voice.
    """
    import tempfile

    import ytclip
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = ytclip.download_clip(body.url, tmp, max_seconds=body.max_seconds)
        except ytclip.DownloadError as exc:
            raise HTTPException(400, str(exc))
        with open(path, "rb") as f:
            data = f.read()
    return Response(content=data, media_type="audio/mpeg",
                    headers={"Content-Disposition": "inline; filename=clip.mp3"})


class UrlEnroll(BaseModel):
    url: str
    actor_id: int
    voice_label: str
    isolate: bool = True
    character_id: Optional[int] = None


@app.post("/enroll/url", tags=["Identification"])
def enroll_url(body: UrlEnroll):
    """Add a clip from a YouTube (or other) URL to a character's fingerprint.

    Downloads a short clip server-side (first ~60s), embeds it, and stores it as
    an additional reference — the same as /enroll but sourced from a link. Pass
    `character_id` to link a character that has no clips yet. The URL is kept as
    the clip's source for provenance.
    """
    actor = db.get_actor(body.actor_id)
    if not actor:
        raise HTTPException(404, f"Actor {body.actor_id} not found")

    character_id = body.character_id
    if character_id is None:
        existing = db.find_voice(body.actor_id, body.voice_label)
        character_id = existing["character_id"] if existing else None

    import tempfile

    import ytclip
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = ytclip.download_clip(body.url, tmp, max_seconds=60)
        except ytclip.DownloadError as exc:
            raise HTTPException(400, str(exc))
        embedding = pipeline.embed_file(path, isolate=body.isolate)
        with open(path, "rb") as f:
            audio_bytes = f.read()

    emb_id = db.add_embedding(
        actor_id=body.actor_id, embedding=embedding, character_id=character_id,
        voice_label=body.voice_label, audio_source="dashboard-enroll-url",
        source_url=body.url, verified=False,
    )
    db.set_embedding_audio(emb_id, _store_audio(emb_id, audio_bytes, ".mp3"))
    samples = next(
        (v["samples"] for v in db.list_voices()
         if v["actor_id"] == body.actor_id and v["voice_label"] == body.voice_label), 1,
    )
    return {"ok": True, "actor_name": actor["name"], "voice_label": body.voice_label,
            "samples": samples, "source_url": body.url}


@app.get("/embeddings/{embedding_id}/audio", tags=["Identification"])
def embedding_audio(embedding_id: int):
    """Stream the stored audio clip for an enrolled embedding (if any)."""
    row = db.get_embedding(embedding_id)
    if not row or not row.get("audio_path"):
        raise HTTPException(404, "No stored audio for this clip")
    fname = os.path.basename(row["audio_path"])
    full = os.path.join(_AUDIO_DIR, fname)
    if not os.path.exists(full):
        raise HTTPException(404, "Audio file missing")
    media = _AUDIO_MEDIA.get(os.path.splitext(fname)[1].lower(), "application/octet-stream")
    return FileResponse(full, media_type=media)


@app.delete("/embeddings/{embedding_id}", tags=["Identification"])
def delete_embedding(embedding_id: int):
    """Remove one embedding (a single clip) from a character's fingerprint."""
    row = db.get_embedding(embedding_id)
    if not db.delete_embedding(embedding_id):
        raise HTTPException(404, f"Embedding {embedding_id} not found")
    _remove_audio(row.get("audio_path") if row else None)
    return {"ok": True, "deleted": embedding_id}


# ── Routes: characters ────────────────────────────────────────────────────────

class CharacterUpdate(BaseModel):
    image_url: Optional[str] = None
    occupation: Optional[str] = None


@app.get("/characters", tags=["Characters"])
def list_characters(request: Request):
    """All characters with actor, show, and voice-sample count (voiced first).

    Content-negotiated so the admin page can live at a clean /characters URL: a
    browser navigation (Accept: text/html) gets the character admin page, while
    the dashboard's fetch() data calls (which send Accept: application/json) get
    the JSON list. The documented API behaviour is the JSON list.
    """
    page = os.path.join(_DIST, "characters.html")
    if "text/html" in request.headers.get("accept", "") and os.path.exists(page):
        return FileResponse(page)
    return db.list_characters()


@app.get("/characters/{character_id}", tags=["Characters"])
def get_character(character_id: int):
    """A character's profile + the source files behind its voice embeddings."""
    character = db.get_character(character_id)
    if not character:
        raise HTTPException(404, f"Character {character_id} not found")
    return character


@app.patch("/characters/{character_id}", tags=["Characters"])
def update_character(character_id: int, body: CharacterUpdate):
    """Edit a character's image_url and/or occupation."""
    if db.get_character(character_id) is None:
        raise HTTPException(404, f"Character {character_id} not found")
    return db.update_character(
        character_id, image_url=body.image_url, occupation=body.occupation,
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
