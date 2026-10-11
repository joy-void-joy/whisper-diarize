"""HTTP API + the built web UI.

Audio only ever travels in request bodies (never URLs or headers, which proxies like to log) and
is decoded straight into memory.

Files:  POST /api/jobs {options, name} -> PUT /api/jobs/{id}/audio <bytes> -> GET /api/jobs/{id}
        until done -> DELETE /api/jobs/{id}.
Live:   POST /api/live {options} -> POST /api/live/{id}/audio <pcm16> (every ~0.5 s; returns new
        lines) -> POST /api/live/{id}/stop (optionally queues a diarization job of the recording).
"""

import asyncio
import io
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from whisper_diarize import pipeline
from whisper_diarize.config import settings
from whisper_diarize.jobs import Job, JobQueue, JobState
from whisper_diarize.live import LiveSession, LiveSessions
from whisper_diarize.models import ModelPool, RunContext
from whisper_diarize.options import Options
from whisper_diarize.voices import VoiceBank

log = logging.getLogger(__name__)

app = FastAPI(title="whisper-diarize", docs_url="/api/docs", openapi_url="/api/openapi.json")
pool = ModelPool()
jobs = JobQueue(pool)
live = LiveSessions(pool)
voice_bank = VoiceBank()


@dataclass
class Activity:
    """What's going on right now, so you can pick a moment to restart (`whisper-diarize status`).
    Open pages hold nothing on the server: only work in progress is lost on a restart."""

    last_request: float = field(default_factory=time.monotonic)
    uploads: int = 0


activity = Activity()


# Registered before require_proxy_secret, so it runs inside it: rejected requests don't count.
@app.middleware("http")
async def note_activity(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path != "/api/status":
        activity.last_request = time.monotonic()
    return await call_next(request)


@app.middleware("http")
async def require_proxy_secret(request: Request, call_next):
    if settings.proxy_secret:
        given = request.headers.get("X-Proxy-Secret", "")
        if not secrets.compare_digest(given.encode(), settings.proxy_secret.encode()):
            return PlainTextResponse("Forbidden", status_code=403)
    return await call_next(request)


def current_user(request: Request) -> str:
    return request.headers.get(settings.user_header) or "local"


User = Annotated[str, Depends(current_user)]


async def read_body(request: Request) -> bytes:
    limit = settings.max_upload_mb * 2**20
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, f"Audio larger than {settings.max_upload_mb} MB")
    return bytes(body)


@app.get("/api/status")
def status() -> dict:
    """Whether anyone is using the server right now (see `whisper-diarize status`). Counts only,
    nothing about who or what."""
    counts = jobs.counts() | {"uploading": activity.uploads, "recording": live.count()}
    busy = any(counts[k] for k in ("uploading", "queued", "running", "recording"))
    return {"busy": busy, **counts, "idle_s": round(time.monotonic() - activity.last_request)}


@app.get("/api/config")
def config() -> dict:
    return {"schema": Options.model_json_schema(), "cuda": torch.cuda.is_available()}


class NewJob(BaseModel):
    options: Options = Options()
    name: str = ""


@app.post("/api/jobs")
def create_job(body: NewJob, user: User) -> dict:
    return {"id": jobs.create(user, body.options, body.name).id}


def get_job(job_id: str, user: User) -> Job:
    job = jobs.get(job_id, user)
    if not job:
        raise HTTPException(404, "No such job (it may have expired)")
    return job


JobDep = Annotated[Job, Depends(get_job)]


@app.put("/api/jobs/{job_id}/audio")
async def upload_audio(job: JobDep, request: Request) -> dict:
    if job.state != JobState.UPLOADING:
        raise HTTPException(409, "Audio already uploaded")
    activity.uploads += 1
    try:
        data = await read_body(request)
    finally:
        activity.uploads -= 1
    try:
        audio = await asyncio.to_thread(pipeline.load_audio, io.BytesIO(data))
    except Exception as e:
        jobs.remove(job)
        raise HTTPException(400, f"Couldn't decode audio: {e}") from e
    del data
    if len(audio) == 0:
        jobs.remove(job)
        raise HTTPException(400, "The file contains no audio")
    if not jobs.exists(job):
        raise HTTPException(410, "Job was cancelled during the upload")
    jobs.submit(job, audio)
    return job.view(jobs.position(job))


@app.get("/api/jobs/{job_id}")
def job_status(job: JobDep) -> dict:
    return job.view(jobs.position(job))


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job: JobDep) -> Response:
    jobs.remove(job)
    return Response(status_code=204)


class NewLive(BaseModel):
    options: Options = Options()


@app.post("/api/live")
def create_live(body: NewLive, user: User) -> dict:
    return {"id": live.create(user, body.options).id}


def get_session(session_id: str, user: User) -> LiveSession:
    session = live.get(session_id, user)
    if not session:
        raise HTTPException(404, "No such live session (it may have timed out)")
    return session


SessionDep = Annotated[LiveSession, Depends(get_session)]


@app.post("/api/live/{session_id}/audio")
async def live_audio(session: SessionDep, request: Request, since: int = 0) -> dict:
    session.feed(await read_body(request))
    return session.view(since)


@app.post("/api/live/{session_id}/stop")
async def stop_live(session_id: str, user: User, since: int = 0, diarize: bool = True) -> dict:
    session = live.pop(session_id, user)
    if not session:
        raise HTTPException(404, "No such live session")
    recording = await asyncio.to_thread(session.stop)
    response = session.view(since) | {"job_id": None}
    if diarize and len(recording) > pipeline.SAMPLE_RATE:
        job = jobs.create(user, session.options.model_copy(update={"diarize": True}), "Recording")
        jobs.submit(job, recording)
        response["job_id"] = job.id
    return response


@app.delete("/api/live/{session_id}", status_code=204)
def discard_live(session_id: str, user: User) -> Response:
    session = live.pop(session_id, user)
    if session:
        session.discard()
    return Response(status_code=204)


# Voice bank: shared by all users (and the CLI). Holds names and voiceprints, never audio.

MIN_ENROLL_SPEECH_S = 3.0


def voice_name(name: str) -> str:
    name = name.strip()
    if not 0 < len(name) <= 60:
        raise HTTPException(400, "Voice names need 1 to 60 characters")
    return name


@app.get("/api/voices")
def list_voices() -> list[dict]:
    return [{"name": v.name, "samples": v.samples} for v in voice_bank.load().values()]


@app.post("/api/voices/enroll")
async def enroll_voice(request: Request, name: str) -> dict:
    """Add a voice (or another sample of it) from a recording of that person talking."""
    name = voice_name(name)
    data = await read_body(request)
    try:
        audio = await asyncio.to_thread(pipeline.load_audio, io.BytesIO(data))
        embedding, speech_s = await asyncio.to_thread(pipeline.voiceprint, audio, RunContext(pool))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if speech_s < MIN_ENROLL_SPEECH_S:
        raise HTTPException(400, f"Only {speech_s:.0f} s of speech found: talk for 10-20 s")
    voice = voice_bank.add(name, embedding)
    return {"name": voice.name, "samples": voice.samples, "speech_s": round(speech_s, 1)}


class SaveVoice(BaseModel):
    name: str
    embedding: list[float]


@app.post("/api/voices")
def save_voice(body: SaveVoice) -> dict:
    """Add a voice (or another sample of it) from a speaker's voiceprint in a transcript."""
    try:
        voice = voice_bank.add(voice_name(body.name), body.embedding)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"name": voice.name, "samples": voice.samples}


class RenameVoice(BaseModel):
    name: str


@app.patch("/api/voices/{name}")
def rename_voice(name: str, body: RenameVoice) -> dict:
    if not voice_bank.rename(name, voice_name(body.name)):
        raise HTTPException(409, "No such voice, or the new name is taken")
    return {"name": body.name.strip()}


@app.delete("/api/voices/{name}", status_code=204)
def delete_voice(name: str) -> Response:
    if not voice_bank.remove(name):
        raise HTTPException(404, "No such voice")
    return Response(status_code=204)


if not settings.frontend_dir.is_dir():
    log.warning(
        "%s not found: build the web UI (`bun run build` in frontend/)", settings.frontend_dir
    )
# check_dir=False: with `serve --dev` the first build may still be running at startup.
app.mount(
    "/", StaticFiles(directory=settings.frontend_dir, html=True, check_dir=False), name="frontend"
)
