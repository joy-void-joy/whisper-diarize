"""In-memory job queue. One worker thread runs jobs one at a time; nothing touches the disk.

Users take turns, so one person queueing twenty files doesn't block everyone else.
A job's audio is dropped as soon as it has been processed, and the job itself (with its
transcript) once the client deletes it or `result_ttl_s` after it finished.
"""

import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from whisper_diarize import pipeline
from whisper_diarize.config import settings
from whisper_diarize.models import Cancelled, ModelPool, RunContext
from whisper_diarize.options import Options

log = logging.getLogger(__name__)

UPLOAD_GRACE_S = 3 * 3600  # jobs created but never uploaded are dropped after this


class JobState(StrEnum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(eq=False)
class Job:
    owner: str
    options: Options
    name: str = ""
    id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    state: JobState = JobState.UPLOADING
    stage: str = ""
    progress: float | None = None
    messages: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    error: str | None = None
    duration: float = 0.0
    audio: np.ndarray | None = None
    result: pipeline.Transcript | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    updated: float = field(default_factory=time.monotonic)

    def report(self, stage: str, progress: float | None, message: str | None) -> None:
        if stage != self.stage:
            self.stage, self.progress = stage, None
        if progress is not None:
            self.progress = progress
        if message and (not self.messages or self.messages[-1] != message):
            self.messages.append(message)

    def view(self, position: int | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "position": position,
            "stage": self.stage,
            "progress": self.progress,
            "messages": list(self.messages),
            "error": self.error,
            "duration": self.duration,
            "result": self.result.to_dict() if self.result else None,
        }


class JobQueue:
    def __init__(self, pool: ModelPool, start: bool = True):
        self.pool = pool
        self._jobs: dict[str, Job] = {}
        self._queue: list[Job] = []
        self._last_served: dict[str, float] = {}
        self._wakeup = threading.Condition()
        if start:
            threading.Thread(target=self._work, daemon=True).start()
            threading.Thread(target=self._forget_old_jobs, daemon=True).start()

    def create(self, owner: str, options: Options, name: str = "") -> Job:
        job = Job(owner, options, name)
        self._jobs[job.id] = job
        return job

    def submit(self, job: Job, audio: np.ndarray) -> None:
        job.audio = audio
        job.duration = len(audio) / pipeline.SAMPLE_RATE
        job.state = JobState.QUEUED
        with self._wakeup:
            self._queue.append(job)
            self._wakeup.notify()

    def get(self, job_id: str, owner: str) -> Job | None:
        job = self._jobs.get(job_id)
        return job if job and job.owner == owner else None

    def schedule(self) -> list[Job]:
        """Queued jobs in the order they'll run: users take turns (least recently served first),
        and each user's jobs run in the order they were submitted."""
        with self._wakeup:
            per_owner: dict[str, deque[Job]] = {}
            for job in self._queue:
                per_owner.setdefault(job.owner, deque()).append(job)
        turns = sorted(per_owner, key=lambda owner: self._last_served.get(owner, 0.0))
        order = []
        while turns:
            for owner in list(turns):
                order.append(per_owner[owner].popleft())
                if not per_owner[owner]:
                    turns.remove(owner)
        return order

    def position(self, job: Job) -> int | None:
        try:
            return self.schedule().index(job) + 1
        except ValueError:
            return None

    def exists(self, job: Job) -> bool:
        return self._jobs.get(job.id) is job

    def remove(self, job: Job) -> None:
        job.cancel.set()
        with self._wakeup:
            if job in self._queue:
                self._queue.remove(job)
        self._jobs.pop(job.id, None)
        job.audio = job.result = None

    def _work(self) -> None:
        while True:
            with self._wakeup:
                while not self._queue:
                    self._wakeup.wait()
                job = self.schedule()[0]
                self._queue.remove(job)
                self._last_served[job.owner] = time.monotonic()
            self._run(job)

    def _run(self, job: Job) -> None:
        job.state = JobState.RUNNING
        ctx = RunContext(self.pool, report=job.report, cancelled=job.cancel.is_set)
        try:
            assert job.audio is not None
            job.result = pipeline.run(job.audio, job.options, ctx)
            job.state = JobState.DONE
        except Cancelled:
            job.state = JobState.CANCELLED
        except Exception as e:
            log.exception("job %s failed", job.id)
            job.state, job.error = JobState.FAILED, str(e) or type(e).__name__
        finally:
            job.audio = None
            job.updated = time.monotonic()

    def _forget_old_jobs(self) -> None:
        while True:
            time.sleep(30)
            now = time.monotonic()
            for job in list(self._jobs.values()):
                if job.state in (JobState.QUEUED, JobState.RUNNING):
                    continue
                ttl = UPLOAD_GRACE_S if job.state == JobState.UPLOADING else settings.result_ttl_s
                if job.updated < now - ttl:
                    self.remove(job)
