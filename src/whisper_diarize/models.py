"""Model loading, sharing and idle unloading, plus the GPU-or-CPU retry policy."""

import gc
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from faster_whisper import WhisperModel

from whisper_diarize.config import settings

log = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

_OOM_MARKERS = ("out of memory", "ALLOC_FAILED", "CUDA_ERROR_OUT_OF_MEMORY")


def is_oom(exc: BaseException) -> bool:
    """CUDA OOMs surface as torch.cuda.OutOfMemoryError (pyannote) or a RuntimeError (CTranslate2)."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or any(
        marker in str(exc) for marker in _OOM_MARKERS
    )


def free_vram_mb() -> int:
    free, _total = torch.cuda.mem_get_info()
    return free // 2**20


class Cancelled(Exception):
    pass


class ModelPool:
    """Loads models on first use, shares them between jobs and live sessions, and unloads them
    once nothing has used them for `idle_unload_s` seconds so the GPU is free for other work."""

    def __init__(self, idle_unload_s: float = settings.idle_unload_s):
        self.idle_unload_s = idle_unload_s
        self._lock = threading.RLock()
        self._models: dict[tuple[str, ...], Any] = {}
        self._in_use = 0
        self._last_used = time.monotonic()
        # The diarization pipeline isn't thread-safe; jobs and voice enrollment take turns.
        self.diarizer_lock = threading.Lock()
        threading.Thread(target=self._unload_when_idle, daemon=True).start()

    @contextmanager
    def using(self) -> Iterator["ModelPool"]:
        """Keeps the idle reaper away while models are being used."""
        with self._lock:
            self._in_use += 1
        try:
            yield self
        finally:
            with self._lock:
                self._in_use -= 1
                self._last_used = time.monotonic()

    def has(self, kind: str, device: str) -> bool:
        return any(key[0] == kind and key[1] == device for key in self._models)

    def whisper(self, name: str, device: str, compute_type: str) -> WhisperModel:
        key = ("whisper", device, name, compute_type)
        with self._lock:
            if key not in self._models:
                # Only keep one whisper model per device around.
                self._drop(lambda k: k[0] == "whisper" and k[1] == device)
                log.info("loading whisper %s on %s (%s)", name, device, compute_type)
                self._models[key] = WhisperModel(name, device=device, compute_type=compute_type)
            return self._models[key]

    def diarizer(self, device: str) -> Any:
        key = ("diarizer", device)
        with self._lock:
            if key not in self._models:
                from pyannote.audio import Pipeline

                log.info("loading %s on %s", DIARIZATION_MODEL, device)
                error = (
                    f"Couldn't load {DIARIZATION_MODEL}. Accept its terms on "
                    f"https://huggingface.co/{DIARIZATION_MODEL} and log in with "
                    "`uvx hf auth login` (or set HF_TOKEN)."
                )
                try:
                    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
                except Exception as e:
                    raise RuntimeError(error) from e
                if pipeline is None:
                    raise RuntimeError(error)
                self._models[key] = pipeline.to(torch.device(device))
            return self._models[key]

    def unload(self, device: str | None = None, keep: str | None = None) -> None:
        """Drop models (on `device`, if given), except those of kind `keep`."""
        with self._lock:
            self._drop(lambda k: (device is None or k[1] == device) and k[0] != keep)

    def _drop(self, predicate: Callable[[tuple[str, ...]], bool]) -> None:
        dropped = [key for key in self._models if predicate(key)]
        for key in dropped:
            log.info("unloading %s", key)
            del self._models[key]
        # Also run when nothing was dropped: after an OOM, cached allocator blocks are worth freeing.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_when_idle(self) -> None:
        while True:
            time.sleep(10)
            with self._lock:
                idle = time.monotonic() - self._last_used > self.idle_unload_s
                if self._models and self._in_use == 0 and idle:
                    self.unload()


Report = Callable[[str, float | None, str | None], None]


class RunContext:
    """What a pipeline run needs besides its inputs: models, progress reporting and cancellation."""

    def __init__(
        self,
        pool: ModelPool,
        report: Report = lambda stage, progress, message: None,
        cancelled: Callable[[], bool] = lambda: False,
    ):
        self.pool = pool
        self.report = report
        self.cancelled = cancelled

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.5, end - time.monotonic()))


def run_on_best_device[T](
    ctx: RunContext,
    stage: str,
    *,
    kind: str,
    vram_mb: int,
    device: str,
    fn: Callable[[str, int], T],
    wait_s: float = settings.gpu_wait_s,
) -> T:
    """Run `fn(device, attempt)` on the GPU, coping with it being busy.

    - waits (up to `wait_s`) for `vram_mb` of free VRAM before loading a model that isn't loaded,
    - on CUDA OOM, unloads our models, backs off and retries; `attempt` lets `fn` shrink batches,
    - if the GPU still isn't usable after `wait_s`, falls back to CPU (when allowed).
    """
    if device == "cpu" or not torch.cuda.is_available():
        return fn("cpu", 0)

    deadline = time.monotonic() + wait_s
    attempt = 0
    while True:
        ctx.check()
        free = free_vram_mb()
        if free < vram_mb and not ctx.pool.has(kind, "cuda"):
            if time.monotonic() >= deadline:
                break
            ctx.report(
                stage,
                None,
                f"Waiting for GPU memory: {free / 1024:.1f} GB free, need {vram_mb / 1024:.1f} GB",
            )
            ctx.sleep(5)
            continue
        try:
            return fn("cuda", attempt)
        except Exception as e:
            if not is_oom(e):
                raise
            log.warning("CUDA OOM during %s (attempt %d): %s", stage, attempt + 1, e)
        # Clean up outside the except block: the traceback keeps the failed stage's tensors
        # alive until it ends, so emptying the cache inside it would free less.
        attempt += 1
        # Free our other models. Dropping the one this stage needs wouldn't help (it gets
        # reloaded right away) and could duplicate it if another thread is still using it.
        ctx.pool.unload("cuda", keep=kind)
        if time.monotonic() >= deadline:
            break
        delay = min(5 * 2**attempt, 60)
        ctx.report(stage, None, f"GPU ran out of memory, retrying in {delay}s")
        ctx.sleep(delay)

    if device == "cuda" or not settings.cpu_fallback:
        raise RuntimeError(f"GPU unavailable for {stage} (out of memory or busy)")
    ctx.report(stage, None, "GPU unavailable, falling back to CPU (slower)")
    return fn("cpu", attempt)
