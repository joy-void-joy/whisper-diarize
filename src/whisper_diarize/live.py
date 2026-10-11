"""Live transcription of a microphone stream.

The browser sends 16 kHz mono PCM16 chunks. Silero VAD (bundled with faster-whisper) cuts the
stream into utterances at pauses: each finished utterance is transcribed once and becomes a line,
and the one still in progress is re-transcribed about every second as a "partial" preview.
The whole recording stays in memory so it can get a full diarization pass once it's stopped.
"""

import logging
import secrets
import threading
import time
from typing import Any

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from whisper_diarize.config import settings
from whisper_diarize.models import Cancelled, ModelPool, RunContext, run_on_best_device
from whisper_diarize.options import Options
from whisper_diarize.pipeline import SAMPLE_RATE, compute_type_for

log = logging.getLogger(__name__)

PAUSE_S = 0.6  # silence that ends an utterance
MAX_UTTERANCE_S = 20  # VAD splits longer stretches at their last short pause
PARTIAL_EVERY_S = 1.0
KEEP_SILENCE_S = 1.0  # silence kept in the buffer so speech onsets aren't clipped
ABANDONED_AFTER_S = 60  # no audio for this long = the browser went away

VAD = VadOptions(
    min_silence_duration_ms=int(PAUSE_S * 1000),
    speech_pad_ms=200,
    max_speech_duration_s=MAX_UTTERANCE_S,
)


class LiveSession:
    def __init__(self, owner: str, options: Options, pool: ModelPool):
        self.id = secrets.token_urlsafe(16)
        self.owner = owner
        self.options = options
        self.pool = pool
        self.lines: list[dict[str, Any]] = []
        self.partial = ""
        self.notice: str | None = None
        self.last_seen = time.monotonic()

        self._recording: list[np.ndarray] = []
        self._pending = np.zeros(0, np.float32)  # audio not turned into a line yet
        self._offset = 0  # samples before `_pending`
        self._language = options.language
        self._last_partial = 0.0
        self._lock = threading.Lock()
        self._new_audio = threading.Event()
        self._stopping = False
        self._discarded = False
        self._ctx = RunContext(pool, report=self._report, cancelled=lambda: self._discarded)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def feed(self, pcm16: bytes) -> None:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        with self._lock:
            self._recording.append(samples)
            self._pending = np.concatenate([self._pending, samples])
        self.last_seen = time.monotonic()
        self._new_audio.set()

    def view(self, since: int) -> dict[str, Any]:
        return {"lines": self.lines[since:], "partial": self.partial, "notice": self.notice}

    def stop(self) -> np.ndarray:
        """Transcribe whatever is left, end the session and hand back the whole recording."""
        self._stopping = True
        self._new_audio.set()
        self._thread.join()
        with self._lock:
            recording, self._recording = self._recording, []
        return np.concatenate(recording) if recording else np.zeros(0, np.float32)

    def discard(self) -> None:
        self._discarded = self._stopping = True
        self._new_audio.set()
        with self._lock:
            self._recording, self._pending = [], np.zeros(0, np.float32)

    def _report(self, stage: str, progress: float | None, message: str | None) -> None:
        if message:
            self.notice = message

    def _loop(self) -> None:
        with self.pool.using():
            while not self._stopping:
                self._new_audio.wait(timeout=1.0)
                self._new_audio.clear()
                self._step(flush=False)
            if not self._discarded:
                self._step(flush=True)

    def _step(self, flush: bool) -> None:
        try:
            self._advance(flush)
        except Cancelled:
            pass
        except Exception as e:
            log.exception("live session %s failed a step", self.id)
            self.notice = f"Error: {e}"

    def _advance(self, flush: bool) -> None:
        window = self._pending
        if len(window) == 0:
            return
        speech = get_speech_timestamps(window, VAD)
        if not speech:
            self.partial = ""
            keep = 0 if flush else int(KEEP_SILENCE_S * SAMPLE_RATE)
            if len(window) > keep:
                self._commit(len(window) - keep)
            return

        start = speech[0]["start"]
        if flush:
            cut = len(window)
        elif len(window) - speech[-1]["end"] >= PAUSE_S * SAMPLE_RATE:
            cut = speech[-1]["end"]
        elif len(speech) > 1:
            cut = speech[-2]["end"]  # a long utterance VAD split up: emit all but the open part
        else:
            if time.monotonic() - self._last_partial >= PARTIAL_EVERY_S:
                self.partial = self._transcribe(window[start:], partial=True)
                self._last_partial = time.monotonic()
            return

        text = self._transcribe(window[start:cut], partial=False)
        if text:
            self.lines.append(
                {
                    "start": (self._offset + start) / SAMPLE_RATE,
                    "end": (self._offset + cut) / SAMPLE_RATE,
                    "text": text,
                }
            )
        self.partial = ""
        self._commit(cut)

    def _commit(self, samples: int) -> None:
        with self._lock:
            self._pending = self._pending[samples:]
            self._offset += samples

    def _transcribe(self, audio: np.ndarray, partial: bool) -> str:
        o = self.options
        kwargs: dict[str, Any] = o.whisper_kwargs() | {
            "condition_on_previous_text": False,
            "vad_filter": False,
            "without_timestamps": True,
        }
        if partial:
            kwargs |= {"beam_size": 1, "temperature": 0.0}

        def run(device: str, attempt: int) -> str:
            model = self.pool.whisper(o.model, device, compute_type_for(o, device))
            language = self._language
            if language is None:
                language, _, _ = model.detect_language(audio)
                # Lock it in after the first full utterance: short ones are easy to misdetect.
                if not partial and not o.multilingual:
                    self._language = language
            # The previous line gives whisper context across utterance boundaries.
            context = self.lines[-1]["text"] if self.lines else None
            prompt = " ".join(p for p in (o.prompt_for(language), context) if p) or None
            call: dict[str, Any] = {
                **kwargs,
                "language": None if o.multilingual else language,
                "initial_prompt": prompt,
            }
            segments, _ = model.transcribe(audio, **call)
            return "".join(s.text for s in segments).strip()

        # Live audio can't wait minutes for VRAM: go to CPU right away if the GPU is full.
        return run_on_best_device(
            self._ctx,
            "live",
            kind="whisper",
            vram_mb=settings.whisper_vram_mb,
            device=o.device,
            fn=run,
            wait_s=0,
        )


class LiveSessions:
    def __init__(self, pool: ModelPool):
        self.pool = pool
        self._sessions: dict[str, LiveSession] = {}
        threading.Thread(target=self._discard_abandoned, daemon=True).start()

    def create(self, owner: str, options: Options) -> LiveSession:
        session = LiveSession(owner, options, self.pool)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str, owner: str) -> LiveSession | None:
        session = self._sessions.get(session_id)
        return session if session and session.owner == owner else None

    def pop(self, session_id: str, owner: str) -> LiveSession | None:
        session = self.get(session_id, owner)
        if session:
            del self._sessions[session_id]
        return session

    def _discard_abandoned(self) -> None:
        while True:
            time.sleep(10)
            cutoff = time.monotonic() - ABANDONED_AFTER_S
            for session in list(self._sessions.values()):
                if session.last_seen < cutoff:
                    self._sessions.pop(session.id, None)
                    session.discard()
