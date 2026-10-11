"""The voice bank: named voiceprints, shared by the server and the CLI, used to put names on
diarized speakers.

A voiceprint is pyannote's speaker embedding: the average of what a speaker's voice "looks like"
to its embedding model. It identifies a voice but can't be turned back into audio. Only names
and voiceprints are stored, never the samples they came from.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from whisper_diarize.config import settings
from whisper_diarize.models import DIARIZATION_MODEL

log = logging.getLogger(__name__)


@dataclass
class Voice:
    name: str
    embedding: np.ndarray  # unit length
    samples: int = 1


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class VoiceBank:
    """A JSON file, read on every use so the server and the CLI see each other's changes."""

    def __init__(self, path: Path | None = None):
        self.path = path or settings.voices_path
        self._lock = threading.Lock()

    def load(self) -> dict[str, Voice]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        if data.get("model") != DIARIZATION_MODEL:
            # Voiceprints from another embedding model aren't comparable.
            log.warning("ignoring %s: made with %s", self.path, data.get("model"))
            return {}
        return {
            v["name"]: Voice(v["name"], unit(v["embedding"]), v["samples"]) for v in data["voices"]
        }

    def add(self, name: str, embedding) -> Voice:
        """Save a voice, or average another sample into an existing one (more samples, e.g. from
        different microphones, make recognition more robust)."""
        with self._lock:
            voices = self.load()
            new = unit(embedding)
            if any(v.embedding.shape != new.shape for v in voices.values()):
                raise ValueError("voiceprint has the wrong size")
            if old := voices.get(name):
                voice = Voice(name, unit(old.embedding * old.samples + new), old.samples + 1)
            else:
                voice = Voice(name, new)
            voices[name] = voice
            self._save(voices)
            return voice

    def remove(self, name: str) -> bool:
        with self._lock:
            voices = self.load()
            if voices.pop(name, None) is None:
                return False
            self._save(voices)
            return True

    def rename(self, old: str, new: str) -> bool:
        with self._lock:
            voices = self.load()
            if old not in voices or new in voices:
                return False
            voice = voices.pop(old)
            voices[new] = Voice(new, voice.embedding, voice.samples)
            self._save(voices)
            return True

    def _save(self, voices: dict[str, Voice]) -> None:
        data = {
            "model": DIARIZATION_MODEL,
            "voices": [
                {"name": v.name, "samples": v.samples, "embedding": v.embedding.round(6).tolist()}
                for v in sorted(voices.values(), key=lambda v: v.name.lower())
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)  # atomic, so a reader never sees half a file


_EXACT_BONUS = 10.0  # outweighs any similarity, so in exact mode every voice finds a speaker
_NEVER = -1e6


def name_speakers(
    centroids: dict[str, np.ndarray],
    voices: dict[str, Voice],
    threshold: float,
    exact: bool = False,
) -> dict[str, str]:
    """Map diarized speaker labels to voice names.

    Each voice names at most one speaker and vice versa, picked to maximise total similarity.
    Outside exact mode a voice needs `threshold` cosine similarity; in exact mode every voice gets
    the closest speaker it can. On AMI meetings, with 30 s enrollment samples, the right voice
    scored 0.88-0.92 and the best wrong one 0.31.
    """
    if not centroids or not voices:
        return {}
    labels, names = list(centroids), list(voices)
    sims = np.array(
        [[unit(centroids[label]) @ voices[n].embedding for n in names] for label in labels]
    )
    if exact:
        weights = sims + _EXACT_BONUS
    else:
        weights = np.where(sims >= threshold, sims, _NEVER)
    # One "stays unnamed" column per speaker, so a weak match never has to be made, and can't
    # distort who gets which name either.
    weights = np.hstack([weights, np.zeros((len(labels), len(labels)))])
    rows, cols = linear_sum_assignment(-weights)
    return {
        labels[r]: names[c]
        for r, c in zip(rows, cols)
        if c < len(names) and weights[r, c] > _NEVER / 2
    }
