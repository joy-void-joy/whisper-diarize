import numpy as np
from fastapi.testclient import TestClient

from whisper_diarize import server
from whisper_diarize.pipeline import Turn, assign_speakers
from whisper_diarize.voices import VoiceBank, name_speakers, unit

from .test_pipeline import words


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def test_bank_averages_samples_and_survives_reload(tmp_path):
    bank = VoiceBank(tmp_path / "voices.json")
    bank.add("Ann", vec(1, 0, 0))
    voice = bank.add("Ann", vec(0, 1, 0))
    assert voice.samples == 2
    assert np.allclose(VoiceBank(bank.path).load()["Ann"].embedding, unit(vec(1, 1, 0)), atol=1e-5)
    assert bank.rename("Ann", "Anne") and not bank.rename("Ann", "X")
    assert bank.remove("Anne") and bank.load() == {}


def test_speakers_get_the_closest_voice_above_the_threshold_once(tmp_path):
    bank = VoiceBank(tmp_path / "voices.json")
    bank.add("Ann", vec(1, 0, 0))
    bank.add("Bob", vec(0, 1, 0))
    centroids = {"A": vec(0.9, 0.1, 0), "B": vec(0.8, 0.2, 0.1), "C": vec(0, 0, 1)}
    # B also resembles Ann, but Ann is taken by the closer A; nobody sounds like C.
    assert name_speakers(centroids, bank.load(), threshold=0.5, exact=False) == {"A": "Ann"}
    # Exact mode names everyone it can, threshold or not.
    exact = name_speakers(centroids, bank.load(), threshold=0.5, exact=True)
    assert exact["A"] == "Ann" and len(exact) == 2


def test_named_and_unnamed_speakers():
    ws = words((0, 1, " Hi."), (2, 3, " Yo."), (4, 5, " Hey."))
    turns = [Turn(0, 1.5, "X"), Turn(1.5, 3.5, "Y"), Turn(3.5, 6, "Z")]
    shown = assign_speakers(ws, turns, {"Y": "Bob"})
    assert [w.speaker for w in ws] == ["Speaker 1", "Bob", "Speaker 2"]
    assert shown == {"X": "Speaker 1", "Y": "Bob", "Z": "Speaker 2"}


def test_voice_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "voice_bank", VoiceBank(tmp_path / "voices.json"))
    client = TestClient(server.app)
    assert client.post("/api/voices", json={"name": " Ann ", "embedding": [1, 0]}).json() == {
        "name": "Ann",
        "samples": 1,
    }
    assert (
        client.post("/api/voices", json={"name": "Bob", "embedding": [1, 2, 3]}).status_code == 400
    )
    assert client.post("/api/voices", json={"name": "", "embedding": [1, 0]}).status_code == 400
    assert client.patch("/api/voices/Ann", json={"name": "Anne"}).status_code == 200
    assert client.get("/api/voices").json() == [{"name": "Anne", "samples": 1}]
    assert client.delete("/api/voices/Anne").status_code == 204
    assert client.delete("/api/voices/Anne").status_code == 404
