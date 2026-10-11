import numpy as np
from fastapi.testclient import TestClient

from whisper_diarize import pipeline, server
from whisper_diarize.options import Options
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
    # B also resembles Ann, but Ann is taken by the closer A; nobody sounds like C or Bob.
    assert name_speakers(centroids, bank.load(), threshold=0.5) == {"A": "Ann"}


def test_exact_mode_names_everyone_it_can(tmp_path):
    bank = VoiceBank(tmp_path / "voices.json")
    bank.add("Ann", vec(1, 0, 0))
    bank.add("Bob", vec(0, 1, 0))
    centroids = {"X": vec(0.45, 0.5, 0.74), "Y": vec(0.1, 0.3, 0.95)}
    assert name_speakers(centroids, bank.load(), threshold=0.6) == {}
    # Both are known to be here, so both get a speaker however weak the match, paired for the
    # highest total similarity (X-Ann 0.45 + Y-Bob 0.3 beats X-Bob 0.5 + Y-Ann 0.1).
    assert name_speakers(centroids, bank.load(), threshold=0.6, exact=True) == {
        "X": "Ann",
        "Y": "Bob",
    }


def test_weak_matches_dont_distort_naming(tmp_path):
    bank = VoiceBank(tmp_path / "voices.json")
    bank.add("Ann", vec(1, 0, 0))
    bank.add("Bob", vec(0, 1, 0))
    # X is clearly Bob (0.8), Y weakly resembles Bob (0.55) and is otherwise nobody. Pairing Ann
    # with X (0.6) would give a bigger total, but Ann is below the 0.7 threshold everywhere.
    centroids = {"X": vec(0.6, 0.8, 0), "Y": vec(0, 0.55, 0.835)}
    assert name_speakers(centroids, bank.load(), threshold=0.7) == {"X": "Bob"}


def test_speaker_count_follows_exact_speakers():
    assert Options(speakers=["Ann", "Bob"]).speaker_kwargs() == {"num_speakers": 2}
    assert Options(num_speakers=3).speaker_kwargs()["num_speakers"] == 3


def test_absent_and_exact_speakers_filter_the_bank(tmp_path, monkeypatch):
    bank = VoiceBank(tmp_path / "voices.json")
    for name, v in (("Ann", vec(1, 0, 0)), ("Bob", vec(0, 1, 0)), ("Cid", vec(0, 0, 1))):
        bank.add(name, v)
    monkeypatch.setattr(pipeline, "VoiceBank", lambda: bank)
    centroids = {"X": vec(1, 0, 0), "Y": vec(0, 1, 0), "Z": vec(0, 0, 1)}
    assert pipeline.identify(centroids, Options()) == {"X": "Ann", "Y": "Bob", "Z": "Cid"}
    assert pipeline.identify(centroids, Options(absent=["Bob"])) == {"X": "Ann", "Z": "Cid"}
    exact = Options(speakers=["Ann", "Cid"])
    assert pipeline.identify(centroids, exact) == {"X": "Ann", "Z": "Cid"}


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
