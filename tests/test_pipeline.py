import pytest

from whisper_diarize import models
from whisper_diarize.models import ModelPool, RunContext, run_on_best_device
from whisper_diarize.pipeline import Sentence, Transcript, Turn, Word, assign_speakers, to_sentences


def words(*specs: tuple[float, float, str]) -> list[Word]:
    return [Word(start, end, text) for start, end, text in specs]


def test_words_get_the_speaker_they_overlap_most_and_names_follow_appearance():
    ws = words((0.0, 0.5, " Hi"), (0.6, 1.0, " there."), (1.2, 1.6, " Hello."), (5.0, 5.2, " Bye."))
    assign_speakers(ws, [Turn(0.0, 1.05, "SPEAKER_07"), Turn(1.1, 2.0, "SPEAKER_02")])
    assert [w.speaker for w in ws] == ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 2"]


def test_sentence_takes_its_majority_speaker():
    ws = words((0.0, 1.0, " One"), (1.0, 2.0, " two"), (2.0, 2.2, " three."), (3.0, 4.0, " Four."))
    for w, s in zip(ws, ["A", "A", "B", "B"]):
        w.speaker = s
    sentences = to_sentences(ws, speaker_per_sentence=True)
    assert [(s.speaker, s.text) for s in sentences] == [("A", "One two three."), ("B", "Four.")]
    unsmoothed = to_sentences(ws, speaker_per_sentence=False)
    assert [(s.speaker, s.text) for s in unsmoothed] == [
        ("A", "One two"),
        ("B", "three."),
        ("B", "Four."),
    ]


def test_outputs():
    t = Transcript(
        "en",
        5.0,
        [
            Sentence(0.0, 1.5, "Speaker 1", "Hi."),
            Sentence(1.5, 3.0, "Speaker 1", "How are you?"),
            Sentence(3661.0, 3662.25, "Speaker 2", "Fine."),
        ],
    )
    assert t.to_text() == "Speaker 1: Hi. How are you?\n\nSpeaker 2: Fine."
    assert "3\n01:01:01,000 --> 01:01:02,250\nSpeaker 2: Fine.\n" in t.to_srt()


class FakeGpu:
    def __init__(self, monkeypatch, free_mb: int):
        self.free_mb = free_mb
        monkeypatch.setattr(models.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(models, "free_vram_mb", lambda: self.free_mb)
        monkeypatch.setattr(RunContext, "sleep", lambda ctx, s: None)


def run(fn, wait_s=60.0, device="auto"):
    messages = []
    ctx = RunContext(ModelPool(), report=lambda stage, p, m: m and messages.append(m))
    result = run_on_best_device(
        ctx, "test", kind="whisper", vram_mb=5000, device=device, fn=fn, wait_s=wait_s
    )
    return result, messages


def test_oom_retries_with_smaller_batches_then_succeeds(monkeypatch):
    FakeGpu(monkeypatch, free_mb=20000)
    attempts = []

    def fn(device, attempt):
        attempts.append((device, attempt))
        if attempt < 2:
            raise RuntimeError("CUDA failed with error out of memory")
        return "ok"

    result, messages = run(fn)
    assert result == "ok"
    assert attempts == [("cuda", 0), ("cuda", 1), ("cuda", 2)]
    assert sum("ran out of memory" in m for m in messages) == 2


def test_full_gpu_waits_then_falls_back_to_cpu(monkeypatch):
    FakeGpu(monkeypatch, free_mb=1000)
    clock = iter(range(0, 10_000, 10))  # every check advances 10 s
    monkeypatch.setattr(models.time, "monotonic", lambda: next(clock))

    result, messages = run(lambda device, attempt: device, wait_s=30)
    assert result == "cpu"
    assert any("Waiting for GPU memory" in m for m in messages)
    assert "falling back to CPU" in messages[-1]


def test_forced_cuda_fails_instead_of_falling_back(monkeypatch):
    FakeGpu(monkeypatch, free_mb=1000)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        run(lambda device, attempt: device, wait_s=0, device="cuda")


def test_other_errors_are_not_retried(monkeypatch):
    FakeGpu(monkeypatch, free_mb=20000)

    def fn(device, attempt):
        raise ValueError("bad audio")

    with pytest.raises(ValueError):
        run(fn)


def test_unlabelled_paragraphs_break_on_long_pauses():
    t = Transcript(
        "en",
        10.0,
        [Sentence(0, 1, None, "A."), Sentence(1.5, 2, None, "B."), Sentence(5, 6, None, "C.")],
    )
    assert t.to_text() == "A. B.\n\nC."
