"""Audio in, speaker-labelled transcript out.

1. transcribe with faster-whisper, keeping word timestamps,
2. diarize with pyannote, using its "exclusive" output (one speaker at a time), which is meant
   for lining up with transcription timestamps,
3. name speakers whose voice matches one in the voice bank,
4. give each word the speaker it overlaps most,
5. cut words into sentences, give each sentence its majority speaker, merge into turns.
"""

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline, decode_audio

from whisper_diarize.config import settings
from whisper_diarize.models import RunContext, run_on_best_device
from whisper_diarize.options import Options
from whisper_diarize.voices import VoiceBank, name_speakers

SAMPLE_RATE = 16000
PARAGRAPH_PAUSE_S = 2.0


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Turn:
    start: float
    end: float
    speaker: str | None


@dataclass
class Sentence:
    start: float
    end: float
    speaker: str | None
    text: str


@dataclass
class Transcript:
    language: str
    duration: float
    sentences: list[Sentence]
    # Speaker name -> voiceprint, so the UI can save a speaker to the voice bank.
    voiceprints: dict[str, list[float]] = field(default_factory=dict)

    def paragraphs(self) -> list[Sentence]:
        """Consecutive sentences by the same speaker, merged. Without speaker labels, a new
        paragraph starts at every pause of PARAGRAPH_PAUSE_S or more."""
        merged: list[Sentence] = []
        for s in self.sentences:
            last = merged[-1] if merged else None
            pause = last and last.speaker is None and s.start - last.end >= PARAGRAPH_PAUSE_S
            if last and last.speaker == s.speaker and not pause:
                merged[-1] = Sentence(
                    merged[-1].start, s.end, s.speaker, f"{merged[-1].text} {s.text}"
                )
            else:
                merged.append(s)
        return merged

    def to_text(self) -> str:
        return "\n\n".join(
            f"{p.speaker}: {p.text}" if p.speaker else p.text for p in self.paragraphs()
        )

    def to_srt(self) -> str:
        cues = []
        for i, s in enumerate(self.sentences, 1):
            text = f"{s.speaker}: {s.text}" if s.speaker else s.text
            cues.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{text}\n")
        return "\n".join(cues)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        data = {k: v for k, v in self.to_dict().items() if k != "voiceprints"}
        return json.dumps(data, ensure_ascii=False, indent=2)


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def load_audio(source) -> np.ndarray:
    """Decode any ffmpeg-readable file path or file object to 16 kHz mono float32."""
    audio = decode_audio(source, sampling_rate=SAMPLE_RATE)
    assert isinstance(audio, np.ndarray)  # it's a tuple only with split_stereo=True
    return audio


def run(audio: np.ndarray, options: Options, ctx: RunContext) -> Transcript:
    duration = len(audio) / SAMPLE_RATE
    with ctx.pool.using():
        words, language = run_on_best_device(
            ctx,
            "transcribing",
            kind="whisper",
            vram_mb=settings.whisper_vram_mb,
            device=options.device,
            fn=lambda device, attempt: transcribe(audio, options, ctx, device, attempt),
        )
        voiceprints = {}
        if options.diarize and words:
            turns, centroids = run_on_best_device(
                ctx,
                "diarizing",
                kind="diarizer",
                vram_mb=settings.diarizer_vram_mb,
                device=options.device,
                fn=lambda device, attempt: diarize(audio, options, ctx, device, attempt),
            )
            shown = assign_speakers(words, turns, identify(centroids, options))
            voiceprints = {
                name: centroids[label].tolist()
                for label, name in shown.items()
                if label in centroids
            }
    ctx.report("done", 1.0, None)
    sentences = to_sentences(words, options.speaker_per_sentence)
    return Transcript(language, duration, sentences, voiceprints)


def identify(centroids: dict[str, np.ndarray], options: Options) -> dict[str, str]:
    """Voice-bank names for diarized speaker labels (see voices.name_speakers)."""
    if not options.identify_speakers:
        return {}
    voices = VoiceBank().load()
    if options.speakers:
        voices = {name: v for name, v in voices.items() if name in options.speakers}
    return name_speakers(centroids, voices, options.voice_threshold, exact=bool(options.speakers))


def voiceprint(audio: np.ndarray, ctx: RunContext) -> tuple[np.ndarray, float]:
    """The main speaker's voiceprint in a sample, and how many seconds of their speech it's from.
    Made the same way as the ones it'll be compared with: by diarizing the sample."""
    options = Options()
    with ctx.pool.using():
        turns, centroids = run_on_best_device(
            ctx,
            "enrolling",
            kind="diarizer",
            vram_mb=settings.diarizer_vram_mb,
            device=options.device,
            fn=lambda device, attempt: diarize(audio, options, ctx, device, attempt),
            wait_s=0,
        )
    talk: dict[str, float] = defaultdict(float)
    for turn in turns:
        if turn.speaker is not None:
            talk[turn.speaker] += turn.end - turn.start
    if not talk:
        raise ValueError("No speech found in the sample")
    main = max(talk, key=lambda label: talk[label])
    return centroids[main], talk[main]


def compute_type_for(options: Options, device: str) -> str:
    if options.compute_type != "default":
        return options.compute_type
    return "float16" if device == "cuda" else "int8"


def transcribe(
    audio: np.ndarray, options: Options, ctx: RunContext, device: str, attempt: int
) -> tuple[list[Word], str]:
    ctx.report("transcribing", 0.0, f"Loading {options.model} on {device}")
    model = ctx.pool.whisper(options.model, device, compute_type_for(options, device))
    kwargs = options.whisper_kwargs()
    if options.suppress_numerals:
        kwargs["suppress_tokens"] = numeral_symbol_tokens(model)
    language = options.language or detect_language(audio, options, model, ctx)
    kwargs["initial_prompt"] = options.prompt_for(language)
    if not options.multilingual:
        kwargs["language"] = language

    # Halve the batch on every OOM retry.
    batch_size = max(1, options.batch_size >> attempt) if options.batch_size else 0
    if batch_size:
        pipeline = BatchedInferencePipeline(model)
        segments, info = pipeline.transcribe(
            audio, batch_size=batch_size, word_timestamps=True, **kwargs
        )
    else:
        segments, info = model.transcribe(audio, word_timestamps=True, **kwargs)

    duration = len(audio) / SAMPLE_RATE
    words = []
    for segment in segments:
        ctx.check()
        words += [Word(w.start, w.end, w.word) for w in segment.words or []]
        ctx.report("transcribing", min(segment.end / duration, 1.0), None)
    return words, info.language


def detect_language(audio: np.ndarray, options: Options, model, ctx: RunContext) -> str:
    language, probability, _ = model.detect_language(
        audio,
        vad_filter=options.vad_filter,
        language_detection_segments=options.language_detection_segments,
        language_detection_threshold=options.language_detection_threshold,
    )
    ctx.report("transcribing", 0.0, f"Detected language: {language} ({probability:.0%})")
    return language


def numeral_symbol_tokens(model) -> list[int]:
    """Token ids containing digits or currency/percent signs, to make whisper spell numbers out."""
    vocab = model.hf_tokenizer.get_vocab()
    return [-1] + [i for token, i in vocab.items() if any(c in "0123456789%$£€" for c in token)]


def diarize(
    audio: np.ndarray, options: Options, ctx: RunContext, device: str, attempt: int
) -> tuple[list[Turn], dict[str, np.ndarray]]:
    """Speaker turns (one speaker at a time) and each speaker's voiceprint."""
    ctx.report("diarizing", 0.0, f"Loading diarization model on {device}")
    pipeline = ctx.pool.diarizer(device)
    # pyannote's default embedding batch (32) peaks at ~11.8 GB of VRAM on a 3090; 4 runs just
    # as fast in ~1.1 GB (8 was 9x slower at ~9.9 GB). Halve both on every OOM retry.
    pipeline.embedding_batch_size = max(1, 4 >> attempt)
    pipeline.segmentation_batch_size = max(1, 32 >> attempt)

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        ctx.check()
        if total:
            ctx.report("diarizing", completed / total, None)
        if completed in (None, 0):
            ctx.report("diarizing", None, f"Diarization: {step_name.replace('_', ' ')}")

    waveform = torch.from_numpy(audio).unsqueeze(0)
    with ctx.pool.diarizer_lock:
        output = pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            hook=hook,
            **options.speaker_kwargs(),
        )
    turns = [
        Turn(segment.start, segment.end, label)
        for segment, _, label in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    # Rows of speaker_embeddings follow the (non-exclusive) diarization's label order.
    centroids = dict(zip(output.speaker_diarization.labels(), output.speaker_embeddings))
    return turns, centroids


def assign_speakers(
    words: list[Word], turns: list[Turn], names: dict[str, str] | None = None
) -> dict[str, str]:
    """Give each word the speaker whose turn overlaps it most, or the nearest turn if none does.
    Speakers in `names` get that name, the others "Speaker 1", "Speaker 2"... in order of
    appearance. Returns label -> name shown."""
    turns = sorted(turns, key=lambda t: t.start)
    first = 0
    for word in words:
        # Turns are exclusive and sorted, so ones ending before this word can be skipped for good.
        while first < len(turns) and turns[first].end < word.start:
            first += 1
        best, best_overlap = None, 0.0
        for turn in turns[first:]:
            if turn.start > word.end:
                break
            overlap = min(word.end, turn.end) - max(word.start, turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn, overlap
        if best is None:
            nearby = turns[max(first - 1, 0) : first + 1]
            best = min(nearby, key=lambda t: _distance(t, word), default=None)
        word.speaker = best.speaker if best else None

    names = names or {}
    shown: dict[str, str] = {}
    unnamed = 0
    for word in words:
        if word.speaker is None:
            continue
        if word.speaker not in shown:
            if word.speaker in names:
                shown[word.speaker] = names[word.speaker]
            else:
                unnamed += 1
                shown[word.speaker] = f"Speaker {unnamed}"
        word.speaker = shown[word.speaker]
    return shown


def _distance(turn: Turn, word: Word) -> float:
    return max(turn.start - word.end, word.start - turn.end, 0.0)


_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
_ACRONYM = re.compile(r"(?:[A-Za-z]\.){2,}")
# Longer "sentences" are usually missing punctuation, so don't force a single speaker on them.
_MAX_SMOOTHED_WORDS = 60


def to_sentences(words: list[Word], speaker_per_sentence: bool) -> list[Sentence]:
    sentences = []
    for group in _split_sentences(words):
        if speaker_per_sentence and len(group) <= _MAX_SMOOTHED_WORDS:
            speaker = _majority_speaker(group)
            pieces = [[Word(w.start, w.end, w.text, speaker) for w in group]]
        else:
            pieces = _split_on_speaker_change(group)
        for piece in pieces:
            text = "".join(w.text for w in piece).strip()
            if text:
                sentences.append(Sentence(piece[0].start, piece[-1].end, piece[0].speaker, text))
    return sentences


def _split_sentences(words: list[Word]) -> list[list[Word]]:
    groups, current = [], []
    for word in words:
        current.append(word)
        text = word.text.strip()
        if text.endswith(_SENTENCE_END) and not _ACRONYM.fullmatch(text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _majority_speaker(words: list[Word]) -> str | None:
    talk_time: dict[str | None, float] = defaultdict(float)
    for w in words:
        talk_time[w.speaker] += max(w.end - w.start, 0.01)
    return max(talk_time, key=lambda speaker: talk_time[speaker])


def _split_on_speaker_change(words: list[Word]) -> list[list[Word]]:
    pieces: list[list[Word]] = []
    for word in words:
        if pieces and pieces[-1][-1].speaker == word.speaker:
            pieces[-1].append(word)
        else:
            pieces.append([word])
    return pieces
