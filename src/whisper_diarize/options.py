"""Every user-tunable knob, shared by the CLI, the HTTP API and the web UI's settings form.

`group` in each field's schema tells the UI where to show it: "basic" fields are always visible,
"advanced" ones are one click away, and the rest sit under "All settings".
"""

from typing import Any, Literal

from faster_whisper.tokenizer import _LANGUAGE_CODES
from pydantic import BaseModel, ConfigDict, Field

MODELS = [
    "large-v3",
    "large-v3-turbo",
    "large-v2",
    "distil-large-v3.5",
    "medium",
    "small",
    "base",
    "tiny",
]
LANGUAGES = sorted(_LANGUAGE_CODES)

# Whisper sometimes slips into all-lowercase, unpunctuated output for a whole window. A short
# punctuated prompt in the spoken language avoids it (measured: 3/22 bad segments -> 0/22).
PUNCTUATION_PRIMERS = {
    "en": "Hello, everyone. Let's get started.",
    "fr": "Bonjour à tous. On commence ?",
    "de": "Hallo zusammen. Fangen wir an.",
    "es": "Hola a todos. ¿Empezamos?",
    "it": "Ciao a tutti. Cominciamo.",
    "pt": "Olá a todos. Vamos começar?",
    "nl": "Hallo allemaal. Zullen we beginnen?",
}


def _field(
    default: Any,
    description: str,
    group: str = "whisper",
    ui: dict[str, Any] | None = None,
    **constraints: Any,
) -> Any:
    extra = {"group": group, **(ui or {})}
    return Field(default, description=description, json_schema_extra=extra, **constraints)


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- basic ---
    language: str | None = _field(
        None,
        "Spoken language. Empty = auto-detect.",
        "basic",
        ui={"choices": LANGUAGES},
        title="Language",
    )
    num_speakers: int | None = _field(
        None, "Exact number of speakers, if known.", "basic", ge=1, title="Number of speakers"
    )
    diarize: bool = _field(
        True,
        "Tell speakers apart (and name the ones in the voice bank). For recordings, this redoes "
        "the transcript of the whole recording once you stop.",
        "basic",
        title="Label speakers",
    )

    # --- advanced ---
    model: str = _field(
        "large-v3",
        "Whisper model. large-v3-turbo is ~2x faster, slightly less accurate. (The CLI also "
        "takes any faster-whisper name or CTranslate2 repo id.)",
        "advanced",
        ui={"choices": MODELS},
        title="Model",
    )
    initial_prompt: str | None = _field(
        None,
        "Names, jargon or spellings to expect, or a sentence in the style you want "
        "(punctuation, casing).",
        "advanced",
        title="Prompt",
    )
    task: Literal["transcribe", "translate"] = _field(
        "transcribe",
        "'translate' outputs English whatever the spoken language.",
        "advanced",
        title="Task",
    )

    # --- diarization ---
    identify_speakers: bool = _field(
        True, "Name speakers who match a voice in the voice bank.", "diarization"
    )
    voice_threshold: float = _field(
        0.5,
        "How similar (0-1) a speaker must be to a saved voice to get its name. The same person "
        "on the same microphone scores ~0.9, different people ~0.3 or less.",
        "diarization",
        ge=0,
        le=1,
    )
    absent: list[str] = _field(
        [], "Saved voices that aren't in the recording: never used.", "voices"
    )
    speakers: list[str] = _field(
        [],
        "Exactly these saved voices are in the recording: split it into that many speakers and "
        "give each one of these names.",
        "voices",
    )
    min_speakers: int | None = _field(None, "Lower bound on the number of speakers.", "diarization")
    max_speakers: int | None = _field(None, "Upper bound on the number of speakers.", "diarization")
    speaker_per_sentence: bool = _field(
        True,
        "Give each sentence a single speaker (majority vote), which hides word-level jitter at "
        "speaker changes.",
        "diarization",
    )

    # --- whisper ---
    prime_punctuation: bool = _field(
        True,
        "When no initial prompt is set, prompt with a short punctuated sentence in the spoken "
        f"language ({', '.join(PUNCTUATION_PRIMERS)}) so whisper keeps casing and punctuation.",
    )
    device: Literal["auto", "cuda", "cpu"] = _field(
        "auto", "Where to run. 'auto' uses the GPU when it has room and falls back to CPU."
    )
    compute_type: Literal[
        "default", "float16", "bfloat16", "int8_float16", "int8_bfloat16", "int8", "float32"
    ] = _field("default", "Weight precision. 'default' is float16 on GPU and int8 on CPU.")
    batch_size: int = _field(
        8,
        "Chunks decoded in parallel. Faster but uses more VRAM. 0 = sequential long-form "
        "decoding (honours condition_on_previous_text).",
        ge=0,
    )
    beam_size: int = _field(5, "Beam search width.", ge=1)
    best_of: int = _field(5, "Candidates when sampling with temperature > 0.", ge=1)
    patience: float = _field(1.0, "Beam search patience factor.")
    length_penalty: float = _field(1.0, "Exponential length penalty.")
    repetition_penalty: float = _field(1.0, "Penalty applied to previously generated tokens.")
    no_repeat_ngram_size: int = _field(0, "Forbid repeating n-grams of this size (0 = off).", ge=0)
    temperature: list[float] = _field(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "Temperatures to fall back through when decoding fails the thresholds below.",
    )
    compression_ratio_threshold: float | None = _field(
        2.4, "Treat output as failed (repetitive) above this gzip compression ratio."
    )
    log_prob_threshold: float | None = _field(
        -1.0, "Treat output as failed below this average log probability."
    )
    no_speech_threshold: float | None = _field(
        0.6, "Treat a window as silence above this no-speech probability."
    )
    condition_on_previous_text: bool = _field(
        True, "Feed previous output as prompt to the next window (sequential mode only)."
    )
    prompt_reset_on_temperature: float = _field(
        0.5, "Stop conditioning on previous text once temperature exceeds this."
    )
    prefix: str | None = _field(None, "Text forced at the start of the first window.")
    hotwords: str | None = _field(None, "Words/phrases to bias decoding towards.")
    suppress_blank: bool = _field(True, "Suppress blank output at the start of sampling.")
    suppress_tokens: list[int] = _field([-1], "Token ids to suppress (-1 = default symbol set).")
    suppress_numerals: bool = _field(
        False, "Write numbers as words, which helps alignment of numbers and symbols."
    )
    max_initial_timestamp: float = _field(1.0, "The first timestamp can't be later than this.")
    hallucination_silence_threshold: float | None = _field(
        None, "Skip silent periods longer than this (s) when a hallucination is detected."
    )
    multilingual: bool = _field(False, "Detect the language again in every window.")
    language_detection_threshold: float = _field(
        0.5, "Accept a detected language above this probability."
    )
    language_detection_segments: int = _field(1, "Windows to look at for language detection.", ge=1)
    chunk_length: int | None = _field(None, "Audio window length in seconds (default 30).")
    max_new_tokens: int | None = _field(None, "Max tokens generated per window.")
    vad_filter: bool = _field(True, "Skip non-speech with Silero VAD before transcribing.")
    vad_threshold: float = _field(0.5, "VAD speech probability threshold.")
    vad_min_silence_ms: int = _field(2000, "Silence needed (ms) to split speech segments.")

    def whisper_kwargs(self) -> dict[str, Any]:
        """Arguments for faster-whisper's transcribe() (minus audio/batching/word timestamps)."""
        kwargs = self.model_dump(
            exclude={
                "model",
                "diarize",
                "num_speakers",
                "min_speakers",
                "max_speakers",
                "speaker_per_sentence",
                "identify_speakers",
                "voice_threshold",
                "absent",
                "speakers",
                "device",
                "compute_type",
                "batch_size",
                "suppress_numerals",
                "prime_punctuation",
                "vad_threshold",
                "vad_min_silence_ms",
            }
        )
        kwargs["vad_parameters"] = {
            "threshold": self.vad_threshold,
            "min_silence_duration_ms": self.vad_min_silence_ms,
        }
        return kwargs

    def prompt_for(self, language: str | None) -> str | None:
        if self.initial_prompt or not self.prime_punctuation:
            return self.initial_prompt
        return PUNCTUATION_PRIMERS.get(language or "")

    def speaker_kwargs(self) -> dict[str, int | None]:
        if self.speakers:
            return {"num_speakers": len(set(self.speakers))}
        return {
            "num_speakers": self.num_speakers,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
        }
