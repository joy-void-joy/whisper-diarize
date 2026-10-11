"""Transcription + speaker diarization with faster-whisper and pyannote."""

import warnings

# pyannote warns at import time when torchcodec can't load (e.g. against FFmpeg 9). We never let
# pyannote decode files itself: it always receives in-memory waveforms.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
