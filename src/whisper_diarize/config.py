"""Runtime settings, read from WD_* environment variables."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WD_")

    host: str = "127.0.0.1"
    port: int = 5801
    # Header the reverse proxy fills with the authenticated user (Caddy: {http.auth.user.id}).
    # Jobs are only visible to the user who created them.
    user_header: str = "X-Remote-User"
    # When set, every request must carry it in X-Proxy-Secret. Use it when something besides the
    # proxy can reach the port (e.g. other users on the far end of an SSH tunnel): without it they
    # could skip the proxy's login and fake the user header.
    proxy_secret: str | None = None
    frontend_dir: Path = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    # Models stay loaded between jobs and are dropped after this many idle seconds. Reloading
    # is cheap (~1.5 s for large-v3), while idle models hold ~4 GB of VRAM.
    idle_unload_s: float = 30
    # Before loading a model onto the GPU, wait until this much VRAM is free (MiB).
    whisper_vram_mb: int = 6500  # large-v3 fp16 peaks at ~6.1 GB with batch_size 8
    diarizer_vram_mb: int = 2000  # pyannote peaks at ~1.1 GB (embedding batch 4)
    # How long a job keeps waiting for VRAM / retrying after OOMs before giving up on the GPU.
    gpu_wait_s: float = 300
    # When the GPU stays unavailable, run on CPU instead of failing.
    cpu_fallback: bool = True

    # Named voiceprints (no audio), shared by everyone using this server and by the CLI.
    voices_path: Path = (
        Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
        / "whisper-diarize"
        / "voices.json"
    )

    # Finished jobs (and their transcript) are forgotten after this many seconds.
    result_ttl_s: float = 900
    max_upload_mb: int = 2048


settings = Settings()
