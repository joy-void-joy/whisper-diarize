"""Command line: batch-transcribe files, list options, or run the web server."""

import json
import logging
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from whisper_diarize.config import settings
from whisper_diarize.options import Options

app = typer.Typer(no_args_is_help=True, help=__doc__)
console = Console(stderr=True)


class Format(StrEnum):
    txt = "txt"
    srt = "srt"
    json = "json"


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """KEY=VALUE pairs, where VALUE is read as JSON when it parses (numbers, lists, true, null)."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--set")
        try:
            overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key.strip()] = value
    return overrides


@app.command()
def transcribe(
    files: Annotated[list[Path] | None, typer.Argument(help="Audio/video files.")] = None,
    files_from: Annotated[
        Path | None, typer.Option(help="Text file with more input paths, one per line.")
    ] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: next to each input).")
    ] = None,
    formats: Annotated[
        list[Format] | None, typer.Option("--format", "-f", help="Default: txt and srt.")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Redo files whose outputs exist.")] = False,
    model: Annotated[str, typer.Option(help="Whisper model.")] = "large-v3",
    language: Annotated[str | None, typer.Option(help="Language code (default: detect).")] = None,
    diarize: Annotated[bool, typer.Option(help="Label speakers.")] = True,
    num_speakers: Annotated[int | None, typer.Option(help="Exact speaker count.")] = None,
    speakers: Annotated[
        list[str] | None,
        typer.Option(
            "--speaker",
            help="Exactly these saved voices are in the recording (repeat for each). Without "
            "it, any saved voice that matches a speaker still names them.",
        ),
    ] = None,
    prompt: Annotated[str | None, typer.Option(help="Initial prompt (names, jargon).")] = None,
    device: Annotated[str, typer.Option(help="auto, cuda or cpu.")] = "auto",
    set_: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Any other option as KEY=VALUE, e.g. --set beam_size=1 "
            "--set temperature='[0, 0.4]'. See `whisper-diarize options`.",
        ),
    ] = None,
) -> None:
    """Transcribe (and diarize) files into .txt/.srt/.json, loading models once per batch."""
    from whisper_diarize import pipeline
    from whisper_diarize.models import ModelPool, RunContext

    inputs = list(files or [])
    if files_from:
        lines = files_from.read_text().splitlines()
        inputs += [Path(line.strip()) for line in lines if line.strip()]
    if not inputs:
        raise typer.BadParameter("no input files")

    try:
        options = Options.model_validate(
            {
                "model": model,
                "language": language,
                "diarize": diarize,
                "num_speakers": num_speakers,
                "speakers": speakers or [],
                "initial_prompt": prompt,
                "device": device,
                **parse_overrides(set_ or []),
            }
        )
    except ValidationError as e:
        raise typer.BadParameter(str(e)) from e

    pool = ModelPool(idle_unload_s=float("inf"))
    failed = 0
    for i, path in enumerate(inputs, 1):
        targets = {
            fmt: (out_dir or path.parent) / f"{path.stem}.{fmt}"
            for fmt in formats or [Format.txt, Format.srt]
        }
        label = f"[{i}/{len(inputs)}] {path.name}"
        if not overwrite and all(t.exists() for t in targets.values()):
            console.print(f"{label}: outputs exist, skipping")
            continue
        try:
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.fields[note]}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(label, total=1.0, note="decoding")

                def report(
                    stage: str, fraction: float | None, message: str | None, task=task
                ) -> None:
                    progress.update(
                        task,
                        completed=fraction if fraction is not None else None,
                        note=message or stage,
                    )

                audio = pipeline.load_audio(str(path))
                result = pipeline.run(audio, options, RunContext(pool, report=report))
            renderers = {"txt": result.to_text, "srt": result.to_srt, "json": result.to_json}
            for fmt, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(renderers[fmt](), encoding="utf-8")
            console.print(f"{label}: wrote {', '.join(str(t) for t in targets.values())}")
        except Exception as e:  # noqa: BLE001 - report it and carry on with the batch
            failed += 1
            console.print(f"[red]{label}: failed: {e}[/red]")
    if failed:
        raise typer.Exit(1)


voices_app = typer.Typer(
    no_args_is_help=True,
    help="The voice bank (shared with the web UI): names that diarized speakers get when their "
    "voice matches.",
)
app.add_typer(voices_app, name="voices")


@voices_app.command("list")
def voices_list() -> None:
    """List saved voices."""
    from whisper_diarize.voices import VoiceBank

    bank = VoiceBank()
    voices = bank.load()
    for voice in voices.values():
        print(f"{voice.name}  ({voice.samples} sample{'s' if voice.samples > 1 else ''})")
    console.print(f"{len(voices)} voices in {bank.path}")


@voices_app.command("add")
def voices_add(
    name: Annotated[str, typer.Argument(help="Who's speaking.")],
    files: Annotated[list[Path], typer.Argument(help="Recordings of them talking (10-30 s).")],
) -> None:
    """Save a voice from recordings of that person (mostly alone); each file adds a sample."""
    from whisper_diarize import pipeline
    from whisper_diarize.models import ModelPool, RunContext
    from whisper_diarize.voices import VoiceBank

    bank, ctx = VoiceBank(), RunContext(ModelPool(idle_unload_s=float("inf")))
    for path in files:
        embedding, speech_s = pipeline.voiceprint(pipeline.load_audio(str(path)), ctx)
        voice = bank.add(name, embedding)
        console.print(
            f"{name}: added {speech_s:.0f} s of speech from {path} ({voice.samples} samples)"
        )


@voices_app.command("remove")
def voices_remove(name: str) -> None:
    """Delete a saved voice."""
    from whisper_diarize.voices import VoiceBank

    if not VoiceBank().remove(name):
        raise typer.BadParameter(f"no voice called {name!r}")


@voices_app.command("rename")
def voices_rename(old: str, new: str) -> None:
    """Rename a saved voice."""
    from whisper_diarize.voices import VoiceBank

    if not VoiceBank().rename(old, new):
        raise typer.BadParameter(f"no voice called {old!r}, or {new!r} is taken")


@app.command()
def options() -> None:
    """List every option with its default (usable with --set and in the web UI)."""
    table = Table("option", "default", "description")
    for name, field in Options.model_fields.items():
        table.add_row(name, json.dumps(field.default), field.description or "")
    Console().print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = settings.host,
    port: Annotated[int, typer.Option(help="Port.")] = settings.port,
    dev: Annotated[
        bool,
        typer.Option(
            help="Rebuild the web UI on every change and reload the server on code changes."
        ),
    ] = False,
) -> None:
    """Run the web UI + API. Put it behind a reverse proxy with auth (see README)."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    package_dir = Path(__file__).resolve().parent
    frontend = package_dir.parents[1] / "frontend"
    watcher = None
    if dev:
        if not (frontend / "node_modules").exists():
            subprocess.run(["bun", "install"], cwd=frontend, check=True)
        watcher = subprocess.Popen(["bun", "run", "build", "--watch"], cwd=frontend)
    try:
        uvicorn.run(
            "whisper_diarize.server:app",
            host=host,
            port=port,
            reload=dev,
            reload_dirs=[str(package_dir)] if dev else None,
        )
    finally:
        if watcher:
            watcher.terminate()
