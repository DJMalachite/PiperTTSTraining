"""Synthesize from a checkpoint without exporting.

``piper.train.infer_torch`` runs the model directly from a ``.ckpt``, so you can
hear arbitrary text at any point during training without a 900 MB export cycle.

TensorBoard already logs audio for the held-back test utterances every validation
epoch, and that is the better first stop. This is for the question TensorBoard
cannot answer: how does it say *this* sentence, and what do the inference scales
sound like.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .. import env as env_mod
from .. import profile as profile_mod
from .. import proc, tui
from ..paths import REPO_ROOT, VoicePaths, venv_python
from . import export as export_mod
from . import launch

# Chosen to exercise the things that break first: sentence-final prosody, a
# question contour, an exclamation, numbers, and a long clause that tests the
# duration predictor.
DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Did you remember to close the door behind you?",
    "Stop right there!",
    "It costs forty-two pounds and fifty pence, as of March 2026.",
    "Although the weather had turned, and the road ahead was longer than "
    "anyone had expected, they carried on walking until the light failed.",
]


class PreviewError(RuntimeError):
    pass


def run(
    profile_name: str | None = None,
    checkpoint: str | None = None,
    sentences: Sequence[str] | None = None,
    noise_scale: float | None = None,
    length_scale: float | None = None,
    noise_w: float | None = None,
    offline: bool = False,
) -> int:
    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    if not paths.piper_config_json.exists():
        raise PreviewError(
            f"no voice config at {paths.piper_config_json}. It is written during "
            f"training, so train for at least one validation epoch first."
        )

    target = Path(checkpoint) if checkpoint else _choose(paths)
    if not target.exists():
        raise PreviewError(f"checkpoint not found: {target}")

    text = list(sentences) if sentences else DEFAULT_SENTENCES
    destination = paths.previews / target.stem[:60]
    destination.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        str(venv_python()),
        "-m",
        "piper.train.infer_torch",
        "--checkpoint",
        str(target),
        "--config",
        str(paths.piper_config_json),
        "--output-dir",
        str(destination),
    ]
    for flag, value, fallback in (
        ("--noise-scale", noise_scale, prof.export.noise_scale),
        ("--length-scale", length_scale, prof.export.length_scale),
        ("--noise-w", noise_w, prof.export.noise_w),
    ):
        argv += [flag, str(value if value is not None else fallback)]

    tui.ok(f"synthesizing {len(text)} sentence(s) from {target.name}")
    tui.hint(f"  scales: {' '.join(argv[-6:])}")

    # infer_torch reads one JSON object per line from stdin — `utt["text"]`,
    # with an optional `speaker_id` — not bare text.
    payload = "\n".join(json.dumps({"text": sentence}) for sentence in text) + "\n"

    env = env_mod.training_env(prof.runtime.env, offline=offline or prof.runtime.offline)
    result = _run_with_stdin(argv, payload, env, paths)

    if not result.ok:
        tui.error("inference failed:")
        for line in result.lines[-20:]:
            print(f"  {line}")
        return result.returncode or 1

    produced = sorted(destination.glob("*.wav"))
    if not produced:
        tui.warn(f"inference reported success but wrote no WAV files to {destination}")
        return 1

    tui.ok(f"wrote {len(produced)} file(s) to {destination}")
    # infer_torch names each file after its stdin line index, so the mapping
    # back to sentences is only obvious if we print it.
    for index, sentence in enumerate(text):
        path = destination / f"{index}.wav"
        if path.exists():
            size = path.stat().st_size / 1024
            tui.bullet(f"{path.name} ({size:.0f} KB) — {sentence[:60]}")
    tui.info("")
    tui.info("Listen with:")
    tui.info(f"  aplay {destination}/*.wav")
    tui.hint(f"  or copy them off the box: scp -r user@host:{destination} .")
    return 0


def _run_with_stdin(
    argv: list[str], payload: str, env: dict[str, str], paths: VoicePaths
) -> proc.Result:
    """infer_torch reads sentences from stdin, one per line."""
    import os
    import subprocess

    merged = dict(os.environ)
    merged.update(env)
    log_path = paths.logs / "preview.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        env=merged,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=3600,
    )
    lines = (completed.stdout or "").splitlines()
    with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(proc.describe(argv) + "\n")
        handle.write("\n".join(lines) + "\n")
    return proc.Result(
        argv=argv, returncode=completed.returncode, lines=lines, log_path=log_path
    )


def _choose(paths: VoicePaths) -> Path:
    candidates = launch.find_checkpoints(paths)
    if not candidates:
        raise PreviewError(
            f"no checkpoints under {paths.lightning_logs}; train first"
        )
    rows = export_mod.describe_checkpoints(paths)[:12]
    tui.heading("Checkpoints")
    tui.table(rows, headers=["file", "size", "val_mel", "val_mos"])
    tui.info("")
    options = [(path.name, path.name) for path in reversed(candidates[-12:])]
    options.append(("__best__", "best by val_mel (recommended)"))
    choice = tui.menu("Which checkpoint?", options, allow_back=False)
    if choice == "__best__":
        return export_mod.pick_checkpoint(paths, "val_mel")
    return next(path for path in candidates if path.name == choice)
