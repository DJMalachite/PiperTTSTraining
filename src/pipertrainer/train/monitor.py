"""Watching a run: TensorBoard, metrics, and checkpoint housekeeping.

TensorBoard is the primary tool because piper already logs *listenable audio*
there every validation epoch — ``on_validation_epoch_end`` calls
``add_audio`` — and for TTS, listening beats any loss curve.

For a headless box where opening a browser is inconvenient, ``report()`` prints
the same information as text: a metric table read from the event files and a
checkpoint leaderboard read from the filenames Lightning writes.

Housekeeping matters here too. Upstream keeps the top five by ``val_mel`` *and*
the top five by ``val_mos`` *and* ``last.ckpt``, at roughly 0.9 GB each — about
10 GB per run.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import env as env_mod
from .. import profile as profile_mod
from .. import proc, tui
from ..paths import VoicePaths, venv_bin, venv_python
from . import export as export_mod
from . import launch


class MonitorError(RuntimeError):
    pass


@dataclass
class Scalars:
    steps: dict[str, list[tuple[int, float]]]

    def latest(self, tag: str) -> tuple[int, float] | None:
        series = self.steps.get(tag)
        return series[-1] if series else None

    def best(self, tag: str, mode: str = "min") -> tuple[int, float] | None:
        series = self.steps.get(tag)
        if not series:
            return None
        return min(series, key=lambda item: item[1]) if mode == "min" else max(
            series, key=lambda item: item[1]
        )


_SCALAR_PROBE = r"""
import json, sys, glob, os
from collections import defaultdict
try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as exc:
    print(json.dumps({"error": "tensorboard unavailable: %s" % exc}))
    raise SystemExit(0)

root = sys.argv[1]
wanted = set(sys.argv[2:])
series = defaultdict(list)
paths = sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True))
for path in paths:
    try:
        acc = event_accumulator.EventAccumulator(
            path, size_guidance={event_accumulator.SCALARS: 0}
        )
        acc.Reload()
    except Exception:
        continue
    for tag in acc.Tags().get("scalars", []):
        if wanted and tag not in wanted:
            continue
        for event in acc.Scalars(tag):
            series[tag].append([int(event.step), float(event.value)])

for tag in series:
    series[tag].sort()
print(json.dumps({"series": series, "files": len(paths)}))
"""

TRACKED_TAGS = (
    "loss_g", "loss_d", "loss_gen_all", "val_loss", "val_mel", "val_mos",
    "loss_disc_all", "grad_norm",
)


def read_scalars(paths: VoicePaths) -> Scalars:
    import json

    if not paths.lightning_logs.exists():
        return Scalars(steps={})
    result = proc.capture(
        [
            venv_python(),
            "-c",
            _SCALAR_PROBE,
            str(paths.lightning_logs),
            *TRACKED_TAGS,
        ],
        timeout=600,
    )
    for line in reversed(result.lines):
        if line.strip().startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in payload:
                tui.warn(payload["error"])
                return Scalars(steps={})
            return Scalars(
                steps={
                    tag: [(int(s), float(v)) for s, v in values]
                    for tag, values in payload.get("series", {}).items()
                }
            )
    return Scalars(steps={})


# --------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------


def report(profile_name: str | None = None) -> int:
    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    checkpoints = launch.find_checkpoints(paths)
    if not checkpoints and not paths.lightning_logs.exists():
        tui.warn(f"no run yet for {prof.voice.name}. Start one with './run train'.")
        return 1

    tui.heading(f"Run status: {prof.voice.name}")
    total = sum(path.stat().st_size for path in checkpoints)
    tui.table(
        [
            ["checkpoints", str(len(checkpoints))],
            ["disk used", f"{total / 1e9:.2f} GB"],
            ["run directory", str(paths.run_root)],
            ["free space", f"{env_mod.free_gib(paths.run_root):.1f} GiB"],
        ]
    )

    tui.heading("Metrics")
    scalars = read_scalars(paths)
    if not scalars.steps:
        tui.hint("  no scalars logged yet (the first validation epoch writes them)")
    else:
        rows: list[list[str]] = []
        for tag in TRACKED_TAGS:
            latest = scalars.latest(tag)
            if latest is None:
                continue
            step, value = latest
            mode = "max" if tag == "val_mos" else "min"
            best = scalars.best(tag, mode)
            rows.append(
                [
                    tag,
                    f"{value:.4f}",
                    f"step {step}",
                    f"{best[1]:.4f} @ {best[0]}" if best else "-",
                ]
            )
        tui.table(rows, headers=["metric", "latest", "at", "best"])
        if "val_mos" not in scalars.steps and "val_mel" in scalars.steps:
            tui.hint(
                "  val_mos is absent: model.mos_metric is 'none', or UTMOS could "
                "not be fetched from torch.hub. Neither affects training."
            )
        tui.info("")
        tui.hint(
            "  val_mel saturates long before the audio stops improving — that is "
            "why upstream does not early-stop on it. Judge by listening."
        )

    tui.heading("Checkpoints")
    rows = export_mod.describe_checkpoints(paths)
    if rows:
        tui.table(rows[:15], headers=["file", "size", "val_mel", "val_mos"])
        if len(rows) > 15:
            tui.hint(f"  ... and {len(rows) - 15} more")
        best_mel = export_mod.pick_checkpoint(paths, "val_mel")
        tui.info("")
        tui.ok(f"best by val_mel: {best_mel.name}")
    else:
        tui.hint("  none written yet")

    tui.info("")
    tui.bullet("./run monitor            serve TensorBoard (audio samples included)")
    tui.bullet("./run preview            synthesize a sentence from a checkpoint")
    tui.bullet("./run monitor --prune    delete checkpoints to reclaim disk")
    return 0


# --------------------------------------------------------------------------
# TensorBoard
# --------------------------------------------------------------------------


def serve(profile_name: str | None = None, port: int = 6006) -> int:
    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    if not paths.lightning_logs.exists():
        raise MonitorError(
            f"no logs at {paths.lightning_logs}. Start a run with './run train'."
        )

    binary = venv_bin("tensorboard")
    if not binary.exists():
        raise MonitorError(
            "tensorboard is not installed in the venv. It comes with "
            "piper1-gpl[train]; run './run setup --force-step piper_install'."
        )

    import socket

    host = socket.gethostname()
    tui.heading("TensorBoard")
    tui.ok(f"serving {paths.lightning_logs} on port {port}")
    tui.info("")
    tui.info("If you are on the machine:")
    tui.info(f"  http://localhost:{port}")
    tui.info("")
    tui.info("If you are on another machine, tunnel first:")
    tui.info(f"  ssh -L {port}:localhost:{port} {host}")
    tui.info("")
    tui.hint(
        "  Open the AUDIO tab — piper logs synthesized test utterances every "
        "validation epoch, which is the most useful signal there is."
    )
    tui.hint("  Ctrl-C to stop.")
    tui.info("")

    result = proc.run(
        [
            binary,
            "--logdir",
            str(paths.lightning_logs),
            "--port",
            str(port),
            "--bind_all",
        ],
        check=False,
    )
    return 0 if result.interrupted else result.returncode


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------


def prune(profile_name: str | None = None) -> int:
    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    checkpoints = launch.find_checkpoints(paths)
    if not checkpoints:
        tui.info("nothing to prune")
        return 0

    total = sum(path.stat().st_size for path in checkpoints)
    tui.heading("Prune checkpoints")
    tui.table(
        export_mod.describe_checkpoints(paths),
        headers=["file", "size", "val_mel", "val_mos"],
    )
    tui.info("")
    tui.info(f"{len(checkpoints)} checkpoints using {total / 1e9:.2f} GB")

    keep_mel = export_mod.pick_checkpoint(paths, "val_mel")
    last = launch.find_last(paths)
    protected = {keep_mel}
    if last:
        protected.add(last)
    try:
        protected.add(export_mod.pick_checkpoint(paths, "val_mos"))
    except Exception:  # noqa: BLE001 - val_mos may never have been logged
        pass

    tui.info("")
    tui.info("These would be kept:")
    for path in sorted(protected):
        tui.bullet(path.name)

    doomed = [path for path in checkpoints if path not in protected]
    if not doomed:
        tui.ok("nothing to delete — every checkpoint is worth keeping")
        return 0

    freed = sum(path.stat().st_size for path in doomed)
    tui.info("")
    tui.warn(f"{len(doomed)} checkpoint(s) would be deleted, freeing {freed / 1e9:.2f} GB")
    if not tui.confirm("delete them?", default=False):
        tui.info("nothing deleted")
        return 0

    for path in doomed:
        path.unlink()
    tui.ok(f"deleted {len(doomed)} checkpoint(s), freed {freed / 1e9:.2f} GB")
    tui.hint(
        "  To keep fewer from the start, lower trainer.checkpoint_save_top_k in "
        "the profile."
    )
    return 0


# --------------------------------------------------------------------------
# Interactive
# --------------------------------------------------------------------------


def interactive(profile_name: str | None = None) -> int:
    while True:
        try:
            choice = tui.menu(
                "Monitor",
                [
                    ("report", "Show metrics and checkpoints as text"),
                    ("serve", "Serve TensorBoard (includes audio samples)"),
                    ("tail", "Follow the training log"),
                    ("prune", "Delete checkpoints to reclaim disk"),
                ],
                allow_back=True,
            )
        except tui.Back:
            return 0

        if choice == "report":
            report(profile_name)
        elif choice == "serve":
            serve(profile_name)
        elif choice == "prune":
            prune(profile_name)
        elif choice == "tail":
            _tail(profile_name)


def _tail(profile_name: str | None) -> None:
    prof, _ = profile_mod.resolve_active(profile_name)
    paths = VoicePaths(prof.voice.name)
    logs = sorted(paths.logs.glob("run-*/train.log"))
    if not logs:
        tui.warn(f"no training log under {paths.logs}")
        return
    newest = logs[-1]
    tui.ok(f"following {newest} — Ctrl-C to stop")
    if shutil.which("tail"):
        proc.run(["tail", "-n", "40", "-f", str(newest)], check=False)
    else:
        tui.info(newest.read_text(encoding="utf-8", errors="replace")[-4000:])
