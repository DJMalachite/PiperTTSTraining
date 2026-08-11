"""The export flow: pick a checkpoint, produce a usable voice, prove it works."""

from __future__ import annotations

from pathlib import Path

from .. import profile as profile_mod
from .. import tui
from ..paths import VoicePaths, in_venv
from ..train import export as export_mod
from ..train import launch
from . import common

EXPORT_PREFIXES = ("export",)


def run(
    profile_name: str | None = None,
    checkpoint: str | None = None,
    best: str = "val_mel",
    verify: bool = True,
    interactive: bool = True,
) -> int:
    if not in_venv():
        tui.error("not set up yet — run './run setup' first")
        return 1

    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    candidates = launch.find_checkpoints(paths)
    if not candidates:
        tui.error(
            f"no checkpoints under {paths.lightning_logs}. Train first with "
            f"'./run train'."
        )
        return 1

    tui.heading(f"Export {prof.voice.name}")

    target: Path
    if checkpoint:
        target = Path(checkpoint)
    elif interactive:
        target = _choose(paths, best)
    else:
        target = export_mod.pick_checkpoint(paths, best)

    if interactive:
        changed = common.walk(prof, EXPORT_PREFIXES, title="Export settings")
        common.save_and_report(prof, changed)

    stem = export_mod.voice_filename(prof)
    destination = export_mod.output_dir(prof, paths)
    tui.info("")
    tui.table(
        [
            ["checkpoint", target.name],
            ["output", str(destination)],
            ["files", f"{stem}.onnx  +  {stem}.onnx.json"],
            ["verify", "yes" if verify else "no"],
        ]
    )
    tui.hint(
        "  The .onnx.json comes from training (piper writes it to "
        "--data.config_path), not from the export step. We copy and correct it."
    )

    if interactive and not tui.confirm("export now?", default=True):
        return 0

    tui.heading("Exporting")
    result = export_mod.run(
        prof, checkpoint=str(target), prefer=best, do_verify=verify
    )

    if verify and not result.verified:
        tui.error("the exported voice failed verification")
        return 1
    return 0


def _choose(paths: VoicePaths, best: str) -> Path:
    rows = export_mod.describe_checkpoints(paths)
    tui.table(rows[:15], headers=["file", "size", "val_mel", "val_mos"])
    if len(rows) > 15:
        tui.hint(f"  ... and {len(rows) - 15} more")
    tui.info("")
    tui.hint(
        "  val_mel tracks reconstruction accuracy and saturates early; val_mos "
        "tracks perceptual quality and its winner often sounds better. Neither "
        "beats listening — './run preview' synthesizes from any checkpoint."
    )
    tui.info("")

    options = [
        ("val_mel", "Best by val_mel (recommended)"),
        ("val_mos", "Best by val_mos (perceptual)"),
        ("last", "Most recent (last.ckpt)"),
        ("pick", "Choose from the list"),
    ]
    choice = tui.menu("Which checkpoint?", options, allow_back=False)
    if choice == "pick":
        candidates = launch.find_checkpoints(paths)
        names = [(path.name, path.name) for path in reversed(candidates[-15:])]
        selected = tui.menu("Checkpoint", names, allow_back=False)
        return next(path for path in candidates if path.name == selected)
    return export_mod.pick_checkpoint(paths, choice)
