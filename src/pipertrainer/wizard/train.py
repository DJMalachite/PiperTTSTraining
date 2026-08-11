"""The training flow: settings, then a confirmation screen, then go.

The confirmation screen exists because the alternative is discovering an hour in
that the batch size was wrong. Everything that can be checked without the GPU is
checked before the subprocess starts, and the summary shows the decisions that
actually matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .. import checkpoints as checkpoints_mod
from .. import env as env_mod
from .. import profile as profile_mod
from .. import tui
from ..paths import VoicePaths, in_venv
from ..train import argmap, launch, presets
from . import common

TRAIN_PREFIXES = ("data", "model", "trainer")


def run(
    profile_name: str | None = None,
    dry_run: bool = False,
    interactive: bool = True,
    offline: bool = False,
    extra_argv: Sequence[str] = (),
) -> int:
    if not in_venv():
        tui.error("not set up yet — run './run setup' first")
        return 1

    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    if not paths.metadata_csv.exists():
        tui.error(
            f"no dataset for {prof.voice.name}. Run './run dataset' first."
        )
        return 1

    offline = offline or prof.runtime.offline
    tui.heading(f"Training {prof.voice.name}")

    usable, shortest, missing = launch.dataset_facts(paths)
    split = argmap.split_sizes(
        usable, float(prof.data.validation_split), int(prof.data.num_test_examples)
    )
    tui.table(
        [
            ["utterances", str(usable)],
            ["split", f"{split.train} train / {split.val} val / {split.test} test"],
            ["max batch size", str(split.max_batch_size)],
            ["shortest clip", f"{shortest:.2f} s" if shortest else "?"],
        ]
    )

    if interactive:
        _hardware(prof, split)
        _finetune(prof, offline)

        changed = common.walk(
            prof,
            ("voice",),
            skip=("voice.quality",),
            title="Voice",
        )
        changed += _quality(prof)
        changed += common.walk(prof, TRAIN_PREFIXES, title="Training settings")
        changed += common.offer_advanced(prof, TRAIN_PREFIXES, "training")
        changed += common.walk(prof, ("runtime",), title="Runtime")
        common.save_and_report(prof, changed)
    else:
        profile_mod.save(prof)

    tui.heading("Preflight")
    try:
        prepared = launch.prepare(prof, offline=offline, extra_argv=extra_argv)
    except argmap.ArgMapError as exc:
        tui.error(str(exc))
        return 2

    common.report_findings(prepared.plan.warnings, prepared.plan.notes)

    tui.heading("Summary")
    tui.table(argmap.summarise(prepared.plan))
    tui.info("")
    tui.hint(f"  config: {prepared.config_path}")

    launch.check_cache(prepared, interactive=interactive)
    launch.print_config_gate(prepared)

    if dry_run:
        tui.info("")
        tui.ok("dry run: every check passed and lightning.yaml is written")
        tui.hint(f"  start for real with: ./run train --profile {prof.slug}")
        return 0

    if interactive and not tui.confirm("start training?", default=True):
        return 0

    return launch.start(prepared, extra_argv=extra_argv)


# --------------------------------------------------------------------------
# Sub-flows
# --------------------------------------------------------------------------


def _hardware(prof: profile_mod.Profile, split) -> None:
    """Offer hardware-derived defaults, stating the assumptions out loud."""
    info = env_mod.SetupState.load().info
    if not info.ok:
        info = env_mod.verify_torch()

    tui.heading("Hardware")
    tui.info(f"  {info.summary()}")

    if not info.usable_gpu:
        tui.warn(
            "no usable GPU — training will run on CPU, which is slower by "
            "orders of magnitude. Run './run doctor' to see why."
        )
        if tui.confirm("set trainer.accelerator to cpu explicitly?", default=True):
            prof.trainer.accelerator = "cpu"
        return

    recommended = env_mod.recommended_batch_size(info)
    capped = min(recommended, split.max_batch_size)
    tui.info(
        f"  {info.total_memory_gib:.1f} GiB of GPU memory suggests batch_size "
        f"{recommended}"
        + (
            f", capped to {capped} by the training split"
            if capped != recommended
            else ""
        )
    )
    tui.hint(
        "  That is a table lookup, not a measurement. If it OOMs, halve it — "
        "the failure message will list the whole ladder."
    )

    if env_mod.low_vram(info) and not prof.runtime.low_vram:
        tui.info("")
        if tui.confirm(
            f"apply the small-GPU preset ({', '.join(presets.LOW_VRAM)})?",
            default=True,
        ):
            for line in argmap.apply_low_vram(prof):
                tui.bullet(line)
            prof.runtime.low_vram = True

    if int(prof.data.batch_size) != capped:
        if tui.confirm(f"set data.batch_size to {capped}?", default=True):
            prof.data.batch_size = capped

    if info.hsa_override and not prof.runtime.env.get("HSA_OVERRIDE_GFX_VERSION"):
        prof.runtime.env["HSA_OVERRIDE_GFX_VERSION"] = info.hsa_override
        tui.ok(f"recorded HSA_OVERRIDE_GFX_VERSION={info.hsa_override} in the profile")


def _quality(prof: profile_mod.Profile) -> list[str]:
    tui.heading("Quality preset")
    for name in presets.preset_names():
        marker = "*" if name == prof.voice.quality else " "
        tui.info(f" {marker} {name}: {presets.PRESET_NOTES[name]}")
    tui.info("")
    before = prof.voice.quality
    prof.voice.quality = tui.ask_choice(
        "quality", presets.preset_names(), prof.voice.quality,
        help_text="Changes the model architecture. Only 'medium' matches the "
        "pretrained checkpoints.",
    )
    changed = ["voice.quality"] if prof.voice.quality != before else []

    expected = presets.PRESET_SAMPLE_RATE[prof.voice.quality]
    if int(prof.audio.sample_rate) != expected:
        tui.warn(
            f"'{prof.voice.quality}' is designed for {expected} Hz but your "
            f"dataset is {prof.audio.sample_rate} Hz."
        )
        tui.hint(
            "  Changing audio.sample_rate now means rebuilding the dataset: the "
            "clips are already written at the old rate."
        )
    return changed


def _finetune(prof: profile_mod.Profile, offline: bool) -> None:
    tui.heading("Starting point")
    current = prof.finetune.checkpoint
    if current and Path(current).exists():
        size = Path(current).stat().st_size / 1e9
        tui.ok(f"{prof.finetune.mode}: {Path(current).name} ({size:.2f} GB)")
    elif current:
        tui.warn(f"finetune.checkpoint does not exist: {current}")
    else:
        tui.info("  no checkpoint selected — training would start from scratch")
        tui.hint(
            "  Fine-tuning is dramatically faster and upstream recommends it "
            "even across languages. A 'medium' checkpoint in any language beats "
            "starting cold."
        )

    options = [
        ("keep", "Keep the current setting"),
        ("browse", "Browse HuggingFace for a checkpoint"),
        ("local", "Use a local .ckpt file"),
        ("scratch", "Train from scratch"),
    ]
    if offline:
        options = [option for option in options if option[0] != "browse"]

    choice = tui.menu("Fine-tuning", options, allow_back=False)
    if choice == "keep":
        pass
    elif choice == "browse":
        checkpoints_mod.browse(profile_name=prof.voice.name, offline=offline)
        reloaded, _ = profile_mod.load_by_name(prof.voice.name)
        prof.finetune.mode = reloaded.finetune.mode
        prof.finetune.checkpoint = reloaded.finetune.checkpoint
    elif choice == "local":
        prof.finetune.checkpoint = tui.ask_path(
            "path to .ckpt", prof.finetune.checkpoint, must_exist=True,
            allow_back=False,
        )
        if prof.finetune.mode == "none":
            prof.finetune.mode = "ckpt_path"
    elif choice == "scratch":
        prof.finetune.mode = "none"
        prof.finetune.checkpoint = ""

    if prof.finetune.mode != "none":
        tui.info("")
        prof.finetune.mode = tui.ask_choice(
            "mode",
            [m for m in profile_mod.FINETUNE_MODES if m != "none"],
            prof.finetune.mode,
            help_text=(
                "ckpt_path: strict resume, needs a matching architecture.\n"
                "warmstart: copies matching-shape parameters, fresh optimizer. "
                "Required when model.use_mrd is on.\n"
                "vocoder_warmstart: copies the vocoder only, so the phoneme "
                "count may differ."
            ),
        )
