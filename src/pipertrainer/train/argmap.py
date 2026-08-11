"""Profile to ``piper.train fit`` translation.

This is the highest-risk module in the repo and deliberately the most testable:
it is a pure function from a profile to a config dict plus an argv list, with no
I/O. Everything it knows about upstream's argument surface is encoded as data
below, so a golden test fails loudly when a pin bump changes the contract.

Three upstream behaviours drive the design.

**Link arguments.** ``VitsLightningCLI.add_arguments_to_parser`` declares eight
``parser.link_arguments`` pairs. jsonargparse computes the target from the
source, and setting the target yourself is a hard error. So the emitted config
must put ``sample_rate`` on ``model`` and ``batch_size`` on ``data``, never the
reverse. ``FORBIDDEN`` below is that rule, enforced before any subprocess runs.

**Manual optimization.** ``VitsModel`` sets ``automatic_optimization = False``,
which makes Lightning reject ``trainer.gradient_clip_val``. Separately,
``model.grad_clip`` is accepted but never applied — ``clip_grad_value_`` exists
in ``vits/commons.py`` and is never called — so offering it would be a lie.

**Config files beat flags.** LightningCLI accepts ``--config file.yaml`` and
``--print_config``. Emitting YAML rather than sixty flags means nested
architecture values go out as native sequences (upstream's ``ast.literal_eval``
calls are all guarded by ``isinstance(..., str)``, so both forms work), and
``--print_config`` gives a free type-check before committing to a real run.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import profile as profile_mod
from ..paths import VoicePaths
from . import presets


class ArgMapError(ValueError):
    """A configuration that upstream would reject, or that would waste a run."""


# --------------------------------------------------------------------------
# Upstream argument surface, as data
# --------------------------------------------------------------------------

# parser.link_arguments(source, target) — the target must never be set by us.
LINKS: tuple[tuple[str, str], ...] = (
    ("data.batch_size", "model.batch_size"),
    ("data.num_symbols", "model.num_symbols"),
    ("model.num_speakers", "data.num_speakers"),
    ("model.sample_rate", "data.sample_rate"),
    ("model.filter_length", "data.filter_length"),
    ("model.hop_length", "data.hop_length"),
    ("model.win_length", "data.win_length"),
    ("model.segment_size", "data.segment_size"),
)

FORBIDDEN: dict[str, str] = {
    target: (
        f"{target} is computed from {source} by a jsonargparse link; "
        f"set {source} instead"
    )
    for source, target in LINKS
}

BLOCKED: dict[str, str] = {
    "trainer.gradient_clip_val": (
        "VitsModel uses manual optimization (automatic_optimization = False), "
        "and Lightning raises MisconfigurationException when gradient clipping "
        "is configured on the trainer in that mode"
    ),
    "trainer.gradient_clip_algorithm": (
        "see trainer.gradient_clip_val — gradient clipping cannot be configured "
        "on the trainer under manual optimization"
    ),
    "model.grad_clip": (
        "accepted by VitsModel.__init__ but never applied in piper1-gpl v1.6.0: "
        "clip_grad_value_ is defined in vits/commons.py and never called, and "
        "training_step does zero_grad/manual_backward/step with no clipping"
    ),
}

NOOP_WARNINGS: dict[str, str] = {
    "trainer.accumulate_grad_batches": (
        "gradient accumulation is almost certainly inert here: Lightning does "
        "not apply it under manual optimization. Raise data.batch_size instead."
    ),
}

# Keys that belong on `data` even though they look model-ish.
DATA_KEYS = (
    "voice_name",
    "csv_path",
    "audio_dir",
    "cache_dir",
    "config_path",
    "espeak_voice",
    "batch_size",
    "num_symbols",
    "validation_split",
    "num_test_examples",
    "num_workers",
    "trim_silence",
    "keep_seconds_before_silence",
    "keep_seconds_after_silence",
    "phoneme_type",
    "dataset_type",
    "phonemes_path",
    "alignments_dir",
    "vowel_clusters",
)

# Upstream's two default ModelCheckpoint callbacks, replicated so that changing
# save_top_k does not silently drop them: setting trainer.callbacks replaces
# trainer_defaults wholesale.
_CHECKPOINT_CLASS = "lightning.pytorch.callbacks.ModelCheckpoint"


def checkpoint_callbacks(save_top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "class_path": _CHECKPOINT_CLASS,
            "init_args": {
                "monitor": "val_mel",
                "mode": "min",
                "save_top_k": save_top_k,
                "save_last": True,
                "filename": "epoch={epoch}-val_mel={val_mel:.4f}",
                "auto_insert_metric_name": False,
            },
        },
        {
            "class_path": _CHECKPOINT_CLASS,
            "init_args": {
                "monitor": "val_mos",
                "mode": "max",
                "save_top_k": save_top_k,
                "save_last": False,
                "filename": "epoch={epoch}-val_mos={val_mos:.4f}",
                "auto_insert_metric_name": False,
            },
        },
    ]


DEFAULT_SAVE_TOP_K = 5


# --------------------------------------------------------------------------
# Dataset split arithmetic
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    total: int
    train: int
    val: int
    test: int

    @property
    def max_batch_size(self) -> int:
        return max(1, self.train)


def split_sizes(
    total: int, validation_split: float, num_test_examples: int
) -> Split:
    """Replicate ``VitsDataModule.setup``'s random_split arithmetic exactly.

    Getting this wrong is not academic: ``train_dataloader`` uses
    ``drop_last=True``, so a train split smaller than the batch size yields
    *zero* batches and Lightning fails with a message that says nothing about
    the real cause.
    """
    valid = int(total * validation_split)
    test = min(num_test_examples, max(0, total - valid - 1))
    train = total - valid - test
    return Split(total=total, train=train, val=valid, test=test)


def check_dataset_math(
    total: int,
    validation_split: float,
    num_test_examples: int,
    batch_size: int,
) -> Split:
    if total <= 0:
        raise ArgMapError(
            "the dataset has no usable utterances — run './run dataset' first "
            "and check data/<voice>/report.md"
        )
    split = split_sizes(total, validation_split, num_test_examples)
    if split.train < 1:
        raise ArgMapError(
            f"only {total} utterances: after holding back "
            f"{split.val} for validation and {split.test} for test samples, "
            f"nothing is left to train on. Lower data.validation_split "
            f"(currently {validation_split}) or data.num_test_examples "
            f"(currently {num_test_examples}), or record more audio."
        )
    if batch_size > split.train:
        raise ArgMapError(
            f"data.batch_size {batch_size} exceeds the training split of "
            f"{split.train} utterances. The dataloader drops the last partial "
            f"batch, so this would produce zero batches per epoch. Use "
            f"{split.max_batch_size} or lower."
        )
    return split


# --------------------------------------------------------------------------
# Architecture invariants
# --------------------------------------------------------------------------


def check_architecture(model: dict[str, Any]) -> list[str]:
    """Validate the model block. Returns warnings; raises on hard errors."""
    warnings: list[str] = []
    rates = list(model["upsample_rates"])
    kernels = list(model["upsample_kernel_sizes"])
    hop = int(model["hop_length"])

    product = reduce(lambda a, b: a * b, rates, 1)
    if product != hop:
        raise ArgMapError(
            f"upsample_rates {tuple(rates)} multiply to {product}, but "
            f"hop_length is {hop}. Upstream raises 'Upsample rates do not match "
            f"hop length'. Either set hop_length to {product}, or choose rates "
            f"whose product is {hop} (medium uses (8,8,4); high uses (8,8,2,2))."
        )

    if len(rates) != len(kernels):
        raise ArgMapError(
            f"upsample_rates has {len(rates)} entries but "
            f"upsample_kernel_sizes has {len(kernels)}; they index the same "
            f"upsample stages and must be the same length"
        )
    for index, (rate, kernel) in enumerate(zip(rates, kernels)):
        if kernel < rate:
            raise ArgMapError(
                f"upsample stage {index}: kernel size {kernel} is smaller than "
                f"the stride {rate}, which leaves gaps in the output"
            )
        if (kernel - rate) % 2 != 0:
            raise ArgMapError(
                f"upsample stage {index}: (kernel {kernel} - stride {rate}) is "
                f"odd, so the transposed convolution cannot be padded "
                f"symmetrically. Use an even difference, e.g. kernel "
                f"{rate * 2} for stride {rate}."
            )

    if len(model["resblock_kernel_sizes"]) != len(model["resblock_dilation_sizes"]):
        raise ArgMapError(
            f"resblock_kernel_sizes has "
            f"{len(model['resblock_kernel_sizes'])} entries but "
            f"resblock_dilation_sizes has "
            f"{len(model['resblock_dilation_sizes'])}; each kernel size needs "
            f"its own dilation list"
        )

    segment = int(model["segment_size"])
    if segment % hop != 0:
        raise ArgMapError(
            f"segment_size {segment} is not a multiple of hop_length {hop}; "
            f"the model divides one by the other. Nearest valid values: "
            f"{(segment // hop) * hop} or {((segment // hop) + 1) * hop}."
        )

    if str(model.get("resblock")) not in ("1", "2"):
        raise ArgMapError(
            f"resblock must be the string '1' or '2', got "
            f"{model.get('resblock')!r}"
        )

    if int(model.get("num_speakers", 1)) > 1:
        warnings.append(
            "num_speakers is above 1, which requires a wav|speaker|text "
            "metadata.csv. This pipeline produces single-speaker datasets."
        )
    return warnings


def check_clip_length(
    model: dict[str, Any], min_clip_seconds: float | None
) -> list[str]:
    """Warn when clips are shorter than the discriminator's window.

    ``UtteranceCollate`` pads every batch up to at least ``segment_size``, so a
    clip below that length is padded with silence the model then learns.
    """
    if min_clip_seconds is None:
        return []
    segment = int(model["segment_size"])
    rate = int(model["sample_rate"])
    threshold = segment / rate
    if min_clip_seconds < threshold:
        return [
            f"the shortest clip is {min_clip_seconds:.2f} s but segment_size "
            f"{segment} at {rate} Hz is {threshold:.2f} s. Shorter clips get "
            f"zero-padded, teaching the model to emit silence. Raise "
            f"dataset.min_seconds and rebuild, or lower model.segment_size."
        ]
    return []


# --------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------

# Checkpoint hyper_parameters that must match for a strict --ckpt_path load.
ARCH_KEYS = (
    "sample_rate",
    "resblock",
    "resblock_kernel_sizes",
    "upsample_rates",
    "upsample_initial_channel",
    "upsample_kernel_sizes",
    "num_symbols",
    "num_speakers",
)


def _normalise(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(item) for item in value)
    if isinstance(value, str):
        return value
    return value


def compare_architecture(
    arch: dict[str, Any], ckpt_hparams: dict[str, Any]
) -> list[str]:
    """Keys where a checkpoint disagrees with the configured architecture.

    ``arch`` must be the *merged* view — the model block plus ``num_symbols``,
    which lives on the data side because of the
    ``data.num_symbols -> model.num_symbols`` link. Keys absent from either side
    are skipped rather than compared against ``None``.
    """
    mismatches: list[str] = []
    for key in ARCH_KEYS:
        if key not in ckpt_hparams or key not in arch:
            continue
        want = _normalise(arch.get(key))
        have = _normalise(ckpt_hparams.get(key))
        if want != have:
            mismatches.append(f"{key}: checkpoint has {have!r}, profile wants {want!r}")
    return mismatches


def check_finetune(
    mode: str,
    checkpoint: str,
    arch: dict[str, Any],
    *,
    ckpt_hparams: dict[str, Any] | None = None,
    offline: bool = False,
) -> list[str]:
    """Validate the fine-tuning choice. ``arch`` is the merged model view."""
    model = arch
    warnings: list[str] = []
    if mode == "none":
        if not model.get("use_mrd"):
            warnings.append(
                "training from scratch. Fine-tuning from a pretrained "
                "checkpoint is dramatically faster and upstream recommends it "
                "even across languages — see './run checkpoints'."
            )
        return warnings

    if not checkpoint:
        raise ArgMapError(
            f"finetune.mode is {mode!r} but finetune.checkpoint is empty. "
            f"Pick one with './run checkpoints', or set mode to 'none'."
        )

    if model.get("use_mrd") and mode == "ckpt_path":
        raise ArgMapError(
            "model.use_mrd adds the multi-resolution discriminator, whose "
            "parameters are absent from every existing checkpoint. --ckpt_path "
            "does a strict load and will fail on the extra keys. Use "
            "finetune.mode 'warmstart' instead, which copies matching-shape "
            "parameters and starts a fresh optimizer."
        )

    if ckpt_hparams and mode == "ckpt_path":
        mismatches = compare_architecture(arch, ckpt_hparams)
        if mismatches:
            phoneme_mismatch = any(m.startswith("num_symbols") for m in mismatches)
            suggestion = (
                "vocoder_warmstart" if phoneme_mismatch else "warmstart"
            )
            raise ArgMapError(
                "the checkpoint's architecture does not match this profile, and "
                "--ckpt_path loads strictly:\n  "
                + "\n  ".join(mismatches)
                + f"\nUse finetune.mode '{suggestion}' instead, or set "
                f"voice.quality to match the checkpoint (pretrained "
                f"checkpoints are all 'medium')."
            )
    return warnings


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass
class Plan:
    """Everything needed to launch, plus what the user should know first."""

    config: dict[str, Any]
    argv: list[str]
    ckpt_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    split: Split | None = None

    @property
    def model(self) -> dict[str, Any]:
        return self.config["model"]

    @property
    def data(self) -> dict[str, Any]:
        return self.config["data"]

    @property
    def trainer(self) -> dict[str, Any]:
        return self.config["trainer"]


def _reject_forbidden(keys: Iterable[str], where: str) -> None:
    for key in keys:
        if key in FORBIDDEN:
            raise ArgMapError(f"{where}: {FORBIDDEN[key]}")
        if key in BLOCKED:
            raise ArgMapError(f"{where}: {key} is not usable — {BLOCKED[key]}")


def check_extra_argv(extra: Sequence[str]) -> list[str]:
    """Validate raw argv the user wants appended, e.g. --trainer.foo bar."""
    warnings: list[str] = []
    for token in extra:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in FORBIDDEN:
            raise ArgMapError(f"--{key} cannot be used: {FORBIDDEN[key]}")
        if key in BLOCKED:
            raise ArgMapError(f"--{key} cannot be used: {BLOCKED[key]}")
        if key in NOOP_WARNINGS:
            warnings.append(f"--{key}: {NOOP_WARNINGS[key]}")
    return warnings


def build(
    prof: profile_mod.Profile,
    *,
    paths: VoicePaths | None = None,
    offline: bool = False,
    total_utterances: int | None = None,
    min_clip_seconds: float | None = None,
    ckpt_hparams: dict[str, Any] | None = None,
    extra_argv: Sequence[str] = (),
) -> Plan:
    """Translate a profile into a ``lightning.yaml`` config plus argv.

    Pure: no filesystem access beyond building path strings. The checks that
    need to read the disk (cache completeness, espeak validation) live in
    ``launch.py`` and call back into the helpers here.
    """
    paths = paths or VoicePaths(prof.voice.name)
    warnings: list[str] = []
    notes: list[str] = []

    # ---- model block -----------------------------------------------------
    model: dict[str, Any] = presets.preset_for(prof.voice.quality)
    model["sample_rate"] = int(prof.audio.sample_rate)
    model["segment_size"] = int(prof.model.segment_size)
    model["num_speakers"] = int(prof.model.num_speakers)
    model["learning_rate"] = float(prof.model.learning_rate)
    model["learning_rate_d"] = float(prof.model.learning_rate_d)
    model["lr_decay"] = float(prof.model.lr_decay)
    model["lr_decay_d"] = float(prof.model.lr_decay_d)
    model["warmup_epochs"] = int(prof.model.warmup_epochs)
    model["c_mel"] = int(prof.model.c_mel)
    model["c_kl"] = float(prof.model.c_kl)
    model["use_sdp"] = bool(prof.model.use_sdp)
    model["use_mrd"] = bool(prof.model.use_mrd)

    mos_metric = prof.model.mos_metric
    if offline and mos_metric not in ("none", ""):
        mos_metric = "none"
        notes.append(
            "offline: model.mos_metric set to 'none'. UTMOS would otherwise be "
            "fetched from torch.hub at first validation. Upstream degrades "
            "gracefully if that fails, so this only saves a timeout and a "
            "warning — it is not a crash fix."
        )
    model["mos_metric"] = mos_metric or "none"

    # Raw passthrough last, so it can override anything above.
    extra_model = dict(prof.model.extra or {})
    _reject_forbidden([f"model.{k}" for k in extra_model], "model.extra")
    if extra_model:
        notes.append(
            "model.extra overrides: " + ", ".join(sorted(extra_model))
        )
    model.update(extra_model)

    # ---- invariants on the model block ----------------------------------
    warnings += check_architecture(model)
    warnings += check_clip_length(model, min_clip_seconds)

    expected_rate = presets.PRESET_SAMPLE_RATE[prof.voice.quality]
    if model["sample_rate"] != expected_rate:
        warnings.append(
            f"voice.quality '{prof.voice.quality}' is designed for "
            f"{expected_rate} Hz but audio.sample_rate is "
            f"{model['sample_rate']} Hz. This trains, but no pretrained "
            f"checkpoint will match."
        )

    # ---- data block ------------------------------------------------------
    data: dict[str, Any] = {
        "voice_name": prof.voice.name,
        "csv_path": str(paths.metadata_csv),
        "audio_dir": str(paths.wavs),
        "cache_dir": str(paths.cache),
        "config_path": str(paths.piper_config_json),
        "espeak_voice": prof.voice.espeak_voice,
        "batch_size": int(prof.data.batch_size),
        "num_symbols": int(prof.data.num_symbols),
        "validation_split": float(prof.data.validation_split),
        "num_test_examples": int(prof.data.num_test_examples),
        "num_workers": int(prof.data.num_workers),
        "trim_silence": bool(prof.data.trim_silence),
        "keep_seconds_before_silence": float(
            prof.data.keep_seconds_before_silence
        ),
        "keep_seconds_after_silence": float(prof.data.keep_seconds_after_silence),
        "phoneme_type": prof.data.phoneme_type,
        "dataset_type": prof.data.dataset_type,
    }
    if prof.data.phonemes_path:
        data["phonemes_path"] = str(prof.data.phonemes_path)

    # Belt and braces: nothing on the data side may be a link target.
    _reject_forbidden([f"data.{key}" for key in data], "generated data block")
    _reject_forbidden([f"model.{key}" for key in model], "generated model block")

    # ---- trainer block ---------------------------------------------------
    trainer: dict[str, Any] = {
        "accelerator": prof.trainer.accelerator,
        "devices": _devices(prof.trainer.devices),
        "precision": prof.trainer.precision,
        "max_epochs": int(prof.trainer.max_epochs),
        "check_val_every_n_epoch": int(prof.trainer.check_val_every_n_epoch),
        "log_every_n_steps": int(prof.trainer.log_every_n_steps),
        "enable_progress_bar": bool(prof.trainer.enable_progress_bar),
        "default_root_dir": str(paths.run_root),
    }
    if int(prof.trainer.max_steps) > 0:
        trainer["max_steps"] = int(prof.trainer.max_steps)
    if int(prof.trainer.accumulate_grad_batches) > 1:
        trainer["accumulate_grad_batches"] = int(prof.trainer.accumulate_grad_batches)
        warnings.append(NOOP_WARNINGS["trainer.accumulate_grad_batches"])
    if int(prof.trainer.checkpoint_save_top_k) != DEFAULT_SAVE_TOP_K:
        trainer["callbacks"] = checkpoint_callbacks(
            int(prof.trainer.checkpoint_save_top_k)
        )
        notes.append(
            f"keeping {prof.trainer.checkpoint_save_top_k} checkpoints per "
            f"metric instead of upstream's {DEFAULT_SAVE_TOP_K}"
        )

    if prof.trainer.precision != "32-true":
        warnings.append(
            f"trainer.precision is {prof.trainer.precision!r}. piper trains a "
            f"GAN with manual optimization, the least-tested combination for "
            f"mixed precision; watch loss_g for divergence. 32-true is the "
            f"safe choice."
        )

    if int(prof.trainer.max_epochs) == -1:
        notes.append(
            "max_epochs is -1 (upstream's default): training runs until you "
            "stop it. That is normal for VITS — mel loss saturates long before "
            "the audio stops improving. Listen to the TensorBoard samples."
        )

    # ---- fine-tuning -----------------------------------------------------
    mode = prof.finetune.mode
    checkpoint = str(prof.finetune.checkpoint or "")
    # num_symbols is emitted on the data side (it is the source of the
    # data.num_symbols -> model.num_symbols link), so the architecture
    # comparison needs a merged view or it would compare against nothing.
    arch = {**model, "num_symbols": data["num_symbols"]}
    warnings += check_finetune(
        mode, checkpoint, arch, ckpt_hparams=ckpt_hparams, offline=offline
    )

    ckpt_path: str | None = None
    if mode == "ckpt_path" and checkpoint:
        ckpt_path = checkpoint
    elif mode == "warmstart" and checkpoint:
        model["warmstart_ckpt"] = checkpoint
        notes.append(
            "warmstart: copies every matching-shape parameter and starts a "
            "fresh optimizer, so early loss values will look worse than a "
            "--ckpt_path resume."
        )
    elif mode == "vocoder_warmstart" and checkpoint:
        model["vocoder_warmstart_ckpt"] = checkpoint
        notes.append(
            "vocoder_warmstart: copies the vocoder but not the phoneme "
            "embedding, so a different phoneme inventory is fine."
        )

    # ---- dataset split arithmetic ---------------------------------------
    split: Split | None = None
    if total_utterances is not None:
        split = check_dataset_math(
            total_utterances,
            float(prof.data.validation_split),
            int(prof.data.num_test_examples),
            int(prof.data.batch_size),
        )
        notes.append(
            f"{split.total} utterances: {split.train} train, {split.val} "
            f"validation, {split.test} test (max batch size {split.max_batch_size})"
        )

    # ---- offline ---------------------------------------------------------
    if offline and checkpoint and not Path(checkpoint).exists():
        raise ArgMapError(
            f"offline mode is on but the checkpoint does not exist locally: "
            f"{checkpoint}"
        )

    warnings += check_extra_argv(extra_argv)

    config = {
        "seed_everything": 1234,
        "trainer": trainer,
        "model": model,
        "data": data,
    }

    return Plan(
        config=config,
        argv=to_argv(config, ckpt_path=ckpt_path, extra=extra_argv),
        ckpt_path=ckpt_path,
        warnings=warnings,
        notes=notes,
        split=split,
    )


def _devices(value: str) -> Any:
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text or "auto"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def to_argv(
    config: dict[str, Any],
    *,
    ckpt_path: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """The flag-by-flag equivalent of ``config``, for the run log.

    We launch with ``--config``; this exists so the log records something a
    human can paste into a shell to reproduce the run by hand.
    """
    argv: list[str] = ["fit"]
    for section in ("trainer", "model", "data"):
        for key, value in config.get(section, {}).items():
            if key == "callbacks":
                argv.append(f"--{section}.{key} <see lightning.yaml>")
                continue
            argv.extend([f"--{section}.{key}", _render(value)])
    if "seed_everything" in config:
        argv.extend(["--seed_everything", _render(config["seed_everything"])])
    if ckpt_path:
        argv.extend(["--ckpt_path", str(ckpt_path)])
    argv.extend(extra)
    return argv


def _render(value: Any) -> str:
    """Render one value as it would be typed in a shell."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        # Tuple-literal form: upstream literal_eval's these when given a string.
        return shlex.quote(presets._as_tuple_literal(value))
    if isinstance(value, str):
        # Quote only when a shell would need it, so paths with spaces survive
        # a copy-paste but ordinary values stay readable.
        return shlex.quote(value)
    return str(value)


def summarise(plan: Plan) -> list[tuple[str, str]]:
    """Key facts for a confirmation screen, in reading order."""
    model, data, trainer = plan.model, plan.data, plan.trainer
    rows = [
        ("voice", str(data["voice_name"])),
        ("espeak voice", str(data["espeak_voice"])),
        ("sample rate", f"{model['sample_rate']} Hz"),
        (
            "architecture",
            f"resblock {model['resblock']}, upsample "
            f"{tuple(model['upsample_rates'])}, "
            f"{model['upsample_initial_channel']} ch",
        ),
        ("batch size", str(data["batch_size"])),
        ("workers", str(data["num_workers"])),
        ("precision", str(trainer["precision"])),
        ("accelerator", f"{trainer['accelerator']} / {trainer['devices']}"),
        (
            "max epochs",
            "unlimited" if int(trainer["max_epochs"]) == -1 else str(trainer["max_epochs"]),
        ),
        ("mos metric", str(model["mos_metric"])),
    ]
    if plan.ckpt_path:
        rows.append(("resume from", plan.ckpt_path))
    elif model.get("warmstart_ckpt"):
        rows.append(("warmstart from", str(model["warmstart_ckpt"])))
    elif model.get("vocoder_warmstart_ckpt"):
        rows.append(("vocoder warmstart", str(model["vocoder_warmstart_ckpt"])))
    else:
        rows.append(("starting from", "scratch"))
    if plan.split:
        rows.append(
            (
                "split",
                f"{plan.split.train} train / {plan.split.val} val / "
                f"{plan.split.test} test",
            )
        )
    return rows


def cache_fingerprint_inputs(plan: Plan, metadata_bytes: bytes) -> str:
    """Hash the inputs that invalidate piper's utterance cache.

    Cache ids are ``f"{row_number}_{sanitize_filename(text)}"[:50]``, so
    reordering rows or editing a transcript orphans the old entries. Anything
    affecting the cached tensors belongs in this hash.
    """
    import hashlib

    model, data = plan.model, plan.data
    parts = [
        str(model["sample_rate"]),
        str(model["filter_length"]),
        str(model["hop_length"]),
        str(model["win_length"]),
        str(data["trim_silence"]),
        str(data["keep_seconds_before_silence"]),
        str(data["keep_seconds_after_silence"]),
        str(data["phoneme_type"]),
        str(data["dataset_type"]),
        str(data["espeak_voice"]),
    ]
    digest = hashlib.sha256()
    digest.update("|".join(parts).encode("utf-8"))
    digest.update(b"\0")
    digest.update(metadata_bytes)
    return digest.hexdigest()


def apply_low_vram(prof: profile_mod.Profile) -> list[str]:
    """Apply the small-GPU overlay in place. Returns what changed."""
    changed: list[str] = []
    for dotted, value in presets.LOW_VRAM.items():
        current = profile_mod.get_path(prof, dotted)
        if current != value:
            profile_mod.set_path(prof, dotted, value)
            changed.append(f"{dotted}: {current} -> {value}")
    return changed


OOM_LADDER = [
    "data.batch_size: halve it (8 -> 4 -> 2), and clamp dataset.max_seconds "
    "(14 -> 10 -> 8) — long clips dominate peak memory",
    "data.num_workers: 2 -> 1 -> 0 (each worker holds a prefetch queue, and on "
    "an APU that is the same physical RAM as VRAM)",
    "close other GPU users — on an APU the desktop compositor takes GTT",
    "model.segment_size: 8192 -> 4096 (must stay a multiple of hop_length)",
    "trainer.precision: 16-mixed (watch loss_g for divergence)",
    "model.mos_metric: none (UTMOS runs a second model during validation)",
    "trainer.check_val_every_n_epoch: 5 — validation synthesizes full "
    "utterances and is often the real peak",
    "voice.quality: low (16 kHz), or runtime.vendor: cpu for a correctness-only "
    "run",
]


def looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "hip out of memory",
            "hiperroroutofmemory",
            "cuda error: out of memory",
            "hsa_status_error_out_of_resources",
        )
    )
