"""Voice quality presets.

piper1-gpl has **no** ``--quality`` flag: the names x-low/low/medium/high
survive only as a filename convention and as directory names in the pretrained
checkpoint repository. ``vits/config.py`` still contains ``low_quality()`` and
``high_quality()``, but nothing references them — they are dead code carried
over from the legacy ``rhasspy/piper`` tree.

So "quality" is our abstraction. Each preset here expands to raw ``--model.*``
values. The defaults in ``VitsModel.__init__`` are byte-identical to legacy
``low_quality()``, which is what legacy Piper shipped as the *medium* preset —
hence ``MEDIUM`` below matches upstream's defaults exactly, and is the only
preset whose architecture matches the pretrained checkpoints on HuggingFace.

The hard constraint every preset must satisfy is
``prod(upsample_rates) == hop_length``; upstream raises
``ValueError("Upsample rates do not match hop length")`` otherwise. There is a
test asserting it for every preset here.
"""

from __future__ import annotations

from typing import Any

# Values common to every preset. Kept separate so a preset only states what it
# actually changes.
_COMMON: dict[str, Any] = {
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": None,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
}

MEDIUM: dict[str, Any] = {
    **_COMMON,
    # resblock is compared as a string upstream (`resblock == "1"`), so it must
    # stay quoted.
    "resblock": "2",
    "resblock_kernel_sizes": [3, 5, 7],
    "resblock_dilation_sizes": [[1, 2], [2, 6], [3, 12]],
    "upsample_rates": [8, 8, 4],
    "upsample_initial_channel": 256,
    "upsample_kernel_sizes": [16, 16, 8],
}

HIGH: dict[str, Any] = {
    **_COMMON,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    # 8*8*2*2 == 256 == hop_length, so the extra upsample stage needs no
    # hop_length change.
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
}

# Legacy "low" was medium's architecture at 16 kHz, nothing more.
LOW: dict[str, Any] = dict(MEDIUM)

PRESETS: dict[str, dict[str, Any]] = {
    "medium": MEDIUM,
    "high": HIGH,
    "low": LOW,
}

# The sample rate each preset is designed around. `audio.sample_rate` in the
# profile is authoritative; this is what the wizard offers and what we warn
# against when they disagree.
PRESET_SAMPLE_RATE: dict[str, int] = {
    "medium": 22050,
    "high": 22050,
    "low": 16000,
}

PRESET_NOTES: dict[str, str] = {
    "medium": (
        "Matches every pretrained checkpoint on HuggingFace, so it is the only "
        "preset you can fine-tune from with --ckpt_path. Start here."
    ),
    "high": (
        "Roughly 2x the vocoder width and an extra upsample stage: better "
        "high-frequency detail, noticeably slower, more VRAM. No pretrained "
        "checkpoint matches it, so fine-tuning needs 'vocoder_warmstart'."
    ),
    "low": (
        "Medium's architecture at 16 kHz. Cheapest to train and run; audibly "
        "duller. Useful on very limited hardware."
    ),
}

# Overlay applied on small GPUs. Deliberately conservative: on a shared-memory
# APU the GPU competes with the desktop for the same physical RAM, so the
# defaults that work on a discrete 24 GB card will OOM.
LOW_VRAM: dict[str, Any] = {
    "data.batch_size": 8,
    "data.num_workers": 2,
    "trainer.precision": "32-true",
}


def preset_for(quality: str) -> dict[str, Any]:
    try:
        return dict(PRESETS[quality])
    except KeyError:
        raise ValueError(
            f"unknown quality {quality!r} (expected one of {', '.join(PRESETS)})"
        ) from None


def preset_names() -> list[str]:
    return list(PRESETS)


def high_quality_argv_hint() -> str:
    """The raw flags equivalent to the 'high' preset, for docs and messages."""
    keys = (
        "resblock",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_initial_channel",
        "upsample_kernel_sizes",
    )
    return " ".join(
        f"--model.{key} {_as_tuple_literal(HIGH[key])}" for key in keys
    )


def _as_tuple_literal(value: Any) -> str:
    """Render a nested list the way a human would type it on the CLI.

    Upstream accepts either a YAML sequence or a Python tuple literal (the
    ``ast.literal_eval`` calls are guarded by ``isinstance(..., str)``). We emit
    YAML sequences in the config file and this form only in logs and docs.
    """
    if isinstance(value, (list, tuple)):
        inner = ",".join(_as_tuple_literal(item) for item in value)
        return f"({inner})"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)
