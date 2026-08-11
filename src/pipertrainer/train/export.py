"""ONNX export and voice packaging.

A Piper voice is two files that must agree: ``<lang>-<name>-<quality>.onnx`` and
``<same>.onnx.json``. They come from different places — the ONNX from
``piper.train.export_onnx``, the JSON from *training* (written to
``--data.config_path`` when the datamodule prepares data). Export does not
produce it, which surprises people.

Two fields in that JSON are wrong as written, verified in piper1-gpl v1.6.0:

* ``hop_length`` is hardcoded to ``DEFAULT_HOP_LENGTH`` (256) because
  ``VitsDataModule`` builds ``PiperConfig`` without passing its own value.
  Harmless at the default, wrong for any other hop length.
* ``piper_version`` is the literal string ``"1.5.0"`` while the package is
  1.6.0.

So we copy rather than move, fix those fields, keep the original alongside as
``.from-training``, and print a diff. Nothing is rewritten silently.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import env as env_mod
from .. import profile as profile_mod
from .. import proc, tui
from ..paths import REPO_ROOT, VoicePaths, venv_python
from . import argmap, launch


class ExportError(RuntimeError):
    pass


@dataclass
class Exported:
    onnx: Path
    config: Path
    original_config: Path
    verified: bool = False
    fixups: list[str] = field(default_factory=list)


def voice_filename(prof: profile_mod.Profile) -> str:
    """``en_US-mariah-medium`` — the naming other Piper tooling expects."""
    if prof.export.filename and prof.export.filename != "auto":
        return prof.export.filename
    language = prof.voice.language.strip() or "und"
    name = prof.voice.name.strip() or "voice"
    return f"{language}-{name}-{prof.voice.quality}"


def output_dir(prof: profile_mod.Profile, paths: VoicePaths) -> Path:
    if prof.export.output_dir.strip():
        return Path(prof.export.output_dir).expanduser()
    return paths.voice_out


# --------------------------------------------------------------------------
# Choosing a checkpoint
# --------------------------------------------------------------------------


def parse_metric(path: Path, metric: str) -> float | None:
    """Read a metric out of Lightning's checkpoint filename.

    Upstream names them ``epoch={epoch}-val_mel={val_mel:.4f}.ckpt`` with
    ``auto_insert_metric_name=False``, so the value is in the name and no torch
    load is needed to rank candidates.
    """
    stem = path.stem
    marker = f"{metric}="
    if marker not in stem:
        return None
    tail = stem.split(marker, 1)[1]
    number = ""
    for char in tail:
        if char.isdigit() or char in ".-+eE":
            number += char
        else:
            break
    try:
        return float(number)
    except ValueError:
        return None


def pick_checkpoint(paths: VoicePaths, prefer: str = "val_mel") -> Path:
    candidates = launch.find_checkpoints(paths)
    if not candidates:
        raise ExportError(
            f"no checkpoints under {paths.lightning_logs}. Train first with "
            f"'./run train'."
        )
    if prefer == "last":
        last = launch.find_last(paths)
        if last:
            return last
        return candidates[-1]

    scored = [
        (value, path)
        for path in candidates
        for value in (parse_metric(path, prefer),)
        if value is not None
    ]
    if not scored:
        tui.warn(
            f"no checkpoint filename carries {prefer}; falling back to the "
            f"newest. (val_mos is only logged when model.mos_metric is 'utmos' "
            f"and UTMOS could be downloaded.)"
        )
        return candidates[-1]
    # val_mel is minimised; val_mos is maximised.
    best = min(scored)[1] if prefer == "val_mel" else max(scored)[1]
    return best


def describe_checkpoints(paths: VoicePaths) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in reversed(launch.find_checkpoints(paths)):
        rows.append(
            [
                path.name,
                f"{path.stat().st_size / 1e9:.2f} GB",
                _format(parse_metric(path, "val_mel")),
                _format(parse_metric(path, "val_mos")),
            ]
        )
    return rows


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


# Our shim rather than `piper.train.export_onnx` directly: torch 2.9 made the
# dynamo exporter the default and VITS does not survive torch.export. The shim
# forces the TorchScript path, then calls upstream's main() unchanged.
# See train/export_shim.py.
EXPORT_ENTRYPOINT = "pipertrainer.train.export_shim"


def export_command(python: str, checkpoint: Path, destination: Path) -> list[str]:
    return [
        python,
        "-m",
        EXPORT_ENTRYPOINT,
        "--checkpoint",
        str(checkpoint),
        "--output-file",
        str(destination),
    ]


def export_onnx(checkpoint: Path, destination: Path, log_path: Path | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc.run(
        export_command(str(venv_python()), checkpoint, destination),
        cwd=REPO_ROOT,
        env=env_mod.training_env(),
        log_path=log_path,
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise ExportError(f"export produced no file at {destination}")


def fix_config(
    source: Path,
    destination: Path,
    prof: profile_mod.Profile,
    plan_model: dict[str, Any],
    ckpt_hparams: dict[str, Any] | None = None,
) -> list[str]:
    """Copy the training config, correcting the fields upstream gets wrong."""
    if not source.exists():
        raise ExportError(
            f"the voice config is missing at {source}.\n"
            f"It is written during training, not by export — so this usually "
            f"means training never reached the point of preparing data. Check "
            f"the run log, then train again."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    fixups: list[str] = []

    trained_hop = int(plan_model.get("hop_length", 256))
    if int(payload.get("hop_length", 256)) != trained_hop:
        fixups.append(
            f"hop_length: {payload.get('hop_length')} -> {trained_hop} "
            f"(upstream always writes DEFAULT_HOP_LENGTH)"
        )
        payload["hop_length"] = trained_hop

    if payload.get("piper_version") == "1.5.0":
        fixups.append('piper_version: "1.5.0" -> "1.6.0" (hardcoded upstream)')
        payload["piper_version"] = "1.6.0"

    # Sample rate disagreeing is not a fixup — it means the wrong config.
    config_rate = int(payload.get("audio", {}).get("sample_rate", 0))
    expected_rate = int(plan_model.get("sample_rate", 22050))
    if config_rate and config_rate != expected_rate:
        raise ExportError(
            f"the voice config says {config_rate} Hz but this profile trains at "
            f"{expected_rate} Hz. That means {source} belongs to a different "
            f"run. Re-train, or point export at the right run directory."
        )
    if ckpt_hparams:
        ckpt_rate = int(ckpt_hparams.get("sample_rate", 0) or 0)
        if ckpt_rate and config_rate and ckpt_rate != config_rate:
            raise ExportError(
                f"the checkpoint was trained at {ckpt_rate} Hz but the config "
                f"says {config_rate} Hz; they are from different runs"
            )

    inference = payload.setdefault("inference", {})
    for key, value in (
        ("noise_scale", float(prof.export.noise_scale)),
        ("length_scale", float(prof.export.length_scale)),
        ("noise_w", float(prof.export.noise_w)),
    ):
        if abs(float(inference.get(key, -1)) - value) > 1e-9:
            fixups.append(f"inference.{key}: {inference.get(key)} -> {value}")
            inference[key] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return fixups


VERIFY_SENTENCE = "This voice was exported and verified automatically."


def verify(onnx: Path, config: Path, destination: Path, log_path: Path | None) -> bool:
    """Load the exported pair with piper on CPU and synthesize one sentence."""
    code = (
        "import sys, wave\n"
        "from piper import PiperVoice\n"
        "voice = PiperVoice.load(sys.argv[1], config_path=sys.argv[2], use_cuda=False)\n"
        "with wave.open(sys.argv[3], 'wb') as handle:\n"
        "    voice.synthesize_wav(sys.argv[4], handle)\n"
        "with wave.open(sys.argv[3], 'rb') as handle:\n"
        "    print('FRAMES:%d RATE:%d CH:%d' % (handle.getnframes(), handle.getframerate(), handle.getnchannels()))\n"
    )
    result = proc.capture(
        [
            venv_python(),
            "-c",
            code,
            str(onnx),
            str(config),
            str(destination),
            VERIFY_SENTENCE,
        ],
        cwd=REPO_ROOT,
        timeout=900,
    )
    line = next((l for l in result.lines if l.startswith("FRAMES:")), "")
    if not result.ok or not line:
        tui.error("the exported voice could not synthesize:")
        for entry in result.lines[-15:]:
            print(f"  {entry}")
        return False

    frames = int(line.split()[0].split(":")[1])
    rate = int(line.split()[1].split(":")[1])
    if frames <= 0:
        tui.error("synthesis produced an empty file")
        return False
    tui.ok(
        f"verified: synthesized {frames / rate:.2f} s at {rate} Hz -> "
        f"{destination.name}"
    )
    return True


def run(
    prof: profile_mod.Profile,
    *,
    checkpoint: str | None = None,
    prefer: str = "val_mel",
    do_verify: bool = True,
) -> Exported:
    paths = VoicePaths(prof.voice.name)
    target = Path(checkpoint) if checkpoint else pick_checkpoint(paths, prefer)
    if not target.exists():
        raise ExportError(f"checkpoint not found: {target}")

    tui.ok(f"exporting from {target.name}")
    hparams = launch.checkpoint_hparams(target)

    # The plan tells us the hop length actually trained at, which is what the
    # config JSON fixup needs. Export must still work if the plan cannot be
    # rebuilt (for instance if the dataset has since been deleted), so fall back
    # to the checkpoint's own hyper-parameters.
    try:
        plan_model = argmap.build(prof, paths=paths).model
    except (argmap.ArgMapError, OSError, ValueError):
        plan_model = {
            "hop_length": int(hparams.get("hop_length", 256) or 256),
            "sample_rate": int(
                hparams.get("sample_rate", prof.audio.sample_rate)
                or prof.audio.sample_rate
            ),
        }

    stem = voice_filename(prof)
    destination = output_dir(prof, paths)
    onnx_path = destination / f"{stem}.onnx"
    config_path = destination / f"{stem}.onnx.json"
    log_path = paths.logs / "export.log"

    export_onnx(target, onnx_path, log_path)
    tui.ok(f"wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    original = destination / f"{stem}.onnx.json.from-training"
    shutil.copy2(paths.piper_config_json, original)
    fixups = fix_config(
        paths.piper_config_json, config_path, prof, plan_model, hparams
    )
    if fixups:
        tui.info("voice config corrections:")
        for line in fixups:
            tui.bullet(line)
        tui.hint(f"  the unmodified original is kept as {original.name}")
    else:
        tui.ok("voice config needed no corrections")

    verified = False
    if do_verify:
        verified = verify(
            onnx_path, config_path, destination / "verify.wav", log_path
        )

    tui.info("")
    tui.ok(f"voice ready in {destination}")
    tui.bullet(f"{onnx_path.name}")
    tui.bullet(f"{config_path.name}")
    tui.info("")
    tui.info("Use it with:")
    tui.info(
        f"  {venv_python()} -m piper -m {onnx_path} -f out.wav -- 'Hello there.'"
    )
    tui.hint(
        "  For Home Assistant or another Piper install, copy both files into "
        "its voices directory keeping the names identical."
    )

    return Exported(
        onnx=onnx_path,
        config=config_path,
        original_config=original,
        verified=verified,
        fixups=fixups,
    )
