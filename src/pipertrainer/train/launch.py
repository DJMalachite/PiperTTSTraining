"""Launching ``piper.train fit``.

Everything expensive is checked before the subprocess starts. Training runs for
hours, so a mistake caught in the first second is worth a lot — and several of
the failure modes here surface downstream as errors that say nothing about their
cause:

* ``batch_size`` above the training split gives *zero* batches, because the
  dataloader drops the last partial batch.
* An interrupted ``prepare_data`` leaves a partial utterance cache. Upstream's
  guard against that is ``if not <Path>:``, which is always false, so the cache
  passes validation and then fails inside the dataloader.
* Editing one transcript changes piper's cache ids (they embed the row number
  and the text), silently orphaning every cached tensor after it.

Then ``--print_config`` runs as a free type-check, and only after that does the
real run begin.
"""

from __future__ import annotations

import json
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .. import env as env_mod
from .. import profile as profile_mod
from .. import proc, tui, yamlio
from ..dataset import metadata as meta
from ..paths import PIPER_DIR, REPO_ROOT, VoicePaths, in_venv, venv_python
from . import argmap


class LaunchError(RuntimeError):
    pass


@dataclass
class Prepared:
    """A validated, ready-to-run plan."""

    profile: profile_mod.Profile
    paths: VoicePaths
    plan: argmap.Plan
    usable: int
    min_clip_seconds: float | None
    config_path: Path


# --------------------------------------------------------------------------
# Inspecting the dataset
# --------------------------------------------------------------------------


def dataset_facts(paths: VoicePaths) -> tuple[int, float | None, list[str]]:
    """``(usable_rows, shortest_clip_seconds, missing_ids)``.

    Rows are resolved exactly the way piper does, including the retry with
    ``.wav`` appended, so this count agrees with what training will see.
    """
    if not paths.metadata_csv.exists():
        raise LaunchError(
            f"no dataset at {paths.metadata_csv}. Run './run dataset' first."
        )
    usable, missing = meta.count_usable(paths.metadata_csv, paths.wavs)

    shortest: float | None = None
    for path in paths.wavs.glob("*.wav"):
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate() or 1
                seconds = handle.getnframes() / float(rate)
        except (OSError, wave.Error):
            continue
        shortest = seconds if shortest is None else min(shortest, seconds)
    return usable, shortest, missing


# --------------------------------------------------------------------------
# Inspecting a checkpoint
# --------------------------------------------------------------------------

_HPARAMS_PROBE = r"""
import json, sys
import torch
path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}))
    raise SystemExit(0)
hparams = payload.get("hyper_parameters") or {}
out = {}
for key, value in hparams.items():
    if isinstance(value, (int, float, str, bool)) or value is None:
        out[key] = value
    elif isinstance(value, (list, tuple)):
        try:
            out[key] = json.loads(json.dumps(value))
        except TypeError:
            pass
print(json.dumps({
    "hparams": out,
    "epoch": payload.get("epoch"),
    "global_step": payload.get("global_step"),
}))
"""


def checkpoint_hparams(path: Path) -> dict[str, Any]:
    """Read a checkpoint's stored architecture. Empty dict if unreadable."""
    if not path.exists():
        raise LaunchError(f"checkpoint not found: {path}")
    result = proc.capture(
        [venv_python(), "-c", _HPARAMS_PROBE, str(path)], timeout=600
    )
    for line in reversed(result.lines):
        if line.strip().startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in payload:
                tui.warn(
                    f"could not read {path.name}'s architecture "
                    f"({payload['error']}); skipping the compatibility check"
                )
                return {}
            hparams = payload.get("hparams", {})
            epoch, step = payload.get("epoch"), payload.get("global_step")
            if epoch is not None:
                tui.hint(f"  checkpoint is at epoch {epoch}, step {step}")
            return hparams
    tui.warn(f"could not inspect {path.name}; skipping the compatibility check")
    return {}


# --------------------------------------------------------------------------
# espeak validation
# --------------------------------------------------------------------------


def validate_espeak_voice(voice: str) -> None:
    """Prove the voice phonemizes. There is no espeak-ng binary to ask."""
    code = (
        "from piper.phonemize_espeak import EspeakPhonemizer\n"
        f"out = EspeakPhonemizer().phonemize({voice!r}, 'The quick brown fox.')\n"
        "print('OK:' + ''.join(''.join(s) for s in out))\n"
    )
    result = env_mod.python_snippet(code)
    line = next((l for l in result.lines if l.startswith("OK:")), "")
    if not result.ok or not line[3:].strip():
        raise LaunchError(
            f"espeak-ng cannot phonemize with voice {voice!r}.\n"
            f"voice.espeak_voice must be an espeak-ng voice name such as en-us, "
            f"en-gb, de or fr — a locale tag like en_US is not valid here.\n"
            + "\n".join(result.lines[-5:])
        )
    tui.ok(f"espeak voice {voice!r} -> {line[3:][:40]}")


# --------------------------------------------------------------------------
# The utterance cache
# --------------------------------------------------------------------------

CACHE_SUFFIXES = (".phonemes.pt", ".audio.pt", ".spec.pt")


def cache_counts(cache_dir: Path) -> dict[str, int]:
    if not cache_dir.exists():
        return {suffix: 0 for suffix in CACHE_SUFFIXES}
    return {
        suffix: len(list(cache_dir.glob(f"*{suffix}")))
        for suffix in CACHE_SUFFIXES
    }


def wipe_cache(cache_dir: Path) -> int:
    removed = 0
    if not cache_dir.exists():
        return 0
    for path in cache_dir.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def check_cache(
    prepared: Prepared, *, interactive: bool = True, assume: str = "keep"
) -> None:
    """Validate the cache fingerprint and completeness.

    Both problems are recoverable, and both are silent without this: a stale
    fingerprint means wasted re-preprocessing at best and mismatched tensors at
    worst; an incomplete cache means a crash inside the dataloader.
    """
    cache_dir = prepared.paths.cache
    fingerprint_path = prepared.paths.cache_fingerprint
    metadata_bytes = prepared.paths.metadata_csv.read_bytes()
    expected = argmap.cache_fingerprint_inputs(prepared.plan, metadata_bytes)

    counts = cache_counts(cache_dir)
    populated = any(counts.values())
    recorded = (
        fingerprint_path.read_text(encoding="utf-8").strip()
        if fingerprint_path.exists()
        else ""
    )

    if populated and recorded and recorded != expected:
        tui.warn(
            "the utterance cache was built with different settings or a "
            "different metadata.csv."
        )
        tui.hint(
            "  piper's cache ids embed the row number and the transcript text, "
            "so editing one line orphans every entry after it. Keeping the "
            "cache is safe — piper skips what exists and recomputes the rest — "
            "but stale entries waste disk."
        )
        choice = assume
        if interactive:
            choice = tui.ask_choice(
                "cache",
                ["keep", "wipe", "abort"],
                "keep",
                help_text=(
                    "keep: reuse what matches and recompute the rest.\n"
                    "wipe: delete the cache and preprocess from scratch.\n"
                    "abort: stop without changing anything."
                ),
                allow_back=False,
            )
        if choice == "abort":
            raise LaunchError("aborted at the cache prompt")
        if choice == "wipe":
            removed = wipe_cache(cache_dir)
            tui.ok(f"removed {removed} cached file(s)")
            counts = cache_counts(cache_dir)

    elif populated and not recorded:
        tui.hint("  cache has no fingerprint (built by an older run); keeping it")

    # Completeness: upstream's own guard never fires, so check it here.
    if populated:
        incomplete = [
            suffix
            for suffix, count in counts.items()
            if 0 < count < prepared.usable
        ]
        if incomplete and min(counts.values()) < prepared.usable:
            tui.warn(
                f"the cache is partial: "
                + ", ".join(
                    f"{count} {suffix}" for suffix, count in counts.items()
                )
                + f" for {prepared.usable} utterances. piper will fill the gaps "
                f"on this run; that is expected after an interrupted start."
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(expected + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


def prepare(
    prof: profile_mod.Profile,
    *,
    offline: bool = False,
    extra_argv: Sequence[str] = (),
    check_espeak: bool = True,
) -> Prepared:
    if not in_venv():
        raise LaunchError("not set up yet — run './run setup'")
    if not PIPER_DIR.exists():
        raise LaunchError(f"piper1-gpl is missing at {PIPER_DIR}; run './run setup'")

    paths = VoicePaths(prof.voice.name)
    paths.ensure_run_dirs()

    usable, shortest, missing = dataset_facts(paths)
    if missing:
        tui.warn(
            f"{len(missing)} row(s) in metadata.csv have no audio file and will "
            f"be skipped: " + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    tui.ok(f"{usable} usable utterances")

    ckpt_hparams: dict[str, Any] | None = None
    if prof.finetune.mode == "ckpt_path" and prof.finetune.checkpoint:
        ckpt_hparams = checkpoint_hparams(Path(prof.finetune.checkpoint))

    plan = argmap.build(
        prof,
        paths=paths,
        offline=offline,
        total_utterances=usable,
        min_clip_seconds=shortest,
        ckpt_hparams=ckpt_hparams,
        extra_argv=extra_argv,
    )

    if check_espeak:
        validate_espeak_voice(prof.voice.espeak_voice)

    yamlio.dump(
        plan.config,
        paths.lightning_yaml,
        header=(
            "Generated by ./run train — do not edit.\n"
            "Edit the profile instead: profiles/"
            f"{prof.slug}.yaml, then re-run.\n"
            "Passed to piper as: python -m piper.train fit --config this-file"
        ),
    )

    return Prepared(
        profile=prof,
        paths=paths,
        plan=plan,
        usable=usable,
        min_clip_seconds=shortest,
        config_path=paths.lightning_yaml,
    )


def print_config_gate(prepared: Prepared) -> None:
    """Free validation: let jsonargparse type-check before we commit hours."""
    result = proc.capture(
        [
            venv_python(),
            "-m",
            "piper.train",
            "fit",
            "--config",
            str(prepared.config_path),
            "--print_config",
        ],
        env=env_mod.training_env(
            prepared.profile.runtime.env,
            offline=prepared.profile.runtime.offline,
        ),
        timeout=900,
    )
    if not result.ok:
        tui.error("piper rejected the generated configuration:")
        for line in result.lines[-25:]:
            print(f"  {line}")
        raise LaunchError(
            "the generated lightning.yaml is not valid for this piper version. "
            "If pins.toml was bumped, re-run the checklist in "
            "docs/UPSTREAM_NOTES.md — the flag mapping in train/argmap.py is "
            "verified against a specific tag."
        )
    tui.ok("configuration accepted by piper (--print_config)")


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------


def record_run(prepared: Prepared, argv: Sequence[str], env: dict[str, str]) -> Path:
    """Snapshot everything needed to reproduce or diff this run."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    directory = prepared.paths.logs / f"run-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "argv.txt").write_text(
        proc.describe(argv) + "\n", encoding="utf-8", newline="\n"
    )
    (directory / "equivalent-flags.txt").write_text(
        "# The same run expressed as flags instead of --config.\n"
        + proc.describe([venv_python(), "-m", "piper.train"] + list(prepared.plan.argv))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (directory / "env.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in sorted(env.items())) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copy2(prepared.config_path, directory / "lightning.yaml")
    if prepared.profile.path.exists():
        shutil.copy2(prepared.profile.path, directory / "profile.yaml")

    versions = []
    for label, path in (("pipertrainer", REPO_ROOT), ("piper1-gpl", PIPER_DIR)):
        head = proc.capture(["git", "rev-parse", "HEAD"], cwd=path, timeout=60)
        versions.append(f"{label}: {head.lines[0].strip() if head.ok and head.lines else '?'}")
    (directory / "git.txt").write_text(
        "\n".join(versions) + "\n", encoding="utf-8", newline="\n"
    )

    freeze = proc.capture([venv_python(), "-m", "pip", "freeze"], timeout=300)
    (directory / "pip-freeze.txt").write_text(
        freeze.output + "\n", encoding="utf-8", newline="\n"
    )
    return directory


# --------------------------------------------------------------------------
# Launching
# --------------------------------------------------------------------------


def start(
    prepared: Prepared,
    *,
    extra_argv: Sequence[str] = (),
    resume_from: str | None = None,
) -> int:
    prof = prepared.profile
    argv: list[str] = [
        str(venv_python()),
        "-m",
        "piper.train",
        "fit",
        "--config",
        str(prepared.config_path),
    ]
    checkpoint = resume_from or prepared.plan.ckpt_path
    if checkpoint:
        argv += ["--ckpt_path", str(checkpoint)]
    argv += list(extra_argv)

    env = env_mod.training_env(
        prof.runtime.env,
        offline=prof.runtime.offline,
        num_workers=int(prof.data.num_workers),
    )
    run_dir = record_run(prepared, argv, env)
    log_path = run_dir / "train.log"

    tui.heading("Training")
    if env:
        for key, value in sorted(env.items()):
            tui.hint(f"  {key}={value}")
    tui.hint(f"  log: {log_path}")
    tui.hint("  Ctrl-C once to stop; Lightning writes last.ckpt on the way out.")
    tui.info("")

    result = proc.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
        check=False,
        keep_lines=4000,
    )

    if result.interrupted:
        tui.info("")
        tui.ok("stopped. Resume with './run resume'.")
        return 0

    if result.ok:
        tui.ok(f"training finished after {result.duration / 60:.1f} minutes")
        tui.bullet("./run monitor    inspect metrics and pick a checkpoint")
        tui.bullet("./run export     export the voice to ONNX")
        return 0

    _diagnose_failure(prepared, result)
    return result.returncode or 1


def _diagnose_failure(prepared: Prepared, result: proc.Result) -> None:
    text = result.output
    tui.info("")
    if argmap.looks_like_oom(text):
        tui.error("the GPU ran out of memory.")
        tui.info("Work down this list, cheapest first:")
        for index, item in enumerate(argmap.OOM_LADDER, start=1):
            tui.info(f"  {index}. {item}")
        split = prepared.plan.split
        if split:
            tui.info("")
            tui.hint(
                f"  your training split is {split.train} utterances, so "
                f"batch_size can be anything up to {split.max_batch_size}"
            )
        return

    if "zero" in text.lower() and "batch" in text.lower():
        tui.error(
            "the dataloader produced no batches. That normally means "
            "data.batch_size exceeds the training split."
        )
        return

    if "HSA_STATUS_ERROR" in text or "Memory access fault" in text:
        tui.error(
            "the ROCm runtime aborted. This is the signature of a GPU "
            "architecture the torch build has no kernels for."
        )
        tui.info(
            "Run './run doctor' — if the matmul check fails, try another ROCm "
            "index from pins.toml with './run setup --force-step torch "
            "--torch-index ... --torch-spec ...'."
        )
        return

    tui.error(f"training exited with code {result.returncode}")
    tui.hint(f"  full log: {result.log_path}")


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def find_checkpoints(paths: VoicePaths) -> list[Path]:
    if not paths.lightning_logs.exists():
        return []
    return sorted(
        paths.lightning_logs.glob("version_*/checkpoints/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )


def find_last(paths: VoicePaths) -> Path | None:
    candidates = [
        path for path in find_checkpoints(paths) if path.name == "last.ckpt"
    ]
    return candidates[-1] if candidates else None


def resume(
    profile_name: str | None = None,
    checkpoint: str | None = None,
    force: bool = False,
    offline: bool = False,
) -> int:
    prof, warnings = profile_mod.resolve_active(profile_name)
    for warning in warnings:
        tui.warn(warning)
    paths = VoicePaths(prof.voice.name)

    target = Path(checkpoint) if checkpoint else find_last(paths)
    if target is None:
        raise LaunchError(
            f"no last.ckpt under {paths.lightning_logs}. Start a run with "
            f"'./run train' first."
        )
    if not target.exists():
        raise LaunchError(f"checkpoint not found: {target}")

    age = (time.time() - target.stat().st_mtime) / 3600.0
    tui.ok(
        f"resuming from {target.name} "
        f"({target.stat().st_size / 1e9:.2f} GB, {age:.1f} h old)"
    )

    previous = None
    if paths.lightning_yaml.exists():
        previous = yamlio.load(paths.lightning_yaml)

    prepared = prepare(prof, offline=offline)

    if previous is not None and previous != prepared.plan.config:
        changed = _config_diff(previous, prepared.plan.config)
        tui.warn("the profile no longer matches the configuration this run started with:")
        for line in changed[:12]:
            tui.bullet(line)
        tui.hint(
            "  A changed architecture makes --ckpt_path's strict load fail, and "
            "changed cache settings invalidate the utterance cache."
        )
        if not force and not tui.confirm("resume anyway?", default=False):
            raise LaunchError("aborted; re-run with --force to override")

    check_cache(prepared, interactive=True)
    print_config_gate(prepared)
    return start(prepared, resume_from=str(target))


def _config_diff(before: dict, after: dict, trail: str = "") -> list[str]:
    lines: list[str] = []
    for key in sorted(set(before) | set(after)):
        path = f"{trail}.{key}" if trail else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            lines += _config_diff(old, new, path)
        elif old != new:
            lines.append(f"{path}: {old!r} -> {new!r}")
    return lines
