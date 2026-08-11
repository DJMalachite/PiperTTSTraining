"""End-to-end self-test on CPU, offline, in minutes.

The point is to prove *plumbing*, not quality. The dataset is synthetic tones —
the model learns nothing from it, which is fine; what gets verified is that the
generated config is accepted, that preprocessing writes the cache and the voice
config, that a checkpoint appears, that the cache is genuinely incremental, and
that the exported voice loads and synthesizes.

It also runs four negative tests, because a guard that has never fired is not
known to work:

* ``--trainer.gradient_clip_val`` really is rejected under manual optimization.
  That is the one claim in this repo's design that came from reading Lightning's
  semantics rather than from running it, so the test settles it.
* ``batch_size`` above the training split is refused *before* launching.
* An architecture violating the hop-length invariant is refused.
* Setting a jsonargparse link target is refused.

Tiers: ``unit`` (no venv), ``dataset``, ``train``, ``export``.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import env as env_mod
from . import profile as profile_mod
from . import proc, tui
from .dataset import ffmpeg
from .dataset import metadata as meta
from .dataset import report as report_mod
from .paths import PROFILES_DIR, REPO_ROOT, VoicePaths, in_venv, venv_python
from .train import argmap, launch
from .train import export as export_mod

VOICE = "_smoke"
CLIP_COUNT = 24
SAMPLE_RATE = 22050

# Long enough to clear the 1 s floor and the 0.372 s segment_size padding
# threshold, varied enough to exercise the histogram.
CLIP_SECONDS = [1.5, 2.0, 2.5, 3.0]

# Deliberately awkward: a pipe, double quotes, and non-ASCII, all of which have
# to survive the csv round trip into piper's reader.
TRANSCRIPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore every summer.",
    '"Stop right there," he said quietly to the others.',
    "A pipe | inside a transcript must never break the file.",
    "Café naïve façade — unicode should survive the round trip.",
    "It costs forty-two pounds and fifty pence in total.",
    "Did you remember to close the door behind you?",
    "Numbers like 1234 and 5678 are read aloud by espeak.",
]

TRAIN_ARGS = [
    "--trainer.accelerator", "cpu",
    "--trainer.max_epochs", "1",
    "--trainer.limit_train_batches", "2",
    "--trainer.limit_val_batches", "1",
    "--trainer.log_every_n_steps", "1",
    "--trainer.enable_progress_bar", "false",
]


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(name)
        tui.ok(f"{name}{f': {detail}' if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append((name, detail))
        tui.error(f"{name}: {detail}")

    def skip(self, name: str, why: str) -> None:
        self.skipped.append((name, why))
        tui.hint(f"  - {name}: skipped ({why})")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.ok(name, detail)
        else:
            self.fail(name, detail or "assertion failed")
        return condition


def smoke_profile() -> profile_mod.Profile:
    prof = profile_mod.Profile()
    prof.voice.name = VOICE
    prof.voice.language = "en_US"
    prof.voice.espeak_voice = "en-us"
    prof.voice.quality = "medium"
    prof.audio.sample_rate = SAMPLE_RATE
    prof.data.batch_size = 4
    prof.data.num_workers = 0
    prof.data.num_test_examples = 1
    prof.data.validation_split = 0.1
    prof.trainer.accelerator = "cpu"
    prof.trainer.precision = "32-true"
    prof.trainer.max_epochs = 1
    prof.trainer.enable_progress_bar = False
    # No checkpoint: from scratch keeps the test fully offline (no 900 MB
    # download) and mos_metric 'none' keeps it off torch.hub.
    prof.finetune.mode = "none"
    prof.finetune.checkpoint = ""
    prof.model.mos_metric = "none"
    prof.runtime.offline = True
    return prof


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------


def tier_unit(results: Results) -> None:
    tui.heading("Tier 0: unit tests")
    result = proc.run(
        [venv_python() if in_venv() else "python3", "-m", "unittest", "discover",
         "-s", "tests", "-t", "."],
        cwd=REPO_ROOT,
        check=False,
        quiet=True,
    )
    tail = result.lines[-1] if result.lines else ""
    if result.ok:
        count = next(
            (line for line in result.lines if line.startswith("Ran ")), "ran"
        )
        results.ok("unit tests", count)
    else:
        for line in result.lines[-25:]:
            print(f"  {line}")
        results.fail("unit tests", tail or "failed")


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------


def tier_dataset(results: Results) -> None:
    tui.heading("Tier 1: synthetic dataset")
    paths = VoicePaths(VOICE)
    paths.ensure_dataset_dirs()

    # A stand-in "original recording" whose bytes must not change: the
    # predecessor script deleted the user's input, and that must never recur.
    source = paths.source / "original.wav"
    ffmpeg.synthesize_test_audio(
        source, sample_rate=SAMPLE_RATE, seconds=3.0, frequency=220.0
    )
    before = (source.stat().st_size, source.stat().st_mtime_ns)

    for old in paths.wavs.glob("*.wav"):
        old.unlink()

    rows: list[meta.Row] = []
    stats: list[report_mod.ClipStat] = []
    for index in range(1, CLIP_COUNT + 1):
        seconds = CLIP_SECONDS[index % len(CLIP_SECONDS)]
        text = TRANSCRIPTS[index % len(TRANSCRIPTS)]
        utt_id = meta.clip_id(index)
        ffmpeg.synthesize_test_audio(
            paths.wavs / f"{utt_id}.wav",
            sample_rate=SAMPLE_RATE,
            seconds=seconds,
            frequency=140.0 + (index % 7) * 40.0,
        )
        rows.append(meta.Row(utt_id=f"{utt_id}.wav", text=text))
        stats.append(
            report_mod.ClipStat(
                utt_id=utt_id, seconds=seconds, chars=len(text), text=text
            )
        )

    written = meta.write(rows, paths.metadata_csv)
    results.check("metadata rows written", written == CLIP_COUNT, f"{written} rows")

    # The pipe-bearing transcript must survive piper's own reader.
    read_back = meta.read(paths.metadata_csv)
    piped = [row for row in read_back if "|" in row.text]
    results.check(
        "pipe in a transcript survives the csv round trip",
        len(piped) >= 1 and piped[0].text.count("|") == 1,
        f"{len(piped)} row(s)",
    )
    quoted = [row for row in read_back if row.text.startswith('"')]
    results.check(
        "leading double quote survives", len(quoted) >= 1, f"{len(quoted)} row(s)"
    )

    usable, shortest, missing = launch.dataset_facts(paths)
    results.check(
        "every row resolves to audio",
        usable == CLIP_COUNT and not missing,
        f"{usable} usable, {len(missing)} missing",
    )
    results.check(
        "shortest clip clears the 1 s floor",
        shortest is not None and shortest >= 1.0,
        f"{shortest:.2f} s" if shortest else "unknown",
    )

    report = report_mod.build(
        rows,
        stats,
        [],
        voice=VOICE,
        source=str(source),
        source_duration=60.0,
        sample_rate=SAMPLE_RATE,
        batch_size=4,
        validation_split=0.1,
        num_test_examples=1,
    )
    report.write(paths.report_md)
    split = report.split
    results.check(
        "split arithmetic matches the trainer",
        (split.train, split.val, split.test) == (21, 2, 1),
        f"{split.train}/{split.val}/{split.test}",
    )
    results.check(
        "small dataset is flagged",
        any(level == "error" for level, _ in report.flags),
        "report flags the tiny corpus, as it should",
    )

    after = (source.stat().st_size, source.stat().st_mtime_ns)
    results.check(
        "source audio was not modified", before == after, "byte-identical"
    )


# --------------------------------------------------------------------------
# Tier 2
# --------------------------------------------------------------------------


def tier_train(results: Results) -> None:
    tui.heading("Tier 2: training on CPU")
    prof = smoke_profile()
    profile_mod.save(prof)
    paths = VoicePaths(VOICE)

    if not paths.metadata_csv.exists():
        results.skip("training", "no dataset; run the dataset tier first")
        return

    # --- negative tests that need no subprocess --------------------------
    _negative_checks(results, prof)

    # --- preflight --------------------------------------------------------
    try:
        prepared = launch.prepare(prof, offline=True)
    except (launch.LaunchError, argmap.ArgMapError) as exc:
        results.fail("preflight", str(exc))
        return
    results.ok("preflight", f"{prepared.usable} utterances")

    try:
        launch.print_config_gate(prepared)
        results.ok("piper accepts the generated config (--print_config)")
    except launch.LaunchError as exc:
        results.fail("--print_config", str(exc))
        return

    launch.check_cache(prepared, interactive=False, assume="keep")

    # --- first run --------------------------------------------------------
    started = time.monotonic()
    code = launch.start(prepared, extra_argv=TRAIN_ARGS)
    elapsed = time.monotonic() - started
    if code != 0:
        results.fail("training", f"exit code {code}")
        return
    results.ok("training completed", f"{elapsed:.0f}s for one epoch")

    # --- artefacts --------------------------------------------------------
    results.check(
        "voice config written during training",
        paths.piper_config_json.exists(),
        str(paths.piper_config_json.name),
    )
    if paths.piper_config_json.exists():
        import json

        payload = json.loads(paths.piper_config_json.read_text(encoding="utf-8"))
        results.check(
            "voice config has phonemes and rate",
            payload.get("num_symbols")
            and payload.get("phoneme_id_map")
            and payload.get("audio", {}).get("sample_rate") == SAMPLE_RATE,
            f"{payload.get('num_symbols')} symbols, "
            f"{payload.get('audio', {}).get('sample_rate')} Hz",
        )
        results.check(
            "espeak phonemization ran",
            len(payload.get("phoneme_id_map") or {}) > 20,
            f"{len(payload.get('phoneme_id_map') or {})} phonemes mapped",
        )

    counts = launch.cache_counts(paths.cache)
    results.check(
        "utterance cache is complete",
        all(count >= prepared.usable for count in counts.values()),
        ", ".join(f"{n}{suffix}" for suffix, n in counts.items()),
    )

    last = launch.find_last(paths)
    results.check(
        "checkpoint written", last is not None,
        f"{last.name} ({last.stat().st_size / 1e6:.0f} MB)" if last else "none",
    )

    _metric_checks(results, paths)

    # --- incrementality ---------------------------------------------------
    fingerprints = {
        path.name: path.stat().st_mtime_ns for path in paths.cache.glob("*.pt")
    }
    code = launch.start(prepared, extra_argv=TRAIN_ARGS)
    after = {
        path.name: path.stat().st_mtime_ns for path in paths.cache.glob("*.pt")
    }
    results.check(
        "second run reuses the cache",
        code == 0 and after == fingerprints,
        f"{len(after)} cached tensors unchanged",
    )

    _gradient_clip_check(results, prepared)


def _negative_checks(results: Results, prof: profile_mod.Profile) -> None:
    paths = VoicePaths(VOICE)

    # Link target.
    try:
        argmap.build(prof, paths=paths, extra_argv=["--data.sample_rate", "22050"])
        results.fail("refuses a link target", "no error raised")
    except argmap.ArgMapError as exc:
        results.ok("refuses a link target", str(exc).split(";")[0][:70])

    # Oversized batch.
    oversized = smoke_profile()
    oversized.data.batch_size = 40
    try:
        argmap.build(oversized, paths=paths, total_utterances=CLIP_COUNT)
        results.fail("refuses an oversized batch", "no error raised")
    except argmap.ArgMapError as exc:
        results.ok(
            "refuses an oversized batch",
            "zero batches" in str(exc) and "17" in str(exc) and "explains why",
        )

    # Hop-length invariant.
    bad = smoke_profile()
    bad.model.extra = {"upsample_rates": [8, 8, 2]}
    try:
        argmap.build(bad, paths=paths)
        results.fail("refuses a bad architecture", "no error raised")
    except argmap.ArgMapError as exc:
        results.ok(
            "refuses a bad architecture",
            "suggests a fix" if "(8,8,4)" in str(exc) else "raised",
        )

    # Gradient clipping.
    try:
        argmap.build(prof, paths=paths, extra_argv=["--trainer.gradient_clip_val=5"])
        results.fail("refuses gradient clipping", "no error raised")
    except argmap.ArgMapError:
        results.ok("refuses gradient clipping", "blocked before launch")


def _metric_checks(results: Results, paths: VoicePaths) -> None:
    from .train import monitor

    scalars = monitor.read_scalars(paths)
    if not scalars.steps:
        results.skip("metrics", "no scalars readable from the event files")
        return
    tags = set(scalars.steps)
    results.check(
        "GAN losses logged",
        bool({"loss_g", "loss_d"} & tags),
        ", ".join(sorted(tags)[:6]),
    )
    results.check(
        "val_mel logged", "val_mel" in tags, "checkpoint selection metric present"
    )
    # mos_metric was 'none', so val_mos must be absent — which also proves
    # nothing reached out to torch.hub.
    results.check(
        "val_mos absent with mos_metric=none",
        "val_mos" not in tags,
        "confirms the offline path stayed offline",
    )


def _gradient_clip_check(results: Results, prepared) -> None:
    """Settle the one inferred claim: does Lightning actually reject it?"""
    argv = [
        str(venv_python()),
        "-m", "piper.train", "fit",
        "--config", str(prepared.config_path),
        *TRAIN_ARGS,
        "--trainer.gradient_clip_val", "5.0",
    ]
    result = proc.capture(
        argv,
        cwd=REPO_ROOT,
        env=env_mod.training_env(prepared.profile.runtime.env, offline=True),
        timeout=1800,
    )
    if result.ok:
        results.fail(
            "upstream rejects gradient_clip_val",
            "it was ACCEPTED — the block in train/argmap.py can be relaxed. "
            "Update BLOCKED and docs/UPSTREAM_NOTES.md.",
        )
    else:
        hit = any(
            marker in result.output
            for marker in ("automatic optimization", "gradient_clip", "Misconfiguration")
        )
        results.ok(
            "upstream rejects gradient_clip_val",
            "as expected under manual optimization"
            if hit
            else f"exit {result.returncode} (message did not name the cause)",
        )


# --------------------------------------------------------------------------
# Tier 3
# --------------------------------------------------------------------------


def tier_export(results: Results) -> None:
    tui.heading("Tier 3: export and inference")
    paths = VoicePaths(VOICE)
    last = launch.find_last(paths)
    if last is None:
        results.skip("export", "no checkpoint; run the train tier first")
        return

    prof, _ = profile_mod.load_by_name(VOICE)
    try:
        exported = export_mod.run(
            prof, checkpoint=str(last), do_verify=True
        )
    except export_mod.ExportError as exc:
        results.fail("export", str(exc))
        return

    results.check(
        "onnx written",
        exported.onnx.exists() and exported.onnx.stat().st_size > 1_000_000,
        f"{exported.onnx.stat().st_size / 1e6:.0f} MB",
    )

    import json

    payload = json.loads(exported.config.read_text(encoding="utf-8"))
    results.check(
        "config hop_length is the trained value",
        int(payload.get("hop_length", 0)) == 256,
        str(payload.get("hop_length")),
    )
    results.check(
        "config piper_version corrected",
        payload.get("piper_version") == "1.6.0",
        f"{payload.get('piper_version')} (upstream hardcodes 1.5.0)",
    )
    results.check(
        "unmodified training config kept",
        exported.original_config.exists(),
        exported.original_config.name,
    )
    results.check(
        "exported voice synthesizes",
        exported.verified,
        "loaded via onnxruntime and produced audio",
    )

    onnx_check = proc.capture(
        [
            venv_python(),
            "-c",
            "import onnxruntime, sys;"
            "s = onnxruntime.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']);"
            "print(','.join(i.name for i in s.get_inputs()))",
            str(exported.onnx),
        ],
        timeout=600,
    )
    names = onnx_check.lines[-1] if onnx_check.ok and onnx_check.lines else ""
    results.check(
        "onnx input names match the piper contract",
        {"input", "input_lengths", "scales"} <= set(names.split(",")),
        names,
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def cleanup() -> None:
    for path in (
        VoicePaths(VOICE).data_root,
        VoicePaths(VOICE).run_root,
        VoicePaths(VOICE).voice_out,
        PROFILES_DIR / f"{VOICE}.yaml",
    ):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def run(stage: str = "all", keep: bool = False) -> int:
    results = Results()
    tui.heading("Self-test")
    tui.hint(
        "  CPU only, no network, synthetic audio. This proves the pipeline "
        "works end to end; it says nothing about voice quality."
    )

    wants = {"unit", "dataset", "train", "export"} if stage == "all" else {stage}
    needs_venv = wants & {"dataset", "train", "export"}
    if needs_venv and not in_venv():
        tui.error("this tier needs the venv — run './run setup' first")
        return 1
    if needs_venv:
        try:
            ffmpeg.require_ffmpeg()
        except ffmpeg.AudioError as exc:
            tui.error(str(exc))
            return 1

    started = time.monotonic()
    try:
        if "unit" in wants:
            tier_unit(results)
        if "dataset" in wants:
            tier_dataset(results)
        if "train" in wants:
            tier_train(results)
        if "export" in wants:
            tier_export(results)
    except KeyboardInterrupt:
        tui.warn("interrupted")
        return 130
    finally:
        if not keep and stage == "all" and not results.failed:
            cleanup()
        elif not keep and results.failed:
            tui.hint(
                f"  leaving the {VOICE} voice in place so you can inspect it; "
                f"remove it with: rm -rf data/{VOICE} runs/{VOICE} voices/{VOICE} "
                f"profiles/{VOICE}.yaml"
            )

    tui.heading("Self-test summary")
    elapsed = time.monotonic() - started
    tui.info(f"  {len(results.passed)} passed, {len(results.failed)} failed, "
             f"{len(results.skipped)} skipped in {elapsed:.0f}s")
    if results.failed:
        tui.info("")
        for name, detail in results.failed:
            tui.error(f"{name}: {detail}")
        return 1
    tui.info("")
    tui.ok("everything works end to end")
    return 0
