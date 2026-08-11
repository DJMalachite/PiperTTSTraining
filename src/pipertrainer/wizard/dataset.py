"""The dataset flow: point at one recording, get clips and transcripts.

Ends by requiring the user to acknowledge the report. That is deliberate: in a
fully automated pipeline the transcripts are the weakest link, and a two-minute
skim of the flagged clips is the highest-value quality check available.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence

from .. import profile as profile_mod
from .. import tui
from ..dataset import ffmpeg, pipeline
from ..dataset import metadata as meta
from ..paths import VoicePaths, in_venv
from . import common

DATASET_PREFIXES = ("dataset", "audio")


def _ensure_profile(profile_name: str | None) -> profile_mod.Profile:
    if profile_name:
        prof, warnings = profile_mod.load_by_name(profile_name)
    else:
        try:
            prof, warnings = profile_mod.resolve_active()
        except profile_mod.ProfileError:
            tui.info("No profile yet — let's make one.")
            prof = profile_mod.Profile()
            prof.voice.name = tui.ask_str(
                "Voice name",
                prof.voice.name,
                allow_back=False,
                help_text="Used for directory names and the exported filename.",
            )
            warnings = []
    for warning in warnings:
        tui.warn(warning)
    return prof


def _choose_input(prof: profile_mod.Profile, paths: VoicePaths, given: str | None) -> Path:
    if given:
        candidate = Path(os.path.expanduser(given))
        if not candidate.exists():
            raise pipeline.PipelineError(f"no such file: {candidate}")
        prof.dataset.input_path = str(candidate)
        return candidate

    try:
        return pipeline.resolve_input(prof, paths)
    except pipeline.PipelineError:
        pass

    tui.info("")
    tui.info(f"Drop your recording into {paths.source}/ or give its path now.")
    tui.hint(
        "  Anything ffmpeg can read: wav, mp3, m4a, flac, opus, or a video file. "
        "The file is only ever read — never modified or deleted."
    )
    while True:
        answer = tui.ask_path(
            "path to the recording", prof.dataset.input_path, must_exist=True,
            allow_back=False,
        )
        candidate = Path(answer)
        if candidate.is_dir():
            tui.error("that is a directory; give the path to a single file")
            continue
        prof.dataset.input_path = str(candidate)
        return candidate


def _offer_copy(
    source: Path, paths: VoicePaths, prof: profile_mod.Profile
) -> None:
    """Optionally keep a copy under the voice, so the dataset is self-contained."""
    if paths.source in source.parents:
        return
    destination = paths.source / source.name
    if destination.exists():
        return
    size_gb = source.stat().st_size / 1e9
    if not tui.confirm(
        f"copy the recording into {paths.source}/ ({size_gb:.1f} GB)?",
        default=False,
    ):
        return
    paths.source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    prof.dataset.input_path = str(destination)
    tui.ok(f"copied to {destination}")


def run(
    profile_name: str | None = None,
    input_path: str | None = None,
    force_stages: Sequence[str] = (),
    dry_run: bool = False,
    interactive: bool = True,
    offline: bool = False,
) -> int:
    if not in_venv():
        tui.error("not set up yet — run './run setup' first")
        return 1

    prof = _ensure_profile(profile_name)
    paths = VoicePaths(prof.voice.name)
    paths.ensure_dataset_dirs()

    tui.heading(f"Dataset for {prof.voice.name}")

    source = _choose_input(prof, paths, input_path)
    info = ffmpeg.probe(source)
    tui.ok(f"{source.name}: {info.pretty}")

    if interactive:
        _offer_copy(source, paths, prof)

        # Sample rate is tied to the quality preset, so ask about quality here
        # rather than making the user discover the coupling later.
        tui.heading("Voice")
        common.walk(prof, ("voice",), title=None)
        from ..train import presets

        recommended = presets.PRESET_SAMPLE_RATE[prof.voice.quality]
        tui.hint(f"  {presets.PRESET_NOTES[prof.voice.quality]}")
        if int(prof.audio.sample_rate) != recommended:
            if tui.confirm(
                f"set audio.sample_rate to {recommended} Hz to match "
                f"'{prof.voice.quality}'?",
                default=True,
            ):
                prof.audio.sample_rate = recommended

        changed = common.walk(
            prof,
            DATASET_PREFIXES,
            skip=("dataset.input_path", "dataset.dry_run"),
            title="Dataset settings",
        )
        changed += common.offer_advanced(prof, DATASET_PREFIXES, "dataset")
        common.save_and_report(prof, changed)

        estimate = _estimate(info.duration, prof)
        tui.info("")
        tui.info(estimate)
        if not tui.confirm("build the dataset now?", default=True):
            return 0
    else:
        profile_mod.save(prof)

    tui.heading("Building")
    result = pipeline.run(
        prof,
        force_stages=force_stages,
        dry_run=dry_run,
        offline=offline or prof.runtime.offline,
    )

    _present(result, paths, prof, interactive=interactive)
    return 0 if result.report.count else 1


def _estimate(duration: float, prof: profile_mod.Profile) -> str:
    """Rough expectations, stated as assumptions rather than promises."""
    from ..dataset.macrosplit import format_duration

    target = float(prof.dataset.target_seconds)
    # Assume roughly 60% of a recording survives as usable speech.
    clips = int((duration * 0.6) / max(1.0, target))
    return (
        f"A {format_duration(duration)} recording usually yields somewhere "
        f"around {clips} clips at ~{target:.0f} s each, assuming about 60% of it "
        f"is speech. Transcription is the slow part and its results are cached, "
        f"so re-running with different clip lengths is cheap."
    )


def _present(
    result: pipeline.DatasetResult,
    paths: VoicePaths,
    prof: profile_mod.Profile,
    *,
    interactive: bool,
) -> None:
    report = result.report
    tui.heading("Dataset report")
    tui.table(report.summary_rows())

    if report.clips:
        tui.info("")
        tui.info("clip length distribution:")
        for line in report.histogram(buckets=10, width=32):
            print(f"  {line}")

    errors = [message for level, message in report.flags if level == "error"]
    warnings = [message for level, message in report.flags if level == "warn"]

    if errors or warnings:
        tui.info("")
        for message in errors:
            tui.error(tui.wrap(message, indent="  ").lstrip())
        for message in warnings:
            tui.warn(tui.wrap(message, indent="  ").lstrip())
    else:
        tui.info("")
        tui.ok("nothing to flag")

    tui.info("")
    tui.info(f"full report: {paths.report_md}")
    if result.rejected:
        tui.info(f"rejected clips and reasons: {paths.rejected_csv}")

    if result.dry_run:
        tui.warn("this was a dry run — no WAV files or metadata.csv were written")
        return

    if not report.clips:
        return

    if interactive:
        _review(paths, report)

    tui.info("")
    tui.info("Next:")
    tui.bullet("./run checkpoints    pick a pretrained checkpoint to fine-tune from")
    tui.bullet("./run train          configure and start training")


def _review(paths: VoicePaths, report) -> None:
    """Make the user look at the outliers before they spend hours training."""
    outliers = report.outliers(limit=8)
    if not outliers:
        return
    tui.info("")
    tui.info("Clips most worth checking (furthest from the norm):")
    tui.table(
        [
            [clip.utt_id, f"{clip.seconds:.1f}s", f"{clip.chars_per_second:.0f} c/s", clip.text[:56]]
            for clip in outliers
        ],
        headers=["clip", "length", "rate", "transcript"],
    )
    tui.hint(f"  the audio is in {paths.wavs}")

    while True:
        choice = tui.menu(
            "Transcripts",
            [
                ("ok", "Looks fine, continue"),
                ("edit", "Open metadata.csv in $EDITOR"),
                ("play", "Print a command to listen to these clips"),
            ],
            allow_back=False,
        )
        if choice == "ok":
            return
        if choice == "play":
            names = " ".join(f"{clip.utt_id}.wav" for clip in outliers[:5])
            tui.info(f"  cd {paths.wavs} && aplay {names}")
            continue
        _edit_metadata(paths)
        return


def _edit_metadata(paths: VoicePaths) -> None:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        for candidate in ("nano", "vim", "vi"):
            if shutil.which(candidate):
                editor = candidate
                break
    if not editor:
        tui.warn(
            f"no $EDITOR set and no nano/vim found. Edit {paths.metadata_csv} "
            f"yourself; the format is wav|text, pipe-delimited."
        )
        return

    from .. import proc

    tui.hint(
        "  Fix only the text column. Do not reorder or renumber rows: piper's "
        "cache ids include the row number, so a reorder invalidates the cache."
    )
    proc.run([editor, str(paths.metadata_csv)], check=False, echo=False)

    try:
        rows = meta.read(paths.metadata_csv)
    except (ValueError, OSError) as exc:
        tui.error(f"metadata.csv no longer parses: {exc}")
        return
    tui.ok(f"metadata.csv still parses ({len(rows)} rows)")
    tui.hint(
        "  The next training run will notice the change and offer to rebuild "
        "the utterance cache."
    )
