"""The interactive menu shown by a bare ``./run``.

The status header is the important part: on a headless box the first question is
always "is this machine actually set up, and how far did I get with this voice".
Everything it reports comes from cached state or a cheap filesystem check, so
opening the menu never costs a GPU probe.
"""

from __future__ import annotations

from pathlib import Path

from . import env, profile as profile_mod, tui
from .paths import PIPER_DIR, VoicePaths, in_venv


def _dataset_status(prof: profile_mod.Profile) -> str:
    paths = VoicePaths(prof.voice.name)
    if not paths.metadata_csv.exists():
        if paths.source.exists() and any(paths.source.iterdir()):
            return "source staged, not yet processed"
        return tui.style("none", "dim")
    rows = sum(
        1
        for line in paths.metadata_csv.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    wavs = len(list(paths.wavs.glob("*.wav"))) if paths.wavs.exists() else 0
    return f"{rows} utterances, {wavs} wavs"


def _run_status(prof: profile_mod.Profile) -> str:
    paths = VoicePaths(prof.voice.name)
    if not paths.lightning_logs.exists():
        return tui.style("not started", "dim")
    checkpoints = sorted(
        paths.lightning_logs.glob("version_*/checkpoints/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        return "started, no checkpoint yet"
    newest = checkpoints[-1]
    total_gib = sum(c.stat().st_size for c in checkpoints) / (1024**3)
    return f"{len(checkpoints)} checkpoints ({total_gib:.1f} GiB), newest {newest.name}"


def _export_status(prof: profile_mod.Profile) -> str:
    paths = VoicePaths(prof.voice.name)
    out = Path(prof.export.output_dir) if prof.export.output_dir else paths.voice_out
    if not out.exists():
        return tui.style("none", "dim")
    models = sorted(out.glob("*.onnx"))
    if not models:
        return tui.style("none", "dim")
    return ", ".join(model.name for model in models)


def status_lines(prof: profile_mod.Profile | None) -> list[str]:
    lines: list[str] = []
    state = env.SetupState.load()

    if not in_venv():
        lines.append(
            f"{tui.style('setup', 'bold')}    "
            f"{tui.style('not installed — run option 1 first', 'yellow')}"
        )
    else:
        info = state.info
        detail = info.summary() if info.ok else "installed, torch not verified"
        marker = "green" if (info.ok and (info.usable_gpu or info.vendor == "cpu")) else "yellow"
        lines.append(f"{tui.style('setup', 'bold')}    {tui.style(detail, marker)}")
        if not PIPER_DIR.exists():
            lines.append(
                f"{tui.style('piper', 'bold')}    "
                f"{tui.style('piper1-gpl missing — re-run setup', 'yellow')}"
            )

    if prof is None:
        lines.append(
            f"{tui.style('profile', 'bold')}  {tui.style('none selected', 'dim')}"
        )
        return lines

    lines.append(
        f"{tui.style('profile', 'bold')}  {prof.voice.name}"
        f"  ({prof.voice.language}, {prof.voice.quality}, "
        f"{prof.audio.sample_rate} Hz, espeak {prof.voice.espeak_voice})"
    )
    lines.append(f"{tui.style('dataset', 'bold')}  {_dataset_status(prof)}")
    lines.append(f"{tui.style('training', 'bold')} {_run_status(prof)}")
    lines.append(f"{tui.style('export', 'bold')}   {_export_status(prof)}")
    if prof.runtime.offline:
        lines.append(tui.style("OFFLINE mode is on for this profile", "yellow"))
    return lines


ITEMS = [
    ("setup", "Set up this machine (torch, piper1-gpl, whisper)"),
    ("doctor", "Check the environment and diagnose problems"),
    ("profile", "Choose or create a voice profile"),
    ("dataset", "Prepare a dataset from one audio file"),
    ("checkpoints", "Browse and download a pretrained checkpoint"),
    ("train", "Configure and start training"),
    ("resume", "Resume the most recent run"),
    ("monitor", "Monitor progress (TensorBoard, metrics, checkpoints)"),
    ("preview", "Synthesize a sentence from a checkpoint"),
    ("export", "Export to ONNX for use with Piper"),
    ("smoke", "Run the CPU self-test"),
]


def _load_active() -> profile_mod.Profile | None:
    name = profile_mod.get_active()
    if name is None:
        names = profile_mod.profile_names()
        if len(names) == 1:
            name = names[0]
            profile_mod.set_active(name)
        else:
            return None
    try:
        prof, warnings = profile_mod.load_by_name(name)
    except profile_mod.ProfileError as exc:
        tui.warn(f"could not load profile {name!r}: {exc}")
        return None
    for warning in warnings:
        tui.warn(warning)
    return prof


def _choose_profile() -> None:
    names = profile_mod.profile_names()
    active = profile_mod.get_active()
    options = [(name, f"{name}{' (active)' if name == active else ''}") for name in names]
    options.append(("__new__", "Create a new profile"))
    try:
        choice = tui.menu("Profiles", options, allow_back=True)
    except tui.Back:
        return
    if choice == "__new__":
        name = tui.ask_str("Voice name", "myvoice", allow_back=False)
        prof = profile_mod.Profile()
        prof.voice.name = name
        prof.voice.language = tui.ask_str(
            "Locale tag (for the exported filename)", prof.voice.language,
            allow_back=False,
        )
        prof.voice.espeak_voice = tui.ask_str(
            "espeak-ng voice", prof.voice.espeak_voice, allow_back=False,
            help_text="Run './run doctor' afterwards to confirm it phonemizes.",
        )
        path = profile_mod.save(prof)
        profile_mod.set_active(name)
        tui.ok(f"created {path}")
        return
    profile_mod.set_active(choice)
    tui.ok(f"active profile: {choice}")


def _dispatch(action: str, prof: profile_mod.Profile | None, offline: bool) -> None:
    """Run one menu action. Errors are reported, not fatal — we return to the menu."""
    name = prof.voice.name if prof else None

    if action == "profile":
        _choose_profile()
        return

    if action == "setup":
        from . import install

        install.run_setup(offline=offline)
        return

    if action == "doctor":
        from . import doctor

        doctor.run(offline=offline)
        return

    if action == "smoke":
        from . import smoke

        smoke.run()
        return

    if action == "checkpoints":
        from . import checkpoints

        checkpoints.browse(profile_name=name, offline=offline)
        return

    # Everything below needs a profile.
    if prof is None:
        tui.warn("select or create a profile first (option 3)")
        return

    if action == "dataset":
        from .wizard import dataset as wizard

        wizard.run(profile_name=name, interactive=True, offline=offline)
    elif action == "train":
        from .wizard import train as wizard

        wizard.run(profile_name=name, interactive=True, offline=offline)
    elif action == "resume":
        from .train import launch

        launch.resume(profile_name=name, offline=offline)
    elif action == "monitor":
        from .train import monitor

        monitor.interactive(profile_name=name)
    elif action == "preview":
        from .train import preview

        preview.run(profile_name=name, offline=offline)
    elif action == "export":
        from .wizard import export as wizard

        wizard.run(profile_name=name)


def run(offline: bool = False) -> int:
    from .pins import PinsError
    from .proc import CommandFailed

    tui.info(tui.style("\npipertrainer — Piper TTS voice training", "bold"))
    tui.hint("one audio file in, one .onnx voice out")

    while True:
        prof = _load_active()
        effective_offline = offline or (prof.runtime.offline if prof else False)
        try:
            action = tui.menu(
                "Main menu", ITEMS, status=status_lines(prof), allow_back=False
            )
        except tui.Quit:
            print()
            return 0

        try:
            _dispatch(action, prof, effective_offline)
        except tui.Back:
            continue
        except tui.Quit:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            tui.warn("cancelled")
        except (profile_mod.ProfileError, PinsError, ValueError, OSError) as exc:
            tui.error(str(exc))
        except CommandFailed as exc:
            tui.error(str(exc))
