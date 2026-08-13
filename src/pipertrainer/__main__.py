"""Command-line entry point.

``./run`` with no arguments opens the interactive menu; every menu action also
exists as a subcommand so the same workflow is scriptable and CI-able.

Imports of the heavier modules are deferred into each handler so that
``./run setup`` works on a fresh clone, before the venv exists and before
anything that needs torch or ffmpeg is importable.
"""

from __future__ import annotations

import argparse
import sys

from . import tui


def _add_profile_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="profile to use; defaults to the active one (see './run profile')",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./run",
        description="Train a Piper TTS voice from a single audio file.",
        epilog="Run with no arguments for an interactive menu.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no network: no downloads, and mos_metric forced to 'none'",
    )
    sub = parser.add_subparsers(dest="command")

    setup = sub.add_parser("setup", help="install the toolchain and piper1-gpl")
    setup.add_argument(
        "--vendor",
        choices=("rocm", "cuda", "cpu"),
        help="force the GPU vendor instead of autodetecting",
    )
    setup.add_argument(
        "--force-step",
        metavar="NAME",
        action="append",
        default=[],
        help="re-run a completed step (repeatable); 'all' redoes everything",
    )
    setup.add_argument(
        "--yes",
        action="store_true",
        help="assume yes for the system-package prompt (still prints the command)",
    )
    setup.add_argument(
        "--torch-index",
        metavar="URL",
        help="override the torch index URL from pins.toml",
    )
    setup.add_argument(
        "--torch-spec",
        metavar="SPEC",
        help=(
            "override the torch requirement from pins.toml, e.g. torch==2.6.0; "
            "a path to a local .whl is installed directly, with no index"
        ),
    )

    sub.add_parser("doctor", help="diagnose the environment and report problems")

    prof = sub.add_parser("profile", help="create, list, select and refresh profiles")
    prof.add_argument("--new", metavar="NAME", help="create a profile from defaults")
    prof.add_argument("--list", action="store_true", help="list profiles")
    prof.add_argument("--select", metavar="NAME", help="make a profile the active one")
    prof.add_argument(
        "--refresh",
        metavar="NAME",
        help="rewrite a profile, regenerating the comments from the schema",
    )
    prof.add_argument("--show", metavar="NAME", help="print a profile's resolved values")

    dataset = sub.add_parser(
        "dataset", help="build wavs/ and metadata.csv from one audio file"
    )
    _add_profile_flag(dataset)
    dataset.add_argument("--input", metavar="FILE", help="source audio or video file")
    dataset.add_argument(
        "--force-stage",
        metavar="STAGE",
        action="append",
        default=[],
        help="re-run a cached stage: decode, macrosplit, transcribe, segment, emit",
    )
    dataset.add_argument(
        "--dry-run",
        action="store_true",
        help="analyse and report without writing any WAV files",
    )
    dataset.add_argument(
        "--non-interactive",
        action="store_true",
        help="use the profile as-is instead of walking the wizard",
    )

    train = sub.add_parser("train", help="train (or fine-tune) a voice")
    _add_profile_flag(train)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="write lightning.yaml, run every preflight check, then stop",
    )
    train.add_argument(
        "--non-interactive",
        action="store_true",
        help="use the profile as-is instead of walking the wizard",
    )
    train.add_argument(
        "--extra",
        metavar="ARG",
        action="append",
        default=[],
        help="extra argv passed straight to 'piper.train fit' (repeatable)",
    )

    resume = sub.add_parser("resume", help="continue from the newest last.ckpt")
    _add_profile_flag(resume)
    resume.add_argument(
        "--checkpoint", metavar="CKPT", help="resume from a specific checkpoint"
    )
    resume.add_argument(
        "--force",
        action="store_true",
        help="resume even if the profile no longer matches the saved lightning.yaml",
    )

    monitor = sub.add_parser("monitor", help="TensorBoard, metrics and checkpoints")
    _add_profile_flag(monitor)
    monitor.add_argument("--port", type=int, default=6006)
    monitor.add_argument(
        "--no-browser",
        action="store_true",
        help="print a metric table and checkpoint leaderboard instead of serving",
    )
    monitor.add_argument(
        "--prune",
        action="store_true",
        help="interactively delete checkpoints to reclaim disk",
    )

    preview = sub.add_parser("preview", help="synthesize from a checkpoint (no export)")
    _add_profile_flag(preview)
    preview.add_argument("--checkpoint", metavar="CKPT")
    preview.add_argument(
        "--text",
        action="append",
        default=[],
        help="sentence to synthesize (repeatable); omit for the default set",
    )
    preview.add_argument("--noise-scale", type=float)
    preview.add_argument("--length-scale", type=float)
    preview.add_argument("--noise-w", type=float)

    export = sub.add_parser("export", help="export ONNX plus its voice config")
    _add_profile_flag(export)
    export.add_argument("--checkpoint", metavar="CKPT")
    export.add_argument(
        "--best",
        choices=("val_mel", "val_mos", "last"),
        default="val_mel",
        help="pick the checkpoint automatically by this metric",
    )
    export.add_argument(
        "--no-verify",
        action="store_true",
        help="skip synthesizing a test sentence from the exported voice",
    )

    ckpt = sub.add_parser("checkpoints", help="browse and download pretrained models")
    _add_profile_flag(ckpt)
    ckpt.add_argument(
        "--list", metavar="PATH", nargs="?", const="", help="list a HuggingFace subtree"
    )
    ckpt.add_argument("--download", metavar="PATH", help="download one checkpoint")
    ckpt.add_argument(
        "--local", action="store_true", help="list already-downloaded checkpoints"
    )

    smoke = sub.add_parser(
        "smoke", help="end-to-end CPU self-test on a synthetic dataset"
    )
    smoke.add_argument(
        "--keep", action="store_true", help="leave the _smoke voice in place afterwards"
    )
    smoke.add_argument(
        "--stage",
        choices=("unit", "dataset", "train", "export", "all"),
        default="all",
    )

    return parser


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def _cmd_setup(args: argparse.Namespace) -> int:
    from . import install

    return install.run_setup(
        vendor=args.vendor,
        offline=args.offline,
        force_steps=args.force_step,
        assume_yes=args.yes,
        torch_index=args.torch_index,
        torch_spec=args.torch_spec,
    )


def _cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    return doctor.run(offline=args.offline)


def _cmd_profile(args: argparse.Namespace) -> int:
    from . import profile as profile_mod

    if args.new:
        prof = profile_mod.Profile()
        prof.voice.name = args.new
        path = profile_mod.save(prof)
        profile_mod.set_active(args.new)
        tui.ok(f"created {path} (now active)")
        return 0
    if args.select:
        profile_mod.load_by_name(args.select)  # validates it parses
        profile_mod.set_active(args.select)
        tui.ok(f"active profile: {args.select}")
        return 0
    if args.refresh:
        prof, warnings = profile_mod.load_by_name(args.refresh)
        for warning in warnings:
            tui.warn(warning)
        path = profile_mod.save(prof)
        tui.ok(f"rewrote {path} with current schema comments")
        return 0
    if args.show:
        prof, warnings = profile_mod.load_by_name(args.show)
        for warning in warnings:
            tui.warn(warning)
        for path, _, value in profile_mod.iter_specs(prof):
            print(f"{path} = {value!r}")
        return 0

    names = profile_mod.profile_names()
    active = profile_mod.get_active()
    if not names:
        tui.info("no profiles yet — create one with: ./run profile --new myvoice")
        return 0
    for name in names:
        marker = "*" if name == active else " "
        print(f" {marker} {name}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from .wizard import dataset as wizard

    return wizard.run(
        profile_name=args.profile,
        input_path=args.input,
        force_stages=args.force_stage,
        dry_run=args.dry_run,
        interactive=not args.non_interactive,
        offline=args.offline,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    from .wizard import train as wizard

    return wizard.run(
        profile_name=args.profile,
        dry_run=args.dry_run,
        interactive=not args.non_interactive,
        offline=args.offline,
        extra_argv=args.extra,
    )


def _cmd_resume(args: argparse.Namespace) -> int:
    from .train import launch

    return launch.resume(
        profile_name=args.profile,
        checkpoint=args.checkpoint,
        force=args.force,
        offline=args.offline,
    )


def _cmd_monitor(args: argparse.Namespace) -> int:
    from .train import monitor

    if args.prune:
        return monitor.prune(profile_name=args.profile)
    if args.no_browser:
        return monitor.report(profile_name=args.profile)
    return monitor.serve(profile_name=args.profile, port=args.port)


def _cmd_preview(args: argparse.Namespace) -> int:
    from .train import preview

    return preview.run(
        profile_name=args.profile,
        checkpoint=args.checkpoint,
        sentences=args.text or None,
        noise_scale=args.noise_scale,
        length_scale=args.length_scale,
        noise_w=args.noise_w,
        offline=args.offline,
    )


def _cmd_export(args: argparse.Namespace) -> int:
    from .wizard import export as wizard

    return wizard.run(
        profile_name=args.profile,
        checkpoint=args.checkpoint,
        best=args.best,
        verify=not args.no_verify,
    )


def _cmd_checkpoints(args: argparse.Namespace) -> int:
    from . import checkpoints

    if args.local:
        return checkpoints.list_local()
    if args.download:
        path = checkpoints.download(args.download, offline=args.offline)
        tui.ok(f"downloaded {path}")
        return 0
    if args.list is not None:
        return checkpoints.list_remote(args.list, offline=args.offline)
    return checkpoints.browse(profile_name=args.profile, offline=args.offline)


def _cmd_smoke(args: argparse.Namespace) -> int:
    from . import smoke

    return smoke.run(stage=args.stage, keep=args.keep)


HANDLERS = {
    "setup": _cmd_setup,
    "doctor": _cmd_doctor,
    "profile": _cmd_profile,
    "dataset": _cmd_dataset,
    "train": _cmd_train,
    "resume": _cmd_resume,
    "monitor": _cmd_monitor,
    "preview": _cmd_preview,
    "export": _cmd_export,
    "checkpoints": _cmd_checkpoints,
    "smoke": _cmd_smoke,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from . import env as env_mod
    from . import paths

    problem = paths.env_name_problem()
    if problem:
        raise ValueError(problem)

    # Environment that setup discovered for this machine (e.g. a working
    # HSA_OVERRIDE_GFX_VERSION). Applied here rather than by the entry-point
    # script, because a shell fragment only one of the two platforms can source
    # is not a place to keep facts both of them need.
    env_mod.apply_persisted_env()

    if not args.command:
        from . import menu

        return menu.run(offline=args.offline)

    return HANDLERS[args.command](args)


def _entry() -> int:
    from .pins import PinsError
    from .proc import CommandFailed
    from .profile import ProfileError

    try:
        return main()
    except tui.Quit:
        print()
        return 130
    except KeyboardInterrupt:
        print()
        tui.warn("interrupted")
        return 130
    except (ProfileError, PinsError, ValueError) as exc:
        tui.error(str(exc))
        return 2
    except CommandFailed as exc:
        tui.error(str(exc))
        return exc.result.returncode or 1
    except FileNotFoundError as exc:
        tui.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(_entry())
