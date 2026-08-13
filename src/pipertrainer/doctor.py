"""Environment diagnostics.

Answers one question: can this machine train right now, and if not, what is the
first thing to fix. Ordered so the earliest failure is the one that matters —
there is no point reporting a Whisper problem when torch cannot see the GPU.

Exits non-zero when something is actually broken, so it can gate a script.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass

from . import env as env_mod
from . import install
from . import pins as pins_mod
from . import proc, profile as profile_mod, tui
from .paths import (
    PIPER_DIR,
    REPO_ROOT,
    TORCH_CONSTRAINT,
    VENV_DIR,
    WINDOWS,
    in_venv,
)

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

_MARKERS = {
    OK: (tui.glyph("ok"), "green"),
    WARN: (tui.glyph("warn"), "yellow"),
    FAIL: (tui.glyph("fail"), "red"),
    SKIP: (tui.glyph("skip"), "dim"),
}


@dataclass
class Check:
    status: str
    name: str
    detail: str
    fix: str = ""


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, status: str, name: str, detail: str, fix: str = "") -> Check:
        check = Check(status, name, detail, fix)
        self.checks.append(check)
        symbol, colour = _MARKERS[status]
        print(f"{tui.style(symbol, colour)} {name}: {detail}")
        if fix and status in (WARN, FAIL):
            print(tui.wrap(f"{tui.glyph('arrow')} {fix}", indent="    "))
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]


_IMPORT_PROBE = r"""
import json
report = {}
def check(name, fn):
    try:
        report[name] = [True, str(fn())]
    except Exception as exc:
        report[name] = [False, "%s: %s" % (type(exc).__name__, exc)]

check("piper", lambda: __import__("piper") and "importable")
check("piper.train", lambda: __import__("piper.train") and "importable")

def _align():
    from piper.train.vits.monotonic_align import maximum_path
    return "importable"
check("monotonic_align", _align)

def _espeak():
    from piper.phonemize_espeak import EspeakPhonemizer
    out = EspeakPhonemizer().phonemize("en-us", "Hello world.")
    flat = "".join("".join(s) for s in out)
    if not flat:
        raise RuntimeError("returned no phonemes")
    return flat[:32]
check("espeak", _espeak)

check("librosa", lambda: __import__("librosa").__version__)
check("lightning", lambda: __import__("lightning").__version__)
check("jsonargparse", lambda: __import__("jsonargparse").__version__)
check("whisper", lambda: __import__("whisper") and "importable")

def _vad():
    from pysilero_vad import SileroVoiceActivityDetector
    return "importable"
check("silero-vad", _vad)

def _onnx():
    import onnxruntime
    return onnxruntime.__version__
check("onnxruntime", _onnx)

def _onnxscript():
    # torch >= 2.9 imports onnxscript inside torch.onnx.export. Missing, it
    # fails only at export time, with a trained checkpoint already on disk.
    import torch.onnx, onnxscript
    return onnxscript.__version__
check("onnxscript", _onnxscript)
print(json.dumps(report))
"""


def _probe_imports() -> dict[str, list] | None:
    result = env_mod.python_snippet(_IMPORT_PROBE, timeout=600)
    for line in reversed(result.lines):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _espeak_voice_check(report: Report, voice: str) -> None:
    """Prove the configured espeak voice actually phonemizes.

    piper embeds espeak-ng rather than shelling out, so there is no `espeak-ng`
    binary to test with — the only honest check is through the phonemizer.
    """
    code = (
        "from piper.phonemize_espeak import EspeakPhonemizer\n"
        f"out = EspeakPhonemizer().phonemize({voice!r}, 'The quick brown fox.')\n"
        "flat = ''.join(''.join(s) for s in out)\n"
        "print('PHONEMES:' + flat)\n"
    )
    result = env_mod.python_snippet(code)
    line = next(
        (l for l in result.lines if l.startswith("PHONEMES:")), ""
    )
    phonemes = line[len("PHONEMES:") :]
    if result.ok and phonemes.strip():
        report.add(OK, f"espeak voice {voice!r}", phonemes[:40])
    else:
        report.add(
            FAIL,
            f"espeak voice {voice!r}",
            "produced no phonemes",
            "check voice.espeak_voice in the profile — it must be an espeak-ng "
            "voice name such as en-us, en-gb, de, fr. A locale tag like en_US "
            "is not valid here.",
        )


def run(offline: bool = False) -> int:
    report = Report()
    pins = pins_mod.load()
    state = env_mod.SetupState.load()

    supported = sys.platform.startswith(install.SUPPORTED_PLATFORMS)
    tui.heading("Machine")
    report.add(
        OK if supported else FAIL,
        "platform",
        platform.platform(),
        ""
        if supported
        else "this tool is developed against Linux and Windows",
    )
    report.add(
        OK if sys.version_info[:2] >= tuple(pins.python_min[:2]) else FAIL,
        "python",
        f"{sys.version.split()[0]} at {sys.executable}",
        f"need {'.'.join(str(p) for p in pins.python_min)}+",
    )
    free = env_mod.free_gib(REPO_ROOT)
    report.add(
        OK if free >= pins.run_free_gib else WARN,
        "disk",
        f"{free:.1f} GiB free at {REPO_ROOT}",
        f"a run keeps up to ~10 GB of checkpoints; {pins.run_free_gib} GiB is "
        f"the recommended floor. './run monitor --prune' reclaims space.",
    )

    tui.heading("Tools")
    for tool, why, fatal in (
        ("git", "cloning piper1-gpl", True),
        ("ffmpeg", "decoding and cutting audio", True),
        ("ffprobe", "reading audio metadata", True),
        ("cmake", "building espeak-ng (pip can supply a wheel)", False),
        ("ninja", "building espeak-ng (pip can supply a wheel)", False),
    ):
        found = shutil.which(tool)
        if found:
            report.add(OK, tool, found)
        else:
            report.add(
                FAIL if fatal else WARN,
                tool,
                "not found",
                f"needed for {why}; " + (
                    "install it with winget — see './run setup'"
                    if WINDOWS
                    else "install it with your package manager"
                ),
            )
    compiler = install.find_compiler()
    report.add(
        OK if compiler else FAIL,
        "c compiler",
        compiler or "none found",
        install.VS_BUILD_TOOLS
        if WINDOWS
        else "install base-devel (Arch) or build-essential (Debian)",
    )

    tui.heading("Installation")
    if not in_venv():
        report.add(
            FAIL,
            "venv",
            f"missing at {VENV_DIR}",
            "run './run setup'",
        )
        _finish(report)
        return 1
    report.add(OK, "venv", str(VENV_DIR))

    if PIPER_DIR.exists():
        head = proc.capture(["git", "rev-parse", "HEAD"], cwd=PIPER_DIR, timeout=60)
        sha = head.lines[0].strip() if head.ok and head.lines else "unknown"
        matches = sha == pins.piper_sha
        report.add(
            OK if matches else WARN,
            "piper1-gpl",
            f"{sha[:12]} (pins.toml wants {pins.piper_tag} / {pins.piper_sha[:12]})",
            "" if matches else (
                "the checkout does not match the pin. The flag mapping in "
                "train/argmap.py was verified against the pinned tag — re-run "
                "the checklist in docs/UPSTREAM_NOTES.md, or "
                "'./run setup --force-step clone'."
            ),
        )
        dirty = proc.capture(
            ["git", "status", "--porcelain"], cwd=PIPER_DIR, timeout=60
        )
        if dirty.ok and dirty.output.strip():
            report.add(
                WARN,
                "piper1-gpl tree",
                "has local modifications",
                "this tool never edits piper1-gpl; local patches make the "
                "verified upstream notes unreliable",
            )
    else:
        report.add(FAIL, "piper1-gpl", "not cloned", "run './run setup'")

    if TORCH_CONSTRAINT.exists():
        report.add(
            OK,
            "torch pin",
            TORCH_CONSTRAINT.read_text(encoding="utf-8").strip().splitlines()[-1],
        )
    else:
        report.add(
            WARN,
            "torch pin",
            "missing",
            "later pip installs could replace the vendor-specific torch build; "
            "run './run setup --force-step constraint'",
        )

    tui.heading("Python packages")
    imports = _probe_imports()
    if imports is None:
        report.add(
            FAIL,
            "imports",
            "the probe produced no output — the venv interpreter may be broken",
            "run './run setup --force-step verify'",
        )
    else:
        fatal_names = {"piper", "piper.train", "monotonic_align", "espeak", "librosa", "lightning"}
        for name, (ok, detail) in imports.items():
            if ok:
                report.add(OK, name, detail)
            else:
                report.add(
                    FAIL if name in fatal_names else WARN,
                    name,
                    detail,
                    _import_fix(name),
                )

    tui.heading("GPU")
    if state.vendor:
        tui.info(f"  configured vendor: {state.vendor}")
    detected, notes = env_mod.detect_vendor()
    for note in notes:
        tui.hint(f"  {note}")
    if state.vendor and detected not in ("ambiguous", state.vendor):
        report.add(
            WARN,
            "vendor",
            f"installed for {state.vendor} but this machine looks like {detected}",
            "moving the repo between machines needs "
            "'./run setup --force-step torch --force-step constraint'",
        )

    info = env_mod.verify_torch()
    if not info.ok:
        report.add(FAIL, "torch", info.error or "not importable", "run './run setup'")
    else:
        report.add(OK, "torch", info.summary())
        if info.vendor == "rocm" and env_mod.render_group_ok() is False:
            report.add(
                FAIL,
                "render group",
                "you are not in 'render' or 'video'",
                "sudo usermod -aG render,video $USER, then log out and back in. "
                "This is the most common cause of ROCm not seeing the GPU.",
            )
        # Asked independently of the matmul below, because the interesting case
        # passes matmul and fails everything else: GEMM comes from rocBLAS, so a
        # device missing from torch's arch list can multiply matrices all day
        # and still have no kernel for `a + b`.
        compiled = info.compiled_for_device
        if compiled is True:
            report.add(
                OK,
                "torch gfx kernels",
                f"built for {env_mod.normalise_gfx(info.gcn_arch)}",
            )
        elif compiled is False:
            report.add(
                FAIL,
                "torch gfx kernels",
                f"this torch has no {env_mod.normalise_gfx(info.gcn_arch)} "
                f"code (built for {', '.join(info.arch_list)})",
                "Every kernel launch outside rocBLAS raises 'invalid device "
                "function', which is what blocks training while leaving matmul "
                "working. " + env_mod.unsupported_arch_advice(info),
            )

        if state.vendor != "cpu" and not info.available:
            report.add(
                FAIL,
                "gpu visible",
                "torch reports no GPU",
                "see docs/GPU_SETUP.md",
            )
        elif info.available and not info.matmul_ok:
            report.add(
                FAIL,
                "gpu usable",
                f"a real matmul failed: {info.matmul_error or 'process aborted'}",
                "torch.cuda.is_available() lies on unsupported ROCm targets. "
                f"Device reports {info.gcn_arch or 'unknown'}; torch was built "
                f"for {', '.join(info.arch_list) or 'unknown'}. "
                + env_mod.unsupported_arch_advice(info),
            )
        elif info.available:
            report.add(
                OK,
                "gpu usable",
                f"matmul verified on {info.device_name} "
                f"({info.total_memory_gib:.1f} GiB)"
                + (f", HSA_OVERRIDE_GFX_VERSION={info.hsa_override}" if info.hsa_override else ""),
            )
            if env_mod.low_vram(info):
                report.add(
                    WARN,
                    "vram",
                    f"{info.total_memory_gib:.1f} GiB is tight for 22.05 kHz "
                    f"training",
                    f"start at batch_size "
                    f"{env_mod.recommended_batch_size(info)} and see the OOM "
                    f"ladder in docs/TROUBLESHOOTING.md",
                )

    _training_probe(report, info)

    tui.heading("Profiles")
    names = profile_mod.profile_names()
    if not names:
        report.add(
            SKIP, "profiles", "none yet", ""
        )
    else:
        active = profile_mod.get_active()
        report.add(
            OK, "profiles", ", ".join(f"{n}*" if n == active else n for n in names)
        )
        for name in names:
            try:
                prof, warnings = profile_mod.load_by_name(name)
            except profile_mod.ProfileError as exc:
                report.add(FAIL, f"profile {name}", str(exc), "fix or delete the file")
                continue
            if warnings:
                report.add(
                    WARN,
                    f"profile {name}",
                    "; ".join(warnings[:3]),
                    "run './run profile --refresh " + name + "' after fixing",
                )
            if name == active and imports and imports.get("espeak", [False])[0]:
                _espeak_voice_check(report, prof.voice.espeak_voice)

    if offline or os.environ.get("PT_OFFLINE"):
        tui.heading("Offline")
        report.add(OK, "mode", "offline — network checks skipped")

    return _finish(report)


def _training_probe(report: Report, info: env_mod.TorchInfo) -> None:
    """The measurement that actually predicts whether a run will survive.

    A matmul proves arithmetic. Only an autograd loop with allocation churn
    proves training, because the failures that stop a run — a missing
    elementwise or convolution kernel, a fault after tens of allocate/free
    cycles — cannot be reached by a matmul.
    """
    tui.heading("Training")
    if not info.usable_gpu:
        report.add(SKIP, "training probe", "no usable GPU to test")
        return

    tui.info("  running a real autograd loop (this takes a few seconds)...")
    probe = env_mod.probe_training(extra_env=env_mod.training_env())
    if probe.ok:
        report.add(OK, "training probe", probe.verdict)
        return

    report.add(
        FAIL,
        "training probe",
        probe.verdict,
        "The GPU can do arithmetic but cannot complete a training step. "
        "Usually this means the torch build has no kernels for this "
        "architecture. " + env_mod.unsupported_arch_advice(info),
    )


def _import_fix(name: str) -> str:
    if name in ("piper", "piper.train"):
        return "run './run setup --force-step piper_install'"
    if name == "monotonic_align":
        return (
            "the VITS alignment kernel did not build; run "
            "'./run setup --force-step monotonic_align'"
        )
    if name == "espeak":
        return (
            "the embedded espeak-ng did not build or the in-place extension is "
            "missing; run './run setup --force-step build_ext'"
        )
    if name == "whisper":
        return (
            "transcription will not work; run './run setup --force-step whisper'"
        )
    if name == "onnxruntime":
        return "export verification needs it; run './run setup --force-step piper_install'"
    if name == "onnxscript":
        return (
            "torch.onnx.export cannot load without it, so export fails after "
            "training; run './run setup --force-step export_deps'"
        )
    return "run './run setup'"


def _finish(report: Report) -> int:
    tui.heading("Summary")
    if report.failures:
        tui.error(f"{len(report.failures)} problem(s) must be fixed:")
        for check in report.failures:
            tui.bullet(f"{check.name}: {check.detail}")
        return 1
    if report.warnings:
        tui.warn(f"{len(report.warnings)} warning(s), but training should work")
        for check in report.warnings:
            tui.bullet(f"{check.name}: {check.detail}")
        return 0
    tui.ok("everything checks out")
    return 0
