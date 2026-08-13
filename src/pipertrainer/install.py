"""Setup: an ordered, idempotent, resumable state machine.

Order is the whole point. The vendor-specific torch wheel must be installed
*before* ``pip install -e './piper1-gpl[train]'``, or pip resolves ``torch>=2,<3``
against PyPI and quietly installs the CUDA build on an AMD box. And every pip
call after that point runs under ``PIP_CONSTRAINT`` pinning the exact local
version, because ``openai-whisper``'s metadata asks for an unpinned torch plus
CUDA-flavoured triton and will otherwise replace it.

Each step records completion in ``.state/setup.json`` so an interrupted setup
resumes instead of rebuilding espeak-ng from source again. ``--force-step NAME``
re-runs one; ``--force-step all`` redoes everything.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import env as env_mod
from . import pins as pins_mod
from . import proc, tui
from .paths import (
    ENV_JSON,
    PIPER_DIR,
    REPO_ROOT,
    STATE_DIR,
    TORCH_CONSTRAINT,
    VENV_DIR,
    venv_python,
)


class SetupError(RuntimeError):
    pass


# Package manager -> the packages piper1-gpl's build actually needs. Upstream
# documents build-essential/cmake/ninja-build for apt; these are the
# equivalents. espeak-ng is deliberately absent: piper builds it from source as
# a CMake ExternalProject and embeds it, so no system package is involved.
PACKAGE_SETS: dict[str, list[str]] = {
    "pacman": ["base-devel", "cmake", "ninja", "git", "python", "ffmpeg"],
    "apt-get": [
        "build-essential",
        "cmake",
        "ninja-build",
        "git",
        "python3-venv",
        "python3-dev",
        "ffmpeg",
    ],
    "dnf": [
        "gcc",
        "gcc-c++",
        "make",
        "cmake",
        "ninja-build",
        "git",
        "python3-devel",
        "ffmpeg",
    ],
    "zypper": [
        "gcc",
        "gcc-c++",
        "make",
        "cmake",
        "ninja",
        "git",
        "python3-devel",
        "ffmpeg",
    ],
}

INSTALL_FLAGS: dict[str, list[str]] = {
    "pacman": ["-S", "--needed", "--noconfirm"],
    "apt-get": ["install", "-y"],
    "dnf": ["install", "-y"],
    "zypper": ["install", "-y"],
}

# Windows equivalents: (winget id, the executable it puts on PATH, why).
# The executable is listed rather than derived from the id so that a package
# being renamed upstream cannot silently turn into a check for the wrong file.
WINGET_PACKAGES: list[tuple[str, str, str]] = [
    ("Git.Git", "git", "cloning piper1-gpl"),
    ("Gyan.FFmpeg", "ffmpeg", "decoding and cutting audio"),
    ("Kitware.CMake", "cmake", "building espeak-ng (3.26+)"),
    ("Ninja-build.Ninja", "ninja", "the CMake generator"),
]
VS_BUILD_TOOLS = (
    'winget install --id Microsoft.VisualStudio.2022.BuildTools '
    '--override "--quiet --wait --add '
    'Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
)

COMPILER_LABEL = (
    "the MSVC C++ build tools" if sys.platform == "win32"
    else "a C compiler (cc/gcc/clang)"
)

#: Where the Visual Studio installer records what is installed. Asking vswhere
#: is the supported way to find MSVC; looking for ``cl.exe`` on PATH only works
#: inside a Developer Command Prompt, while CMake locates the toolchain by
#: itself from an ordinary shell.
_VSWHERE = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio"
    / "Installer"
    / "vswhere.exe"
)
_VC_TOOLS = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"


def find_compiler() -> str | None:
    """A usable C/C++ toolchain, described for the user, or None."""
    if sys.platform != "win32":
        return next((n for n in ("cc", "gcc", "clang") if shutil.which(n)), None)

    on_path = shutil.which("cl")
    if on_path:
        return on_path
    if not _VSWHERE.is_file():
        return None
    result = proc.capture(
        [
            str(_VSWHERE),
            "-latest",
            "-products", "*",
            "-requires", _VC_TOOLS,
            "-property", "installationPath",
        ],
        timeout=60,
    )
    if not result.ok or not result.lines:
        return None
    path = result.lines[-1].strip()
    return f"MSVC at {path}" if path else None

NETWORK_HOSTS = [
    ("github.com", 443),
    ("pypi.org", 443),
    ("download.pytorch.org", 443),
    ("huggingface.co", 443),
]


@dataclass
class Context:
    pins: pins_mod.Pins
    state: env_mod.SetupState
    vendor: str = "cpu"
    offline: bool = False
    assume_yes: bool = False
    #: Argv of the interpreter used to create the venv. A list because Windows
    #: selects a version with `py -3.13`.
    bootstrap_python: list[str] = field(default_factory=lambda: [sys.executable])
    torch_index: str = ""
    torch_spec: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return STATE_DIR / "setup.log"

    def pip(self, *args: str, offline_ok: bool = False) -> proc.Result:
        if self.offline and not offline_ok:
            raise SetupError(
                "offline mode: this step needs the network. See the staging "
                "instructions printed at the end of './run setup --offline'."
            )
        return proc.run(
            [venv_python(), "-m", "pip", *args],
            env=env_mod.pip_env(),
            log_path=self.log_path,
        )

    def note(self, text: str) -> None:
        self.notes.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        tui.warn(text)


@dataclass
class Step:
    name: str
    title: str
    action: Callable[[Context], None]


# --------------------------------------------------------------------------
# Step 0: preflight
# --------------------------------------------------------------------------


def _reachable(host: str, port: int, timeout: float = 6.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _version_of(argv: Sequence[str]) -> tuple[int, ...] | None:
    result = proc.capture(
        [*argv, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        timeout=30,
    )
    if not result.ok or not result.lines:
        return None
    try:
        return tuple(int(p) for p in result.lines[-1].strip().split("."))
    except ValueError:
        return None


def _pick_exact_python(ctx: Context, wanted: str) -> list[str]:
    """Find one specific ``major.minor``, because only it has a wheel.

    The native-Windows ROCm build is published as cp312 and nothing else, so
    "newest available" is the wrong rule there — a 3.13 venv would fail at the
    torch step with a bare "no matching distribution", which names neither the
    cause nor the fix.
    """
    target = tuple(int(p) for p in wanted.split("."))
    candidates: list[list[str]] = [[sys.executable], *ctx.pins.python_prefer]
    if sys.platform == "win32":
        candidates.insert(1, ["py", f"-{wanted}"])
    else:
        candidates.insert(1, [f"python{wanted}"])

    for candidate in candidates:
        found = shutil.which(candidate[0])
        if not found:
            continue
        argv = [found, *candidate[1:]]
        if _version_of(argv) == target:
            return argv

    how = (
        f"winget install Python.Python.{wanted}"
        if sys.platform == "win32"
        else f"install python{wanted} with your package manager"
    )
    raise SetupError(
        f"the {ctx.vendor} torch build for this platform is published for "
        f"Python {wanted} only, and no Python {wanted} was found.\n"
        f"  {how}\n"
        f"Then re-run setup. If you would rather not install it, "
        f"'./run setup --vendor cpu' works on any supported Python."
    )


def _pick_python(ctx: Context) -> list[str]:
    """Newest supported interpreter available, preferring the running one.

    Returns an argv, not a path: on Windows a specific version is selected as
    ``py -3.13``, which is two arguments and cannot be collapsed to one.
    """
    exact = ctx.pins.torch(ctx.vendor).requires_python
    if exact:
        return _pick_exact_python(ctx, exact)

    minimum = ctx.pins.python_min
    if sys.version_info[: len(minimum)] >= minimum:
        # Running under the entry point with a good interpreter already.
        candidate_version = ".".join(str(p) for p in sys.version_info[:2])
        if sys.version_info[:2] <= (3, 13):
            return [sys.executable]
        ctx.warn(
            f"running on Python {candidate_version}; numba (needed by "
            f"openai-whisper) often lags new releases. Looking for 3.13 or older."
        )
    for candidate in ctx.pins.python_prefer:
        found = shutil.which(candidate[0])
        if not found:
            continue
        argv = [found, *candidate[1:]]
        parts = _version_of(argv)
        if parts is not None and parts >= minimum:
            return argv
    if sys.version_info[: len(minimum)] >= minimum:
        return [sys.executable]
    tried = ", ".join(" ".join(c) for c in ctx.pins.python_prefer)
    raise SetupError(
        f"need Python {'.'.join(str(p) for p in minimum)} or newer; none found. "
        f"Tried: {tried}"
    )


SUPPORTED_PLATFORMS = ("linux", "win32", "darwin")


def step_preflight(ctx: Context) -> None:
    if not sys.platform.startswith(SUPPORTED_PLATFORMS):
        raise SetupError(
            f"unsupported platform {sys.platform!r}. This tool is developed "
            f"against Linux and Windows."
        )

    ctx.bootstrap_python = _pick_python(ctx)
    version = proc.capture([*ctx.bootstrap_python, "--version"], timeout=30)
    tui.ok(
        f"python: {proc.describe(ctx.bootstrap_python)} "
        f"({version.output.strip()})"
    )
    tui.ok(f"machine: {platform.platform()}")

    required = {
        "git": "cloning piper1-gpl",
        "ffmpeg": "decoding and cutting audio",
        "ffprobe": "reading audio metadata",
    }
    missing = [name for name in required if not shutil.which(name)]
    compiler = find_compiler()
    if compiler is None:
        missing.append(COMPILER_LABEL)
    else:
        tui.ok(f"compiler: {compiler}")

    if missing:
        tui.warn("missing tools: " + ", ".join(missing))
        ctx.note("install the system packages in step 'system_deps'")
    else:
        tui.ok("git, ffmpeg and ffprobe are present")

    # cmake and ninja come from pip's isolated build environment as wheels, so
    # they are recommended rather than required.
    for optional in ("cmake", "ninja"):
        if not shutil.which(optional):
            ctx.note(
                f"{optional} is not on PATH; pip will fetch a wheel for the "
                f"build (piper needs cmake 3.26+)"
            )

    free = env_mod.free_gib(REPO_ROOT)
    needed = ctx.pins.setup_free_gib
    if free < needed:
        raise SetupError(
            f"only {free:.1f} GiB free at {REPO_ROOT}; need about {needed} GiB "
            f"(torch unpacks to ~4 GB, one pretrained checkpoint is ~0.9 GB, "
            f"and the utterance cache is a few times the dataset size)"
        )
    tui.ok(f"disk: {free:.1f} GiB free")

    if ctx.offline:
        tui.warn("offline mode: skipping network checks")
        return
    unreachable = [
        host for host, port in NETWORK_HOSTS if not _reachable(host, port)
    ]
    if unreachable:
        raise SetupError(
            "cannot reach: "
            + ", ".join(unreachable)
            + ". Setup needs github.com (piper1-gpl and espeak-ng), pypi.org "
            "and download.pytorch.org. Use --offline with pre-staged files if "
            "this machine has no network."
        )
    tui.ok("network: github, pypi, download.pytorch.org, huggingface reachable")


# --------------------------------------------------------------------------
# Step 1: system packages
# --------------------------------------------------------------------------


def detect_package_manager() -> str | None:
    for manager in ("pacman", "apt-get", "dnf", "zypper"):
        if shutil.which(manager):
            return manager
    return None


def step_system_deps(ctx: Context) -> None:
    if sys.platform == "win32":
        _windows_system_deps(ctx)
        return

    manager = detect_package_manager()
    if manager is None:
        tui.warn(
            "no known package manager found. Install a C/C++ toolchain, cmake "
            "3.26+, ninja, git and ffmpeg yourself, then re-run."
        )
        return

    packages = PACKAGE_SETS[manager]
    command = ["sudo", manager, *INSTALL_FLAGS[manager], *packages]
    tui.info("system packages needed for the piper build:")
    tui.info("  " + proc.describe(command))

    if os.geteuid() == 0:
        command = command[1:]  # already root

    if not ctx.assume_yes:
        if not tui.confirm("run this now?", default=True):
            tui.warn("skipped — re-run setup after installing them yourself")
            ctx.note("system packages were not installed by this tool")
            return

    proc.run(command, log_path=ctx.log_path, check=False)
    tui.ok("system packages step finished")


def _windows_system_deps(ctx: Context) -> None:
    """Install what the piper build needs on Windows, via winget.

    The MSVC build tools are deliberately never installed automatically: the
    download is several gigabytes, it wants a reboot often enough to matter,
    and getting the workload selection wrong leaves a Visual Studio that looks
    installed and cannot compile. Printing the exact command is more use than
    running the wrong one.
    """
    missing = [
        (package, why)
        for package, executable, why in WINGET_PACKAGES
        if not shutil.which(executable)
    ]

    if find_compiler() is None:
        tui.warn("no MSVC C++ toolchain found")
        tui.info(
            "  piper builds espeak-ng and two C extensions from source, so it "
            "needs the Visual Studio C++ build tools. Install them with:"
        )
        tui.info("")
        tui.info(f"    {VS_BUILD_TOOLS}")
        tui.info("")
        tui.hint(
            "  the --override is what selects the C++ workload; without it "
            "winget installs the launcher and no compiler. Re-run setup after."
        )
        ctx.note("the MSVC build tools were not installed by this tool")
    else:
        tui.ok(f"compiler: {find_compiler()}")

    if not missing:
        tui.ok("git, ffmpeg, cmake and ninja are present")
        return

    if not shutil.which("winget"):
        tui.warn(
            "winget is not available. Install these yourself and re-run: "
            + ", ".join(package for package, _ in missing)
        )
        ctx.note("system packages were not installed by this tool")
        return

    for package, why in missing:
        command = ["winget", "install", "--id", package, "-e", "--silent",
                   "--accept-package-agreements", "--accept-source-agreements"]
        tui.info(f"{package} — {why}")
        tui.info("  " + proc.describe(command))
        if not ctx.assume_yes and not tui.confirm("install it now?", default=True):
            ctx.note(f"{package} was not installed by this tool")
            continue
        proc.run(command, log_path=ctx.log_path, check=False)

    tui.ok("system packages step finished")
    tui.hint(
        "  winget updates PATH for new shells only — if a tool still looks "
        "missing, open a new terminal and re-run setup."
    )


# --------------------------------------------------------------------------
# Step 2: virtualenv
# --------------------------------------------------------------------------


def step_venv(ctx: Context) -> None:
    if venv_python().exists():
        tui.ok(f"venv already exists at {VENV_DIR}")
    else:
        proc.run(
            [*ctx.bootstrap_python, "-m", "venv", str(VENV_DIR)],
            log_path=ctx.log_path,
        )
        tui.ok(f"created {VENV_DIR}")
    # Deliberately not under PIP_CONSTRAINT: there is no torch yet.
    proc.run(
        [venv_python(), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        log_path=ctx.log_path,
        env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )


# --------------------------------------------------------------------------
# Step 3: vendor-specific torch
# --------------------------------------------------------------------------


def local_wheel(spec: str) -> Path | None:
    """A ``--torch-spec`` naming a wheel on disk, or None for a requirement.

    Not every torch comes from an index. A card with no published wheel needs
    one built locally, and pointing ``--torch-spec`` at it has to install *that
    file* rather than resolve its name against a package index that has never
    heard of it.

    Deliberately narrow: only an existing path ending in ``.whl``. Anything
    else stays a requirement string, so ``torch==2.6.0`` cannot be mistaken for
    a filename.
    """
    if not spec or not spec.endswith(".whl"):
        return None
    path = Path(spec).expanduser()
    return path if path.is_file() else None


def step_torch(ctx: Context) -> None:
    pin = ctx.pins.torch(ctx.vendor)

    # A pin delivered as wheel URLs has no index and no requirement to resolve,
    # so the index/spec path below cannot express it. An explicit --torch-index
    # or --torch-spec still wins: that is the escape hatch for trying something
    # the pins do not know about.
    if pin.from_urls and not (ctx.torch_index or ctx.torch_spec):
        _install_torch_from_urls(ctx, pin)
        return

    index = ctx.torch_index or pin.index
    spec = ctx.torch_spec or pin.spec

    wheel = local_wheel(spec)
    if wheel is not None:
        # No --index-url: there is no index involved, and torch's own
        # dependencies (filelock, sympy, jinja2 ...) come from PyPI as usual.
        tui.info(f"installing {wheel.name} for {ctx.vendor} from {wheel.parent}")
        tui.hint(
            "  a locally built wheel, so nothing is resolved against an index. "
            "PIP_CONSTRAINT still pins whatever version it turns out to be for "
            "every later step."
        )
        ctx.pip("install", str(wheel.resolve()))
        index = "local wheel"
        spec = str(wheel.resolve())
    elif spec.endswith(".whl"):
        raise SetupError(
            f"--torch-spec looks like a wheel path, but there is no file at "
            f"{spec!r}. Pass an existing .whl, or a requirement such as "
            f"'torch==2.9.1'."
        )
    else:
        tui.info(f"installing {spec} for {ctx.vendor} from {index}")
        tui.hint(
            "  --index-url replaces PyPI entirely for this step. That is what "
            "stops a CUDA wheel being resolved on an AMD box; the PyTorch index "
            "mirrors torch's own dependencies, so nothing is missing."
        )
        ctx.pip("install", "--index-url", index, spec)

    ctx.state.vendor = ctx.vendor
    ctx.state.torch_index = index
    ctx.state.torch_spec = spec
    ctx.state.save()


def _install_torch_from_urls(ctx: Context, pin: pins_mod.TorchPin) -> None:
    """Install a torch published as wheel URLs rather than through an index.

    This is how AMD ships ROCm for native Windows. The runtime wheels go first
    and in their own pip call: torch links against them, and letting pip
    resolve one set is a smaller surface than letting it resolve both at once.
    """
    if pin.driver:
        tui.info(f"  this build needs the AMD {pin.driver} graphics driver or newer")
    if pin.docs:
        tui.hint(f"  {pin.docs}")

    if pin.prerequisites:
        tui.info(f"installing the ROCm runtime ({len(pin.prerequisites)} packages)")
        tui.hint(
            "  the runtime ships as wheels of its own and must be present "
            "before torch, which links against it."
        )
        ctx.pip("install", "--no-cache-dir", *pin.prerequisites)

    tui.info(f"installing {pin.describe}")
    tui.hint(
        "  wheel URLs, not an index: download.pytorch.org publishes no ROCm "
        "build for Windows, so there is nothing to point --index-url at. "
        "PIP_CONSTRAINT still pins the result for every later step."
    )
    ctx.pip("install", "--no-cache-dir", *pin.wheels)

    ctx.state.vendor = ctx.vendor
    ctx.state.torch_index = pin.wheels[0].rsplit("/", 1)[0]
    ctx.state.torch_spec = pin.wheels[0].rsplit("/", 1)[-1]
    ctx.state.save()


# --------------------------------------------------------------------------
# Step 4: freeze the torch pin
# --------------------------------------------------------------------------


def _installed_torch_version() -> str:
    result = env_mod.python_snippet("import torch; print(torch.__version__)")
    if not result.ok or not result.lines:
        raise SetupError(
            "torch is not importable after installation:\n"
            + "\n".join(result.lines[-10:])
        )
    return result.lines[-1].strip()


#: Pinned alongside torch when present. They are not piper dependencies, but a
#: vendor build installs the three as a tested set, and a later pip call that
#: replaced one of them would pull a torch to match it.
TORCH_FAMILY = ("torchvision", "torchaudio")

_FAMILY_PROBE = r"""
import json
from importlib.metadata import PackageNotFoundError, version
found = {}
for name in %r:
    try:
        found[name] = version(name)
    except PackageNotFoundError:
        pass
print(json.dumps(found))
"""


def _installed_family() -> dict[str, str]:
    import json

    result = env_mod.python_snippet(_FAMILY_PROBE % (TORCH_FAMILY,))
    for line in reversed(result.lines):
        if line.strip().startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def step_constraint(ctx: Context) -> None:
    version = _installed_torch_version()
    pins_written = {"torch": version, **_installed_family()}

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_CONSTRAINT.write_text(
        "# Written by ./run setup. Passed as PIP_CONSTRAINT to every later pip\n"
        "# call so that nothing can replace the vendor-specific torch build.\n"
        + "".join(f"{name}=={value}\n" for name, value in pins_written.items()),
        encoding="utf-8",
        newline="\n",
    )
    ctx.state.torch_local_version = version
    ctx.state.save()
    tui.ok(
        "pinned "
        + ", ".join(f"{n}=={v}" for n, v in pins_written.items())
        + " for all later pip calls"
    )


def assert_torch_unchanged(ctx: Context, after: str) -> None:
    expected = ctx.state.torch_local_version
    if not expected:
        return
    actual = _installed_torch_version()
    if actual != expected:
        raise SetupError(
            f"torch changed from {expected} to {actual} during '{after}'. "
            f"That usually means a dependency pulled a different build — on "
            f"AMD it silently costs you the GPU. Re-run "
            f"'./run setup --force-step torch --force-step {after}'."
        )


# --------------------------------------------------------------------------
# Step 5: clone piper1-gpl at the pinned tag
# --------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> proc.Result:
    return proc.capture(["git", *args], cwd=cwd, check=check, timeout=600)


def step_clone(ctx: Context) -> None:
    tag = ctx.pins.piper_tag
    want_sha = ctx.pins.piper_sha

    if PIPER_DIR.exists():
        head = _git("rev-parse", "HEAD", cwd=PIPER_DIR, check=False)
        current = head.lines[0].strip() if head.ok and head.lines else ""
        if current == want_sha:
            tui.ok(f"piper1-gpl already at {tag} ({want_sha[:12]})")
            return
        dirty = _git("status", "--porcelain", cwd=PIPER_DIR, check=False)
        if dirty.ok and dirty.output.strip():
            raise SetupError(
                f"{PIPER_DIR} has local modifications and is not at {tag}. This "
                f"tool never edits piper1-gpl — workarounds live on our side. "
                f"Move or delete the directory, or check out {tag} yourself."
            )
        if ctx.offline:
            raise SetupError(
                f"offline: {PIPER_DIR} is at {current[:12] or 'unknown'} but "
                f"{tag} ({want_sha[:12]}) is pinned"
            )
        tui.info(f"updating piper1-gpl to {tag}")
        proc.run(
            ["git", "fetch", "--depth", "1", "origin", "tag", tag],
            cwd=PIPER_DIR,
            log_path=ctx.log_path,
        )
        proc.run(["git", "checkout", "-q", tag], cwd=PIPER_DIR, log_path=ctx.log_path)
    else:
        if ctx.offline:
            raise SetupError(
                f"offline: {PIPER_DIR} does not exist. Stage a clone of "
                f"{ctx.pins.piper_repo} at {tag} there first."
            )
        proc.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                ctx.pins.piper_repo,
                str(PIPER_DIR),
            ],
            log_path=ctx.log_path,
        )

    head = _git("rev-parse", "HEAD", cwd=PIPER_DIR)
    actual = head.lines[0].strip()
    if actual != want_sha:
        ctx.warn(
            f"piper1-gpl HEAD is {actual[:12]} but pins.toml records "
            f"{want_sha[:12]} for {tag}. The tag may have moved upstream; "
            f"re-verify docs/UPSTREAM_NOTES.md before trusting the flag mapping."
        )
    ctx.state.piper_tag = tag
    ctx.state.piper_sha = actual
    ctx.state.save()
    tui.ok(f"piper1-gpl at {tag} ({actual[:12]})")


# --------------------------------------------------------------------------
# Step 6: editable install of piper1-gpl[train]
# --------------------------------------------------------------------------


def step_piper_install(ctx: Context) -> None:
    tui.info("installing piper1-gpl[train] (editable)")
    tui.hint(
        "  This is the long step: CMake builds espeak-ng from source "
        f"(pinned {ctx.pins.espeak_ng_tag}) and embeds it, so it needs a "
        "toolchain, cmake 3.26+, and network access."
    )
    try:
        ctx.pip("install", "-e", f"{PIPER_DIR}[train]")
    except proc.CommandFailed as exc:
        _explain_build_failure(exc.result)
        raise
    assert_torch_unchanged(ctx, "piper_install")


def _explain_build_failure(result: proc.Result) -> None:
    text = result.output.lower()
    if "no module named 'skbuild'" in text:
        # The bare traceback sends people to their distro package manager,
        # which cannot help: 'skbuild' is the *import* name of the PyPI
        # package 'scikit-build', and a venv cannot see system site-packages.
        tui.error(
            "piper1-gpl's setup.py needs scikit-build, which must be installed "
            "into .venv — 'skbuild' is the import name of the PyPI package "
            "'scikit-build', so no distro package provides it. Re-run "
            "'./run setup --force-step build_ext'."
        )
    elif "cmake" in text and ("version" in text or "3.26" in text):
        tui.error(
            "the espeak-ng build needs CMake 3.26 or newer. Install a newer "
            "cmake, or let pip fetch one by removing the system cmake from PATH."
        )
    elif "could not resolve host" in text or "network" in text:
        tui.error(
            "the build could not reach github.com to fetch espeak-ng. Check "
            "your network or proxy settings (pip's build runs in isolation and "
            "does not inherit every proxy variable)."
        )
    elif "microsoft visual c++" in text or "cl.exe" in text or "msvc" in text:
        tui.error(
            "the MSVC C++ toolchain is missing or incomplete. Install the "
            "Visual Studio Build Tools with the C++ workload:\n"
            f"    {VS_BUILD_TOOLS}"
        )
    elif "no such file or directory: 'cc'" in text or "compiler" in text:
        tui.error(
            f"no working C/C++ compiler ({COMPILER_LABEL}). Re-run the "
            f"system_deps step, or install base-devel (Arch) / build-essential "
            f"(Debian)."
        )
    tui.hint("  see docs/TROUBLESHOOTING.md for the full list of build failures")


# --------------------------------------------------------------------------
# Step 7: the monotonic_align Cython extension
# --------------------------------------------------------------------------


#: Where upstream keeps the Cython alignment kernel, relative to the checkout.
MONOTONIC_ALIGN_DIR = Path("src/piper/train/vits/monotonic_align")


def step_monotonic_align(ctx: Context) -> None:
    """Build the VITS alignment kernel.

    Upstream ships ``build_monotonic_align.sh``, which we deliberately do not
    call: it is four commands wrapped in bash, and bash is not a thing we can
    assume on Windows. Doing those four steps directly also removes upstream's
    "activate piper1-gpl/.venv if it happens to exist" behaviour, which would
    otherwise build against the wrong interpreter.

    ``cythonize`` is invoked as a module rather than as the console script,
    because the script lands in ``bin`` or ``Scripts`` depending on platform.
    """
    workdir = PIPER_DIR / MONOTONIC_ALIGN_DIR
    if not (workdir / "core.pyx").is_file():
        raise SetupError(
            f"missing {workdir / 'core.pyx'} — is piper1-gpl checked out?"
        )

    (workdir / "monotonic_align").mkdir(exist_ok=True)
    # cythonize skips regeneration when core.c is newer than core.pyx, which
    # makes a retry after a failed compile a no-op. Upstream removes it too.
    (workdir / "core.c").unlink(missing_ok=True)

    proc.run(
        [venv_python(), "-m", "Cython.Build.Cythonize", "-i", "core.pyx"],
        cwd=workdir,
        env=env_mod.pip_env(),
        log_path=ctx.log_path,
    )

    # The extension lands beside core.pyx but has to be importable as
    # monotonic_align.core. Suffix differs per platform (.so / .pyd) and
    # carries the ABI tag, so match on the stem instead of naming it.
    built = [
        path
        for path in workdir.iterdir()
        if path.is_file()
        and path.name.startswith("core.")
        and path.suffix in (".so", ".pyd")
    ]
    if not built:
        raise SetupError(
            f"cythonize reported success but produced no extension module in "
            f"{workdir}. Expected a core*.so or core*.pyd."
        )
    for path in built:
        shutil.move(str(path), str(workdir / "monotonic_align" / path.name))

    result = env_mod.python_snippet(
        "from piper.train.vits.monotonic_align import maximum_path; print('ok')"
    )
    if not result.ok or "ok" not in result.output:
        raise SetupError(
            "the monotonic_align extension did not build. Training cannot "
            "start without it (it is the VITS alignment search kernel).\n"
            + "\n".join(result.lines[-10:])
        )
    tui.ok(f"monotonic_align extension built ({built[0].name})")


# --------------------------------------------------------------------------
# Step 8: in-place C extension build
# --------------------------------------------------------------------------


def step_build_ext(ctx: Context) -> None:
    # Needed because we run piper from its source tree: this builds the
    # espeakbridge C extension in place.
    #
    # setup.py is invoked directly rather than through pip, so the isolated
    # build environment that supplied skbuild during piper_install does not
    # exist here. Upstream's build requirements have to be in our own venv.
    tui.info("installing piper's build requirements into the venv")
    tui.hint(
        "  piper1-gpl's setup.py does 'from skbuild import setup'. pip's build "
        "isolation supplied that for the editable install and then threw it "
        "away; running setup.py by hand needs it installed for real."
    )
    ctx.pip("install", *ctx.pins.build_requires)
    try:
        proc.run(
            [venv_python(), "setup.py", "build_ext", "--inplace"],
            cwd=PIPER_DIR,
            log_path=ctx.log_path,
        )
    except proc.CommandFailed as exc:
        _explain_build_failure(exc.result)
        raise
    tui.ok("espeakbridge extension built in place")


# --------------------------------------------------------------------------
# Step 9: whisper, without letting it touch torch
# --------------------------------------------------------------------------


def step_export_deps(ctx: Context) -> None:
    # Deliberately its own step, after torch is pinned by the constraint file.
    # The failure this prevents lands at the end of a training run, when the
    # checkpoint exists and the export is the only thing left.
    tui.info("installing what torch.onnx.export needs")
    tui.hint(
        "  torch 2.9 made the dynamo exporter the default, and torch.onnx "
        "imports onnxscript on the way in. piper does not declare it, so "
        "export would otherwise fail only after training finished."
    )
    ctx.pip("install", *ctx.pins.export_requires)
    assert_torch_unchanged(ctx, "export_deps")
    result = env_mod.python_snippet("import onnxscript; print(onnxscript.__version__)")
    if not result.ok:
        raise SetupError(
            "onnxscript is not importable after installation:\n"
            + "\n".join(result.lines[-10:])
        )
    tui.ok(f"onnxscript {result.lines[-1].strip() if result.lines else 'importable'}")


def step_whisper(ctx: Context) -> None:
    tui.info(f"installing {ctx.pins.whisper_package} without its dependencies")
    tui.hint(
        "  openai-whisper's metadata wants an unpinned torch plus triton, and "
        "the PyPI triton is CUDA-only. Installing --no-deps and naming the "
        "dependencies ourselves is what keeps a ROCm torch intact."
    )
    ctx.pip("install", "--no-deps", ctx.pins.whisper_package)
    ctx.pip("install", *ctx.pins.whisper_deps)
    assert_torch_unchanged(ctx, "whisper")
    result = env_mod.python_snippet("import whisper; print(whisper.__version__ if hasattr(whisper, '__version__') else 'ok')")
    if not result.ok:
        raise SetupError(
            "whisper is not importable after installation:\n"
            + "\n".join(result.lines[-10:])
        )
    tui.ok("whisper importable")


# --------------------------------------------------------------------------
# Step 10: verify everything
# --------------------------------------------------------------------------

_VERIFY = r"""
import json
report = {}
def check(name, fn):
    try:
        report[name] = {"ok": True, "detail": fn()}
    except Exception as exc:
        report[name] = {"ok": False, "detail": "%s: %s" % (type(exc).__name__, exc)}

def _piper():
    import piper
    return getattr(piper, "__version__", "installed")

def _train():
    import piper.train
    return "importable"

def _align():
    from piper.train.vits.monotonic_align import maximum_path
    return "importable"

def _espeak():
    from piper.phonemize_espeak import EspeakPhonemizer
    phonemes = EspeakPhonemizer().phonemize("en-us", "Hello world.")
    flat = "".join("".join(sentence) for sentence in phonemes)
    if not flat:
        raise RuntimeError("phonemizer returned nothing")
    return flat[:40]

def _librosa():
    import librosa
    return librosa.__version__

def _vad():
    from pysilero_vad import SileroVoiceActivityDetector
    return "importable"

def _whisper():
    import whisper
    return ", ".join(sorted(whisper.available_models())[:4]) + ", ..."

def _lightning():
    import lightning
    return lightning.__version__

def _onnxscript():
    # Import torch.onnx rather than onnxscript alone: the failure we care
    # about is torch's exporter refusing to load, and it imports onnxscript
    # from inside torch.onnx.export.
    import torch.onnx, onnxscript
    return onnxscript.__version__

check("piper", _piper)
check("piper.train", _train)
check("monotonic_align", _align)
check("espeak-ng phonemizer", _espeak)
check("librosa", _librosa)
check("silero vad", _vad)
check("whisper", _whisper)
check("lightning", _lightning)
check("onnxscript", _onnxscript)
print(json.dumps(report))
"""


def step_verify(ctx: Context) -> None:
    import json

    result = env_mod.python_snippet(_VERIFY, timeout=600)
    payload = None
    for line in reversed(result.lines):
        if line.strip().startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        raise SetupError(
            "verification produced no output:\n" + "\n".join(result.lines[-20:])
        )

    failures = []
    for name, entry in payload.items():
        if entry["ok"]:
            tui.ok(f"{name}: {entry['detail']}")
        else:
            tui.error(f"{name}: {entry['detail']}")
            failures.append(name)

    info = env_mod.verify_torch()
    tui.info(f"torch: {info.summary()}")
    env_mod.record_torch(ctx.state, info)

    if info.hsa_override:
        tui.ok(
            f"HSA_OVERRIDE_GFX_VERSION={info.hsa_override} was needed and has "
            f"been saved to {ENV_JSON.relative_to(REPO_ROOT).as_posix()}"
        )
    if ctx.vendor != "cpu" and not info.usable_gpu:
        detail = info.matmul_error or info.error or "no GPU visible"
        ctx.warn(
            f"torch cannot use the GPU: {detail}. Training will fall back to "
            f"CPU, which is far slower. {env_mod.unsupported_arch_advice(info)}"
        )
    elif info.usable_gpu:
        tui.ok(
            f"GPU verified with a real matmul: {info.device_name} "
            f"({info.total_memory_gib:.1f} GiB)"
        )
        _verify_training(ctx, info)

    if failures:
        raise SetupError("verification failed for: " + ", ".join(failures))


def _verify_training(ctx: Context, info: env_mod.TorchInfo) -> None:
    """Prove the GPU can train, not merely multiply.

    A matmul exercises the GEMM library and nothing else. The failure that
    matters here is a missing elementwise or convolution kernel, or a fault
    after tens of allocate/free cycles — neither of which a matmul can reach.
    Better to spend ten seconds now than to find out an hour into a run.
    """
    tui.info("  running a real autograd loop to check training works...")
    probe = env_mod.probe_training(extra_env=env_mod.training_env())
    if probe.ok:
        tui.ok(f"training verified: {probe.verdict}")
        return

    ctx.warn(f"this GPU cannot complete a training step: {probe.verdict}")
    if info.compiled_for_device is False:
        tui.info(
            f"torch has no {env_mod.normalise_gfx(info.gcn_arch)} kernels "
            f"(built for {', '.join(info.arch_list) or 'unknown'}), which is "
            f"exactly what blocks training while leaving matmul working."
        )
    tui.info(env_mod.unsupported_arch_advice(info))
    tui.info(
        "Everything else still works on this machine: dataset preparation, "
        "export, and CPU training."
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

STEPS: list[Step] = [
    Step("preflight", "Check the machine can build piper", step_preflight),
    Step("system_deps", "Install system packages", step_system_deps),
    Step("venv", "Create the virtualenv", step_venv),
    Step("torch", "Install the vendor-specific torch wheel", step_torch),
    Step("constraint", "Pin torch for every later pip call", step_constraint),
    Step("clone", "Clone piper1-gpl at the pinned tag", step_clone),
    Step("piper_install", "Install piper1-gpl[train] (builds espeak-ng)", step_piper_install),
    Step("monotonic_align", "Build the monotonic_align extension", step_monotonic_align),
    Step("build_ext", "Build the espeakbridge extension in place", step_build_ext),
    Step("whisper", "Install whisper without disturbing torch", step_whisper),
    Step("export_deps", "Install onnxscript for ONNX export", step_export_deps),
    Step("verify", "Verify the whole toolchain", step_verify),
]

# Steps that must always re-run: they are cheap and they validate rather than
# mutate, so skipping them would report a stale picture.
ALWAYS_RUN = {"verify"}


def _choose_vendor(ctx: Context, explicit: str | None) -> str:
    vendor, notes = env_mod.resolve_vendor(explicit)
    for note in notes:
        tui.info(f"  {note}")
    if vendor == "ambiguous":
        tui.warn("both NVIDIA and AMD GPUs are visible; pick which to build for")
        vendor = tui.ask_choice(
            "GPU vendor", list(env_mod.VENDORS), "cuda", allow_back=False
        )
    return vendor


def run_setup(
    vendor: str | None = None,
    offline: bool = False,
    force_steps: Sequence[str] = (),
    assume_yes: bool = False,
    torch_index: str | None = None,
    torch_spec: str | None = None,
) -> int:
    pins = pins_mod.load()
    state = env_mod.SetupState.load()
    ctx = Context(
        pins=pins,
        state=state,
        offline=offline or bool(os.environ.get("PT_OFFLINE")),
        assume_yes=assume_yes,
        torch_index=torch_index or "",
        torch_spec=torch_spec or "",
    )

    tui.heading("Setup")
    if ctx.offline:
        tui.warn("offline mode: no downloads will be attempted")
    ctx.vendor = _choose_vendor(ctx, vendor)
    tui.info(f"  building for: {tui.style(ctx.vendor, 'bold')}")

    force = set(force_steps)
    if "all" in force:
        force = {step.name for step in STEPS}
    unknown = force - {step.name for step in STEPS}
    if unknown:
        raise SetupError(
            f"unknown step(s): {', '.join(sorted(unknown))}. Valid: "
            + ", ".join(step.name for step in STEPS)
        )

    for index, step in enumerate(STEPS):
        done = ctx.state.steps.get(step.name) == "ok"
        if done and step.name not in force and step.name not in ALWAYS_RUN:
            tui.hint(f"  [{index}/{len(STEPS) - 1}] {step.title} — already done")
            continue
        tui.heading(f"[{index}/{len(STEPS) - 1}] {step.title}")
        try:
            step.action(ctx)
        except (SetupError, proc.CommandFailed) as exc:
            ctx.state.steps[step.name] = "failed"
            ctx.state.save()
            tui.error(str(exc))
            tui.info("")
            tui.info(
                f"Setup stopped at step '{step.name}'. Fix the problem and "
                f"re-run './run setup' — completed steps are skipped."
            )
            return 1
        ctx.state.steps[step.name] = "ok"
        ctx.state.save()

    _print_summary(ctx)
    return 0


def _print_summary(ctx: Context) -> None:
    tui.heading("Setup complete")
    info = ctx.state.info
    tui.table(
        [
            ["vendor", ctx.vendor],
            ["torch", ctx.state.torch_local_version or "unknown"],
            ["torch index", ctx.state.torch_index],
            ["piper1-gpl", f"{ctx.state.piper_tag} ({ctx.state.piper_sha[:12]})"],
            ["gpu", info.device_name or "none (CPU)"],
            ["gpu usable", "yes" if info.usable_gpu else "no"],
        ]
    )
    for note in ctx.notes:
        tui.bullet(note)
    if ctx.warnings:
        tui.info("")
        for warning in ctx.warnings:
            tui.warn(warning)

    tui.info("")
    tui.info("Next:")
    tui.bullet("./run doctor          re-check the environment any time")
    tui.bullet("./run dataset         build a dataset from one audio file")
    tui.bullet("./run smoke           prove the whole pipeline on CPU (~5 min)")

    if ctx.offline:
        tui.info("")
        tui.info("To prepare an offline machine, stage these from a networked box:")
        tui.bullet(f"a clone of {ctx.pins.piper_repo} at {ctx.pins.piper_tag} -> ./piper1-gpl")
        tui.bullet("a pip wheelhouse for torch, piper1-gpl[train] and whisper")
        tui.bullet("~/.cache/whisper/<model>.pt for the Whisper model you want")
        tui.bullet("any pretrained .ckpt files -> ./checkpoints/")
