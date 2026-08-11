"""Hardware profiles.

The core of this tool is deliberately generic: a GPU is a GPU, and the only
thing that decides which one trains is the torch wheel. But some boards need
more than a wheel, and hardcoding their quirks into the vendor logic would make
the common path worse.

So hardware-specific knowledge lives here, as named profiles. ``generic`` is the
default and applies nothing. ``bc250`` carries what is known about the AMD BC-250
(gfx1013 "Cyan Skillfish"), a board with no official ROCm support.

Set it with ``runtime.hardware`` in the profile, or let it be detected from the
reported gfx target.

Sources for the BC-250 entries are cited inline; they come from
https://github.com/akandr/bc250-rocm, which is the most thorough public
characterisation of the board. That work covers **one** board, and the author
says reproducibility elsewhere is unknown — so everything here is framed as
"check this on your machine", and ``./run doctor`` measures rather than assumes.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import proc

# How much of the training path a profile is expected to support.
SUPPORTED = "supported"        # ordinary, no known blockers
UNPROVEN = "unproven"          # may work; measure before committing
BLOCKED = "blocked"            # known not to work in the documented config

BC250_REFERENCE = "https://github.com/akandr/bc250-rocm"


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    title: str
    summary: str
    #: gfx targets that select this profile automatically.
    gfx_targets: tuple[str, ...] = ()
    #: Environment variables applied to training and inference subprocesses.
    env: dict[str, str] = field(default_factory=dict)
    #: Profile settings forced when this hardware is active, as dotted paths.
    settings: dict[str, Any] = field(default_factory=dict)
    #: Things the user needs to know but that no check can enforce.
    caveats: tuple[str, ...] = ()
    #: Environment variables this hardware must NOT have set, with reasons.
    banned_env: dict[str, str] = field(default_factory=dict)
    training: str = SUPPORTED
    reference: str = ""
    #: Our own document for this board, if it has one.
    doc: str = ""
    #: Script that builds what no stock wheel provides for this board.
    build_script: str = ""

    @property
    def is_generic(self) -> bool:
        return self.name == "generic"


GENERIC = HardwareProfile(
    name="generic",
    title="Generic GPU or CPU",
    summary=(
        "No hardware-specific handling. Correct for every officially supported "
        "NVIDIA and AMD GPU, and for CPU-only runs."
    ),
    training=SUPPORTED,
)


BC250 = HardwareProfile(
    name="bc250",
    title="AMD BC-250 (gfx1013, Cyan Skillfish)",
    summary=(
        "A salvaged console APU with no official ROCm support. Getting compute "
        "working at all needs a recent kernel, two amdgpu module parameters and "
        "an environment variable; full training additionally needs gfx1013 "
        "kernels that the stock torch wheel does not ship."
    ),
    gfx_targets=("gfx1013",),
    env={
        # The SDMA host-to-device path is broken for bulk transfers on this
        # board; disabling it is part of the documented working configuration.
        "HSA_ENABLE_SDMA": "0",
        # hipBLASLt has no gfx1013 support at all, and torch reaches for it
        # first for addmm/linear. Force the rocBLAS path, which is the one that
        # can actually be built for this target.
        "TORCH_BLAS_PREFER_HIPBLASLT": "0",
        # MIOpen has no tuning entries for an architecture nobody supports, so
        # the default search compiles and benchmarks every solver at runtime on
        # first use — reported to hang outright on unsupported targets. FAST
        # takes the first workable solver instead.
        "MIOPEN_FIND_MODE": "FAST",
    },
    settings={
        # fp16 batch-GEMM crashes inside rocBLAS on this board, and mixed
        # precision with a GAN under manual optimization is the least-tested
        # combination anyway.
        "trainer.precision": "32-true",
        # The GPU shares physical memory with the system and the compositor.
        "data.num_workers": 1,
        "data.batch_size": 4,
        "dataset.max_seconds": 10.0,
    },
    banned_env={
        # gfx1010 and gfx1013 share an ISA, so overriding is an old and
        # tempting workaround. It does not survive real workloads: the memory
        # aperture layout differs, so anything touching scratch or private
        # addressing raises HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
        "HSA_OVERRIDE_GFX_VERSION": (
            "gfx1013 has a different memory-aperture layout, so the override "
            "produces HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION as soon as "
            "anything uses scratch or private addressing. It is a dead end on "
            "this board, not a fix"
        ),
    },
    training=BLOCKED,
    reference=BC250_REFERENCE,
    doc="docs/BC250.md",
    build_script="scripts/bc250/build.sh",
    caveats=(
        "Training is blocked with the stock torch ROCm wheel: it ships no "
        "gfx1013 elementwise kernels, so autograd raises 'invalid device "
        "function'. Preallocated-buffer matmul reaches about 4.5 TFLOP/s "
        "regardless, because GEMM comes from rocBLAS — that is inference's "
        "shape, not training's.",
        "Loops that allocate and free GPU tensors every iteration fault after "
        "20 to 40 iterations unless the amdgpu module carries the "
        "bc250_flush_by_runlist fix. Training does exactly that.",
        "Compute needs kernel 7.1.5 or newer plus a patched amdgpu module. On "
        "older kernels the correctness fix forces the board to 24 CU, where "
        "compute wedges.",
        "After a compute wedge, power-cycle rather than soft-reboot.",
        "A torch built from source with PYTORCH_ROCM_ARCH=gfx1013 is the only "
        "known way to lift the kernel blocker; scripts/bc250/build.sh does it. "
        "Nobody has published a result either way, so treat './run doctor's "
        "autograd probe as the verdict rather than this text.",
    ),
)


PROFILES: dict[str, HardwareProfile] = {
    GENERIC.name: GENERIC,
    BC250.name: BC250,
}

NAMES = ("auto",) + tuple(PROFILES)


def get(name: str) -> HardwareProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown hardware profile {name!r} (expected one of "
            f"{', '.join(NAMES)})"
        ) from None


def detect(gcn_arch: str | None) -> HardwareProfile:
    """Pick a profile from the reported gfx target."""
    if not gcn_arch:
        return GENERIC
    target = gcn_arch.split(":")[0].strip().lower()
    for profile in PROFILES.values():
        if target in profile.gfx_targets:
            return profile
    return GENERIC


def resolve(configured: str, gcn_arch: str | None) -> HardwareProfile:
    """``runtime.hardware`` plus autodetection. Explicit beats detected."""
    if configured and configured != "auto":
        return get(configured)
    return detect(gcn_arch)


GENERIC_ARCH_ADVICE = (
    "See docs/GPU_SETUP.md, and try a different ROCm index from pins.toml: "
    "'./run setup --force-step torch --torch-index "
    "https://download.pytorch.org/whl/rocm6.3 --torch-spec torch==2.6.0'."
)


def unsupported_arch_advice(profile: HardwareProfile) -> str:
    """What to actually try when torch ships no kernels for this device.

    Swapping ROCm wheels is the right first move for a card that is merely
    missing from one build's arch list. It is a dead end for hardware carrying
    a BLOCKED profile: no stock wheel of any ROCm version ships those kernels,
    so the swap costs a multi-gigabyte download and then fails identically.
    Defer to the profile there, the same way we refuse to suggest
    HSA_OVERRIDE_GFX_VERSION on gfx1013.
    """
    if profile.is_generic or profile.training != BLOCKED:
        return GENERIC_ARCH_ADVICE
    where = " and ".join(part for part in (profile.doc, profile.reference) if part)
    advice = (
        f"This is {profile.title}. No stock ROCm wheel ships kernels for it, "
        f"so swapping the torch index will not help"
    )
    advice += f" — see {where}." if where else "."
    if profile.build_script:
        advice += (
            f" Building one that does is what {profile.build_script} is for."
        )
    return advice


def apply(profile: HardwareProfile, target: Any) -> list[str]:
    """Apply a hardware profile's settings to a training profile, in place.

    Returns a human-readable list of what changed. Existing values are
    overwritten: these settings exist because the hardware requires them.
    """
    from . import profile as profile_mod

    changed: list[str] = []
    for dotted, value in profile.settings.items():
        current = profile_mod.get_path(target, dotted)
        if current != value:
            profile_mod.set_path(target, dotted, value)
            changed.append(f"{dotted}: {current} -> {value}")

    for key, value in profile.env.items():
        if target.runtime.env.get(key) != value:
            target.runtime.env[key] = value
            changed.append(f"runtime.env.{key} = {value}")

    for key in profile.banned_env:
        if key in target.runtime.env:
            del target.runtime.env[key]
            changed.append(f"runtime.env.{key} removed")

    return changed


# --------------------------------------------------------------------------
# System checks
# --------------------------------------------------------------------------

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass(frozen=True)
class SystemCheck:
    status: str
    name: str
    detail: str
    fix: str = ""


def kernel_release() -> str:
    return os.uname().release if hasattr(os, "uname") else ""


def kernel_version_tuple(release: str) -> tuple[int, ...]:
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", release or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def read_cmdline() -> str:
    try:
        return Path("/proc/cmdline").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def cmdline_param(cmdline: str, key: str) -> str | None:
    """Value of ``key=value`` on the kernel command line, if present."""
    for token in cmdline.split():
        name, sep, value = token.partition("=")
        if name == key:
            return value if sep else ""
    return None


def modparam(module: str, name: str) -> str | None:
    """Live value of a module parameter from sysfs."""
    path = Path(f"/sys/module/{module}/parameters/{name}")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


# Prefixes that can hold a rocBLAS Tensile library directory. A pip torch wheel
# bundles its own copy under ``torch/lib``; a torch built from source links
# against the system one instead, so both have to be searched or the answer
# depends on how torch was installed rather than on what is on disk.
# ``/opt/bc250`` is where scripts/bc250/build.sh installs its gfx1013 build.
SYSTEM_ROCBLAS_PREFIXES = (
    "/opt/rocm/lib",
    "/opt/bc250/rocm/lib",
    "/usr/lib64",
    "/usr/lib",
)


def rocblas_library_dirs(torch_lib_dirs: Sequence[Path] = ()) -> list[Path]:
    """Every directory that could hold rocBLAS Tensile libraries, in order.

    ``ROCBLAS_TENSILE_LIBPATH`` is rocBLAS's own override and points straight at
    the library directory; everywhere else it sits under ``rocblas/library``.
    """
    candidates: list[Path] = []
    override = os.environ.get("ROCBLAS_TENSILE_LIBPATH", "").strip()
    if override:
        candidates.append(Path(override))
    for base in (*torch_lib_dirs, *(Path(p) for p in SYSTEM_ROCBLAS_PREFIXES)):
        candidates.append(Path(base) / "rocblas" / "library")

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def rocblas_has_gfx1013(torch_lib_dirs: Sequence[Path] = ()) -> bool | None:
    """Whether a rocBLAS with gfx1013 kernels is present.

    The stock ROCm wheel ships no gfx1013 Tensile libraries. A native build
    produces ``Kernels.so-000-gfx1013.hsaco``. ``None`` means we could not tell
    — no rocBLAS library directory existed anywhere we looked, which is the
    normal answer on a CUDA or CPU install.
    """
    searched = False
    for library in rocblas_library_dirs(torch_lib_dirs):
        if not library.is_dir():
            continue
        searched = True
        try:
            entries = list(library.iterdir())
        except OSError:
            continue
        for entry in entries:
            if "gfx1013" in entry.name:
                return True
    return False if searched else None


# --------------------------------------------------------------------------
# BC-250 specific checks
# --------------------------------------------------------------------------

MIN_BC250_KERNEL = (7, 1, 5)
BC250_MODPARAMS = {
    "bc250_cc_write_mode": "3",
    "bc250_flush_by_runlist": "1",
}


def bc250_checks(torch_lib_dirs: Sequence[Path] = ()) -> list[SystemCheck]:
    """Everything the BC-250 needs that this tool cannot do for you.

    Kernel parameters and a patched amdgpu module require root and a reboot, so
    these are reported rather than applied.
    """
    checks: list[SystemCheck] = []

    release = kernel_release()
    version = kernel_version_tuple(release)
    if not version:
        checks.append(
            SystemCheck(WARN, "kernel", "could not determine the version", "")
        )
    elif version >= MIN_BC250_KERNEL:
        checks.append(SystemCheck(OK, "kernel", release))
    else:
        checks.append(
            SystemCheck(
                FAIL,
                "kernel",
                f"{release} is older than "
                f"{'.'.join(str(p) for p in MIN_BC250_KERNEL)}",
                "On older kernels the correctness fix forces the board to 24 CU, "
                "where compute wedges. CachyOS ships recent kernels; update and "
                "reboot.",
            )
        )

    cmdline = read_cmdline()
    for name, expected in BC250_MODPARAMS.items():
        live = modparam("amdgpu", name)
        from_cmdline = cmdline_param(cmdline, f"amdgpu.{name}")
        value = live if live is not None else from_cmdline
        if value == expected:
            checks.append(SystemCheck(OK, f"amdgpu.{name}", value))
        elif value is None:
            checks.append(
                SystemCheck(
                    FAIL,
                    f"amdgpu.{name}",
                    "not set (the module may not support it)",
                    f"This parameter comes from a patched amdgpu module. Add "
                    f"amdgpu.{name}={expected} to the kernel command line once "
                    f"the patched module is installed. See {BC250_REFERENCE}.",
                )
            )
        else:
            checks.append(
                SystemCheck(
                    FAIL,
                    f"amdgpu.{name}",
                    f"is {value!r}, expected {expected!r}",
                    f"Set amdgpu.{name}={expected} on the kernel command line "
                    f"and reboot.",
                )
            )

    # Hardware scheduling must stay at the default on 7.1.5+. sched_policy=2 was
    # the old workaround and is now the difference between a wedge and a clean
    # run.
    sched = modparam("amdgpu", "sched_policy") or cmdline_param(
        cmdline, "amdgpu.sched_policy"
    )
    if sched in (None, ""):
        checks.append(
            SystemCheck(OK, "amdgpu.sched_policy", "default (hardware scheduling)")
        )
    elif sched == "2":
        checks.append(
            SystemCheck(
                FAIL,
                "amdgpu.sched_policy",
                "is 2",
                "Remove amdgpu.sched_policy=2 from the kernel command line. It "
                "was the workaround for older kernels; with the patched module "
                "on 7.1.5 it causes compute to wedge.",
            )
        )
    else:
        checks.append(SystemCheck(WARN, "amdgpu.sched_policy", f"is {sched!r}"))

    checks.append(_cu_check())

    sdma = os.environ.get("HSA_ENABLE_SDMA")
    if sdma == "0":
        checks.append(SystemCheck(OK, "HSA_ENABLE_SDMA", "0"))
    else:
        checks.append(
            SystemCheck(
                INFO,
                "HSA_ENABLE_SDMA",
                "not set in this shell",
                "This is applied automatically to training and inference "
                "subprocesses when runtime.hardware is 'bc250'; it only matters "
                "here if you run torch by hand.",
            )
        )

    has = rocblas_has_gfx1013(torch_lib_dirs)
    if has is True:
        checks.append(SystemCheck(OK, "rocBLAS gfx1013 kernels", "present"))
    elif has is False:
        # Deliberately a warning, not a failure. rocBLAS is the GEMM library:
        # building it for gfx1013 fixes matmul throughput and nothing else. The
        # training blocker is missing elementwise and convolution kernels, which
        # come from torch's own device code, so this check going green does not
        # unblock training and this check being red is not what blocks it.
        checks.append(
            SystemCheck(
                WARN,
                "rocBLAS gfx1013 kernels",
                "absent",
                "Matmul falls back to something much slower. Building rocBLAS "
                "for gfx1013 fixes GEMM only — see 'torch gfx1013 kernels' "
                "above for the check that governs training. "
                "scripts/bc250/build.sh does both.",
            )
        )

    return checks


def _cu_check() -> SystemCheck:
    """Confirm the board came up with all 40 compute units unlocked."""
    if not shutil.which("dmesg"):
        return SystemCheck(INFO, "active_cu_number", "dmesg not available")
    result = proc.capture(["dmesg"], timeout=30)
    if not result.ok:
        return SystemCheck(
            INFO,
            "active_cu_number",
            "could not read dmesg (usually needs root or "
            "kernel.dmesg_restrict=0)",
        )
    for line in result.lines:
        if "active_cu_number" in line:
            match = re.search(r"active_cu_number\s*[:=]?\s*(\d+)", line)
            if match:
                count = int(match.group(1))
                if count >= 40:
                    return SystemCheck(OK, "active_cu_number", str(count))
                return SystemCheck(
                    WARN,
                    "active_cu_number",
                    f"{count}, expected 40",
                    "The board did not unlock all compute units this boot. "
                    "Power-cycle rather than soft-reboot and check again.",
                )
    return SystemCheck(INFO, "active_cu_number", "not reported in dmesg")


def checks_for(
    profile: HardwareProfile, torch_lib_dirs: Sequence[Path] = ()
) -> list[SystemCheck]:
    if profile.name == "bc250":
        return bc250_checks(torch_lib_dirs)
    return []


def summarise(profile: HardwareProfile) -> list[str]:
    """Lines describing what this profile does, for a status screen."""
    lines = [profile.summary]
    if profile.env:
        lines.append(
            "sets " + ", ".join(f"{k}={v}" for k, v in sorted(profile.env.items()))
        )
    if profile.settings:
        lines.append(
            "forces " + ", ".join(f"{k}={v}" for k, v in sorted(profile.settings.items()))
        )
    return lines
