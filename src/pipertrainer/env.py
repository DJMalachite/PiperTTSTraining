"""GPU vendor detection, torch verification, and environment assembly.

This is where NVIDIA/AMD/CPU portability actually lives. piper1-gpl itself is
vendor-agnostic — it never mentions ROCm — so the only thing that decides which
hardware trains is *which torch wheel is installed*. Everything here exists to
get that decision right and then prove it.

The proof matters more than it looks, and comes in two parts.

On ROCm, ``torch.cuda.is_available()`` returns True for GPU architectures the
build has no kernels for, and the process then aborts at the first real kernel
launch. So ``probe_torch`` runs an actual matmul and synchronises, and treats a
probe that printed nothing as a failure rather than a success.

A matmul is still not proof that the GPU can *train*. It exercises rocBLAS and
nothing else, while the failures that stop a run are missing elementwise or
convolution kernels (``invalid device function``) and faults after tens of
allocate/free cycles. ``probe_training`` therefore runs a real autograd loop
with allocation churn, using the layer types a VITS vocoder actually uses.

``HSA_OVERRIDE_GFX_VERSION`` is applied automatically only for targets it
genuinely rescues — see ``GFX_OVERRIDES``. Boards where it is known to make
things worse declare it in ``hardware.HardwareProfile.banned_env``, and it is
never suggested for those.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import pins, proc
from .paths import (
    ENV_SH,
    REPO_ROOT,
    SETUP_STATE,
    STATE_DIR,
    TORCH_CONSTRAINT,
    venv_python,
)

if TYPE_CHECKING:
    # Imported for real inside the functions that need it, matching how the
    # rest of the package reaches for hardware profiles.
    from .hardware import HardwareProfile

VENDORS = ("rocm", "cuda", "cpu")

# gfx targets that an override genuinely rescues, mapped to the ISA whose
# kernels actually cover them. RDNA1 parts borrow gfx1010's, RDNA2 parts
# gfx1030's.
#
# gfx1013 (BC-250, Cyan Skillfish) is deliberately ABSENT. It shares an ISA
# with gfx1010, which makes the override tempting, but the memory-aperture
# layout differs: anything using scratch or private addressing then raises
# HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. See hardware.BC250.banned_env.
GFX_OVERRIDES = {
    "gfx1011": "10.1.0",
    "gfx1012": "10.1.0",
    "gfx1031": "10.3.0",
    "gfx1032": "10.3.0",
    "gfx1033": "10.3.0",
    "gfx1034": "10.3.0",
    "gfx1035": "10.3.0",
    "gfx1036": "10.3.0",
}

_PROBE = r"""
import json, sys
info = {"python": sys.version.split()[0]}
try:
    import torch
except Exception as exc:
    info["error"] = "cannot import torch: %s" % exc
    print(json.dumps(info)); raise SystemExit(0)

info["torch"] = torch.__version__
info["cuda"] = torch.version.cuda
info["hip"] = getattr(torch.version, "hip", None)
try:
    info["arch_list"] = list(torch.cuda.get_arch_list())
except Exception:
    info["arch_list"] = []
try:
    info["available"] = bool(torch.cuda.is_available())
except Exception as exc:
    info["available"] = False
    info["error"] = "is_available raised: %s" % exc

if info.get("available"):
    try:
        info["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["gcn_arch"] = getattr(props, "gcnArchName", None)
        info["total_memory"] = int(props.total_memory)
    except Exception as exc:
        info["error"] = "device query failed: %s" % exc
    # The real test. An unsupported ISA passes every check above and then
    # aborts here, often with SIGABRT rather than a Python exception -- which
    # is why the caller also treats "no JSON at all" as a failed matmul.
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        total = (a @ b).sum().item()
        torch.cuda.synchronize()
        info["matmul_ok"] = total == total  # NaN would mean garbage kernels
    except Exception as exc:
        info["matmul_ok"] = False
        info["matmul_error"] = str(exc)
else:
    info["device_count"] = 0

print(json.dumps(info))
"""


@dataclass
class TorchInfo:
    """Result of probing torch. ``usable`` is the only field worth trusting."""

    ok: bool = False
    error: str | None = None
    torch_version: str | None = None
    cuda: str | None = None
    hip: str | None = None
    available: bool = False
    device_count: int = 0
    device_name: str | None = None
    gcn_arch: str | None = None
    total_memory: int = 0
    arch_list: list[str] = field(default_factory=list)
    matmul_ok: bool = False
    matmul_error: str | None = None
    hsa_override: str | None = None
    aborted: bool = False

    @property
    def vendor(self) -> str:
        """Which build this is — never inferred from is_available()."""
        if self.hip:
            return "rocm"
        if self.cuda:
            return "cuda"
        return "cpu"

    @property
    def usable_gpu(self) -> bool:
        return self.available and self.matmul_ok

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory / (1024**3)

    def summary(self) -> str:
        if not self.ok:
            return f"torch unavailable ({self.error or 'unknown'})"
        parts = [f"torch {self.torch_version}"]
        if self.hip:
            parts.append(f"ROCm {self.hip}")
        elif self.cuda:
            parts.append(f"CUDA {self.cuda}")
        else:
            parts.append("CPU build")
        if self.available:
            name = self.device_name or "gpu"
            if self.gcn_arch:
                name = f"{name} ({self.gcn_arch})"
            parts.append(name)
            if self.total_memory:
                parts.append(f"{self.total_memory_gib:.1f} GiB")
            parts.append("matmul OK" if self.matmul_ok else "matmul FAILED")
            if self.hsa_override:
                parts.append(f"HSA_OVERRIDE_GFX_VERSION={self.hsa_override}")
        else:
            parts.append("no GPU visible")
        return " / ".join(parts)


@dataclass
class SetupState:
    """Persisted results of setup, so the menu need not re-probe every launch."""

    vendor: str = ""
    torch_index: str = ""
    torch_spec: str = ""
    torch_local_version: str = ""
    piper_tag: str = ""
    piper_sha: str = ""
    steps: dict[str, Any] = field(default_factory=dict)
    torch_info: dict[str, Any] = field(default_factory=dict)
    hsa_override: str = ""
    checked_at: str = ""

    @classmethod
    def load(cls) -> "SetupState":
        if not SETUP_STATE.exists():
            return cls()
        try:
            data = json.loads(SETUP_STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        SETUP_STATE.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @property
    def info(self) -> TorchInfo:
        known = {f for f in TorchInfo.__dataclass_fields__}
        return TorchInfo(**{k: v for k, v in self.torch_info.items() if k in known})


# --------------------------------------------------------------------------
# Vendor detection
# --------------------------------------------------------------------------


def has_nvidia() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    result = proc.capture(["nvidia-smi", "-L"], timeout=15)
    return result.ok and any(line.startswith("GPU ") for line in result.lines)


def has_amd() -> bool:
    """AMD GPU present and the kernel driver is loaded.

    Reads sysfs rather than shelling out to lspci, which needs pciutils.
    """
    if not Path("/dev/kfd").exists():
        return False
    for vendor_file in glob.glob("/sys/class/drm/card*/device/vendor"):
        try:
            if Path(vendor_file).read_text(encoding="utf-8").strip() == "0x1002":
                return True
        except OSError:
            continue
    return False


def render_group_ok() -> bool | None:
    """Whether the user can talk to /dev/kfd. ``None`` if we cannot tell.

    Missing render/video membership is the most common cause of "ROCm sees no
    GPU" on a fresh Arch install.
    """
    result = proc.capture(["id", "-nG"], timeout=10)
    if not result.ok or not result.lines:
        return None
    groups = set(result.lines[0].split())
    return bool(groups & {"render", "video"}) or os.geteuid() == 0


def detect_vendor() -> tuple[str, list[str]]:
    """Return ``(vendor, notes)``. ``vendor`` may be 'ambiguous'."""
    notes: list[str] = []
    nvidia, amd = has_nvidia(), has_amd()
    if nvidia and amd:
        notes.append("both an NVIDIA and an AMD GPU are visible")
        return "ambiguous", notes
    if nvidia:
        return "cuda", notes
    if amd:
        if render_group_ok() is False:
            notes.append(
                "you are not in the 'render' or 'video' group — ROCm will not "
                "see the GPU until you are (log out and back in after adding)"
            )
        return "rocm", notes
    if Path("/dev/kfd").exists():
        notes.append("/dev/kfd exists but no AMD PCI device was found")
    notes.append("no supported GPU detected; falling back to CPU")
    return "cpu", notes


def resolve_vendor(
    explicit: str | None = None,
    profile_vendor: str | None = None,
) -> tuple[str, list[str]]:
    """Vendor precedence: flag > env > profile > saved state > autodetect."""
    for candidate, source in (
        (explicit, "--vendor"),
        (os.environ.get("PT_VENDOR"), "PT_VENDOR"),
        (profile_vendor, "profile"),
        (SetupState.load().vendor, "saved setup state"),
    ):
        if candidate:
            if candidate not in VENDORS:
                raise ValueError(
                    f"{source}: unknown vendor {candidate!r} "
                    f"(expected one of {', '.join(VENDORS)})"
                )
            return candidate, [f"vendor {candidate} (from {source})"]
    return detect_vendor()


# --------------------------------------------------------------------------
# Torch verification
# --------------------------------------------------------------------------


def python_snippet(
    code: str,
    *,
    python: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> proc.Result:
    """Run a short snippet with the venv interpreter.

    ``PYTHONNOUSERSITE`` keeps a stray ``~/.local`` install of torch or whisper
    from shadowing the venv, which is otherwise a confusing failure.
    """
    interpreter = python or venv_python()
    env = {"PYTHONNOUSERSITE": "1"}
    env.update(extra_env or {})
    return proc.capture([interpreter, "-c", code], env=env, timeout=timeout)


def probe_torch(
    python: Path | None = None, extra_env: dict[str, str] | None = None
) -> TorchInfo:
    interpreter = python or venv_python()
    if not Path(interpreter).exists():
        return TorchInfo(ok=False, error=f"no interpreter at {interpreter}")

    env = {"PYTHONNOUSERSITE": "1"}
    env.update(extra_env or {})
    result = proc.capture([interpreter, "-c", _PROBE], env=env, timeout=300)

    payload: dict[str, Any] | None = None
    for line in reversed(result.lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break

    if payload is None:
        # The probe died without printing. On ROCm this is the signature of an
        # unsupported ISA: the runtime aborts the process outright.
        tail = "; ".join(result.lines[-4:]) or "no output"
        return TorchInfo(
            ok=False,
            error=f"probe crashed (exit {result.returncode}): {tail}",
            aborted=True,
            matmul_ok=False,
            hsa_override=(extra_env or {}).get("HSA_OVERRIDE_GFX_VERSION"),
        )

    info = TorchInfo(
        ok="torch" in payload,
        error=payload.get("error"),
        torch_version=payload.get("torch"),
        cuda=payload.get("cuda"),
        hip=payload.get("hip"),
        available=bool(payload.get("available")),
        device_count=int(payload.get("device_count") or 0),
        device_name=payload.get("device_name"),
        gcn_arch=payload.get("gcn_arch"),
        total_memory=int(payload.get("total_memory") or 0),
        arch_list=list(payload.get("arch_list") or []),
        matmul_ok=bool(payload.get("matmul_ok")),
        matmul_error=payload.get("matmul_error"),
        hsa_override=(extra_env or {}).get("HSA_OVERRIDE_GFX_VERSION"),
    )
    return info


def needs_gfx_override(info: TorchInfo) -> str | None:
    """Suggested HSA_OVERRIDE_GFX_VERSION, or None if it would not help."""
    if not info.hip or not info.available:
        return None
    from . import hardware

    arch = (info.gcn_arch or "").split(":")[0]

    # Some boards are known to be made *worse* by an override. Never suggest
    # one for those, however tempting the ISA similarity looks.
    profile = hardware.detect(arch)
    if "HSA_OVERRIDE_GFX_VERSION" in profile.banned_env:
        return None

    if arch in GFX_OVERRIDES:
        return GFX_OVERRIDES[arch]
    if arch and info.arch_list and arch not in [
        a.split(":")[0] for a in info.arch_list
    ]:
        # An unlisted RDNA2 target is usually rescued by presenting as gfx1030.
        # Anything else is a guess we should not make on the user's behalf.
        return "10.3.0" if arch.startswith("gfx103") else None
    return None


def verify_torch(python: Path | None = None) -> TorchInfo:
    """Probe torch, retrying once with a gfx override when that might help."""
    info = probe_torch(python)
    if info.ok and info.available and info.matmul_ok:
        return info

    override = needs_gfx_override(info)
    if override is None and info.aborted and SetupState.load().vendor == "rocm":
        # The probe died before it could report gcnArchName, so we cannot ask
        # the hardware profile whether an override is appropriate. Try gfx1030
        # blind — it is the single most likely rescue on an unlisted RDNA2 part
        # — but never persist it unless the retry genuinely succeeds.
        override = "10.3.0"
    if override is None:
        return info

    retry = probe_torch(python, {"HSA_OVERRIDE_GFX_VERSION": override})
    if retry.ok and retry.matmul_ok:
        retry.hsa_override = override
        return retry
    return info if info.ok else retry


# A matmul proves the GPU can multiply. It does not prove it can *train*: the
# failure mode on an unsupported target is usually a missing elementwise or
# convolution kernel ("invalid device function"), or a fault after tens of
# allocate/free cycles. Both appear only under a real autograd loop, so this
# probe runs one — deliberately with a fresh allocation every iteration, and
# with the layer types a VITS vocoder actually uses.
_TRAIN_PROBE = r"""
import json, sys
info = {"stage": "import", "iterations": 0}
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    info["torch"] = torch.__version__
    if not torch.cuda.is_available():
        info["error"] = "no GPU visible"
        print(json.dumps(info)); raise SystemExit(0)

    wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    torch.manual_seed(0)

    info["stage"] = "build"
    model = nn.Sequential(
        nn.Conv1d(32, 64, 5, padding=2),
        nn.LeakyReLU(0.1),
        nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
        nn.LeakyReLU(0.1),
        nn.Conv1d(32, 16, 3, padding=1),
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    info["stage"] = "train"
    for step in range(wanted):
        # Fresh tensors each iteration on purpose: allocation churn is the
        # documented failure mode on some boards.
        x = torch.randn(4, 32, 256, device="cuda")
        y = model(x)
        target = torch.randn_like(y)
        loss = F.l1_loss(y, target) + (y * y).mean() + torch.tanh(y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.item())
        if value != value:
            info["error"] = "loss became NaN at iteration %d" % step
            break
        del x, y, target, loss
        info["iterations"] = step + 1
    torch.cuda.synchronize()
    info["stage"] = "done"
    info["ok"] = info["iterations"] == wanted and "error" not in info
except Exception as exc:
    info["error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(info))
"""


@dataclass
class TrainingProbe:
    """Can this GPU actually run a training step, not just a matmul?"""

    ok: bool = False
    iterations: int = 0
    requested: int = 60
    error: str | None = None
    stage: str = ""
    aborted: bool = False

    @property
    def verdict(self) -> str:
        if self.ok:
            return f"completed {self.iterations} autograd iterations"
        if self.aborted:
            return (
                f"the process died after {self.iterations} iteration(s) — the "
                f"GPU runtime aborted rather than raising"
            )
        if self.error and "invalid device function" in self.error.lower():
            return (
                f"missing GPU kernels: {self.error}. This build of torch has no "
                f"kernels compiled for this architecture, which is exactly what "
                f"blocks training while leaving simple matmuls working"
            )
        if self.iterations and self.error:
            return (
                f"failed after {self.iterations} of {self.requested} iterations: "
                f"{self.error}"
            )
        return self.error or "failed for an unknown reason"


def probe_training(
    python: Path | None = None,
    *,
    iterations: int = 60,
    extra_env: dict[str, str] | None = None,
) -> TrainingProbe:
    """Run a real autograd loop on the GPU and report how far it got."""
    interpreter = python or venv_python()
    if not Path(interpreter).exists():
        return TrainingProbe(error=f"no interpreter at {interpreter}")

    env = {"PYTHONNOUSERSITE": "1"}
    env.update(extra_env or {})
    result = proc.capture(
        [interpreter, "-c", _TRAIN_PROBE, str(iterations)],
        env=env,
        timeout=900,
    )

    payload: dict[str, Any] | None = None
    for line in reversed(result.lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break

    if payload is None:
        tail = "; ".join(result.lines[-4:]) or "no output"
        return TrainingProbe(
            ok=False,
            requested=iterations,
            error=f"probe crashed (exit {result.returncode}): {tail}",
            aborted=True,
        )

    return TrainingProbe(
        ok=bool(payload.get("ok")),
        iterations=int(payload.get("iterations") or 0),
        requested=iterations,
        error=payload.get("error"),
        stage=str(payload.get("stage") or ""),
    )


def record_torch(state: SetupState, info: TorchInfo) -> SetupState:
    import time

    state.torch_info = asdict(info)
    state.hsa_override = info.hsa_override or ""
    state.checked_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    if info.torch_version:
        state.torch_local_version = info.torch_version
    state.save()
    if info.hsa_override:
        write_env_sh({"HSA_OVERRIDE_GFX_VERSION": info.hsa_override})
    return state


# --------------------------------------------------------------------------
# Environment assembly
# --------------------------------------------------------------------------


def write_env_sh(values: dict[str, str]) -> None:
    """Persist env vars that ``./run`` sources on every invocation."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if ENV_SH.exists():
        for line in ENV_SH.read_text(encoding="utf-8").splitlines():
            if line.startswith("export "):
                key, _, value = line[len("export ") :].partition("=")
                existing[key.strip()] = value.strip().strip('"')
    existing.update(values)
    body = [
        "# Written by ./run setup. Sourced by ./run on every invocation.",
        "# Delete this file to forget the detected settings.",
    ]
    body += [f'export {key}="{value}"' for key, value in sorted(existing.items())]
    ENV_SH.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")


def torch_constraint_file() -> Path | None:
    """Path to the pip constraint pinning torch, if setup recorded one.

    Every pip call after torch is installed passes this as ``PIP_CONSTRAINT``.
    Without it, ``openai-whisper`` — whose metadata wants an unpinned torch plus
    CUDA-flavoured triton — will happily replace a ROCm build.
    """
    return TORCH_CONSTRAINT if TORCH_CONSTRAINT.exists() else None


def pip_env() -> dict[str, str]:
    env = {"PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    constraint = torch_constraint_file()
    if constraint:
        env["PIP_CONSTRAINT"] = str(constraint)
    return env


def resolved_hardware(hardware_name: str = "auto") -> HardwareProfile:
    """The hardware profile in force: configuration plus the recorded target.

    Reads the gfx target from setup state rather than probing, so this is cheap
    enough to call from an error path.
    """
    from . import hardware as hardware_mod

    return hardware_mod.resolve(hardware_name, SetupState.load().info.gcn_arch)


def training_env(
    runtime_env: dict[str, str] | None = None,
    *,
    offline: bool = False,
    num_workers: int | None = None,
    hardware_name: str = "auto",
) -> dict[str, str]:
    """Environment for a training/inference subprocess.

    Layered, most general first: inferred defaults, then the hardware profile,
    then whatever the user put in ``runtime.env``. The user always wins — but a
    setting the hardware profile bans is never *inferred*, only ever set
    deliberately.
    """
    state = SetupState.load()
    profile = resolved_hardware(hardware_name)
    env: dict[str, str] = {}

    # lightning.yaml can name a class_path inside this package (the checkpoint
    # policy in train/callbacks.py). ./run already exports this, but making it
    # explicit means the recorded command reproduces by hand and --print_config
    # can resolve the class the same way the real run does.
    src = str(REPO_ROOT / "src")
    inherited = os.environ.get("PYTHONPATH", "")
    parts = [p for p in inherited.split(os.pathsep) if p]
    if src not in parts:
        parts.insert(0, src)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    if state.hsa_override and "HSA_OVERRIDE_GFX_VERSION" not in profile.banned_env:
        env["HSA_OVERRIDE_GFX_VERSION"] = state.hsa_override

    if state.info.hip and profile.is_generic:
        # Biggest single anti-fragmentation win on a shared-memory APU, where
        # the GPU and the desktop compete for the same physical RAM. Left to
        # hardware profiles otherwise: on a board whose failure mode is
        # allocation churn, remapping virtual segments is not obviously safe.
        env.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    env.update(profile.env)

    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"

    # Dataloader workers plus torch's own thread pool oversubscribe small CPUs.
    if num_workers:
        env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // max(1, num_workers))))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    # User overrides win over everything we inferred.
    for key, value in (runtime_env or {}).items():
        env[str(key)] = str(value)
    return env


def is_offline(profile_offline: bool = False) -> bool:
    return bool(profile_offline or os.environ.get("PT_OFFLINE"))


def free_gib(path: Path) -> float:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024**3)


def recommended_batch_size(info: TorchInfo) -> int:
    """Starting batch size from device memory. States its own assumptions."""
    if not info.usable_gpu:
        return 4  # CPU: correctness runs only
    gib = info.total_memory_gib
    if gib <= 6:
        return 4
    if gib <= 10:
        return 8
    if gib <= 16:
        return 8
    if gib <= 24:
        return 16
    return 32


def low_vram(info: TorchInfo) -> bool:
    return info.usable_gpu and info.total_memory_gib <= 16.0


def python_candidates() -> list[str]:
    return pins.load().python_prefer
