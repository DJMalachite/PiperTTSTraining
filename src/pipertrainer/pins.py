"""Reader for ``pins.toml``.

Kept separate and tiny so that ``doctor`` can report pin drift without
importing anything that needs the venv.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .paths import PINS_FILE

MIN_PYTHON = (3, 11)


class PinsError(RuntimeError):
    pass


def _load_toml(path) -> dict[str, Any]:
    if sys.version_info < MIN_PYTHON:
        raise PinsError(
            f"this tool needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
            f"(running {sys.version.split()[0]}); tomllib is unavailable"
        )
    import tomllib

    with open(path, "rb") as handle:
        return tomllib.load(handle)


@dataclass(frozen=True)
class TorchPin:
    """How to install torch for one vendor on one platform.

    Two delivery mechanisms, because vendors do not agree on one:

    * ``index`` + ``spec`` — a package index that replaces PyPI for this
      install. What download.pytorch.org offers, and the common case.
    * ``wheels`` — explicit wheel URLs, installed as files. AMD publishes the
      native-Windows ROCm build this way, and there is no index to point at.

    ``prerequisites`` are installed first and separately: the Windows ROCm
    runtime is itself a set of wheels that torch links against.
    """

    index: str = ""
    spec: str = ""
    wheels: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    #: Exact ``major.minor`` this delivery is built for, when only one exists.
    requires_python: str = ""
    #: Minimum GPU driver version, reported rather than enforced.
    driver: str = ""
    #: Vendor documentation for this particular path.
    docs: str = ""

    @property
    def from_urls(self) -> bool:
        """Whether this pin installs wheel files rather than resolving a name."""
        return bool(self.wheels)

    @property
    def version(self) -> str:
        """`torch==2.9.1` -> `2.9.1`."""
        if self.from_urls:
            # torch-2.9.1%2Brocm7.2.1-cp312-...whl -> 2.9.1+rocm7.2.1
            name = self.wheels[0].rsplit("/", 1)[-1].replace("%2B", "+")
            parts = name.split("-")
            return parts[1] if len(parts) > 1 else name
        return self.spec.partition("==")[2] or self.spec

    @property
    def describe(self) -> str:
        if self.from_urls:
            host = self.wheels[0].split("/")[2]
            return f"torch {self.version} from {host}"
        return f"{self.spec} from {self.index}"


@dataclass(frozen=True)
class Pins:
    raw: dict[str, Any]

    # --- piper1-gpl -------------------------------------------------------
    @property
    def piper_repo(self) -> str:
        return self.raw["piper"]["repo"]

    @property
    def piper_tag(self) -> str:
        return self.raw["piper"]["tag"]

    @property
    def piper_sha(self) -> str:
        return self.raw["piper"]["sha"]

    @property
    def espeak_ng_tag(self) -> str:
        return self.raw["piper"]["espeak_ng_tag"]

    # --- python -----------------------------------------------------------
    @property
    def python_min(self) -> tuple[int, ...]:
        return tuple(int(p) for p in str(self.raw["python"]["min"]).split("."))

    @property
    def python_prefer(self) -> list[list[str]]:
        """Candidate interpreters to probe, as argv lists, best first.

        Argv rather than a bare name because Windows has no versioned
        interpreter on PATH: ``py -3.13`` is how you ask for one, and that is
        two arguments. ``prefer_windows`` is optional so an older pins.toml
        still loads on Linux.
        """
        entry = self.raw["python"]
        key = "prefer_windows" if sys.platform == "win32" else "prefer"
        return [list(candidate) for candidate in entry.get(key, entry["prefer"])]

    # --- build backend ----------------------------------------------------
    @property
    def build_requires(self) -> list[str]:
        """What piper1-gpl's setup.py needs when run outside pip's isolation."""
        return list(self.raw["build"]["requires"])

    # --- export -----------------------------------------------------------
    @property
    def export_requires(self) -> list[str]:
        """What `torch.onnx.export` needs that piper does not declare."""
        return list(self.raw["export"]["requires"])

    # --- whisper ----------------------------------------------------------
    @property
    def whisper_package(self) -> str:
        return self.raw["whisper"]["package"]

    @property
    def whisper_deps(self) -> list[str]:
        return list(self.raw["whisper"]["deps"])

    # --- torch ------------------------------------------------------------
    def torch(self, vendor: str) -> TorchPin:
        """The pin for ``vendor`` on the running platform.

        A vendor may need a different delivery per platform — ROCm does, since
        AMD ships native-Windows wheels from their own repository rather than
        through download.pytorch.org. A ``[torch.<vendor>.windows]`` sub-table
        overrides the vendor default there.
        """
        try:
            entry = self.raw["torch"][vendor]
        except KeyError as exc:
            raise PinsError(f"unknown vendor {vendor!r} in pins.toml") from exc

        override = entry.get("windows") if sys.platform == "win32" else None
        if override:
            return TorchPin(
                index=str(override.get("index", "")),
                spec=str(override.get("spec", "")),
                wheels=tuple(override.get("wheels", ())),
                prerequisites=tuple(override.get("prerequisites", ())),
                requires_python=str(override.get("requires_python", "")),
                driver=str(override.get("driver", "")),
                docs=str(override.get("docs", "")),
            )
        return TorchPin(index=entry["index"], spec=entry["spec"])

    def torch_alternatives(self, vendor: str) -> list[TorchPin]:
        entry = self.raw["torch"].get(vendor, {})
        if sys.platform == "win32" and entry.get("windows"):
            # The Windows delivery is a single published set, not a menu of
            # ROCm versions to try. Offering the Linux indexes here would send
            # someone after wheels that do not exist for their platform.
            return []
        return [
            TorchPin(index=alt["index"], spec=alt["spec"])
            for alt in entry.get("alternatives", [])
        ]

    @property
    def vendors(self) -> list[str]:
        return sorted(self.raw["torch"])

    # --- checkpoints ------------------------------------------------------
    @property
    def checkpoint_api(self) -> str:
        return self.raw["checkpoints"]["api"]

    @property
    def checkpoint_resolve(self) -> str:
        return self.raw["checkpoints"]["resolve"]

    @property
    def checkpoint_repo(self) -> str:
        return self.raw["checkpoints"]["repo"]

    # --- disk -------------------------------------------------------------
    @property
    def setup_free_gib(self) -> int:
        return int(self.raw["disk"]["setup_free_gib"])

    @property
    def run_free_gib(self) -> int:
        return int(self.raw["disk"]["run_free_gib"])


@lru_cache(maxsize=1)
def load(path=None) -> Pins:
    target = path or PINS_FILE
    if not target.exists():
        raise PinsError(f"missing {target}")
    return Pins(raw=_load_toml(target))
