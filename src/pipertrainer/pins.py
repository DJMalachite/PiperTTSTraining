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
    """One (index-url, spec) pair. See pins.toml on why --index-url."""

    index: str
    spec: str

    @property
    def version(self) -> str:
        """`torch==2.9.1` -> `2.9.1`."""
        return self.spec.partition("==")[2] or self.spec


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
    def python_prefer(self) -> list[str]:
        return list(self.raw["python"]["prefer"])

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
        try:
            entry = self.raw["torch"][vendor]
        except KeyError as exc:
            raise PinsError(f"unknown vendor {vendor!r} in pins.toml") from exc
        return TorchPin(index=entry["index"], spec=entry["spec"])

    def torch_alternatives(self, vendor: str) -> list[TorchPin]:
        entry = self.raw["torch"].get(vendor, {})
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
