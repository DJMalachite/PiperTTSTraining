"""Filesystem layout.

Every path the tool reads or writes is derived here, so the layout is
described in exactly one place. Two rules hold throughout:

* ``VoicePaths.source`` is **read-only**. The dataset pipeline never writes to
  or deletes anything under it — the user's original recording is theirs.
* ``piper1-gpl/`` is a pinned upstream clone and is never edited. Workarounds
  for upstream bugs live on our side of the boundary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# src/pipertrainer/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(name: str) -> str:
    """Collapse arbitrary text into something safe for a path component."""
    cleaned = _UNSAFE.sub("-", name.strip()).strip("-._")
    return cleaned or "voice"


def env_suffix() -> str:
    """``PIPERTRAINER_ENV=bc250`` -> ``-bc250``; unset -> ``""``.

    One repo can need more than one installed environment. The BC-250 builds a
    gfx1013 torch inside a Fedora container that shares ``$HOME`` with the host
    (see docs/BC250.md), and a venv built there must not collide with the
    host's — different distribution, different libc, different torch. Naming
    the environment gives each its own ``.venv-<name>`` and ``.state-<name>``.

    Unset is the overwhelmingly common case and must keep the historic layout
    byte for byte, so that existing clones see no change at all.
    """
    name = os.environ.get("PIPERTRAINER_ENV", "").strip()
    return f"-{slug(name)}" if name else ""


PINS_FILE = REPO_ROOT / "pins.toml"
VENV_DIR = REPO_ROOT / f".venv{env_suffix()}"
STATE_DIR = REPO_ROOT / f".state{env_suffix()}"
PIPER_DIR = REPO_ROOT / "piper1-gpl"
PROFILES_DIR = REPO_ROOT / "profiles"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
VOICES_DIR = REPO_ROOT / "voices"
DOCS_DIR = REPO_ROOT / "docs"

SETUP_STATE = STATE_DIR / "setup.json"
ENV_SH = STATE_DIR / "env.sh"
TORCH_CONSTRAINT = STATE_DIR / "torch-constraint.txt"
ACTIVE_PROFILE = STATE_DIR / "active-profile"

def venv_python() -> Path:
    """Interpreter inside the project venv (may not exist before setup)."""
    return VENV_DIR / "bin" / "python"


def venv_bin(name: str) -> Path:
    return VENV_DIR / "bin" / name


def in_venv() -> bool:
    return venv_python().exists()


def profile_path(voice: str) -> Path:
    return PROFILES_DIR / f"{slug(voice)}.yaml"


@dataclass(frozen=True)
class VoicePaths:
    """All paths belonging to one voice, derived from its slug."""

    name: str

    # --- dataset side: data/<voice>/ ---------------------------------------
    @property
    def data_root(self) -> Path:
        return DATA_DIR / slug(self.name)

    @property
    def source(self) -> Path:
        """Read-only. The user's original audio lives here; we never mutate it."""
        return self.data_root / "source"

    @property
    def wavs(self) -> Path:
        return self.data_root / "wavs"

    @property
    def metadata_csv(self) -> Path:
        return self.data_root / "metadata.csv"

    @property
    def rejected_csv(self) -> Path:
        """Clips the pipeline refused, with a reason column. Never silent."""
        return self.data_root / "rejected.csv"

    @property
    def report_md(self) -> Path:
        return self.data_root / "report.md"

    @property
    def manifest_json(self) -> Path:
        """Per-stage fingerprints so re-runs skip completed stages."""
        return self.data_root / "manifest.json"

    @property
    def asr_cache(self) -> Path:
        """One JSON of Whisper words per macro-segment, keyed by index."""
        return self.data_root / ".cache" / "asr"

    @property
    def decoded_cache(self) -> Path:
        return self.data_root / ".cache" / "decoded"

    # --- training side: runs/<voice>/ -------------------------------------
    @property
    def run_root(self) -> Path:
        return RUNS_DIR / slug(self.name)

    @property
    def cache(self) -> Path:
        """piper's --data.cache_dir. Incremental; fingerprinted by us."""
        return self.run_root / "cache"

    @property
    def cache_fingerprint(self) -> Path:
        return self.cache / ".fingerprint"

    @property
    def lightning_yaml(self) -> Path:
        return self.run_root / "lightning.yaml"

    @property
    def piper_config_json(self) -> Path:
        """piper's --data.config_path. Written by training, not by export."""
        return self.run_root / "config.json"

    @property
    def lightning_logs(self) -> Path:
        return self.run_root / "lightning_logs"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def previews(self) -> Path:
        return self.run_root / "previews"

    # --- output side: voices/<voice>/ -------------------------------------
    @property
    def voice_out(self) -> Path:
        return VOICES_DIR / slug(self.name)

    def ensure_dataset_dirs(self) -> None:
        for path in (self.source, self.wavs, self.asr_cache):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_run_dirs(self) -> None:
        for path in (self.cache, self.logs):
            path.mkdir(parents=True, exist_ok=True)
