"""Browsing and downloading pretrained checkpoints from HuggingFace.

Fine-tuning from an existing checkpoint is the single biggest saving available —
upstream recommends it even across languages — so this exists to make finding one
a menu rather than a research task.

The tree is ``<lang>/<lang_REGION>/<speaker>/<quality>/``, and filenames are
*not* uniform: ``epoch=6679-step=1554200.ckpt``, ``last.ckpt`` and
``bryce-3499.ckpt`` all appear. So the listing always comes from the API rather
than being constructed. Files are 845-935 MB, which is why free space is checked
before starting and downloads resume.

Only ``medium`` checkpoints match this repo's default architecture; anything else
needs ``finetune.mode`` set to a warmstart. ``train/argmap.py`` enforces that by
comparing the checkpoint's stored hyper-parameters.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import env as env_mod
from . import pins as pins_mod
from . import profile as profile_mod
from . import tui
from .paths import CHECKPOINTS_DIR

USER_AGENT = "pipertrainer/0.1 (+https://github.com/DJMalachite/PiperTTSTraining)"
TIMEOUT = 60
BLOCK = 1024 * 256

# A language-agnostic starting point when no voice in the target language
# exists. Documented upstream as the base model.
BASE_MODEL = "_base_model/base_model.ckpt"


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "directory" | "file"
    size: int = 0

    @property
    def name(self) -> str:
        return self.path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def is_dir(self) -> bool:
        return self.kind == "directory"

    @property
    def size_mb(self) -> float:
        return self.size / 1e6


def _fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise CheckpointError(f"HuggingFace returned {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise CheckpointError(
            f"could not reach HuggingFace ({exc.reason}). Use --offline and "
            f"stage checkpoints under {CHECKPOINTS_DIR}/ if this machine has no "
            f"network."
        ) from exc


def list_tree(path: str = "", *, recursive: bool = False) -> list[Entry]:
    """List one level of the checkpoint repository."""
    pins = pins_mod.load()
    url = f"{pins.checkpoint_api}/tree/main"
    if path:
        url += f"/{path.strip('/')}"
    if recursive:
        url += "?recursive=true"

    payload = _fetch_json(url)
    entries = [
        Entry(
            path=item.get("path", ""),
            kind=item.get("type", "file"),
            size=int(item.get("size") or 0),
        )
        for item in payload
        if item.get("path")
    ]
    return sorted(entries, key=lambda entry: (not entry.is_dir, entry.path))


def local_path(remote_path: str) -> Path:
    """Mirror the remote layout locally so files are reusable and obvious."""
    return CHECKPOINTS_DIR / remote_path.strip("/")


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------


def download(remote_path: str, *, offline: bool = False, force: bool = False) -> Path:
    if offline:
        raise CheckpointError(
            f"offline mode: downloads are disabled. Copy the file to "
            f"{local_path(remote_path)} yourself."
        )

    pins = pins_mod.load()
    destination = local_path(remote_path)
    if destination.exists() and not force:
        tui.ok(f"already downloaded: {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
        return destination

    url = f"{pins.checkpoint_resolve}/{remote_path.strip('/')}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    already = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if already:
        headers["Range"] = f"bytes={already}-"
        tui.info(f"resuming from {already / 1e6:.0f} MB")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and already:
            # Range not satisfiable: the partial file is already complete.
            partial.replace(destination)
            return destination
        raise CheckpointError(f"download failed ({exc.code}) for {url}") from exc
    except urllib.error.URLError as exc:
        raise CheckpointError(f"download failed: {exc.reason}") from exc

    with response:
        declared = int(response.headers.get("Content-Length") or 0)
        expected = declared + (already if response.status == 206 else 0)
        if expected:
            free = env_mod.free_gib(CHECKPOINTS_DIR)
            need = (expected - already) / (1024**3)
            if free < need + 1.0:
                raise CheckpointError(
                    f"need about {need:.1f} GiB but only {free:.1f} GiB is free "
                    f"at {CHECKPOINTS_DIR}"
                )

        mode = "ab" if response.status == 206 and already else "wb"
        if mode == "wb":
            already = 0
        written = already
        with open(partial, mode) as handle:
            while True:
                chunk = response.read(BLOCK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if expected:
                    _progress(written, expected)
        if expected:
            print()

    if expected and written != expected:
        raise CheckpointError(
            f"download is short: got {written} bytes, expected {expected}. "
            f"The partial file is kept at {partial}; re-run to resume."
        )

    partial.replace(destination)
    tui.ok(f"downloaded {destination.name} ({written / 1e6:.0f} MB)")
    return destination


def _progress(written: int, total: int) -> None:
    fraction = written / total if total else 0
    width = 30
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r  [{bar}] {written / 1e6:6.0f} / {total / 1e6:.0f} MB "
        f"({fraction * 100:3.0f}%)",
        end="",
        flush=True,
    )


def download_sidecar(remote_dir: str) -> dict | None:
    """Fetch a checkpoint's config.json so we can report its architecture.

    Worth doing before committing to a 900 MB download: it carries the sample
    rate and phoneme count, which is what decides whether ``--ckpt_path`` will
    load at all.
    """
    pins = pins_mod.load()
    url = f"{pins.checkpoint_resolve}/{remote_dir.strip('/')}/config.json"
    try:
        return _fetch_json(url)
    except CheckpointError:
        return None


# --------------------------------------------------------------------------
# Listing commands
# --------------------------------------------------------------------------


def list_remote(path: str = "", *, offline: bool = False) -> int:
    if offline:
        tui.warn("offline mode: cannot list the remote repository")
        return list_local()
    entries = list_tree(path)
    if not entries:
        tui.warn(f"nothing at {path or '/'}")
        return 1
    tui.heading(f"rhasspy/piper-checkpoints:{path or '/'}")
    rows = [
        [
            entry.name + ("/" if entry.is_dir else ""),
            "" if entry.is_dir else f"{entry.size_mb:.0f} MB",
        ]
        for entry in entries
    ]
    tui.table(rows)
    return 0


def list_local() -> int:
    if not CHECKPOINTS_DIR.exists():
        tui.info(f"no checkpoints downloaded yet ({CHECKPOINTS_DIR} does not exist)")
        return 0
    found = sorted(CHECKPOINTS_DIR.rglob("*.ckpt"))
    if not found:
        tui.info(f"no checkpoints under {CHECKPOINTS_DIR}")
        return 0
    tui.heading("Downloaded checkpoints")
    tui.table(
        [
            [
                str(path.relative_to(CHECKPOINTS_DIR)),
                f"{path.stat().st_size / 1e6:.0f} MB",
            ]
            for path in found
        ]
    )
    total = sum(path.stat().st_size for path in found)
    tui.info("")
    tui.info(f"{len(found)} file(s), {total / 1e9:.2f} GB")
    return 0


# --------------------------------------------------------------------------
# Interactive browsing
# --------------------------------------------------------------------------


def _suggest(prof: profile_mod.Profile) -> list[str]:
    """Where to look first, given the voice's language and quality."""
    language = prof.voice.language.strip()
    short = language.split("_")[0] if language else ""
    return [part for part in (short, language) if part]


def browse(profile_name: str | None = None, offline: bool = False) -> int:
    prof: profile_mod.Profile | None = None
    try:
        prof, _ = profile_mod.resolve_active(profile_name)
    except profile_mod.ProfileError:
        pass

    if offline:
        tui.warn(
            "offline mode: browsing is disabled. Stage .ckpt files under "
            f"{CHECKPOINTS_DIR}/ and point finetune.checkpoint at one."
        )
        return list_local()

    tui.heading("Pretrained checkpoints")
    tui.hint(
        "  Fine-tuning from one of these is dramatically faster than training "
        "from scratch, and upstream recommends it even across languages."
    )
    if prof is not None:
        tui.hint(
            f"  Your profile is {prof.voice.quality!r} quality at "
            f"{prof.audio.sample_rate} Hz. Only 'medium' checkpoints match the "
            f"default architecture for a strict --ckpt_path resume."
        )

    path = ""
    trail: list[str] = []
    while True:
        try:
            entries = list_tree(path)
        except CheckpointError as exc:
            tui.error(str(exc))
            return 1

        files = [entry for entry in entries if not entry.is_dir]
        directories = [entry for entry in entries if entry.is_dir]
        checkpoint_files = [entry for entry in files if entry.path.endswith(".ckpt")]

        options: list[tuple[str, str]] = []
        for entry in directories:
            hint = ""
            if prof is not None and entry.name in _suggest(prof):
                hint = "  <- matches your voice.language"
            options.append((f"d:{entry.path}", f"{entry.name}/{hint}"))
        for entry in checkpoint_files:
            local = local_path(entry.path)
            marker = " (downloaded)" if local.exists() else ""
            options.append(
                (f"f:{entry.path}", f"{entry.name} — {entry.size_mb:.0f} MB{marker}")
            )
        if not path:
            options.append((f"f:{BASE_MODEL}", "_base_model/base_model.ckpt — language-agnostic fallback"))

        if not options:
            tui.warn("nothing selectable here")
            if not trail:
                return 1
            path = trail.pop()
            continue

        title = f"/{path}" if path else "/ (language)"
        try:
            choice = tui.menu(title, options, allow_back=True)
        except tui.Back:
            if not trail:
                return 0
            path = trail.pop()
            continue

        kind, _, target = choice.partition(":")
        if kind == "d":
            trail.append(path)
            path = target
            continue

        return _confirm_and_download(target, prof)


def _confirm_and_download(
    remote_path: str, prof: profile_mod.Profile | None
) -> int:
    remote_dir = remote_path.rsplit("/", 1)[0]
    sidecar = download_sidecar(remote_dir)
    if sidecar:
        rate = sidecar.get("audio", {}).get("sample_rate")
        tui.info("")
        tui.table(
            [
                ["sample rate", str(rate)],
                ["phonemes", str(sidecar.get("num_symbols"))],
                ["speakers", str(sidecar.get("num_speakers"))],
                ["espeak voice", str(sidecar.get("espeak", {}).get("voice"))],
            ]
        )
        if prof is not None and rate and int(rate) != int(prof.audio.sample_rate):
            tui.warn(
                f"this checkpoint is {rate} Hz but your profile trains at "
                f"{prof.audio.sample_rate} Hz. A strict --ckpt_path resume will "
                f"be refused; use finetune.mode 'vocoder_warmstart', or change "
                f"audio.sample_rate to match."
            )
    else:
        tui.hint("  no config.json alongside this checkpoint; architecture unknown")

    tui.info("")
    if not tui.confirm(f"download {remote_path}?", default=True):
        return 0

    path = download(remote_path)

    if prof is not None and tui.confirm(
        f"set it as {prof.voice.name}'s fine-tuning checkpoint?", default=True
    ):
        prof.finetune.checkpoint = str(path)
        if prof.finetune.mode == "none":
            prof.finetune.mode = "ckpt_path"
        profile_mod.save(prof)
        tui.ok(f"profile updated: finetune.mode={prof.finetune.mode}")
    return 0
