"""Audio I/O via ffmpeg and the stdlib.

Deliberately not pydub or ffmpeg-python: both are thin wrappers that add a
dependency and hide the command line, and pydub in particular loads whole files
into Python objects. Here, ffmpeg is invoked once per *recording* to decode,
downmix, resample and filter in a single pass, writing raw float32 to a cache
file we then memory-map. Clips are cut from that map and written with the stdlib
``wave`` module — a dataset of 2000 clips would otherwise mean 2000 ffmpeg
processes.

Normalisation is applied to the **whole recording**, not per clip. Per-clip peak
normalisation flattens the natural loudness differences between a statement and
an aside, which is prosody the model should learn.

The source file is only ever read. Nothing here writes to or deletes it.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .. import proc

FLOAT_BYTES = 4


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str
    format_name: str
    size_bytes: int

    @property
    def pretty(self) -> str:
        minutes, seconds = divmod(int(self.duration), 60)
        hours, minutes = divmod(minutes, 60)
        clock = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
        return (
            f"{clock}, {self.sample_rate} Hz, "
            f"{'mono' if self.channels == 1 else f'{self.channels} ch'}, "
            f"{self.codec} in {self.format_name}"
        )


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise AudioError(
            f"{' and '.join(missing)} not found on PATH. Install ffmpeg: "
            f"'sudo pacman -S ffmpeg' on Arch/CachyOS, "
            f"'sudo apt-get install ffmpeg' on Debian."
        )


def probe(path: Path) -> AudioInfo:
    """Read container and stream metadata. Refuses files with no audio."""
    require_ffmpeg()
    if not path.exists():
        raise AudioError(f"no such file: {path}")

    result = proc.capture(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=300,
    )
    if not result.ok:
        raise AudioError(
            f"ffprobe could not read {path.name}:\n" + "\n".join(result.lines[-6:])
        )
    try:
        payload: dict[str, Any] = json.loads(result.output)
    except json.JSONDecodeError as exc:
        raise AudioError(f"ffprobe returned unparseable output for {path}") from exc

    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        kinds = {s.get("codec_type") for s in payload.get("streams", [])}
        raise AudioError(
            f"{path.name} has no audio stream (found: "
            f"{', '.join(sorted(k for k in kinds if k)) or 'nothing'})"
        )
    stream = streams[0]
    container = payload.get("format", {})

    duration = _first_float(
        stream.get("duration"), container.get("duration"), default=0.0
    )
    return AudioInfo(
        path=path,
        duration=duration,
        sample_rate=int(_first_float(stream.get("sample_rate"), default=0) or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or "unknown"),
        format_name=str(container.get("format_name") or "unknown"),
        size_bytes=int(_first_float(container.get("size"), default=0) or 0),
    )


def _first_float(*values: Any, default: float | None = None) -> float:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    if default is None:
        raise AudioError("expected a number, found none")
    return default


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def _filters(highpass_hz: int, loudnorm: bool) -> list[str]:
    chain: list[str] = []
    if highpass_hz > 0:
        # Second-order high-pass; removes rumble and handling noise the vocoder
        # would otherwise faithfully reproduce.
        chain.append(f"highpass=f={highpass_hz}:poles=2")
    if loudnorm:
        # Single-pass loudnorm. Two-pass is more accurate but needs the whole
        # file measured first; for dataset prep the difference is inaudible.
        chain.append("loudnorm=I=-23:TP=-2:LRA=7")
    return chain


def decode_key(
    path: Path, sample_rate: int, highpass_hz: int, loudnorm: bool
) -> str:
    """Cache key covering everything that changes the decoded samples."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8"))
    digest.update(f"|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8"))
    digest.update(f"|{sample_rate}|{highpass_hz}|{loudnorm}".encode("utf-8"))
    return digest.hexdigest()[:16]


def decode_to_raw(
    path: Path,
    destination: Path,
    *,
    sample_rate: int,
    highpass_hz: int = 0,
    loudnorm: bool = False,
    log_path: Path | None = None,
) -> Path:
    """Decode to raw float32 mono at ``sample_rate``. Returns the raw file.

    Materialised rather than streamed so that re-running the pipeline, or
    resuming after an interrupted transcription, does not decode a multi-hour
    recording again.
    """
    require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-map", "0:a:0",
    ]
    chain = _filters(highpass_hz, loudnorm)
    if chain:
        command += ["-af", ",".join(chain)]
    command += [
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-y",
        str(partial),
    ]

    proc.run(command, log_path=log_path, quiet=True, tail=20)
    if not partial.exists() or partial.stat().st_size < FLOAT_BYTES:
        raise AudioError(f"decoding {path.name} produced no samples")
    partial.replace(destination)
    return destination


def open_samples(raw_path: Path):
    """Memory-map a raw float32 file as a numpy array."""
    import numpy as np

    count = raw_path.stat().st_size // FLOAT_BYTES
    if count == 0:
        raise AudioError(f"{raw_path} is empty")
    return np.memmap(raw_path, dtype="<f4", mode="r", shape=(count,))


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


def peak_dbfs(samples) -> float:
    import numpy as np

    peak = float(np.max(np.abs(np.asarray(samples, dtype="f4")))) if len(samples) else 0.0
    if peak <= 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def peak_gain(samples, target_dbfs: float) -> float:
    """Gain that brings the whole recording's peak to ``target_dbfs``."""
    current = peak_dbfs(samples)
    if current == -math.inf:
        return 1.0
    return float(10.0 ** ((target_dbfs - current) / 20.0))


def clipped_sample_count(samples, threshold: float = 0.999) -> int:
    import numpy as np

    return int(np.count_nonzero(np.abs(np.asarray(samples)) >= threshold))


# --------------------------------------------------------------------------
# Writing clips
# --------------------------------------------------------------------------


def _to_int16(chunk, gain: float):
    import numpy as np

    scaled = np.asarray(chunk, dtype="f4") * gain
    np.clip(scaled, -1.0, 1.0, out=scaled)
    return (scaled * 32767.0).astype("<i2")


def find_zero_crossing(samples, index: int, window: int) -> int:
    """Nearest sign change to ``index``, to avoid a click at the cut."""
    import numpy as np

    total = len(samples)
    index = max(0, min(total - 1, index))
    low = max(0, index - window)
    high = min(total, index + window)
    if high - low < 3:
        return index
    region = np.asarray(samples[low:high], dtype="f4")
    signs = np.signbit(region)
    changes = np.flatnonzero(signs[:-1] != signs[1:])
    if changes.size == 0:
        return index
    absolute = changes + low
    return int(absolute[np.argmin(np.abs(absolute - index))])


def write_clip(
    samples,
    path: Path,
    *,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    gain: float = 1.0,
) -> int:
    """Write one 16-bit mono WAV. Returns the number of frames written."""
    start_sample = max(0, start_sample)
    end_sample = min(len(samples), end_sample)
    if end_sample <= start_sample:
        raise AudioError(
            f"empty clip for {path.name}: samples {start_sample}..{end_sample}"
        )
    chunk = _to_int16(samples[start_sample:end_sample], gain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(chunk.tobytes())
    return len(chunk)


def read_wav_info(path: Path) -> tuple[int, int, int]:
    """``(frames, sample_rate, channels)`` for a WAV file."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes(), handle.getframerate(), handle.getnchannels()


def write_silence_wav(path: Path, sample_rate: int, seconds: float) -> None:
    """Used by the self-test to make a placeholder without ffmpeg."""
    frames = int(sample_rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))


def synthesize_test_audio(
    path: Path,
    *,
    sample_rate: int,
    seconds: float,
    frequency: float,
    amplitude: float = 0.3,
    log_path: Path | None = None,
) -> Path:
    """Generate a tone plus noise with ffmpeg, for the CPU self-test.

    The smoke test needs audio with real structure but no speech; the model
    learns nothing from it, which is the point — the test proves plumbing.
    """
    require_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={seconds}",
        "-f", "lavfi",
        "-i", f"anoisesrc=sample_rate={sample_rate}:amplitude=0.02:duration={seconds}",
        "-filter_complex", f"[0:a][1:a]amix=inputs=2:duration=shortest,volume={amplitude}",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-y",
        str(path),
    ]
    proc.run(command, log_path=log_path, quiet=True)
    return path


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def samples_to_seconds(count: int, sample_rate: int) -> float:
    return count / float(sample_rate)


def concat_wavs(paths: Sequence[Path], destination: Path, sample_rate: int) -> None:
    """Join WAVs for a preview reel. Assumes matching format."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        for item in paths:
            with wave.open(str(item), "rb") as handle:
                out.writeframes(handle.readframes(handle.getnframes()))
