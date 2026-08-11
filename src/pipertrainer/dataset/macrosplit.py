"""Coarse first pass: break a long recording at obvious silences.

Two reasons this exists, both about long recordings.

**Whisper drift.** Given hours of audio in one call, Whisper's timestamps drift
and it becomes prone to repetition loops. Bounding each call to ten minutes of
audio that starts and ends in silence keeps alignment tight.

**Resumability.** Transcription is the slow stage. Macro segments are the unit
of work that gets cached, so an interrupted run resumes at the segment boundary
instead of starting over.

The silence threshold is estimated from the recording's own noise floor by
default. A fixed dBFS value — the approach the legacy script hardcoded at -40 —
is wrong for both a quiet studio capture and a noisy phone recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FRAME_SECONDS = 0.02  # 20 ms; fine enough for word gaps, cheap over hours


@dataclass(frozen=True)
class Segment:
    """A coarse chunk, in seconds relative to the start of the recording."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SilenceProfile:
    threshold_dbfs: float
    floor_dbfs: float
    speech_dbfs: float
    estimated: bool

    def describe(self) -> str:
        how = "estimated" if self.estimated else "configured"
        return (
            f"silence below {self.threshold_dbfs:.1f} dBFS ({how}; noise floor "
            f"{self.floor_dbfs:.1f}, speech {self.speech_dbfs:.1f})"
        )


def frame_levels(samples, sample_rate: int, frame_seconds: float = FRAME_SECONDS):
    """RMS level per frame, in dBFS. Returns a numpy array."""
    import numpy as np

    frame = max(1, int(sample_rate * frame_seconds))
    usable = (len(samples) // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype="f4")
    # Read through the memmap in blocks so a multi-hour file never lands in RAM
    # all at once.
    block_frames = max(1, int(60.0 / frame_seconds))  # ~1 minute per block
    out = np.empty(usable // frame, dtype="f4")
    position = 0
    while position < usable // frame:
        count = min(block_frames, usable // frame - position)
        chunk = np.asarray(
            samples[position * frame : (position + count) * frame], dtype="f4"
        ).reshape(count, frame)
        rms = np.sqrt(np.mean(np.square(chunk), axis=1))
        out[position : position + count] = rms
        position += count
    # -100 dBFS stands in for digital silence, avoiding log(0).
    np.maximum(out, 1e-5, out=out)
    return 20.0 * np.log10(out)


def estimate_silence(levels, configured_dbfs: float = 0.0) -> SilenceProfile:
    """Pick a silence threshold from the level distribution.

    The 5th percentile approximates the noise floor and the 85th approximates
    speech. Sitting the threshold a fixed margin above the floor, but never
    within 6 dB of speech, works across both quiet and noisy recordings.
    """
    import numpy as np

    if len(levels) == 0:
        return SilenceProfile(-40.0, -100.0, 0.0, estimated=True)

    floor = float(np.percentile(levels, 5))
    speech = float(np.percentile(levels, 85))

    if configured_dbfs < 0.0:
        return SilenceProfile(
            threshold_dbfs=configured_dbfs,
            floor_dbfs=floor,
            speech_dbfs=speech,
            estimated=False,
        )

    threshold = floor + 8.0
    # Keep clear of the speech level, and never above a sane ceiling.
    threshold = min(threshold, speech - 6.0, -20.0)
    threshold = max(threshold, floor + 2.0, -75.0)
    return SilenceProfile(
        threshold_dbfs=threshold,
        floor_dbfs=floor,
        speech_dbfs=speech,
        estimated=True,
    )


def silent_runs(
    levels, threshold_dbfs: float, frame_seconds: float = FRAME_SECONDS
) -> list[tuple[float, float]]:
    """Silence spans as ``(start_seconds, end_seconds)``."""
    import numpy as np

    if len(levels) == 0:
        return []
    quiet = np.asarray(levels) < threshold_dbfs
    # Find run boundaries by looking at transitions.
    padded = np.concatenate(([False], quiet, [False]))
    edges = np.flatnonzero(padded[:-1] != padded[1:])
    runs: list[tuple[float, float]] = []
    for start, end in zip(edges[0::2], edges[1::2]):
        runs.append((float(start * frame_seconds), float(end * frame_seconds)))
    return runs


def speech_spans(
    levels,
    threshold_dbfs: float,
    *,
    frame_seconds: float = FRAME_SECONDS,
    min_silence: float = 0.35,
    total_duration: float | None = None,
) -> list[tuple[float, float]]:
    """Inverse of ``silent_runs``: spans that contain speech.

    Silences shorter than ``min_silence`` are not treated as boundaries, so
    normal inter-word gaps do not fragment a sentence. Used by the ``vad``
    strategy and by the report's coverage figure.
    """
    duration = total_duration
    if duration is None:
        duration = len(levels) * frame_seconds

    boundaries = [
        run for run in silent_runs(levels, threshold_dbfs, frame_seconds)
        if (run[1] - run[0]) >= min_silence
    ]
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in boundaries:
        if start > cursor:
            spans.append((cursor, start))
        cursor = end
    if cursor < duration:
        spans.append((cursor, duration))
    return [(a, b) for a, b in spans if b > a]


def split(
    levels,
    *,
    total_duration: float,
    silence: SilenceProfile,
    min_silence_seconds: float = 1.5,
    max_seconds: float = 600.0,
    frame_seconds: float = FRAME_SECONDS,
) -> list[Segment]:
    """Cut the recording into chunks of at most ``max_seconds``.

    Boundaries land at the midpoint of a long silence. If a stretch of speech
    has no long-enough silence within the limit, it is cut at the quietest frame
    available rather than at a fixed offset — a hard cut mid-word would corrupt
    both neighbouring transcripts.
    """
    import numpy as np

    if total_duration <= max_seconds:
        return [Segment(index=0, start=0.0, end=total_duration)]

    candidates = [
        (start + end) / 2.0
        for start, end in silent_runs(levels, silence.threshold_dbfs, frame_seconds)
        if (end - start) >= min_silence_seconds
    ]

    segments: list[Segment] = []
    cursor = 0.0
    array = np.asarray(levels)

    while total_duration - cursor > max_seconds:
        limit = cursor + max_seconds
        # Prefer the last long silence that fits, but not one so early that the
        # chunk is tiny.
        usable = [
            point
            for point in candidates
            if cursor + min(30.0, max_seconds / 4) < point <= limit
        ]
        if usable:
            cut = max(usable)
        else:
            # No long silence: take the quietest frame in the last quarter of
            # the window, which is the least-bad place to break.
            low = int((limit - max_seconds / 4) / frame_seconds)
            high = min(len(array), int(limit / frame_seconds))
            if high <= low:
                cut = limit
            else:
                cut = float((low + int(np.argmin(array[low:high]))) * frame_seconds)
        cut = min(max(cut, cursor + 1.0), limit)
        segments.append(Segment(index=len(segments), start=cursor, end=cut))
        cursor = cut

    if total_duration - cursor > 0.01:
        segments.append(
            Segment(index=len(segments), start=cursor, end=total_duration)
        )
    return segments


def coverage(spans: list[tuple[float, float]], total_duration: float) -> float:
    """Fraction of the recording that contains speech."""
    if total_duration <= 0:
        return 0.0
    covered = sum(max(0.0, end - start) for start, end in spans)
    return min(1.0, covered / total_duration)


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (math.inf, -math.inf):
        return "?"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
