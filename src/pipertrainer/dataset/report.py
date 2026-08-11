"""Dataset quality report.

An automated pipeline can produce a dataset that trains fine and sounds wrong,
and the cause is nearly always visible in statistics rather than in the audio:
clips that are too short, transcripts whose length does not match their
duration, near-duplicate lines from a Whisper repetition loop, clipped samples.

The report is also where the split arithmetic is surfaced, because
``batch_size`` larger than the training split silently yields zero batches —
which Lightning reports as something else entirely.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..train.argmap import split_sizes
from .macrosplit import format_duration
from .metadata import Rejected, Row


@dataclass(frozen=True)
class ClipStat:
    utt_id: str
    seconds: float
    chars: int
    text: str
    peak_dbfs: float = 0.0
    clipped_samples: int = 0

    @property
    def chars_per_second(self) -> float:
        return self.chars / self.seconds if self.seconds > 0 else float("inf")


@dataclass
class Report:
    voice: str = ""
    source: str = ""
    source_duration: float = 0.0
    sample_rate: int = 22050
    strategy: str = "align"
    speech_coverage: float = 0.0
    silence_note: str = ""
    whisper_model: str = ""
    language: str = ""
    clips: list[ClipStat] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    flags: list[tuple[str, str]] = field(default_factory=list)
    validation_split: float = 0.1
    num_test_examples: int = 5
    batch_size: int = 16
    min_clip_floor: float = 1.0

    # -- aggregates -------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.clips)

    @property
    def total_seconds(self) -> float:
        return sum(clip.seconds for clip in self.clips)

    @property
    def durations(self) -> list[float]:
        return [clip.seconds for clip in self.clips]

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.durations) if self.clips else 0.0

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.durations) if self.clips else 0.0

    @property
    def shortest(self) -> float:
        return min(self.durations) if self.clips else 0.0

    @property
    def longest(self) -> float:
        return max(self.durations) if self.clips else 0.0

    @property
    def yield_fraction(self) -> float:
        if self.source_duration <= 0:
            return 0.0
        return self.total_seconds / self.source_duration

    @property
    def split(self):
        return split_sizes(self.count, self.validation_split, self.num_test_examples)

    # -- checks -----------------------------------------------------------
    def add_flag(self, level: str, message: str) -> None:
        self.flags.append((level, message))

    def analyse(self) -> None:
        """Populate ``flags`` from the collected statistics."""
        self.flags.clear()

        if not self.clips:
            self.add_flag(
                "error",
                "no usable clips were produced. Check that the source file "
                "contains speech, and try lowering dataset.min_seconds or "
                "raising dataset.max_seconds.",
            )
            return

        total_minutes = self.total_seconds / 60.0
        if total_minutes < 10:
            self.add_flag(
                "error",
                f"only {total_minutes:.1f} minutes of speech. Fine-tuning a "
                f"pretrained checkpoint can work from about 10 minutes, but "
                f"expect artefacts; 30+ minutes is a realistic floor and "
                f"1-2 hours is comfortable.",
            )
        elif total_minutes < 30:
            self.add_flag(
                "warn",
                f"{total_minutes:.1f} minutes of speech is on the low side. "
                f"Fine-tune from a pretrained checkpoint rather than training "
                f"from scratch.",
            )

        short = [clip for clip in self.clips if clip.seconds < self.min_clip_floor]
        if short:
            self.add_flag(
                "warn",
                f"{len(short)} clip(s) are shorter than {self.min_clip_floor} s. "
                f"piper pads every clip up to segment_size with silence, which "
                f"the model then learns to reproduce.",
            )

        empty = [clip for clip in self.clips if not clip.text.strip()]
        if empty:
            self.add_flag(
                "error", f"{len(empty)} clip(s) have an empty transcript"
            )

        single = [clip for clip in self.clips if len(clip.text.split()) == 1]
        if len(single) > max(3, self.count // 20):
            self.add_flag(
                "warn",
                f"{len(single)} clip(s) contain a single word. Consider raising "
                f"dataset.min_seconds.",
            )

        rates = sorted(clip.chars_per_second for clip in self.clips)
        if rates:
            low = [clip for clip in self.clips if clip.chars_per_second < 3.0]
            high = [clip for clip in self.clips if clip.chars_per_second > 30.0]
            if low:
                self.add_flag(
                    "warn",
                    f"{len(low)} clip(s) have unusually little text for their "
                    f"length — likely a transcript missing words.",
                )
            if high:
                self.add_flag(
                    "warn",
                    f"{len(high)} clip(s) have unusually much text for their "
                    f"length — likely text belonging to neighbouring audio.",
                )

        clipped = [clip for clip in self.clips if clip.clipped_samples > 8]
        if clipped:
            self.add_flag(
                "warn",
                f"{len(clipped)} clip(s) contain clipped samples. Distortion is "
                f"reproduced faithfully by the vocoder; consider re-recording "
                f"or setting audio.normalize to 'loudnorm'.",
            )

        seen: dict[str, int] = {}
        for clip in self.clips:
            key = clip.text.strip().lower()
            seen[key] = seen.get(key, 0) + 1
        duplicates = {text: n for text, n in seen.items() if n > 2 and text}
        if duplicates:
            worst = max(duplicates.items(), key=lambda item: item[1])
            self.add_flag(
                "warn",
                f"{len(duplicates)} transcript(s) repeat more than twice "
                f"(worst: {worst[1]}x {worst[0][:40]!r}). This is what a Whisper "
                f"repetition loop looks like.",
            )

        if self.source_duration > 0 and self.yield_fraction < 0.35:
            self.add_flag(
                "warn",
                f"only {self.yield_fraction * 100:.0f}% of the recording became "
                f"clips. Long music or noise sections, or a silence threshold "
                f"that is too high, are the usual causes.",
            )

        split = self.split
        if self.batch_size > split.train:
            self.add_flag(
                "error",
                f"data.batch_size {self.batch_size} exceeds the training split "
                f"of {split.train}. The dataloader drops the last partial batch, "
                f"so this yields zero batches per epoch. Use "
                f"{split.max_batch_size} or lower.",
            )

    # -- rendering --------------------------------------------------------
    def histogram(self, buckets: int = 12, width: int = 40) -> list[str]:
        if not self.clips:
            return []
        low, high = self.shortest, self.longest
        if high - low < 1e-6:
            return [f"{low:5.1f}s  {'#' * width}  {self.count}"]
        size = (high - low) / buckets
        counts = [0] * buckets
        for value in self.durations:
            index = min(buckets - 1, int((value - low) / size))
            counts[index] += 1
        peak = max(counts) or 1
        lines = []
        for index, count in enumerate(counts):
            start = low + index * size
            bar = "#" * int(round(width * count / peak))
            lines.append(f"{start:5.1f}s  {bar:<{width}}  {count}")
        return lines

    def summary_rows(self) -> list[list[str]]:
        split = self.split
        return [
            ["clips", str(self.count)],
            ["total speech", format_duration(self.total_seconds)],
            ["source length", format_duration(self.source_duration)],
            ["yield", f"{self.yield_fraction * 100:.0f}% of the recording"],
            [
                "clip length",
                f"{self.shortest:.1f}-{self.longest:.1f} s "
                f"(mean {self.mean_seconds:.1f}, median {self.median_seconds:.1f})",
            ],
            ["rejected", str(len(self.rejected))],
            [
                "split",
                f"{split.train} train / {split.val} val / {split.test} test",
            ],
            ["max batch size", str(split.max_batch_size)],
        ]

    def to_markdown(self) -> str:
        split = self.split
        lines: list[str] = [
            f"# Dataset report: {self.voice}",
            "",
            f"- source: `{self.source}` ({format_duration(self.source_duration)})",
            f"- strategy: `{self.strategy}`",
            f"- sample rate: {self.sample_rate} Hz",
            f"- whisper: {self.whisper_model}"
            + (f" (language {self.language})" if self.language else ""),
        ]
        if self.silence_note:
            lines.append(f"- {self.silence_note}")
        lines += ["", "## Summary", "", "| metric | value |", "| --- | --- |"]
        for name, value in self.summary_rows():
            lines.append(f"| {name} | {value} |")

        lines += [
            "",
            "## Training split",
            "",
            f"With `data.validation_split = {self.validation_split}` and "
            f"`data.num_test_examples = {self.num_test_examples}`:",
            "",
            f"- train: **{split.train}**",
            f"- validation: {split.val}",
            f"- test samples (logged to TensorBoard): {split.test}",
            "",
            f"`data.batch_size` must not exceed **{split.max_batch_size}**: the "
            f"dataloader drops the last partial batch, so a larger value gives "
            f"zero batches per epoch.",
        ]

        if self.clips:
            lines += ["", "## Clip length distribution", "", "```"]
            lines += self.histogram()
            lines += ["```"]

        if self.flags:
            lines += ["", "## Findings", ""]
            for level, message in self.flags:
                marker = {"error": "**error**", "warn": "warning", "info": "note"}[level]
                lines.append(f"- {marker}: {message}")
        else:
            lines += ["", "## Findings", "", "- nothing to report"]

        outliers = self.outliers()
        if outliers:
            lines += [
                "",
                "## Clips worth listening to",
                "",
                "Sorted by how far each is from the norm. Listening to the top "
                "few is the cheapest quality check there is.",
                "",
                "| clip | seconds | chars/s | transcript |",
                "| --- | --- | --- | --- |",
            ]
            for clip in outliers:
                text = clip.text.replace("|", "/")[:70]
                lines.append(
                    f"| `{clip.utt_id}` | {clip.seconds:.2f} | "
                    f"{clip.chars_per_second:.1f} | {text} |"
                )

        if self.rejected:
            lines += [
                "",
                "## Rejected",
                "",
                f"{len(self.rejected)} candidate(s) were rejected. The full list "
                f"with reasons is in `rejected.csv`.",
                "",
                "| reason | count |",
                "| --- | --- |",
            ]
            reasons: dict[str, int] = {}
            for item in self.rejected:
                key = item.reason.split("(")[0].split("—")[0].strip()
                reasons[key] = reasons.get(key, 0) + 1
            for reason, count in sorted(
                reasons.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"| {reason} | {count} |")

        lines += [
            "",
            "## Next",
            "",
            "1. Skim the clips above; fix any wrong transcript by editing "
            "`metadata.csv` directly.",
            "2. `./run train` to configure and start training.",
            "",
            "Editing `metadata.csv` changes piper's cache keys (they include the "
            "row number and the text), so the next run will offer to rebuild the "
            "utterance cache. That is expected.",
            "",
        ]
        return "\n".join(lines)

    def outliers(self, limit: int = 10) -> list[ClipStat]:
        """Clips most likely to be wrong, worst first."""
        if not self.clips:
            return []
        rates = [clip.chars_per_second for clip in self.clips if clip.seconds > 0]
        if not rates:
            return []
        median = statistics.median(rates)
        scored = sorted(
            self.clips,
            key=lambda clip: -abs(clip.chars_per_second - median),
        )
        return scored[:limit]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8", newline="\n")


def build(
    rows: Sequence[Row],
    stats: Sequence[ClipStat],
    rejected: Sequence[Rejected],
    **kwargs,
) -> Report:
    report = Report(clips=list(stats), rejected=list(rejected), **kwargs)
    report.analyse()
    return report
