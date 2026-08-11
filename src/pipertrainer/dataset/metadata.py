"""Reading and writing ``metadata.csv``.

The format is piper's: pipe-delimited, no header, ``wav|text`` (or
``wav|speaker|text`` for multi-speaker). Two details decide whether it survives
a round trip.

**Dialect.** Upstream reads with ``csv.reader(csv_file, delimiter="|")`` — that
is, the *default* dialect: ``quotechar='"'``, ``QUOTE_MINIMAL``. So we write
with the matching default writer. A transcript starting with a double quote is
then correctly quoted and doubled on write and correctly unquoted on read.
Writing with ``QUOTE_NONE`` would look cleaner and silently corrupt exactly
those rows, because upstream's reader would still treat the leading quote as a
quote character. The legacy dataset script built these lines with an f-string,
which has the same bug.

**Ordering.** piper's cache id is ``f"{row_number}_{sanitize_filename(text)}"``
truncated to 50 characters. Row *number* is part of the key, so reordering rows
orphans every cached tensor after the first change. Rows are therefore always
written in a stable sorted order, and ``train/argmap.py`` fingerprints the file
so a reorder is reported rather than discovered as a surprise re-preprocess.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DELIMITER = "|"


@dataclass(frozen=True)
class Row:
    utt_id: str
    text: str
    speaker: str = ""

    @property
    def columns(self) -> list[str]:
        if self.speaker:
            return [self.utt_id, self.speaker, self.text]
        return [self.utt_id, self.text]


@dataclass(frozen=True)
class Rejected:
    utt_id: str
    reason: str
    text: str = ""
    start: float = 0.0
    end: float = 0.0


_NUMERIC = re.compile(r"^(\d+)")


def clip_id(index: int, prefix: str = "", width: int = 6) -> str:
    """Zero-padded, sortable clip name. ``000001`` sorts before ``000010``."""
    return f"{prefix}{index:0{width}d}"


def sort_key(row: Row) -> tuple:
    """Stable order: numeric when the ids are numeric, lexical otherwise."""
    match = _NUMERIC.match(row.utt_id)
    if match:
        return (0, int(match.group(1)), row.utt_id)
    return (1, 0, row.utt_id)


def write(rows: Iterable[Row], path: Path, *, sort: bool = True) -> int:
    """Write metadata.csv. Returns the number of rows written."""
    items = list(rows)
    if sort:
        items.sort(key=sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required by the csv module; lineterminator keeps LF on every
    # platform so the file is identical from Windows and Linux.
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=DELIMITER, lineterminator="\n")
        for row in items:
            writer.writerow(row.columns)
    return len(items)


def read(path: Path, *, multispeaker: bool = False) -> list[Row]:
    """Read metadata.csv exactly the way piper does."""
    rows: list[Row] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter=DELIMITER):
            if not fields or not fields[0].strip():
                continue
            if multispeaker:
                if len(fields) < 3:
                    raise ValueError(
                        f"{path}: expected wav|speaker|text, got {len(fields)} "
                        f"column(s): {fields!r}"
                    )
                rows.append(
                    Row(utt_id=fields[0], speaker=fields[1], text=fields[-1])
                )
            else:
                if len(fields) < 2:
                    raise ValueError(
                        f"{path}: expected wav|text, got {len(fields)} "
                        f"column(s): {fields!r}"
                    )
                rows.append(Row(utt_id=fields[0], text=fields[-1]))
    return rows


def write_rejected(items: Sequence[Rejected], path: Path) -> int:
    """Record what was thrown away and why. Never let a drop be silent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["utt_id", "start", "end", "reason", "text"])
        for item in items:
            writer.writerow(
                [
                    item.utt_id,
                    f"{item.start:.3f}",
                    f"{item.end:.3f}",
                    item.reason,
                    item.text,
                ]
            )
    return len(items)


def resolve_audio(audio_dir: Path, utt_id: str) -> Path | None:
    """Find a row's audio file the way piper does.

    piper joins ``audio_dir / utt_id`` and retries with ``.wav`` appended, then
    warns and skips if neither exists. Counting usable rows any other way would
    disagree with what training actually sees.
    """
    direct = audio_dir / utt_id
    if direct.exists():
        return direct
    with_extension = audio_dir / f"{utt_id}.wav"
    if with_extension.exists():
        return with_extension
    return None


def count_usable(path: Path, audio_dir: Path, *, multispeaker: bool = False) -> tuple[int, list[str]]:
    """``(usable_rows, missing_ids)`` using piper's own resolution rules."""
    rows = read(path, multispeaker=multispeaker)
    missing = [
        row.utt_id for row in rows if resolve_audio(audio_dir, row.utt_id) is None
    ]
    return len(rows) - len(missing), missing
