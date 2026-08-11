"""Grouping Whisper words into training utterances.

This is where dataset quality is won or lost, and it is the main improvement
over splitting on silence alone. Cutting on silence gives clips of arbitrary
length whose transcripts are produced separately, so a clipped word yields a
transcript that does not match its audio. Grouping *words* instead means each
clip's boundaries and its text come from the same alignment, and the length
bounds are enforced rather than emergent.

Three rules matter for VITS:

* **A floor of 1 s.** ``UtteranceCollate`` pads every batch up to at least
  ``segment_size`` (0.372 s at 22.05 kHz), so shorter clips are padded with
  silence the model then learns to produce.
* **A ceiling.** Long clips dominate peak GPU memory, which is what actually
  limits batch size on a small board.
* **Cut in silence, not in speech.** Cuts land at the midpoint of the pause
  between utterances, clamped by the padding, and optionally snapped to a zero
  crossing to avoid clicks.

Pure functions over word lists: no audio, no Whisper, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .textnorm import CLAUSE_PUNCTUATION, TERMINAL_PUNCTUATION


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Utterance:
    """One candidate clip: cut points plus the text that belongs to it."""

    start: float
    end: float
    text: str
    word_start: float
    word_end: float
    flags: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_duration(self) -> float:
        return self.word_end - self.word_start


@dataclass(frozen=True)
class Dropped:
    start: float
    end: float
    text: str
    reason: str


@dataclass
class SegmentRules:
    min_seconds: float = 1.0
    target_seconds: float = 8.0
    max_seconds: float = 14.0
    boundary_gap: float = 0.35
    pad_before: float = 0.15
    pad_after: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ValueError(
                f"segment bounds must satisfy 0 < min ({self.min_seconds}) <= "
                f"target ({self.target_seconds}) <= max ({self.max_seconds})"
            )


@dataclass
class SegmentResult:
    utterances: list[Utterance] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)


def _ends_with(text: str, marks: str) -> bool:
    stripped = text.rstrip("\"')]} ")
    return bool(stripped) and stripped[-1] in marks


def _duration(group: Sequence[Word]) -> float:
    return group[-1].end - group[0].start


def _text_of(group: Iterable[Word]) -> str:
    return " ".join(word.text.strip() for word in group if word.text.strip())


# --------------------------------------------------------------------------
# Pass 1: greedy accumulation
# --------------------------------------------------------------------------


def _greedy_groups(words: Sequence[Word], rules: SegmentRules) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []

    for index, word in enumerate(words):
        current.append(word)
        duration = _duration(current)
        following = words[index + 1] if index + 1 < len(words) else None
        gap = (following.start - word.end) if following else float("inf")

        close = False
        if duration >= rules.max_seconds:
            close = True  # hard ceiling; the group is split again below
        elif duration >= rules.min_seconds and _ends_with(
            word.text, TERMINAL_PUNCTUATION
        ):
            close = True
        elif duration >= rules.min_seconds and gap >= rules.boundary_gap:
            close = True
        elif duration >= rules.target_seconds and _ends_with(
            word.text, CLAUSE_PUNCTUATION
        ):
            close = True

        if close:
            groups.append(current)
            current = []

    if current:
        groups.append(current)
    return groups


# --------------------------------------------------------------------------
# Pass 2: split anything still over the ceiling
# --------------------------------------------------------------------------


def _best_split(group: Sequence[Word]) -> int | None:
    """Index to split after: the widest pause, preferring the middle.

    Restricting the search to the central 60% first keeps the two halves
    balanced. Taking the globally widest gap instead tends to shave a one-word
    fragment off the front and recurse, which produces uneven clips.
    """
    if len(group) < 2:
        return None
    gaps = [
        (group[i + 1].start - group[i].end, i) for i in range(len(group) - 1)
    ]
    span = _duration(group)
    origin = group[0].start
    central = [
        (gap, index)
        for gap, index in gaps
        if 0.2 * span <= (group[index].end - origin) <= 0.8 * span
    ]
    candidates = central or gaps
    return max(candidates)[1]


def _split_overlong(
    group: list[Word], rules: SegmentRules, dropped: list[Dropped]
) -> list[list[Word]]:
    if _duration(group) <= rules.max_seconds:
        return [group]

    index = _best_split(group)
    if index is None:
        # A single word longer than max_seconds is a Whisper timestamp glitch,
        # not speech we can use.
        word = group[0]
        dropped.append(
            Dropped(
                word.start,
                word.end,
                word.text,
                f"single word spans {word.duration:.1f} s (> max_seconds "
                f"{rules.max_seconds}); Whisper timestamp glitch",
            )
        )
        return []

    left = _split_overlong(group[: index + 1], rules, dropped)
    right = _split_overlong(group[index + 1 :], rules, dropped)
    return left + right


# --------------------------------------------------------------------------
# Pass 3: absorb anything under the floor
# --------------------------------------------------------------------------


def _merge_short(
    groups: Sequence[list[Word]], rules: SegmentRules, dropped: list[Dropped]
) -> list[list[Word]]:
    out: list[list[Word]] = []
    pending: list[Word] = []

    for group in groups:
        candidate = pending + group
        pending = []
        if _duration(candidate) >= rules.min_seconds:
            out.append(candidate)
            continue
        # Too short. Prefer appending to the previous group, since that keeps
        # sentence order; otherwise carry it into the next one.
        if out and (candidate[-1].end - out[-1][0].start) <= rules.max_seconds:
            out[-1].extend(candidate)
        else:
            pending = candidate

    if pending:
        if out and (pending[-1].end - out[-1][0].start) <= rules.max_seconds:
            out[-1].extend(pending)
        else:
            dropped.append(
                Dropped(
                    pending[0].start,
                    pending[-1].end,
                    _text_of(pending),
                    f"{_duration(pending):.2f} s is below min_seconds "
                    f"{rules.min_seconds} and cannot be merged with a neighbour",
                )
            )
    return out


# --------------------------------------------------------------------------
# Pass 4: turn groups into cut points
# --------------------------------------------------------------------------


def _cut_points(
    groups: Sequence[list[Word]], rules: SegmentRules, total_duration: float | None
) -> list[Utterance]:
    utterances: list[Utterance] = []
    for index, group in enumerate(groups):
        first, last = group[0], group[-1]

        start = first.start - rules.pad_before
        if index > 0:
            previous_end = groups[index - 1][-1].end
            # Never reach back past the middle of the pause: neighbouring clips
            # must not contain each other's speech.
            start = max(start, (previous_end + first.start) / 2.0)
        start = max(0.0, start)

        end = last.end + rules.pad_after
        if index + 1 < len(groups):
            next_start = groups[index + 1][0].start
            end = min(end, (last.end + next_start) / 2.0)
        if total_duration is not None:
            end = min(end, total_duration)

        if end <= start:  # pathological timestamps
            continue

        utterances.append(
            Utterance(
                start=start,
                end=end,
                text=_text_of(group),
                word_start=first.start,
                word_end=last.end,
            )
        )
    return utterances


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def group_words(
    words: Sequence[Word],
    rules: SegmentRules,
    total_duration: float | None = None,
) -> SegmentResult:
    """Group aligned words into utterances honouring min/target/max."""
    usable = [
        word
        for word in words
        if word.text.strip() and word.end > word.start
    ]
    if not usable:
        return SegmentResult()

    usable.sort(key=lambda word: word.start)
    dropped: list[Dropped] = []

    groups = _greedy_groups(usable, rules)

    split: list[list[Word]] = []
    for group in groups:
        split.extend(_split_overlong(group, rules, dropped))

    merged = _merge_short(split, rules, dropped)

    # Merging can push a group back over the ceiling, so split once more. The
    # two passes cannot fight forever: merging only ever combines adjacent
    # groups whose combined span already fits inside max_seconds, so this second
    # split is a safety net rather than a loop.
    final: list[list[Word]] = []
    for group in merged:
        for piece in _split_overlong(group, rules, dropped):
            if _duration(piece) >= rules.min_seconds:
                final.append(piece)
            else:
                dropped.append(
                    Dropped(
                        piece[0].start,
                        piece[-1].end,
                        _text_of(piece),
                        f"{_duration(piece):.2f} s is below min_seconds "
                        f"{rules.min_seconds} after splitting an overlong group",
                    )
                )

    return SegmentResult(
        utterances=_cut_points(final, rules, total_duration), dropped=dropped
    )


# --------------------------------------------------------------------------
# The VAD strategy's counterpart: turn speech spans into utterances
# --------------------------------------------------------------------------


def spans_to_utterances(
    spans: Sequence[tuple[float, float]],
    rules: SegmentRules,
    total_duration: float | None = None,
) -> SegmentResult:
    """Enforce the same length bounds on VAD speech spans.

    Used by the ``vad`` strategy, where each clip is transcribed separately so
    there are no word timestamps to group. Spans become single pseudo-words,
    which lets the same splitting and merging logic apply — except a span over
    the ceiling has no internal gap to split at, so it is trimmed instead.
    """
    words: list[Word] = []
    for start, end in spans:
        if end <= start:
            continue
        words.append(Word(text="", start=start, end=end))

    result = SegmentResult()
    for word in words:
        duration = word.duration
        if duration < rules.min_seconds:
            result.dropped.append(
                Dropped(
                    word.start,
                    word.end,
                    "",
                    f"speech span is {duration:.2f} s, below min_seconds "
                    f"{rules.min_seconds}",
                )
            )
            continue
        pieces = _trim_span(word, rules)
        for piece_start, piece_end in pieces:
            start = max(0.0, piece_start - rules.pad_before)
            end = piece_end + rules.pad_after
            if total_duration is not None:
                end = min(end, total_duration)
            result.utterances.append(
                Utterance(
                    start=start,
                    end=end,
                    text="",
                    word_start=piece_start,
                    word_end=piece_end,
                    flags=("vad",),
                )
            )
    return result


def _trim_span(word: Word, rules: SegmentRules) -> list[tuple[float, float]]:
    """Chop an overlong speech span into equal pieces within the bounds."""
    duration = word.duration
    if duration <= rules.max_seconds:
        return [(word.start, word.end)]
    pieces = int(duration // rules.target_seconds) + 1
    while pieces > 1 and duration / pieces < rules.min_seconds:
        pieces -= 1
    size = duration / pieces
    return [
        (word.start + index * size, word.start + (index + 1) * size)
        for index in range(pieces)
    ]
