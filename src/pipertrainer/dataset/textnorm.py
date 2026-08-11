"""Transcript normalisation and rejection.

Whisper's output is good but not clean: it emits non-speech annotations like
``[Music]``, occasionally hallucinates a line during silence, and sometimes
attaches text to the wrong clip. None of that is visible from the audio, so it
has to be caught here — a transcript that does not match its clip teaches the
model the wrong alignment, which is the single biggest quality killer in a fully
automated pipeline.

Everything in this module is a pure function so it can be tested exhaustively
without audio. Rejections always carry a reason and are written to
``rejected.csv``; nothing is dropped silently.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Whisper's non-speech annotations. Bracketed spans only — parentheses in real
# speech are rare in transcripts but bracket annotations are common.
_BRACKETED = re.compile(r"[\[\(\{][^\]\)\}]{0,40}[\]\)\}]")
_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

TERMINAL_PUNCTUATION = ".!?…"
CLAUSE_PUNCTUATION = ",;:"

_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
}

# Utterances that are only a filler carry no useful phonetic content and are
# usually mistimed.
FILLERS = {
    "uh", "um", "umm", "uhh", "mm", "mmm", "hmm", "hm", "er", "err", "ah",
    "aah", "eh", "mhm", "mmhmm", "uh-huh", "uh huh", "huh", "oh",
}


@dataclass(frozen=True)
class Normalized:
    """A cleaned transcript, or a rejection with a reason."""

    text: str
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


@dataclass(frozen=True)
class TextRules:
    ensure_terminal_punctuation: bool = True
    drop_bracketed: bool = True
    normalize_quotes: bool = True
    min_chars: int = 2
    cps_min: float = 3.0
    cps_max: float = 30.0


def normalize(raw: str, rules: TextRules) -> Normalized:
    """Clean one transcript and decide whether to keep it."""
    if raw is None:
        return Normalized("", False, "empty transcript")

    text = unicodedata.normalize("NFC", str(raw))
    text = _CONTROL.sub(" ", text)

    # The metadata delimiter. The csv module would quote a field containing it,
    # and piper's reader would handle that correctly, but a pipe inside a
    # transcript is always an artefact rather than speech.
    text = text.replace("|", " ")

    if rules.drop_bracketed:
        text = _BRACKETED.sub(" ", text)

    if rules.normalize_quotes:
        text = "".join(_QUOTE_MAP.get(char, char) for char in text)

    text = _WHITESPACE.sub(" ", text).strip()
    text = text.strip("-–— ").strip()

    if not text:
        return Normalized("", False, "empty after cleaning")
    if not _HAS_LETTER.search(text):
        return Normalized(text, False, "no letters (punctuation or digits only)")

    stripped = text.rstrip(TERMINAL_PUNCTUATION + CLAUSE_PUNCTUATION + " ")
    if stripped.lower() in FILLERS:
        return Normalized(text, False, f"filler only ({stripped!r})")

    if len(stripped) < rules.min_chars:
        return Normalized(
            text, False, f"shorter than min_chars ({len(stripped)} < {rules.min_chars})"
        )

    if rules.ensure_terminal_punctuation and text[-1] not in TERMINAL_PUNCTUATION:
        # Strip a trailing clause mark first: "and then," -> "and then."
        text = text.rstrip(CLAUSE_PUNCTUATION + " ") + "."

    return Normalized(text, True)


def chars_per_second(text: str, duration: float) -> float:
    if duration <= 0:
        return float("inf")
    return len(text) / duration


def check_rate(text: str, duration: float, rules: TextRules) -> str | None:
    """Flag clips whose text length does not match their duration.

    A low rate means words are missing from the transcript; a high rate means it
    carries text belonging to neighbouring audio. Both are misalignment, and
    both are invisible without this check.
    """
    rate = chars_per_second(text, duration)
    if rate < rules.cps_min:
        return (
            f"only {rate:.1f} chars/s over {duration:.2f} s — the transcript "
            f"looks incomplete"
        )
    if rate > rules.cps_max:
        return (
            f"{rate:.1f} chars/s over {duration:.2f} s — the transcript looks "
            f"too long for the audio"
        )
    return None


def looks_repetitive(text: str, threshold: int = 4) -> bool:
    """Detect Whisper's repetition loop, e.g. 'thank you' twenty times.

    Only triggers on immediate repeats of the same token run, which is what the
    failure actually looks like; ordinary repeated words survive.
    """
    words = text.lower().split()
    if len(words) < threshold * 2:
        return False
    for size in (1, 2, 3):
        if len(words) < size * threshold:
            continue
        run = 1
        for index in range(size, len(words) - size + 1, size):
            if words[index : index + size] == words[index - size : index]:
                run += 1
                if run >= threshold:
                    return True
            else:
                run = 1
    return False
