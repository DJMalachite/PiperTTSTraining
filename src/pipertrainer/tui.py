"""Terminal prompts.

Deliberately line-based ``input()`` rather than curses: this has to work over
SSH on a headless box, inside ``tmux``, with a piped stdin in CI, and on a
serial console. Every prompt shows the current value as the default, so
pressing Enter through a whole wizard is always safe.

Universal keys at any prompt: ``?`` shows the help text, ``b`` goes back,
``q`` quits.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Callable, Iterable, Sequence


class Back(Exception):
    """User asked to step back one screen."""


class Quit(Exception):
    """User asked to leave."""


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------


def _colour_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_ON = _colour_enabled()
_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def _encodable(text: str) -> bool:
    """Whether stdout can actually represent ``text``.

    A cp1252 console — or any non-UTF-8 locale — raises UnicodeEncodeError on a
    check mark, which would turn a status line into a crash. Ask first.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE = _encodable("✓✗·─→")

GLYPHS = {
    "ok": "✓" if _UNICODE else "+",
    "fail": "✗" if _UNICODE else "x",
    "warn": "!",
    "bullet": "·" if _UNICODE else "-",
    "rule": "─" if _UNICODE else "-",
    "arrow": "→" if _UNICODE else "->",
    "skip": "-",
}


def glyph(name: str) -> str:
    return GLYPHS[name]


def style(text: str, *names: str) -> str:
    if not _ON or not names:
        return text
    codes = ";".join(_CODES[name] for name in names if name in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


def heading(text: str) -> None:
    print()
    print(style(text, "bold"))
    print(style(GLYPHS["rule"] * min(len(text), 72), "dim"))


def info(text: str) -> None:
    print(text)


def hint(text: str) -> None:
    print(style(text, "dim"))


def ok(text: str) -> None:
    print(f"{style(GLYPHS['ok'], 'green')} {text}")


def warn(text: str) -> None:
    print(f"{style(GLYPHS['warn'], 'yellow')} {text}")


def error(text: str) -> None:
    print(f"{style(GLYPHS['fail'], 'red')} {text}", file=sys.stderr)


def bullet(text: str) -> None:
    print(f"  {style(GLYPHS['bullet'], 'dim')} {text}")


def wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(
            line, width=76, initial_indent=indent, subsequent_indent=indent
        )
        for line in text.splitlines()
    )


def table(rows: Sequence[Sequence[str]], headers: Sequence[str] | None = None) -> None:
    """Print a left-aligned table. Values are stringified as given."""
    body = [[str(cell) for cell in row] for row in rows]
    if headers:
        body.insert(0, [str(h) for h in headers])
    if not body:
        return
    widths = [max(len(row[i]) for row in body) for i in range(len(body[0]))]
    for index, row in enumerate(body):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        if headers and index == 0:
            print(style(line, "bold"))
            print(style("  ".join("─" * w for w in widths), "dim"))
        else:
            print(line)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def _read(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        # Piped stdin ran out: treat as quit rather than looping forever.
        print()
        raise Quit() from None
    except KeyboardInterrupt:
        print()
        raise Quit() from None


def _handle_universal(raw: str, help_text: str | None, allow_back: bool) -> None:
    lowered = raw.strip().lower()
    if lowered == "?":
        print(wrap(help_text or "No help for this setting."))
        raise _Retry()
    if lowered == "q":
        raise Quit()
    if lowered == "b":
        if allow_back:
            raise Back()
        warn("nothing to go back to here")
        raise _Retry()


class _Retry(Exception):
    pass


def _label(prompt: str, default: Any, unit: str | None) -> str:
    shown = "" if default is None else str(default)
    if shown == "":
        shown = "(empty)"
    suffix = f" {unit}" if unit else ""
    return f"{prompt} [{style(shown, 'cyan')}{suffix}] "


def ask(
    prompt: str,
    default: Any = None,
    *,
    parse: Callable[[str], Any] | None = None,
    help_text: str | None = None,
    unit: str | None = None,
    allow_back: bool = True,
) -> Any:
    """Core prompt. Empty input keeps ``default``."""
    while True:
        raw = _read(_label(prompt, default, unit))
        try:
            _handle_universal(raw, help_text, allow_back)
        except _Retry:
            continue
        text = raw.strip()
        if not text:
            return default
        if parse is None:
            return text
        try:
            return parse(text)
        except (ValueError, TypeError) as exc:
            error(str(exc))


def ask_str(prompt: str, default: str = "", **kwargs: Any) -> str:
    return ask(prompt, default, **kwargs)


def ask_int(
    prompt: str,
    default: int,
    minimum: float | None = None,
    maximum: float | None = None,
    **kwargs: Any,
) -> int:
    def parse(text: str) -> int:
        value = int(text)
        _check_range(value, minimum, maximum)
        return value

    return ask(prompt, default, parse=parse, **kwargs)


def ask_float(
    prompt: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    **kwargs: Any,
) -> float:
    def parse(text: str) -> float:
        value = float(text)
        _check_range(value, minimum, maximum)
        return value

    return ask(prompt, default, parse=parse, **kwargs)


def _check_range(value: float, minimum: float | None, maximum: float | None) -> None:
    if minimum is not None and value < minimum:
        raise ValueError(f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"must be at most {maximum}")


def ask_bool(prompt: str, default: bool, **kwargs: Any) -> bool:
    """Yes/no prompt. The default is displayed as yes/no rather than True/False."""

    def parse(text: str) -> bool:
        lowered = text.lower()
        if lowered in ("y", "yes", "true", "on", "1"):
            return True
        if lowered in ("n", "no", "false", "off", "0"):
            return False
        raise ValueError("answer y or n")

    result = ask(prompt, "yes" if default else "no", parse=parse, **kwargs)
    if isinstance(result, bool):
        return result
    # Enter pressed: `ask` handed back the displayed default unchanged.
    return default


def ask_choice(
    prompt: str,
    choices: Sequence[Any],
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Numbered choice; the value itself is also accepted."""
    rendered = [("(empty)" if c == "" else str(c)) for c in choices]
    hint("  " + "  ".join(f"{i + 1}) {name}" for i, name in enumerate(rendered)))

    def parse(text: str) -> Any:
        if text.isdigit():
            index = int(text) - 1
            if 0 <= index < len(choices):
                return choices[index]
            raise ValueError(f"pick 1-{len(choices)}")
        for choice, name in zip(choices, rendered):
            if text.lower() in (str(choice).lower(), name.lower()):
                return choice
        raise ValueError(f"not one of: {', '.join(rendered)}")

    return ask(prompt, default, parse=parse, **kwargs)


def ask_path(
    prompt: str,
    default: str = "",
    *,
    must_exist: bool = False,
    **kwargs: Any,
) -> str:
    def parse(text: str) -> str:
        expanded = os.path.expanduser(text)
        if must_exist and not os.path.exists(expanded):
            raise ValueError(f"no such path: {expanded}")
        return expanded

    return ask(prompt, default, parse=parse, **kwargs)


def ask_list(prompt: str, default: Sequence[Any], **kwargs: Any) -> list[Any]:
    """Comma-separated list, parsed as numbers when possible."""

    def parse(text: str) -> list[Any]:
        items: list[Any] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                items.append(int(part))
            except ValueError:
                try:
                    items.append(float(part))
                except ValueError:
                    items.append(part)
        return items

    shown = ", ".join(str(item) for item in default)
    result = ask(prompt, shown, parse=parse, **kwargs)
    return list(default) if result == shown else result


def ask_dict(prompt: str, default: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """``KEY=value, KEY2=value2``."""

    def parse(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition("=")
            if not sep:
                raise ValueError(f"expected KEY=value, got {part!r}")
            result[key.strip()] = value.strip()
        return result

    shown = ", ".join(f"{k}={v}" for k, v in default.items())
    result = ask(prompt, shown, parse=parse, **kwargs)
    return dict(default) if result == shown else result


def confirm(prompt: str, default: bool = False) -> bool:
    return ask_bool(prompt, default, allow_back=False)


def menu(
    title: str,
    items: Sequence[tuple[str, str]],
    *,
    status: Iterable[str] = (),
    allow_back: bool = False,
) -> str:
    """Show a numbered menu; return the key of the chosen item.

    ``items`` is a sequence of ``(key, label)``.
    """
    heading(title)
    for line in status:
        print(f"  {line}")
    if status:
        print()
    for index, (_, label) in enumerate(items, start=1):
        print(f"  {style(str(index), 'bold')}) {label}")
    print()
    choices = [key for key, _ in items]
    extra = "b) back  " if allow_back else ""
    hint(f"  {extra}q) quit")

    while True:
        raw = _read("choice: ").strip().lower()
        if raw == "q":
            raise Quit()
        if raw == "b":
            if allow_back:
                raise Back()
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(items):
                return choices[index]
        if raw in choices:
            return raw
        error(f"pick 1-{len(items)}")


def pause(message: str = "press Enter to continue") -> None:
    _read(style(f"  {message} ", "dim"))
