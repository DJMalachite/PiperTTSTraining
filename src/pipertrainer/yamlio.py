"""YAML I/O for profiles and generated ``lightning.yaml`` files.

Reading uses PyYAML when it is importable — inside the project venv it always
is, since both ``lightning`` and ``jsonargparse`` depend on it — and falls back
to a small parser for the bootstrap path, where a fresh clone has no venv yet.

Writing is *always* ours, because the whole point of the profile files is that
a human can read and edit them: we emit the help text from the profile schema
as comments above each key, and PyYAML cannot do that.

The supported subset is what profiles actually contain: nested mappings, block
and flow sequences, and scalars. No anchors, tags, or multi-line scalars.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - depends on environment
    import yaml as _pyyaml
except ImportError:  # pragma: no cover
    _pyyaml = None


class YamlError(ValueError):
    pass


def using_pyyaml() -> bool:
    return _pyyaml is not None


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

_PLAIN_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./+-]*$")
_NUMBERISH = re.compile(r"^[-+]?(\d+\.?\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?)$")
_RESERVED = {
    "true", "false", "yes", "no", "on", "off", "null", "none", "~", "y", "n",
}


def _quote(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr keeps round-trip precision; avoid bare `inf`/`nan` which YAML
        # spells differently.
        if value != value or value in (float("inf"), float("-inf")):
            raise YamlError(f"cannot serialise non-finite float {value!r}")
        return repr(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if (
            _PLAIN_SAFE.match(value)
            and value.lower() not in _RESERVED
            and not _NUMBERISH.match(value)
        ):
            return value
        return _quote(value)
    raise YamlError(f"unsupported scalar type {type(value).__name__}")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, Path))


def _flow(items: Iterable[Any]) -> str:
    return "[" + ", ".join(_render_flow(item) for item in items) + "]"


def _render_flow(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return _flow(value)
    return _scalar(value)


def _all_scalars_or_flat(value: Any) -> bool:
    """True when a sequence is short and simple enough for flow style.

    Architecture tuples like ``[[1, 2], [2, 6], [3, 12]]`` read far better on
    one line than as nine lines of block sequence.
    """
    if not isinstance(value, (list, tuple)):
        return False
    for item in value:
        if _is_scalar(item):
            continue
        if isinstance(item, (list, tuple)) and all(_is_scalar(x) for x in item):
            continue
        return False
    return len(_flow(value)) <= 78


def _emit(
    value: Any,
    out: list[str],
    indent: int,
    comments: dict[str, str],
    trail: str,
) -> None:
    pad = "  " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{trail}.{key}" if trail else str(key)
            comment = comments.get(path)
            if comment:
                if out and out[-1].strip():
                    out.append("")
                for line in comment.splitlines():
                    out.append(f"{pad}# {line}".rstrip())
            if isinstance(item, dict):
                if not item:
                    out.append(f"{pad}{key}: {{}}")
                else:
                    out.append(f"{pad}{key}:")
                    _emit(item, out, indent + 1, comments, path)
            elif isinstance(item, (list, tuple)):
                if not item:
                    out.append(f"{pad}{key}: []")
                elif _all_scalars_or_flat(item):
                    out.append(f"{pad}{key}: {_flow(item)}")
                else:
                    out.append(f"{pad}{key}:")
                    _emit(item, out, indent + 1, comments, path)
            else:
                out.append(f"{pad}{key}: {_scalar(item)}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                nested: list[str] = []
                _emit(item, nested, indent + 1, comments, trail)
                first, *rest = nested
                out.append(f"{pad}- {first.strip()}")
                out.extend(rest)
            elif isinstance(item, (list, tuple)):
                out.append(f"{pad}- {_flow(item)}")
            else:
                out.append(f"{pad}- {_scalar(item)}")
    else:
        out.append(f"{pad}{_scalar(value)}")


def dumps(
    data: Any,
    comments: dict[str, str] | None = None,
    header: str | None = None,
) -> str:
    """Serialise ``data``, emitting ``comments[dotted.path]`` above each key."""
    out: list[str] = []
    if header:
        for line in header.splitlines():
            out.append(f"# {line}".rstrip())
        out.append("")
    _emit(data, out, 0, comments or {}, "")
    text = "\n".join(out).strip("\n")
    return text + "\n"


def dump(
    data: Any,
    path: Path,
    comments: dict[str, str] | None = None,
    header: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data, comments, header), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment, respecting quotes."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            if char == "\\" and quote == '"':
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text[0] in "'\"" and len(text) >= 2 and text[-1] == text[0]:
        body = text[1:-1]
        if text[0] == '"':
            return (
                body.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return body
    if text.startswith("[") and text.endswith("]"):
        return _parse_flow(text[1:-1])
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for part in _split_flow(inner):
            key, sep, value = part.partition(":")
            if not sep:
                raise YamlError(f"bad inline mapping entry: {part!r}")
            result[key.strip().strip("'\"")] = _parse_scalar(value)
        return result
    lowered = text.lower()
    if lowered in ("null", "~", "none"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_flow(text: str) -> list[str]:
    """Split on commas at bracket depth zero, respecting quotes."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            current.append(char)
        elif char in "[{":
            depth += 1
            current.append(char)
        elif char in "]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _parse_flow(text: str) -> list[Any]:
    return [_parse_scalar(part) for part in _split_flow(text)]


def _parse_block(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    index = start
    if index < len(lines) and lines[index][1].startswith("- "):
        items: list[Any] = []
        while index < len(lines) and lines[index][0] == indent:
            level, content = lines[index]
            if not content.startswith("- "):
                break
            body = content[2:].strip()
            index += 1
            if ":" in body and not body.startswith(("[", "{", "'", '"')):
                # A mapping whose first key shares the dash's line.
                key, _, rest = body.partition(":")
                mapping: dict[str, Any] = {}
                if rest.strip():
                    mapping[key.strip()] = _parse_scalar(rest)
                else:
                    nested, index = _parse_block(lines, index, level + 1)
                    mapping[key.strip()] = nested
                while index < len(lines) and lines[index][0] > level:
                    extra, index = _parse_block(lines, index, lines[index][0])
                    if isinstance(extra, dict):
                        mapping.update(extra)
                items.append(mapping)
            else:
                items.append(_parse_scalar(body))
        return items, index

    mapping = {}
    while index < len(lines) and lines[index][0] == indent:
        level, content = lines[index]
        if content.startswith("- "):
            break
        key, sep, rest = content.partition(":")
        if not sep:
            raise YamlError(f"expected 'key: value', got {content!r}")
        key = key.strip().strip("'\"")
        index += 1
        if rest.strip():
            mapping[key] = _parse_scalar(rest)
        elif index < len(lines) and lines[index][0] > level:
            mapping[key], index = _parse_block(lines, index, lines[index][0])
        else:
            mapping[key] = None
    return mapping, index


def loads(text: str) -> Any:
    if _pyyaml is not None:
        return _pyyaml.safe_load(text)
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if raw.strip().startswith("#"):
            continue
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        if stripped.strip() in ("---", "..."):
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent // 2, stripped.strip()))
    if not lines:
        return None
    value, consumed = _parse_block(lines, 0, lines[0][0])
    if consumed != len(lines):
        raise YamlError(f"could not parse line {consumed + 1}: {lines[consumed][1]!r}")
    return value


def load(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))
