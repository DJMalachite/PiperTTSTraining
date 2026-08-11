"""Turning the profile schema into prompts.

Every prompt is generated from the ``Spec`` attached to the field, so the wizard
and the YAML file can never disagree about what a setting is called, what values
it accepts, or what it means. Adding a setting to ``profile.py`` makes it appear
here automatically.

Fields marked ``advanced`` are hidden behind an explicit opt-in: a first-time
user should not have to form an opinion about ``lr_decay_d`` to build a dataset.
"""

from __future__ import annotations

from typing import Any, Callable

from .. import profile as profile_mod
from .. import tui


def prompt_field(
    prof: profile_mod.Profile, path: str, field_spec: profile_mod.Spec
) -> bool:
    """Prompt for one field. Returns True if the value changed."""
    current = profile_mod.get_path(prof, path)
    label = field_spec.label or path.rsplit(".", 1)[-1].replace("_", " ")
    kwargs: dict[str, Any] = {
        "help_text": field_spec.help,
        "unit": field_spec.unit,
    }

    if field_spec.choices:
        value = tui.ask_choice(label, list(field_spec.choices), current, **kwargs)
    elif field_spec.kind == "bool":
        value = tui.ask_bool(label, bool(current), **kwargs)
    elif field_spec.kind == "int":
        value = tui.ask_int(
            label, int(current), field_spec.minimum, field_spec.maximum, **kwargs
        )
    elif field_spec.kind == "float":
        value = tui.ask_float(
            label, float(current), field_spec.minimum, field_spec.maximum, **kwargs
        )
    elif field_spec.kind == "path":
        value = tui.ask_path(label, str(current or ""), **kwargs)
    elif field_spec.kind == "list":
        value = tui.ask_list(label, list(current or []), **kwargs)
    elif field_spec.kind == "dict":
        value = tui.ask_dict(label, dict(current or {}), **kwargs)
    else:
        value = tui.ask_str(label, str(current or ""), **kwargs)

    if value != current:
        profile_mod.set_path(prof, path, value)
        return True
    return False


def walk(
    prof: profile_mod.Profile,
    prefixes: tuple[str, ...],
    *,
    advanced: bool = False,
    skip: tuple[str, ...] = (),
    title: str | None = None,
) -> list[str]:
    """Prompt for every field under ``prefixes``. Returns the paths that changed."""
    fields = [
        (path, field_spec)
        for path, field_spec, _ in profile_mod.iter_specs(prof)
        if any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)
        and path not in skip
        and (advanced or not field_spec.advanced)
    ]
    if not fields:
        return []
    if title:
        tui.heading(title)
    changed: list[str] = []
    for path, field_spec in fields:
        if prompt_field(prof, path, field_spec):
            changed.append(path)
    return changed


def count_advanced(prof: profile_mod.Profile, prefixes: tuple[str, ...]) -> int:
    return sum(
        1
        for path, field_spec, _ in profile_mod.iter_specs(prof)
        if any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)
        and field_spec.advanced
    )


def offer_advanced(prof: profile_mod.Profile, prefixes: tuple[str, ...], label: str) -> list[str]:
    """Ask whether to walk the advanced settings, then do it."""
    total = count_advanced(prof, prefixes)
    if not total:
        return []
    tui.info("")
    if not tui.confirm(
        f"review the {total} advanced {label} setting(s)?", default=False
    ):
        return []
    return walk(prof, prefixes, advanced=True, title=f"Advanced {label}")


def show_current(prof: profile_mod.Profile, prefixes: tuple[str, ...], title: str) -> None:
    """Print the current values under ``prefixes`` as a table."""
    rows = [
        [path, _render(value)]
        for path, _, value in profile_mod.iter_specs(prof)
        if any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)
    ]
    if rows:
        tui.heading(title)
        tui.table(rows)


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) or "(none)"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    text = str(value)
    return text if text else "(empty)"


def save_and_report(prof: profile_mod.Profile, changed: list[str]) -> None:
    path = profile_mod.save(prof)
    profile_mod.set_active(prof.voice.name)
    if changed:
        tui.ok(f"saved {len(changed)} change(s) to {path}")
    else:
        tui.ok(f"no changes; profile is at {path}")


def report_findings(warnings: list[str], notes: list[str]) -> None:
    for note in notes:
        tui.info(tui.wrap(f"{tui.glyph('bullet')} {note}", indent="  "))
    for warning in warnings:
        tui.warn(warning)


def confirm_or_edit(
    prompt: str, editor: Callable[[], None], *, default: bool = True
) -> bool:
    """Confirm, or loop back into ``editor`` to change something first."""
    while True:
        choice = tui.menu(
            prompt,
            [("go", "Continue"), ("edit", "Change a setting first")],
            allow_back=True,
        )
        if choice == "go":
            return True
        editor()
