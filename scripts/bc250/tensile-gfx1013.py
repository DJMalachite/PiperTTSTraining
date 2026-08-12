#!/usr/bin/env python3
"""Teach the Tensile that rocBLAS fetched for itself about gfx1013.

Why this exists: rocBLAS re-fetches its own copy of Tensile into
``build/release/virtualenv`` during cmake configure, and that copy lists
gfx1013 nowhere. Tensile then declines to generate assembly kernels for it —

    UserWarning: Did not detect SupportedISA: [... (10,1,2), (10,3,0) ...];
    cannot benchmark assembly kernels.

— the manifest asks for files that were never produced, the verify step fails,
and nothing links. The upstream script's step 4 says to make these edits by
hand and then only prints the instruction; the reference repo never writes the
edits down. This is that step, done.

Method: append to the modules rather than splice their literals. Both targets
are ordinary module-level containers, so a few lines at the end of the file
mutate them at import time and keep working when upstream reformats the
literals — which is exactly what a regex over a 19-entry list would not.

gfx1013 is (10, 1, 3) and shares its ISA with gfx1010/gfx1012, so it takes
gfx1012's cached assembler capabilities and navi10's tuned logic. That is the
same reasoning the reference work used: the parts differ in memory aperture
layout and CU count, not in what the assembler can emit.

Idempotent, and refuses rather than guesses: if a structure this expects is not
there, the Tensile version has moved and someone has to look again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "PiperTTSTraining: gfx1013"

COMMON_PATCH = f'''

# --- {MARKER} -------------------------------------------------
# BC-250 / Cyan Skillfish. Same ISA family as gfx1010 and gfx1012, which are
# both already here; it is absent only because no AMD product line shipped it.
# navi10 logic because gfx1010 is the closest tuned architecture.
if (10, 1, 3) not in globalParameters["SupportedISA"]:
    globalParameters["SupportedISA"].append((10, 1, 3))
if "gfx1013" not in architectureMap:
    architectureMap["gfx1013"] = "navi10"
'''

ASMCAPS_PATCH = f'''

# --- {MARKER} -------------------------------------------------
# gfx1013 assembles exactly as gfx1012 does; copy rather than re-derive, so the
# cached and derived caps cannot disagree.
if (10, 1, 3) not in CACHED_ASM_CAPS:
    CACHED_ASM_CAPS[(10, 1, 3)] = dict(CACHED_ASM_CAPS[(10, 1, 2)])
'''

# Each target: file, the names that must already exist, and what to append.
TARGETS = (
    ("Common.py", ('globalParameters["SupportedISA"]', "architectureMap"), COMMON_PATCH),
    ("AsmCaps.py", ("CACHED_ASM_CAPS",), ASMCAPS_PATCH),
)


def find_tensile(build_dir: Path) -> Path:
    """The Tensile package inside rocBLAS's own virtualenv, not the system one."""
    roots = sorted(build_dir.glob("virtualenv/lib*/python*/site-packages/Tensile"))
    if not roots:
        raise SystemExit(
            f"error: no Tensile under {build_dir}/virtualenv\n"
            "       That directory is created by rocBLAS's first cmake configure.\n"
            "       Run the build once before patching it."
        )
    return roots[0]


def patch(tensile: Path, dry_run: bool) -> int:
    changed = 0
    for name, required, addition in TARGETS:
        path = tensile / name
        if not path.is_file():
            raise SystemExit(f"error: {path} does not exist")

        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"  already patched: {path.name}")
            continue

        missing = [token for token in required if token not in text]
        if missing:
            raise SystemExit(
                f"error: {path.name} does not define {', '.join(missing)}\n"
                "       This Tensile is not the version these edits were written\n"
                "       against. Refusing to guess; see docs/BC250.md."
            )

        if dry_run:
            print(f"  would patch: {path}")
        else:
            path.write_text(text + addition, encoding="utf-8")
            print(f"  patched: {path}")
        changed += 1
    return changed


def verify(tensile: Path) -> None:
    """Import the patched modules and confirm gfx1013 is actually visible."""
    sys.path.insert(0, str(tensile.parent))
    try:
        from Tensile.Common import globalParameters  # type: ignore
        from Tensile.AsmCaps import CACHED_ASM_CAPS  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the fetched tree
        raise SystemExit(f"error: the patched Tensile does not import: {exc}")

    if (10, 1, 3) not in globalParameters["SupportedISA"]:
        raise SystemExit("error: (10,1,3) still missing from SupportedISA")
    if (10, 1, 3) not in CACHED_ASM_CAPS:
        raise SystemExit("error: (10,1,3) still missing from CACHED_ASM_CAPS")
    print("  verified: Tensile now reports gfx1013 as a supported ISA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "build_dir",
        type=Path,
        help="rocBLAS build directory, e.g. ~/rocBLAS/build/release",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tensile = find_tensile(args.build_dir.expanduser())
    print(f"  Tensile: {tensile}")
    patch(tensile, args.dry_run)
    if not args.dry_run:
        verify(tensile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
