#!/bin/sh
# Bootstrap wrapper around `./run setup`.
#
# `./run` already works from a bare clone (the tool is stdlib-only until the
# venv exists), so this script exists for one job: pick a supported interpreter
# explicitly and give a distro-specific hint when there isn't one. Python 3.11
# is the floor because we read pins.toml with tomllib.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

need_major=3
need_minor=11

supported() {
    "$1" - "$need_major" "$need_minor" <<'PY' 2>/dev/null
import sys
major, minor = int(sys.argv[1]), int(sys.argv[2])
sys.exit(0 if sys.version_info[:2] >= (major, minor) else 1)
PY
}

python_bin=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && supported "$candidate"; then
        python_bin=$(command -v "$candidate")
        break
    fi
done

if [ -z "$python_bin" ]; then
    echo "error: no Python ${need_major}.${need_minor}+ found on PATH." >&2
    echo >&2
    if command -v pacman >/dev/null 2>&1; then
        echo "  sudo pacman -S --needed python" >&2
    elif command -v apt-get >/dev/null 2>&1; then
        echo "  sudo apt-get install python3 python3-venv" >&2
    elif command -v dnf >/dev/null 2>&1; then
        echo "  sudo dnf install python3" >&2
    fi
    exit 1
fi

echo "using $python_bin ($("$python_bin" --version 2>&1))"

PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
exec "$python_bin" -m pipertrainer setup "$@"
