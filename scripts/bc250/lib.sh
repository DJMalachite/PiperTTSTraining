#!/bin/sh
# Shared helpers for the BC-250 build stages. Sourced, never executed.
#
# Three rules the stages inherit from the rest of this repo:
#
#   * every root command is printed in full and confirmed before it runs;
#   * nothing is dropped silently — a skip says why, a refusal says what to do;
#   * every stage is idempotent, so re-running after a failure is always safe.

# shellcheck shell=sh

BC250_REPO=${BC250_REPO:?lib.sh must be sourced from a stage script}
BC250_STATE="$BC250_REPO/.state/bc250"
BC250_SRC="$BC250_STATE/src"
BC250_WHEELS="$BC250_STATE/wheels"
BC250_LOG="$BC250_STATE/build.log"

BC250_DRY_RUN=${BC250_DRY_RUN:-0}
BC250_YES=${BC250_YES:-0}

# Exit code meaning "stopped on purpose, not broken" — a reboot is needed, or a
# stage has to continue somewhere else. The driver treats it specially.
BC250_EX_PAUSE=75

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*" >&2; }
bad()  { printf '  [FAIL] %s\n' "$*" >&2; }
head_() {
    printf '\n== %s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

# A refusal always explains what to do instead.
refuse() {
    printf 'error: %s\n' "$1" >&2
    shift
    [ $# -gt 0 ] && printf '       %s\n' "$*" >&2
    exit 1
}

# --------------------------------------------------------------------------
# pins.toml — the single source of truth, read by the scripts too so that they
# cannot disagree with the Python side about which commit we are applying.
# --------------------------------------------------------------------------

# Every caller does `x=$(pin name)` under `set -e`, so a silent non-zero exit
# here would abort a stage with no output at all. Fail loudly instead: the
# message still reaches stderr from inside the command substitution.
pin() {
    command -v python3 >/dev/null 2>&1 ||
        refuse "python3 is not on PATH" \
            "Everything here reads pins.toml through it: 'sudo pacman -S python'."
    _pin_value=$(
        python3 - "$1" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["BC250_REPO"], "src"))
from pipertrainer import pins
value = getattr(pins.load().bc250, sys.argv[1])
print(value)
PY
    ) || refuse "could not read '$1' from pins.toml [bc250]" \
        "The Python error is above; the section may be missing a key."
    printf '%s\n' "$_pin_value"
}

# --------------------------------------------------------------------------
# Running things
# --------------------------------------------------------------------------

# Print, then run. In --dry-run, print only.
runcmd() {
    printf '  $ %s\n' "$*"
    [ "$BC250_DRY_RUN" = "1" ] && return 0
    "$@"
}

confirm() {
    if [ "$BC250_YES" = "1" ]; then
        info "(--yes) $1"
        return 0
    fi
    printf '  %s [y/N] ' "$1"
    read -r reply || return 1
    case "$reply" in
        y | Y | yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

# Root commands are printed in full and confirmed before they run. --yes skips
# the question but never the printing: you can always see what touched the
# system by reading the transcript.
run_root() {
    if [ "$(id -u)" = "0" ]; then
        runcmd "$@"
        return $?
    fi
    printf '  # sudo %s\n' "$*"
    if [ "$BC250_DRY_RUN" = "1" ]; then
        return 0
    fi
    confirm "run the above as root?" || refuse \
        "declined: $*" \
        "Run it yourself and re-run this stage; every stage is idempotent."
    sudo "$@"
}

# --------------------------------------------------------------------------
# Stage state
# --------------------------------------------------------------------------

stage_done() {
    [ -f "$BC250_STATE/done-$1" ]
}

mark_done() {
    mkdir -p "$BC250_STATE"
    date -u '+%Y-%m-%dT%H:%M:%SZ' >"$BC250_STATE/done-$1"
}

clear_done() {
    rm -f "$BC250_STATE/done-$1"
}

# --------------------------------------------------------------------------
# Environment facts
# --------------------------------------------------------------------------

# distrobox and toolbox both leave this behind; podman writes .containerenv.
in_container() {
    [ -n "${CONTAINER_ID:-}" ] || [ -f /run/.containerenv ] || [ -f /.dockerenv ]
}

# The board, by PCI id, without needing lspci installed.
is_bc250() {
    for dev in /sys/bus/pci/devices/*; do
        [ -r "$dev/vendor" ] && [ -r "$dev/device" ] || continue
        if [ "$(cat "$dev/vendor")" = "0x1002" ] &&
            [ "$(cat "$dev/device")" = "0x13fe" ]; then
            return 0
        fi
    done
    return 1
}

kernel_base() {
    # 7.1.5-arch1-1 -> 7.1.5
    uname -r | sed 's/^\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/'
}

# Newest-first comparison without bc or awk arithmetic: sort -V.
version_at_least() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

modparam() {
    if [ -r "/sys/module/amdgpu/parameters/$1" ]; then
        cat "/sys/module/amdgpu/parameters/$1"
    else
        printf ''
    fi
}

free_gib() {
    df -P -k "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

# --------------------------------------------------------------------------
# Fetching the reference repositories
# --------------------------------------------------------------------------

# Cloned at the SHA pinned in pins.toml rather than vendored into this repo:
# these are somebody else's kernel patches and the attribution belongs to them.
fetch_pinned() {
    name=$1
    url=$2
    sha=$3
    dest="$BC250_SRC/$name"

    mkdir -p "$BC250_SRC"
    if [ -d "$dest/.git" ]; then
        if [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
            ok "$name already at $(printf '%.12s' "$sha")"
            return 0
        fi
        info "$name is at the wrong commit; fetching $(printf '%.12s' "$sha")"
        runcmd git -C "$dest" fetch --quiet origin "$sha" ||
            runcmd git -C "$dest" fetch --quiet origin
        runcmd git -C "$dest" checkout --quiet "$sha"
    else
        runcmd git clone --quiet "$url" "$dest" || die "could not clone $url"
        runcmd git -C "$dest" checkout --quiet "$sha" ||
            die "$url has no commit $sha — pins.toml may need updating"
    fi
    ok "$name at $(printf '%.12s' "$sha")"
}
