#!/bin/sh
# Build a gfx1013 ROCm stack for the AMD BC-250, in stages.
#
#   scripts/bc250/build.sh                 run every stage that is not done
#   scripts/bc250/build.sh --list          show the stages and their state
#   scripts/bc250/build.sh --stage torch   run exactly one stage
#   scripts/bc250/build.sh --from rocblas  run from that stage onwards
#   scripts/bc250/build.sh --force ...     re-run stages already marked done
#   scripts/bc250/build.sh --reset torch   delete one stage's artefacts, then stop
#   scripts/bc250/build.sh --reset all     delete every stage's artefacts
#   scripts/bc250/build.sh --dry-run       print every command, change nothing
#   scripts/bc250/build.sh --yes           do not ask before root commands
#
# Interrupted part-way? Just run it again. A stage is only marked done once it
# exits cleanly, so the next run resumes at the one that did not finish, and the
# expensive stages resume rather than restart. --reset is for when a stage's own
# artefacts are the problem, not for ordinary interruptions.
#
# Read docs/BC250.md first. The short version: the stock torch ROCm wheel
# contains no gfx1013 device code, so every kernel launch outside rocBLAS
# raises 'invalid device function'. This builds one that does.
#
# It is not quick and it is not guaranteed. Stages 1-2 touch the host kernel and
# need a reboot; 3-6 happen inside a Fedora container and the torch build alone
# is hours. Every stage is idempotent and records its own completion, so an
# interrupted run resumes rather than restarts.
set -eu

BC250_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export BC250_REPO
cd "$BC250_REPO"

# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

# name:where:script. "host" stages touch the kernel and the container runtime;
# "box" stages build and install inside the container.
STAGES='detect:any:stage-01-detect.sh
kernel:host:stage-02-kernel.sh
container:host:stage-03-container.sh
rocblas:box:stage-04-rocblas.sh
torch:box:stage-05-torch.sh
install:box:stage-06-install.sh'

stage_names() { printf '%s\n' "$STAGES" | cut -d: -f1; }
stage_where() { printf '%s\n' "$STAGES" | awk -F: -v n="$1" '$1==n {print $2}'; }
stage_file()  { printf '%s\n' "$STAGES" | awk -F: -v n="$1" '$1==n {print $3}'; }

usage() {
    # The header comment, up to the first line that is not one.
    sed -n '2,/^[^#]/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# What each stage leaves behind, for --reset. An interrupted stage does not
# need this: its done- marker was never written, so re-running resumes there.
# This is for the case where a stage's own artefacts are the problem — a
# half-applied patch, a wedged build tree — and only that case, because some of
# these represent hours.
#
# Deliberately absent: the downloaded kernel tarball (expensive to fetch, cheap
# to re-extract) and the patched amdgpu module already installed on the host,
# which is undone by restoring the backup rather than by deleting anything.
stage_artefacts() {
    case "$1" in
        detect) ;;
        kernel) printf '%s\n' "$BC250_STATE"/linux-* ;;
        container) printf 'container:%s\n' "$container_name" ;;
        rocblas) printf '%s\n%s\n' "$HOME/rocBLAS" "$HOME/Tensile" ;;
        torch)
            printf '%s\n%s\n%s\n' \
                "$HOME/pytorch" "$BC250_STATE/build-venv" "$BC250_WHEELS"
            ;;
        install)
            printf '%s\n%s\n' \
                "$BC250_REPO/.venv-bc250" "$BC250_REPO/.state-bc250"
            ;;
    esac
}

reset_stage() {
    name=$1
    targets=""
    for target in $(stage_artefacts "$name"); do
        case "$target" in
            container:*)
                command -v distrobox >/dev/null 2>&1 &&
                    distrobox list 2>/dev/null |
                    grep -q "[[:space:]]${target#container:}[[:space:]]" &&
                    targets="$targets $target"
                ;;
            *)
                # An empty directory is not an artefact: a later build recreates
                # the scaffolding, and offering to delete nothing trains people
                # to answer 'y' without reading.
                if [ -d "$target" ]; then
                    [ -n "$(ls -A "$target" 2>/dev/null)" ] &&
                        targets="$targets $target"
                elif [ -e "$target" ]; then
                    targets="$targets $target"
                fi
                ;;
        esac
    done

    if [ -z "$targets" ] && ! stage_done "$name"; then
        say "-- $name: nothing to reset"
        return 0
    fi

    say ""
    say "reset '$name' would remove:"
    stage_done "$name" && say "    the 'done' marker"
    for target in $targets; do
        case "$target" in
            container:*) say "    the '${target#container:}' container" ;;
            *) say "    $target" ;;
        esac
    done
    case "$name" in
        torch) say "  NOTE: that includes the built wheel. Hours of compiling." ;;
        kernel)
            say "  NOTE: this removes the patched source tree only. The module"
            say "  already installed on this host is untouched; restore it from"
            say "  its .bc250-backup-* file if that is what you need to undo."
            ;;
    esac

    if [ "$BC250_DRY_RUN" = "1" ]; then
        say "  (--dry-run: nothing removed)"
        return 0
    fi
    confirm "remove the above?" || { say "  left alone"; return 0; }

    for target in $targets; do
        case "$target" in
            container:*) runcmd distrobox rm --force "${target#container:}" ;;
            *)
                # Never let an unset variable turn this into rm -rf /
                case "$target" in
                    "$BC250_REPO"/* | "$HOME"/*) runcmd rm -rf "$target" ;;
                    *) warn "refusing to remove $target (outside the repo and \$HOME)" ;;
                esac
                ;;
        esac
    done
    clear_done "$name"
    say "-- $name: reset"
}

only=""
from=""
reset=""
force=0

while [ $# -gt 0 ]; do
    case "$1" in
        --reset)
            reset=${2:?--reset needs a stage name, or 'all'}
            shift 2
            ;;
        --list)
            printf '%-11s %-5s %s\n' STAGE WHERE STATE
            for name in $(stage_names); do
                state=pending
                stage_done "$name" && state="done  $(cat "$BC250_STATE/done-$name")"
                printf '%-11s %-5s %s\n' "$name" "$(stage_where "$name")" "$state"
            done
            exit 0
            ;;
        --stage)
            only=${2:?--stage needs a stage name}
            shift 2
            ;;
        --from)
            from=${2:?--from needs a stage name}
            shift 2
            ;;
        --force) force=1; shift ;;
        --dry-run) BC250_DRY_RUN=1; export BC250_DRY_RUN; shift ;;
        --yes) BC250_YES=1; export BC250_YES; shift ;;
        -h | --help) usage; exit 0 ;;
        *) refuse "unknown argument: $1" "See scripts/bc250/build.sh --help." ;;
    esac
done

for name in $only $from; do
    [ -n "$(stage_where "$name")" ] ||
        refuse "unknown stage: $name" "Known stages: $(stage_names | tr '\n' ' ')"
done
if [ -n "$reset" ] && [ "$reset" != "all" ] && [ -z "$(stage_where "$reset")" ]; then
    refuse "unknown stage: $reset" \
        "Known stages: $(stage_names | tr '\n' ' ') (or 'all')"
fi

container_name=$(pin container_name)

# Before the mkdir below, or resetting a stage and then recreating its empty
# directories would leave --reset with something to offer every single time.
if [ -n "$reset" ]; then
    if [ "$reset" = "all" ]; then
        for name in $(stage_names); do
            reset_stage "$name"
        done
    else
        reset_stage "$reset"
    fi
    say ""
    say "Re-run scripts/bc250/build.sh to build again."
    exit 0
fi

mkdir -p "$BC250_STATE" "$BC250_SRC" "$BC250_WHEELS"

# Re-enter the container for stages that belong there. distrobox shares $HOME,
# so this repo is the same repo on both sides of the boundary; only the
# environment around it differs, which is exactly why the venv is separate.
enter_box() {
    file=$1
    if in_container; then
        sh "$BC250_REPO/scripts/bc250/$file"
        return $?
    fi
    command -v distrobox >/dev/null 2>&1 ||
        refuse "distrobox is not installed on the host" \
            "Run the 'container' stage first, or install distrobox."
    say "  entering the '$container_name' container"
    distrobox enter "$container_name" -- \
        env BC250_DRY_RUN="$BC250_DRY_RUN" BC250_YES="$BC250_YES" \
        sh "$BC250_REPO/scripts/bc250/$file"
}

run_stage() {
    name=$1
    file=$(stage_file "$name")
    where=$(stage_where "$name")

    if stage_done "$name" && [ "$force" = "0" ]; then
        say "-- $name: already done ($(cat "$BC250_STATE/done-$name"))"
        return 0
    fi

    say ""
    say "== stage $name ($where)"
    status=0
    if [ "$where" = "box" ]; then
        enter_box "$file" || status=$?
    else
        if [ "$where" = "host" ] && in_container; then
            refuse \
                "the '$name' stage touches the host and cannot run inside the container" \
                "Exit the container and run it there."
        fi
        sh "$BC250_REPO/scripts/bc250/$file" || status=$?
    fi

    if [ "$status" = "$BC250_EX_PAUSE" ]; then
        say ""
        say "stopped after '$name' on purpose — see the message above."
        say "Re-run scripts/bc250/build.sh when you are ready to continue."
        exit "$BC250_EX_PAUSE"
    fi
    [ "$status" = "0" ] || exit "$status"

    [ "$BC250_DRY_RUN" = "1" ] || mark_done "$name"
}

if [ -n "$only" ]; then
    [ "$force" = "1" ] && clear_done "$only"
    run_stage "$only"
    exit 0
fi

started=0
[ -n "$from" ] || started=1
for name in $(stage_names); do
    [ "$name" = "$from" ] && started=1
    [ "$started" = "1" ] || continue
    [ "$force" = "1" ] && clear_done "$name"
    run_stage "$name"
done

say ""
say "all stages complete. The verdict is the autograd probe:"
say "    PIPERTRAINER_ENV=bc250 ./run doctor"
