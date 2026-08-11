#!/bin/sh
# Build a gfx1013 ROCm stack for the AMD BC-250, in stages.
#
#   scripts/bc250/build.sh                 run every stage that is not done
#   scripts/bc250/build.sh --list          show the stages and their state
#   scripts/bc250/build.sh --stage torch   run exactly one stage
#   scripts/bc250/build.sh --from rocblas  run from that stage onwards
#   scripts/bc250/build.sh --force ...     re-run stages already marked done
#   scripts/bc250/build.sh --dry-run       print every command, change nothing
#   scripts/bc250/build.sh --yes           do not ask before root commands
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

only=""
from=""
force=0

while [ $# -gt 0 ]; do
    case "$1" in
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

mkdir -p "$BC250_STATE" "$BC250_SRC" "$BC250_WHEELS"

container_name=$(pin container_name)

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
