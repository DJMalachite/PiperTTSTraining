#!/bin/sh
# Acceptance gate: prove the whole pipeline works on CPU, offline, in minutes.
#
#   scripts/smoke_test.sh              every tier
#   scripts/smoke_test.sh unit         unit tests only (no venv needed)
#   scripts/smoke_test.sh --keep       leave the _smoke voice behind to inspect
#
# Tiers: unit (pure functions) -> dataset (synthetic clips) -> train (one CPU
# epoch) -> export (ONNX plus a synthesized sentence). Also runs the negative
# tests, including the one that confirms upstream really does reject
# --trainer.gradient_clip_val under manual optimization.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

stage=all
keep=""
for arg in "$@"; do
    case "$arg" in
        --keep) keep=--keep ;;
        unit|dataset|train|export|all) stage=$arg ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

echo "running self-test (stage: $stage)"
exec ./run smoke --stage "$stage" ${keep:+$keep}
