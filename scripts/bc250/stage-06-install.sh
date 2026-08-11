#!/bin/sh
# Stage 6: install the wheel into a venv of its own, then measure.
#
# PIPERTRAINER_ENV=bc250 puts this environment in .venv-bc250 and .state-bc250,
# beside — not on top of — whatever the host has. They cannot share one: this
# torch links against the container's ROCm and the host's does not have it.
#
# The last thing this does is run './run doctor', whose autograd probe is the
# only answer that counts. It builds the layer types a VITS vocoder actually
# uses (Conv1d, ConvTranspose1d, LeakyReLU, elementwise arithmetic) and runs 60
# optimizer steps allocating fresh tensors each iteration, which reaches both
# documented failure modes. A matmul reaches neither.
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

in_container || refuse "this stage installs into the container's environment" \
    "Run scripts/bc250/build.sh from the host and it will enter the container for you."

cd "$BC250_REPO"

wheel=$(ls "$BC250_WHEELS"/torch-*.whl 2>/dev/null | tail -n1 || true)
[ -n "$wheel" ] || refuse "no torch wheel in $BC250_WHEELS" \
    "Run the 'torch' stage first, or copy a wheel built elsewhere into that directory."

PIPERTRAINER_ENV=bc250
export PIPERTRAINER_ENV

head_ "Installing"
info "wheel:  $(basename "$wheel")"
info "venv:   .venv-bc250"
info "state:  .state-bc250"
runcmd ./run setup --vendor rocm --torch-spec "$wheel" ||
    refuse "setup failed" \
        "It is resumable: fix what it reported and re-run this stage. Individual steps can be redone with './run setup --force-step NAME'."

head_ "Checking what torch was actually built for"
if [ "$BC250_DRY_RUN" != "1" ]; then
    archs=$(./run doctor 2>/dev/null | grep -i 'torch gfx kernels' || true)
    [ -n "$archs" ] && info "$archs"
fi

head_ "The verdict"
say ""
say "  Running the full diagnostic. The line to read is 'training probe'."
say ""
status=0
runcmd ./run doctor || status=$?

say ""
if [ "$status" = "0" ]; then
    say "Training is not blocked on this board. Carry on with:"
    say "    PIPERTRAINER_ENV=bc250 ./run dataset"
    say "    PIPERTRAINER_ENV=bc250 ./run train"
    say ""
    say "Start small — one short run — and watch for a fault around iteration"
    say "20 to 40. That signature means the runlist-flush fix is not active,"
    say "not that training is impossible."
else
    say "doctor reported problems. If the autograd probe is what failed, that is"
    say "a real result and worth writing down: nobody has published a gfx1013"
    say "training result either way. Capture the error and the output of"
    say ""
    say "    PIPERTRAINER_ENV=bc250 ./run doctor"
    say ""
    say "and report it to $(pin reference_repo)."
    say ""
    say "Meanwhile everything else on this board still works: dataset"
    say "preparation, export, synthesis, and CPU training. See docs/BC250.md."
fi
exit "$status"
