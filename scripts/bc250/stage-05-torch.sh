#!/bin/sh
# Stage 5: torch, built from source with PYTORCH_ROCM_ARCH=gfx1013.
#
# This is the stage that matters. torch's HIP kernels are compiled ahead of
# time, once per target in torch.cuda.get_arch_list(); gfx1013 is in no stock
# wheel's list, so every elementwise and convolution launch raises 'invalid
# device function' while matmul keeps working, because GEMM comes from rocBLAS.
# Compiling for gfx1013 is the only known way to remove that.
#
# Precedent, not proof: the same approach works for gfx1010, which is the same
# ISA family and equally unsupported (Efenstor/PyTorch-ROCm-gfx1010). Nobody has
# published a gfx1013 result either way. Stage 6 measures it.
#
# The version deliberately matches [torch.rocm] in pins.toml so that the
# constraint file and every existing version check stay coherent whether the
# wheel came from download.pytorch.org or from here.
#
# Hours, not minutes. It is also portable: the container is the build
# environment, so the same image on a faster machine produces a wheel that works
# here. Copy it into .state/bc250/wheels/ and skip straight to stage 6.
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

in_container || refuse "this stage builds against the container's ROCm" \
    "Run scripts/bc250/build.sh from the host and it will enter the container for you."

repo=$(pin torch_repo)
tag=$(pin torch_tag)
local_version=$(pin torch_local_version)
target=$(pin gfx_target)
python=$(pin container_python)

SRC="$HOME/pytorch"
BUILD_VENV="$BC250_STATE/build-venv"

existing=$(ls "$BC250_WHEELS"/torch-*.whl 2>/dev/null | tail -n1 || true)
if [ -n "$existing" ]; then
    ok "wheel already built: $(basename "$existing")"
    say "  Delete it, or pass --force, to build again."
    exit 0
fi

# --------------------------------------------------------------------------
# Job count
# --------------------------------------------------------------------------
#
# Every core by default. The thing to know is that memory, not cores, is what
# actually ends this build: torch's C++ link steps are the peak, and on a BC-250
# the GPU has already taken half of the 16 GB, so a machine with 16 threads may
# have well under a gigabyte of headroom per job. An OOM-killed link three hours
# in loses those three hours.
#
# So: use the cores, measure the budget, and say plainly when the two do not
# agree. --jobs (or BC250_MAX_JOBS) overrides in either direction.

mem_gib=$(awk '/^MemTotal:/ {printf "%d", $2/1024/1024}' /proc/meminfo || echo 8)
[ -n "$mem_gib" ] || mem_gib=8
swap_gib=$(awk '/^SwapTotal:/ {printf "%d", $2/1024/1024}' /proc/meminfo || echo 0)
[ -n "$swap_gib" ] || swap_gib=0
cores=$(nproc)

jobs=${BC250_MAX_JOBS:-$cores}
case "$jobs" in
    '' | *[!0-9]*) refuse "--jobs must be a number (got '$jobs')" "e.g. --jobs 8" ;;
esac
[ "$jobs" -ge 1 ] || jobs=1

budget=$((mem_gib + swap_gib))
# Roughly 1.5 GiB per parallel job at the peak, which is where torch's larger
# translation units and the final links sit.
want=$((jobs * 3 / 2))
info "${cores} cores, ${mem_gib} GiB RAM + ${swap_gib} GiB swap -> MAX_JOBS=$jobs"

if [ "$budget" -lt "$want" ]; then
    say ""
    warn "${jobs} jobs wants about ${want} GiB at peak; this machine has ${budget} GiB"
    info "The failure mode is the OOM killer taking a link step after hours of"
    info "compiling, not a slow build. Two ways to keep all ${cores} threads:"
    info ""
    info "  add swap on the host (it is only touched at the peak):"
    info "    sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile"
    info "    sudo mkswap /swapfile && sudo swapon /swapfile"
    info ""
    info "  or build with fewer jobs:"
    info "    scripts/bc250/build.sh --stage torch --jobs $((budget * 2 / 3))"
    say ""
    confirm "continue with ${jobs} jobs?" ||
        refuse "stopped" "Add swap or pass --jobs, then re-run this stage."
fi

# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------

head_ "pytorch $tag"
if [ -d "$SRC/.git" ]; then
    ok "already cloned at $SRC"
else
    runcmd git clone --depth 1 --branch "$tag" --recurse-submodules \
        --shallow-submodules "$repo" "$SRC" ||
        refuse "could not clone $repo at $tag" "Check pins.toml's torch_tag."
fi

head_ "Build environment"
if [ ! -x "$BUILD_VENV/bin/python" ]; then
    runcmd "$python" -m venv "$BUILD_VENV"
fi
runcmd "$BUILD_VENV/bin/pip" install --quiet --upgrade pip setuptools wheel
runcmd "$BUILD_VENV/bin/pip" install --quiet -r "$SRC/requirements.txt"
runcmd "$BUILD_VENV/bin/pip" install --quiet cmake ninja
ok "$BUILD_VENV"

# --------------------------------------------------------------------------
# Hipify and build
# --------------------------------------------------------------------------

head_ "Hipifying"
cd "$SRC"
runcmd "$BUILD_VENV/bin/python" tools/amd_build/build_amd.py

head_ "Building torch for $target"
say "  Hours. Log: $BC250_LOG"
say ""
say "  Flags worth knowing about:"
say "    USE_ROCM_CK_GEMM/SDPA=0  Composable Kernel has no gfx10 support at all."
say "    USE_FLASH_ATTENTION=0    likewise, and flash attention on this board is"
say "                             reported intermittent by boot even where it builds."
say "    PYTORCH_BUILD_VERSION    so the wheel is self-describing rather than"
say "                             2.9.1a0+git<sha>."
say ""

if [ "$BC250_DRY_RUN" = "1" ]; then
    info "\$ PYTORCH_ROCM_ARCH=$target ... python setup.py bdist_wheel"
else
    mkdir -p "$(dirname "$BC250_LOG")"
    PYTORCH_ROCM_ARCH="$target" \
    PYTORCH_BUILD_VERSION="${tag#v}+$local_version" \
    PYTORCH_BUILD_NUMBER=1 \
    USE_ROCM=1 \
    USE_CUDA=0 \
    USE_ROCM_CK_GEMM=0 \
    USE_ROCM_CK_SDPA=0 \
    USE_FLASH_ATTENTION=0 \
    USE_MEM_EFF_ATTENTION=0 \
    USE_KINETO=0 \
    BUILD_TEST=0 \
    MAX_JOBS="$jobs" \
        "$BUILD_VENV/bin/python" setup.py bdist_wheel 2>&1 | tee -a "$BC250_LOG" ||
        refuse "the torch build failed" \
            "The tail of $BC250_LOG says why. An OOM-killed link step is the most likely cause; lower MAX_JOBS or add swap."
fi

head_ "Wheel"
if [ "$BC250_DRY_RUN" != "1" ]; then
    wheel=$(ls "$SRC"/dist/torch-*.whl 2>/dev/null | tail -n1 || true)
    [ -n "$wheel" ] || die "the build reported success but produced no wheel"
    mkdir -p "$BC250_WHEELS"
    cp "$wheel" "$BC250_WHEELS/"
    ok "$BC250_WHEELS/$(basename "$wheel")"
fi

say ""
say "Stage 6 installs it and runs the autograd probe. That probe, not this"
say "build succeeding, is what tells you whether training works."
