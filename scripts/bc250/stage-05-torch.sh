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
# Job count, from memory rather than cores
# --------------------------------------------------------------------------
#
# The BC-250 has 16 GB shared between CPU and GPU and torch's C++ link steps are
# the memory peak, not the compiles. Sizing by nproc is how this build gets
# OOM-killed six hours in.

mem_gib=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
cores=$(nproc)
jobs=$((mem_gib / 2))
[ "$jobs" -lt 1 ] && jobs=1
[ "$jobs" -gt "$cores" ] && jobs="$cores"
info "${mem_gib} GiB RAM, ${cores} cores -> MAX_JOBS=$jobs"

swap_gib=$(awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$swap_gib" -lt 8 ]; then
    warn "only ${swap_gib} GiB of swap"
    info "A link step that runs out of memory kills the build after hours of"
    info "work. Consider 'sudo systemd-run --scope swapon' on a file of 16 GiB"
    info "on the host before continuing."
    confirm "continue anyway?" || refuse "stopped" "Add swap and re-run this stage."
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
