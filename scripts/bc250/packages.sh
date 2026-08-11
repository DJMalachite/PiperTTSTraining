#!/bin/sh
# Package resolution, run *inside* the container by stage 3.
#
# Separate from stage-03 because it has to execute in the box, and because
# hardcoding one long `dnf install` line is how the container stage dead-ends:
# one renamed package and nothing installs. Fedora ships MIOpen as `miopen`
# with no -devel subpackage, for instance, so an earlier version of this asking
# for `miopen-hip-devel` failed outright and took the whole list with it.
#
# So: probe first, then split. A missing REQUIRED package is a refusal that
# names it. A missing OPTIONAL one is a warning that says what is lost — never
# a silent skip, and never a reason to abandon the other twenty.
set -eu

# Without these there is nothing to build torch or rocBLAS with.
REQUIRED="rocm-hip-devel rocblas-devel rocm-cmake rocminfo
gcc gcc-c++ make cmake ninja-build git patch which
python3-pip python3-devel"

# Wanted, but the build survives without them:
#   miopen          torch's convolutions go through MIOpen when it is present.
#                   Fedora ships no headers for it, so torch will most likely
#                   build without it and use its own im2col+GEMM path instead —
#                   which on gfx1013 is arguably what you want anyway, since
#                   MIOpen has no kernels for this architecture either.
#   hipblas*        torch probes for them; TORCH_BLAS_PREFER_HIPBLASLT=0 in the
#                   hardware profile means we do not use hipBLASLt regardless.
#   ffmpeg-free     only needed to build datasets inside the container.
#   pyyaml/msgpack  Tensile's Python dependencies; pip has them too.
OPTIONAL="miopen hipblas-devel hipblaslt-devel rocm-comgr-devel rocm-smi
ffmpeg-free zstd xz findutils
python3-pyyaml python3-msgpack python3-joblib"

# The interpreter is passed in, because it is pinned in pins.toml rather than
# being whatever the image defaults to: numba, which openai-whisper needs, lags
# new CPython and Fedora 43 defaults to 3.14.
PYTHON=${1:?usage: packages.sh <python-package-name>}
REQUIRED="$REQUIRED $PYTHON $PYTHON-devel"

exists() {
    dnf -q info "$1" >/dev/null 2>&1
}

install=""
missing_required=""
missing_optional=""

printf 'resolving packages against the enabled repositories...\n'
for pkg in $REQUIRED; do
    if exists "$pkg"; then
        install="$install $pkg"
    else
        missing_required="$missing_required $pkg"
    fi
done
for pkg in $OPTIONAL; do
    if exists "$pkg"; then
        install="$install $pkg"
    else
        missing_optional="$missing_optional $pkg"
    fi
done

if [ -n "$missing_optional" ]; then
    printf '\n  [warn] not in the repositories, continuing without:\n'
    for pkg in $missing_optional; do
        printf '         %s\n' "$pkg"
    done
    printf '         See the notes in scripts/bc250/packages.sh for what each\n'
    printf '         one would have provided; none of them stops the build.\n'
fi

if [ -n "$missing_required" ]; then
    printf '\nerror: these are needed and are not in the repositories:\n' >&2
    for pkg in $missing_required; do
        printf '       %s\n' "$pkg" >&2
    done
    printf '       Fedora renames ROCm packages between releases. Find the\n' >&2
    printf '       current name with "dnf search rocm" inside the container,\n' >&2
    printf '       then update REQUIRED in scripts/bc250/packages.sh.\n' >&2
    exit 1
fi

printf '\ninstalling:%s\n\n' "$install"
# shellcheck disable=SC2086
sudo dnf install -y $install
