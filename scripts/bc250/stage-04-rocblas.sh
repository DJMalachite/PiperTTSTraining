#!/bin/sh
# Stage 4: rocBLAS with native gfx1013 kernels.
#
# Runs akandr's build_rocblas_gfx1013.sh, which adds gfx1013 to Tensile's
# SupportedISA and AsmCaps and to the Tensile and rocBLAS C++ architecture
# enums, then builds against the container's system ROCm. That is exactly the
# environment it was written for, which is the point of stage 3.
#
# Be clear about what this buys. rocBLAS is the GEMM library: it fixes matmul
# throughput. The training blocker is missing elementwise and convolution
# kernels, which come from torch's own device code and are stage 5's problem.
# Expect this check to go green and the autograd probe to still fail until
# stage 5 lands.
#
# Installed to its own prefix rather than over /usr, so it is reversible and so
# that removing one file undoes it.
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

in_container || refuse "this stage builds against the container's ROCm" \
    "Run scripts/bc250/build.sh from the host and it will enter the container for you."

PREFIX=/opt/bc250/rocm
BUILD="$HOME/rocBLAS/build/release"

head_ "Reference sources"
fetch_pinned bc250-rocm "$(pin reference_repo)" "$(pin reference_sha)"
script="$BC250_SRC/bc250-rocm/scripts/build_rocblas_gfx1013.sh"
[ -f "$script" ] || die "missing $script"

head_ "Building rocBLAS for gfx1013"
say "  This clones Tensile and rocBLAS and compiles 56 Tensile libraries."
say "  It takes a long time and it is somebody else's script — read it first:"
say "    $script"
say ""
if [ "$BC250_DRY_RUN" = "1" ]; then
    info "\$ sh $script"
elif [ -d "$BUILD/Tensile/library" ] &&
    [ "$(find "$BUILD/Tensile/library" -name '*gfx1013*' | wc -l)" -gt 0 ]; then
    ok "already built at $BUILD"
else
    confirm "run it now?" || refuse "declined" \
        "Run it yourself, then re-run this stage; it will pick up the artefacts."
    sh "$script" || refuse "the rocBLAS build failed" \
        "The log is above. This is the stage most sensitive to the ROCm version; see docs/BC250.md."
fi

head_ "Checking the artefacts"
if [ "$BC250_DRY_RUN" != "1" ]; then
    count=$(find "$BUILD/Tensile/library" -name '*gfx1013*' 2>/dev/null | wc -l)
    [ "$count" -gt 0 ] || refuse "the build produced no gfx1013 code objects" \
        "Without them this stage has done nothing; do not install it."
    ok "$count gfx1013 Tensile libraries"
fi

head_ "Installing to $PREFIX"
run_root mkdir -p "$PREFIX/lib/rocblas/library"
if [ "$BC250_DRY_RUN" != "1" ]; then
    for so in "$BUILD"/library/src/librocblas.so*; do
        [ -e "$so" ] || continue
        run_root cp -a "$so" "$PREFIX/lib/"
    done
    run_root cp -a "$BUILD/Tensile/library/." "$PREFIX/lib/rocblas/library/"
fi
ok "installed"

# rocBLAS finds its Tensile libraries through ROCBLAS_TENSILE_LIBPATH, and the
# loader finds the patched library through LD_LIBRARY_PATH. Both are read by
# './run doctor' too, so the check and the reality stay in agreement.
head_ "Environment"
profile=/etc/profile.d/bc250.sh
if [ "$BC250_DRY_RUN" != "1" ]; then
    cat >"$BC250_STATE/bc250-profile.sh" <<EOF
# Written by scripts/bc250/build.sh. Points at the gfx1013 rocBLAS built from
# source; delete this file to fall back to the system one.
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export ROCBLAS_TENSILE_LIBPATH="$PREFIX/lib/rocblas/library"
EOF
    run_root cp "$BC250_STATE/bc250-profile.sh" "$profile"
fi
ok "$profile"

say ""
say "Open a new shell in the container, or source $profile, before stage 5."
say "Nothing here unblocks training on its own — that is stage 5."
