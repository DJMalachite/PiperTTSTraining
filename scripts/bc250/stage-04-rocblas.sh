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
# The library itself, not the Tensile code objects beside it. Those are two
# different questions and conflating them is how this stage used to report
# success on a build that had failed: Tensile generates its gfx1013 kernels
# early and they survive a link that never happens.
librocblas() {
    ls "$BUILD"/library/src/librocblas.so* 2>/dev/null | head -n1
}

if [ "$BC250_DRY_RUN" = "1" ]; then
    info "\$ sh $script"
elif [ -n "$(librocblas)" ]; then
    ok "already built: $(librocblas)"
else
    confirm "run it now?" || refuse "declined" \
        "Run it yourself, then re-run this stage; it will pick up the artefacts."
    # The script exits 0 whether or not it produced a library, so its exit code
    # is not evidence of anything. The artefact check below is.
    sh "$script" || true
fi

# --------------------------------------------------------------------------
# Second pass: the gfx1013 edits upstream only prints
# --------------------------------------------------------------------------
#
# rocBLAS fetches its own Tensile into build/release/virtualenv at configure
# time, and that copy knows nothing about gfx1013, so no assembly kernels are
# generated and the manifest verify fails. The upstream script's step 4 says to
# fix that by hand and then only echoes the instruction. tensile-gfx1013.py is
# that step. It can only run after the first configure has created the
# virtualenv, which is why this is a second pass rather than a prerequisite.

if [ "$BC250_DRY_RUN" != "1" ] && [ -z "$(librocblas)" ]; then
    head_ "Adding gfx1013 to the Tensile rocBLAS fetched for itself"
    say "  The first pass was expected to stop at the manifest verify: the"
    say "  fetched Tensile has no gfx1013 in SupportedISA, so it generated no"
    say "  assembly kernels and the manifest asked for files nobody made."
    say ""
    python3 "$BC250_REPO/scripts/bc250/tensile-gfx1013.py" "$BUILD" ||
        refuse "could not patch Tensile for gfx1013" \
            "The reason is above. Skip this stage with 'scripts/bc250/build.sh --skip rocblas' — it governs matmul speed, not whether training works."

    head_ "Rebuilding with gfx1013 known"
    runcmd make -C "$BUILD" -j"$(nproc)" install || true
fi

head_ "Checking the artefacts"
if [ "$BC250_DRY_RUN" != "1" ]; then
    count=$(find "$BUILD/Tensile/library" -name '*gfx1013*' 2>/dev/null | wc -l)
    info "$count gfx1013 Tensile code objects"
    if [ -z "$(librocblas)" ]; then
        say ""
        say "  Still no library after the Tensile patch. The Python side is done,"
        say "  so what remains is the C++ side the upstream note also lists: the"
        say "  Processor enum in AMDGPU.hpp, LazyLoadingInit in"
        say "  PlaceholderLibrary.hpp, and the gfx1013 enum and deviceString"
        say "  branches in rocBLAS's handle.hpp, handle.cpp and tensile_host.cpp."
        say "  Read the error above to see which of those it is actually asking"
        say "  for — it is worth knowing before editing six files."
        say ""
        say "  This stage is OPTIONAL. rocBLAS is the GEMM library: it governs"
        say "  matmul speed, not whether training works. The kernels that block"
        say "  training are torch's own, and those come from the next stage."
        say ""
        say "  To carry on without it:"
        say "      scripts/bc250/build.sh --skip rocblas"
        say "      scripts/bc250/build.sh"
        say ""
        refuse "no librocblas.so was produced" \
            "Nothing has been installed. Skipping costs matmul throughput, not correctness."
    fi
    ok "librocblas.so: $(librocblas)"
fi

head_ "Installing to $PREFIX"
run_root mkdir -p "$PREFIX/lib/rocblas/library"
if [ "$BC250_DRY_RUN" != "1" ]; then
    copied=0
    for so in "$BUILD"/library/src/librocblas.so*; do
        [ -e "$so" ] || continue
        run_root cp -a "$so" "$PREFIX/lib/"
        copied=$((copied + 1))
    done
    [ "$copied" -gt 0 ] || die "no library was copied; refusing to claim success"
    run_root cp -a "$BUILD/Tensile/library/." "$PREFIX/lib/rocblas/library/"
    ok "installed $copied library file(s) and the Tensile code objects"
fi

# rocBLAS finds its Tensile libraries through ROCBLAS_TENSILE_LIBPATH, and the
# loader finds the patched library through LD_LIBRARY_PATH. Both are read by
# './run doctor' too, so the check and the reality stay in agreement.
head_ "Environment"
profile=/etc/profile.d/bc250-rocblas.sh
if [ "$BC250_DRY_RUN" != "1" ]; then
    cat >"$BC250_STATE/bc250-rocblas.sh" <<EOF
# Written by scripts/bc250/build.sh. Points at the gfx1013 rocBLAS built from
# source; delete this file to fall back to the system one.
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export ROCBLAS_TENSILE_LIBPATH="$PREFIX/lib/rocblas/library"
EOF
    run_root cp "$BC250_STATE/bc250-rocblas.sh" "$profile"
fi
ok "$profile"

say ""
say "Open a new shell in the container, or source $profile, before stage 5."
say "Nothing here unblocks training on its own — that is stage 5."
