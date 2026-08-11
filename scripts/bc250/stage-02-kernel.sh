#!/bin/sh
# Stage 2: build a patched amdgpu module carrying all three BC-250 fixes.
#
# The fixes come from two separate projects and both target Fedora. This applies
# them on Arch/CachyOS, where three things differ and each of them will cost you
# a working display if got wrong:
#
#   * modules are compressed with zstd, not xz. The upstream 'xz --check=crc32'
#     advice is a Fedora detail; here we match whatever the installed module
#     already is.
#   * kernel modules must be built by the same toolchain family as the kernel.
#     CachyOS ships clang/LTO builds, so LLVM=1 is detected, not assumed.
#   * the initramfs generator is mkinitcpio, not dracut.
#
# The original module is backed up next to itself before anything is installed,
# and the path is printed. Keep it: restoring it is how you get your display
# back if the new one does not load.
#
#   40-CU unlock            duggasco/bc250-40cu-unlock
#   runlist flush on unmap  akandr/bc250-rocm, scripts/apply_runlist_flush.py
#   PASID flush via MMIO    akandr/bc250-rocm, patches/amdgpu-flush-pasid-mmio.patch
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

is_bc250 || refuse "no BC-250 on this machine" "Nothing here applies."
command -v make >/dev/null 2>&1 || refuse "make is not installed" \
    "sudo pacman -S base-devel"

KVER=$(uname -r)
KBASE=$(kernel_base)
KBUILD="/usr/lib/modules/$KVER/build"
[ -d "$KBUILD" ] || KBUILD="/lib/modules/$KVER/build"
[ -d "$KBUILD" ] || refuse \
    "no kernel build tree at /usr/lib/modules/$KVER/build" \
    "Install the headers for the running kernel, e.g. 'sudo pacman -S linux-cachyos-headers'."

TREE="$BC250_STATE/linux-$KBASE"
AMDGPU="$TREE/drivers/gpu/drm/amd/amdgpu"

head_ "Reference patches"
fetch_pinned bc250-40cu-unlock "$(pin cu_unlock_repo)" "$(pin cu_unlock_sha)"
fetch_pinned bc250-rocm "$(pin reference_repo)" "$(pin reference_sha)"

# --------------------------------------------------------------------------
# Kernel source
# --------------------------------------------------------------------------
#
# The headers package is not enough: building drivers/gpu/drm/amd/amdgpu needs
# the driver *source*, which no distribution ships in -headers. Only the AMD
# subtree is extracted, which is a few tens of MB rather than a gigabyte.

head_ "Kernel source for $KBASE"
if [ -d "$AMDGPU" ]; then
    ok "already extracted at $TREE"
else
    major=${KBASE%%.*}
    tarball="$BC250_STATE/linux-$KBASE.tar.xz"
    url="https://cdn.kernel.org/pub/linux/kernel/v${major}.x/linux-${KBASE}.tar.xz"
    if [ ! -f "$tarball" ]; then
        info "downloading $url"
        runcmd curl -fL --progress-bar -o "$tarball.part" "$url" ||
            refuse "could not download $url" \
                "Check that $KBASE is a released kernel; a -rc or a distribution-only version has no tarball there."
        runcmd mv "$tarball.part" "$tarball"
    fi
    runcmd mkdir -p "$TREE"
    info "extracting the AMD driver subtree only"
    runcmd tar -xf "$tarball" -C "$TREE" --strip-components=1 \
        --wildcards '*/drivers/gpu/drm/amd/' '*/include/drm/'
    ok "extracted to $TREE"
fi

# --------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------
#
# All three go on one tree, then the module is built once. Order matters only in
# that the 40-CU patch has to be present before akandr's build script would
# accept the tree; applying it first also matches how the reference work was done.
#
# Note the tree is freshly extracted from kernel.org, so it carries none of the
# three even if the *running* module already has the 40-CU unlock.

head_ "Applying patches"
cd "$TREE"

cu_patch="$BC250_SRC/bc250-40cu-unlock/patch/bc250-40cu-amdgpu.patch"
[ -f "$cu_patch" ] || die "missing $cu_patch"
if grep -q bc250_cc_write_mode "$AMDGPU/gfx_v10_0.c" 2>/dev/null; then
    ok "40-CU unlock already applied"
else
    runcmd patch -p1 --forward --silent -i "$cu_patch" ||
        refuse "the 40-CU patch did not apply to $KBASE" \
            "It was written against a different kernel. Check duggasco/bc250-40cu-unlock for a newer revision and bump cu_unlock_sha in pins.toml."
    ok "40-CU unlock applied"
fi

# akandr's script hardcodes ~/k715/linux-7.1.5. Rewrite that one assignment
# rather than forking the script: it stays theirs, and its own marker string
# makes it idempotent.
flush_py="$BC250_SRC/bc250-rocm/scripts/apply_runlist_flush.py"
[ -f "$flush_py" ] || die "missing $flush_py"
if [ "$BC250_DRY_RUN" = "1" ]; then
    info "\$ python3 apply_runlist_flush.py (retargeted at $TREE)"
else
    tmp_py="$BC250_STATE/apply_runlist_flush.retargeted.py"
    sed "s|^S = .*|S = \"$TREE/drivers/gpu/drm/amd/amdkfd\"|" "$flush_py" >"$tmp_py"
    grep -q "$TREE" "$tmp_py" ||
        refuse "could not retarget apply_runlist_flush.py" \
            "Its path assignment no longer looks like 'S = ...'; apply it by hand against $TREE."
    python3 "$tmp_py" || refuse "apply_runlist_flush.py failed" \
        "Its anchors did not match this kernel version. See akandr/bc250-rocm."
    ok "runlist flush applied"
fi

pasid_patch="$BC250_SRC/bc250-rocm/patches/amdgpu-flush-pasid-mmio.patch"
[ -f "$pasid_patch" ] || die "missing $pasid_patch"
if grep -q 'flush_pasid_uses_kiq = false' "$AMDGPU/gmc_v10_0.c" 2>/dev/null; then
    ok "PASID MMIO flush already applied"
else
    runcmd patch -p1 --forward --silent -i "$pasid_patch" ||
        refuse "the PASID MMIO patch did not apply to $KBASE" \
            "It targets the 7.1.5 tree. Apply the one-line change to gmc_v10_0.c by hand, or pin a kernel it matches."
    ok "PASID MMIO flush applied"
fi

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

head_ "Building amdgpu"
make_args="-C $KBUILD M=$AMDGPU"
if grep -qi clang /proc/version; then
    info "this kernel was built with clang; building the module with LLVM=1"
    make_args="$make_args LLVM=1"
fi

# amdgpu's tracepoint header is looked up through the kernel tree, which the
# headers package does not carry. Staged temporarily, exactly as
# bc250-enable-40cu-arch.sh does.
trace_dst="$KBUILD/drivers/gpu/drm/amd/amdgpu"
if [ ! -f "$trace_dst/amdgpu_trace.h" ]; then
    run_root mkdir -p "$trace_dst"
    run_root cp "$AMDGPU/amdgpu_trace.h" "$trace_dst/amdgpu_trace.h"
fi

# shellcheck disable=SC2086
runcmd make $make_args -j"$(nproc)" modules ||
    refuse "the module did not build" \
        "The full error is above. A toolchain mismatch is the usual cause; see docs/BC250.md."
[ "$BC250_DRY_RUN" = "1" ] || [ -f "$AMDGPU/amdgpu.ko" ] ||
    die "make reported success but produced no amdgpu.ko"
ok "built $AMDGPU/amdgpu.ko"

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
#
# Match the compression of the module already installed. Guessing here is how
# you end up with a kernel that cannot decompress its own graphics driver.

head_ "Installing"
installed=$(modinfo -n amdgpu 2>/dev/null || true)
[ -n "$installed" ] || refuse "modinfo cannot find the installed amdgpu module" \
    "Nothing to replace or back up; refusing to guess where it should go."
info "installed module: $installed"

backup="$installed.bc250-backup-$(date +%Y%m%d%H%M%S)"
run_root cp -a "$installed" "$backup"
ok "backup: $backup"
info "if the board comes back without a display, boot a fallback entry and"
info "restore that file over $installed"

staged="$BC250_STATE/amdgpu.ko"
runcmd cp "$AMDGPU/amdgpu.ko" "$staged"
runcmd strip --strip-debug "$staged"

case "$installed" in
    *.zst)
        runcmd zstd -q -f -19 -o "$staged.zst" "$staged"
        run_root cp "$staged.zst" "$installed"
        ;;
    *.xz)
        # crc32, not the xz default of crc64: the in-kernel decompressor cannot
        # read crc64, and the failure only shows up at boot.
        if [ "$BC250_DRY_RUN" = "1" ]; then
            info "\$ xz -c -f --check=crc32 $staged > $staged.xz"
        else
            xz -c -f --check=crc32 --lzma2=preset=6,dict=1MiB "$staged" >"$staged.xz"
            xz -t "$staged.xz" || die "the compressed module fails its own integrity check"
        fi
        run_root cp "$staged.xz" "$installed"
        ;;
    *.ko)
        run_root cp "$staged" "$installed"
        ;;
    *)
        refuse "unrecognised module compression: $installed" \
            "Install it by hand with the same compression as the backup."
        ;;
esac
ok "installed"

run_root depmod -a "$KVER"

head_ "Module parameters"
conf=/etc/modprobe.d/bc250.conf
line="options amdgpu bc250_cc_write_mode=3 bc250_flush_by_runlist=1"
if [ -f "$conf" ] && grep -qF "$line" "$conf" 2>/dev/null; then
    ok "$conf already correct"
else
    info "writing $conf:"
    info "  $line"
    if [ "$BC250_DRY_RUN" != "1" ]; then
        printf '# Written by scripts/bc250/build.sh. See docs/BC250.md.\n%s\n' \
            "$line" >"$BC250_STATE/bc250.conf"
        run_root cp "$BC250_STATE/bc250.conf" "$conf"
    fi
fi

# amdgpu loads from the initramfs, so the parameters have to be visible there.
# Whether modprobe.d makes it in depends on the generator's configuration, which
# is why the kernel command line is the belt-and-braces answer the reference
# documentation gives.
if command -v mkinitcpio >/dev/null 2>&1; then
    run_root mkinitcpio -P
elif command -v dracut >/dev/null 2>&1; then
    run_root dracut -f --kver "$KVER"
else
    warn "no mkinitcpio or dracut found — rebuild your initramfs by hand"
fi

say ""
say "Add these to the kernel command line as well, through whatever bootloader"
say "this machine uses (CachyOS is usually systemd-boot or limine, not GRUB):"
say ""
say "    amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_by_runlist=1"
say ""
say "Do NOT add amdgpu.sched_policy=2. Hardware scheduling must stay at the"
say "default; on 7.1.5 with this module it is the difference between a wedge"
say "and a clean run."
say ""
say "Then reboot and re-run scripts/bc250/build.sh. After the reboot check:"
say "    dmesg | grep active_cu_number         # expect 40"
say "    cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist   # expect 1"
exit "$BC250_EX_PAUSE"
