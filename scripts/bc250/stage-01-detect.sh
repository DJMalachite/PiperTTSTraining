#!/bin/sh
# Stage 1: measure. Read-only — this stage changes nothing.
#
# Everything here is also checked by './run doctor' once a venv exists; the
# duplication is deliberate, because this has to work on a bare clone before
# there is any venv to run doctor from. Both read the same constants where they
# can, and both defer to the hardware rather than to the documentation.
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

missing=0

head_ "Board"
if is_bc250; then
    ok "AMD BC-250 found (PCI 1002:13fe)"
else
    refuse "no BC-250 on this machine (no PCI device 1002:13fe)" \
        "This build is only useful on that board. Everything else in this repo works as normal."
fi

head_ "Kernel"
release=$(uname -r)
base=$(kernel_base)
if version_at_least "$base" 7.1.5; then
    ok "kernel $release"
else
    bad "kernel $release is older than 7.1.5"
    info "Below 7.1.5 the correctness fix and the 40-CU unlock cannot coexist:"
    info "the board drops to 24 CU, where compute wedges. CachyOS ships recent"
    info "kernels — update and reboot before going any further."
    missing=$((missing + 1))
fi

head_ "amdgpu module parameters"
for pair in bc250_cc_write_mode:3 bc250_flush_by_runlist:1; do
    name=${pair%:*}
    want=${pair#*:}
    have=$(modparam "$name")
    if [ "$have" = "$want" ]; then
        ok "$name = $have"
    elif [ -z "$have" ]; then
        warn "$name is absent — the running amdgpu module has no such parameter"
        missing=$((missing + 1))
    else
        warn "$name = $have, wanted $want"
        missing=$((missing + 1))
    fi
done

sched=$(modparam sched_policy)
if [ -z "$sched" ] || [ "$sched" = "0" ]; then
    ok "sched_policy = ${sched:-default} (hardware scheduling)"
elif [ "$sched" = "2" ]; then
    bad "sched_policy = 2"
    info "That was the workaround for older kernels. With the patched module on"
    info "7.1.5 it is the difference between a wedge and a clean run: remove it."
    missing=$((missing + 1))
else
    warn "sched_policy = $sched (expected the default)"
fi

head_ "Compute units"
# sed rather than grep, because grep exits 1 when it matches nothing and this
# script runs under `set -e`: a bare `count=$(... | grep ...)` would abort the
# whole stage silently. Most distributions set kernel.dmesg_restrict, so the
# no-match path is the common one, not the exceptional one.
if dmesg >/dev/null 2>&1; then
    count=$(dmesg 2>/dev/null |
        sed -n 's/.*active_cu_number[^0-9]*\([0-9][0-9]*\).*/\1/p' | tail -n1)
    if [ -z "$count" ]; then
        info "dmesg has nothing about active_cu_number (the ring buffer may have wrapped)"
    elif [ "$count" -ge 40 ]; then
        ok "active_cu_number $count"
    else
        warn "active_cu_number $count — the 40-CU unlock is not active"
        info "Power-cycle rather than soft-reboot if this follows a compute wedge."
        missing=$((missing + 1))
    fi
else
    info "dmesg is unreadable here (kernel.dmesg_restrict); to check it yourself:"
    info "  sudo dmesg | grep active_cu_number"
fi

head_ "Build environment"
need=$(pin build_free_gib)
have=$(free_gib "$BC250_REPO")
if [ "${have:-0}" -ge "$need" ]; then
    ok "${have} GiB free (needs ~${need} GiB for kernel source, rocBLAS and pytorch)"
else
    bad "${have:-0} GiB free, and the builds need about ${need} GiB"
    missing=$((missing + 1))
fi

for tool in git make gcc python3; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool"
    else
        bad "$tool is not installed"
        missing=$((missing + 1))
    fi
done

for tool in podman distrobox; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool"
    else
        warn "$tool is not installed — the 'container' stage needs it"
        info "  sudo pacman -S $tool"
    fi
done

head_ "Already built"
container_name=$(pin container_name)
if command -v distrobox >/dev/null 2>&1 &&
    distrobox list 2>/dev/null | grep -q "[[:space:]]$container_name[[:space:]]"; then
    ok "container '$container_name' exists"
else
    info "container '$container_name' not created yet"
fi

wheel=$(ls "$BC250_WHEELS"/torch-*.whl 2>/dev/null | tail -n1 || true)
if [ -n "$wheel" ]; then
    ok "torch wheel: $(basename "$wheel")"
else
    info "no torch wheel built yet"
fi

say ""
if [ "$missing" -gt 0 ]; then
    say "$missing thing(s) above still need doing. The stages that follow do the"
    say "kernel-side and userspace work in order; nothing here is fatal on its own."
else
    say "Nothing outstanding on the host side."
fi
exit 0
