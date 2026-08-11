#!/bin/sh
# Stage 3: a Fedora container to build in, and to train in afterwards.
#
# Fedora is not a preference. The reference work (akandr/bc250-rocm) was done on
# Fedora 43, its rocBLAS build script expects a system ROCm laid out the way
# Fedora lays it out, and Fedora packages ROCm 6.4 in its own repositories. On
# any other base those scripts become something you adapt rather than run.
#
# Training happens in here too. A torch built from source links against the
# container's /usr ROCm rather than bundling its own, so the wheel is only
# meaningful where those libraries are — which also means there is nothing to
# graft into the host and no ABI to match. The cost is a second venv, which is
# what PIPERTRAINER_ENV=bc250 selects.
set -eu

BC250_REPO=${BC250_REPO:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
export BC250_REPO
# shellcheck source=scripts/bc250/lib.sh
. "$BC250_REPO/scripts/bc250/lib.sh"

name=$(pin container_name)
image=$(pin container_image)
python=$(pin container_python)

command -v distrobox >/dev/null 2>&1 ||
    refuse "distrobox is not installed" "sudo pacman -S distrobox podman"
command -v podman >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 ||
    refuse "neither podman nor docker is installed" "sudo pacman -S podman"

head_ "Container"
if distrobox list 2>/dev/null | grep -q "[[:space:]]$name[[:space:]]"; then
    ok "'$name' already exists"
else
    # /dev/kfd is the compute device and /dev/dri the render nodes; without both
    # ROCm sees no GPU at all. The group adds match the host groups that own
    # them, and seccomp=unconfined is needed because the HSA runtime makes
    # syscalls the default profile blocks.
    runcmd distrobox create --name "$name" --image "$image" --yes \
        --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined"
    ok "created '$name' from $image"
fi

head_ "Packages"
# ROCm development stack, a toolchain, and an interpreter that is not Fedora's
# default: numba (which openai-whisper needs) lags new CPython releases, and
# pins.toml's [python].prefer already puts 3.13 first for the same reason.
packages="rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake rocm-comgr-devel \
miopen-hip-devel rocminfo rocm-smi \
gcc gcc-c++ make cmake ninja-build git patch which \
$python $python-devel python3-devel python3-pip \
python3-pyyaml python3-msgpack python3-joblib \
ffmpeg-free zstd xz findutils"

# shellcheck disable=SC2086
runcmd distrobox enter "$name" -- sudo dnf install -y $packages ||
    refuse "package installation failed inside '$name'" \
        "Run 'distrobox enter $name' and install them by hand; the list is above."

head_ "Verifying the GPU is visible in the container"
if [ "$BC250_DRY_RUN" = "1" ]; then
    info "\$ distrobox enter $name -- rocminfo"
else
    arch=$(distrobox enter "$name" -- sh -c \
        "rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -n1" || true)
    case "$arch" in
        gfx1013) ok "rocminfo reports $arch" ;;
        "")
            refuse "rocminfo sees no GPU inside the container" \
                "Check that you are in the render and video groups on the host: 'sudo usermod -aG render,video \$USER', then log out and back in."
            ;;
        *)
            warn "rocminfo reports $arch, expected gfx1013"
            info "Continuing — the tool defers to your hardware over the documentation."
            ;;
    esac
fi

say ""
say "The container shares \$HOME, so this repo is the same repo inside it."
say "Enter it with:  distrobox enter $name"
