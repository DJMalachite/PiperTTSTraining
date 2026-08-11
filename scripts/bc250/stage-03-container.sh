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
# The list, and the decision about what is required versus merely wanted, lives
# in packages.sh because it has to run inside the box. It probes the repos
# first: Fedora renames ROCm packages between releases, and one bad name in a
# single dnf line takes the other twenty with it.
runcmd distrobox enter "$name" -- \
    sh "$BC250_REPO/scripts/bc250/packages.sh" "$python" ||
    refuse "package installation failed inside '$name'" \
        "The missing package is named above. Run 'distrobox enter $name' to look around."

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
