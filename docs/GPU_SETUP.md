# GPU setup

`piper1-gpl` is vendor-agnostic — it contains no ROCm or HIP code at all, and its
only vendor-specific lines are two TF32 flags that are no-ops off NVIDIA. So the
*entire* question of which hardware trains your voice comes down to **which torch
wheel is installed**, and that is decided in one step of `./run setup`.

## The wheel matrix

From [`pins.toml`](../pins.toml):

| Vendor | Index URL | Default pin |
| --- | --- | --- |
| `rocm` | `https://download.pytorch.org/whl/rocm6.4` | `torch==2.9.1` |
| `cuda` | `https://download.pytorch.org/whl/cu128` | `torch==2.9.1` |
| `cpu` | `https://download.pytorch.org/whl/cpu` | `torch==2.9.1` |

Two details that matter more than they look:

**`--index-url`, not `--extra-index-url`.** The PyTorch index mirrors torch's
generic dependencies, so replacing PyPI entirely for that one step is safe — and
it is what actually prevents a CUDA wheel being resolved on an AMD box.

**`PIP_CONSTRAINT` on every later pip call.** Right after torch is installed,
setup writes the exact local version (e.g. `torch==2.9.1+rocm6.4`) to
`.state/torch-constraint.txt` and passes it as a constraint from then on. Without
this, `openai-whisper` — whose metadata wants an unpinned torch plus
CUDA-flavoured `triton` — will happily replace your ROCm build with the PyPI one.
That failure is silent: everything imports, and the GPU quietly stops being used.
Setup re-checks `torch.__version__` after each step and aborts loudly on drift.

## Detection

`./run setup` picks a vendor in this order: `--vendor` flag, `PT_VENDOR`
environment variable, the profile's `runtime.vendor`, the saved setup state, then
autodetection:

1. `cuda` if `nvidia-smi -L` exits 0 and lists a GPU.
2. `rocm` if `/dev/kfd` exists **and** some `/sys/class/drm/card*/device/vendor`
   reads `0x1002`. Reading sysfs avoids needing `pciutils`.
3. `cpu` otherwise.

Both present, and it asks.

## Verification is a real matmul, not `is_available()`

This is the important part on AMD. `torch.cuda.is_available()` returns **True**
for GPU architectures the ROCm build has no compiled kernels for; the process
then aborts at the first real kernel launch, often with `SIGABRT` rather than a
Python exception.

So the probe reports `torch.version.hip` / `torch.version.cuda` (never
`is_available()` alone, which is True for both vendors), `gcnArchName`,
`get_arch_list()`, and then runs a 1024×1024 matmul followed by
`torch.cuda.synchronize()`. If the probe process dies without printing, that is
itself treated as a failure.

Run it any time:

```bash
./run doctor
```

## Verification is a real training step, not just a matmul

A matmul exercises the GEMM library and nothing else. The failure that actually
stops a run is a missing elementwise or convolution kernel — `invalid device
function` — or a fault after tens of allocate/free cycles. Neither is reachable
by a matmul.

So `./run setup` and `./run doctor` also run a **real autograd loop**: a small
model built from the layer types a VITS vocoder uses (`Conv1d`,
`ConvTranspose1d`, LeakyReLU, elementwise arithmetic), 60 optimizer steps,
allocating fresh tensors every iteration. Ten seconds now beats an hour into a
run.

## AMD on Windows

ROCm runs natively on Windows, but it is delivered differently from Linux and
the differences are load-bearing.

On Linux, ROCm torch comes from `download.pytorch.org` like any other wheel, and
several ROCm versions are available to try. On Windows, AMD publishes one build
at fixed URLs on `repo.radeon.com`. `./run setup` installs it automatically when
it detects an AMD GPU; the pins live under `[torch.rocm.windows]` in
`pins.toml`. Three requirements come with it:

- **Python 3.12, exactly.** The wheels are cp312 and there is no other build.
  Setup looks for 3.12 specifically on this path instead of taking the newest
  interpreter, and if it cannot find one it says so and tells you how to get it
  — rather than building a 3.13 venv that fails later with a bare "no matching
  distribution".
- **A recent AMD graphics driver** (the version is recorded in `pins.toml`).
  This cannot be checked from here, so it is reported rather than enforced.
- **A supported GPU.** The Windows list is shorter than the Linux one. Rather
  than reproduce a list that will go stale, setup installs and then *proves* the
  device with the same real matmul and real autograd loop it uses everywhere
  else — see below.

The ROCm runtime itself ships as four more wheels, installed before torch
because torch links against them. Setup does that in its own step.

If your card turns out not to be supported, the fallbacks are CPU training
(everything except the training step runs at full speed) or WSL2, where the
Linux ROCm wheels apply. If you share one checkout between Windows and WSL2, use
`PIPERTRAINER_ENV` to give each its own venv — they cannot occupy the same
`.venv`.

**Upstream instructions:**
<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html>

## The torch build has no kernels for your card

This is the failure worth understanding, because everything reports healthy
right up until training starts.

torch's GPU kernels are compiled ahead of time, once per target in
`torch.cuda.get_arch_list()`. A device missing from that list has no elementwise
or convolution kernels, so every launch raises `invalid device function` — while
`torch.cuda.is_available()` returns True *and a matmul still succeeds*, because
GEMM comes from rocBLAS, a separate library with its own target list.

`./run doctor` asks this question independently of the matmul and reports it as
`torch gfx kernels`. If it fails, a different ROCm build is the first thing to
try — see below.

## Overrides for boards where they do work

`HSA_OVERRIDE_GFX_VERSION` genuinely rescues several unsupported targets, and the
probe applies it automatically for those: gfx1011 and gfx1012 present as gfx1010
(`10.1.0`), and gfx1031 through gfx1036 present as gfx1030 (`10.3.0`). When the
retry succeeds it is persisted to `.state/env.json` (applied on every
invocation, on both platforms) and recorded in the profile's `runtime.env`.

An override is only ever applied for a target known to be rescued by it. It is
never guessed for a merely-unlisted device: an override makes the card claim an
ISA it does not have, and where that is wrong it turns a legible "no kernels for
this architecture" into a fault deep inside the runtime.

If no override helps, the CPU path stays fully functional and the next thing to
try is a different ROCm build. This applies on Linux; the native-Windows build
is a single published set, so there is nothing to swap to there. ROCm 7.x has
been progressively dropping gfx10 targets, which is exactly why the Linux
default is 6.4 rather than the newest:

```bash
./run setup --force-step torch --force-step constraint --torch-index https://download.pytorch.org/whl/rocm6.3 --torch-spec torch==2.6.0
```

Then `./run doctor` again. Each attempt takes about ten minutes. `pins.toml`
lists the alternatives under `[torch.rocm]`.

## Being in the right groups

Linux only. The single most common cause of "ROCm sees no GPU" on a fresh Arch
install is not being able to open `/dev/kfd`:

```bash
sudo usermod -aG render,video "$USER"
```

Then log out and back in — group changes do not apply to an existing session.
`./run doctor` checks this and says so explicitly.

## ROCm environment knobs

These are exposed through the profile's `runtime.env` rather than hardcoded, so
they end up in the run log alongside everything else:

| Variable | Why |
| --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | Present an unsupported gfx target as a supported one. Set automatically when needed. |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | Biggest single anti-fragmentation win on a shared-memory APU. Defaulted on for ROCm. |
| `HIP_VISIBLE_DEVICES` | Restrict which GPUs are visible. |
| `GPU_MAX_HW_QUEUES=1` | Reduce queue overhead; sometimes helps stability on small boards. |
| `HSA_ENABLE_SDMA=0` | Troubleshooting DMA issues on some APUs. |
| `AMD_SERIALIZE_KERNEL=3` | Debugging only — serialises kernels so a fault points at the right one. Very slow. |

## Memory on an APU

An integrated GPU or APU has no discrete VRAM: it and the rest of the system
share the same physical memory, and so does your desktop compositor. Two
consequences:

- `data.num_workers` costs real memory. piper hardcodes
  `persistent_workers=True, prefetch_factor=4` whenever it is above 0, so each
  worker holds a prefetch queue. The small-GPU preset uses 2; `0` is the smallest
  footprint.
- Closing other GPU users genuinely helps, which is not true of a discrete card
  with its own VRAM.

`./run train` reads the device's reported memory and suggests a batch size, then
says out loud that it is a table lookup rather than a measurement. If it runs out
of memory, the failure prints the whole ladder — see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## CPU-only

Perfectly supported, and the right choice for verifying the pipeline:

```bash
./run setup --vendor cpu
./run smoke
```

Training a real voice on CPU is not practical, but proving that every stage works
takes a few minutes.
