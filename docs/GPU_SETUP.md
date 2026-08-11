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

A matmul exercises rocBLAS and nothing else. The failure that actually stops a
run is a missing elementwise or convolution kernel — `invalid device function` —
or a fault after tens of allocate/free cycles. Neither is reachable by a matmul.

So `./run setup` and `./run doctor` also run a **real autograd loop**: a small
model built from the layer types a VITS vocoder uses (`Conv1d`,
`ConvTranspose1d`, LeakyReLU, elementwise arithmetic), 60 optimizer steps,
allocating fresh tensors every iteration. Ten seconds now beats an hour into a
run.

## The BC-250 (gfx1013) — see docs/BC250.md

The BC-250 needs enough special handling that it has [its own
document](BC250.md). The short version:

- **`HSA_OVERRIDE_GFX_VERSION` is a dead end on this board**, despite being the
  most common advice. gfx1010 and gfx1013 share an ISA, but the memory-aperture
  layout differs, so anything touching scratch or private addressing raises
  `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. The tool refuses to set it.
- Compute needs kernel 7.1.5+, a patched amdgpu module with
  `amdgpu.bc250_cc_write_mode=3` and `amdgpu.bc250_flush_by_runlist=1`,
  `HSA_ENABLE_SDMA=0`, and `amdgpu.sched_policy` left at its default.
- **Full training is blocked with a stock wheel**, because torch's kernels are
  compiled ahead of time per gfx target and no published wheel lists gfx1013.
  Dataset preparation, export and CPU training all work regardless.
- `scripts/bc250/build.sh` builds a torch that does carry gfx1013 kernels, in
  six resumable stages. Whether that is enough is unproven; the autograd probe
  in `./run doctor` is the verdict.

Set `runtime.hardware: bc250` (or leave it at `auto`) to get the environment,
the conservative settings, and the extra checks.

## Overrides for boards where they do work

`HSA_OVERRIDE_GFX_VERSION` genuinely rescues several unsupported targets, and the
probe applies it automatically for those: gfx1011 and gfx1012 present as gfx1010
(`10.1.0`), and gfx1031 through gfx1036 present as gfx1030 (`10.3.0`). When the
retry succeeds it is persisted to `.state/env.sh` (sourced by `./run`) and
recorded in the profile's `runtime.env`.

If no override helps, the CPU path stays fully functional and the next thing to
try is a different ROCm build. ROCm 7.x has been progressively dropping gfx10
targets, which is exactly why the default is 6.4 rather than the newest:

```bash
./run setup --force-step torch --force-step constraint --torch-index https://download.pytorch.org/whl/rocm6.3 --torch-spec torch==2.6.0
```

Then `./run doctor` again. Each attempt takes about ten minutes. `pins.toml`
lists the alternatives under `[torch.rocm]`.

## Being in the right groups

The single most common cause of "ROCm sees no GPU" on a fresh Arch install is not
being able to open `/dev/kfd`:

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

A board like the BC-250 has no discrete VRAM: the GPU and the rest of the system
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
