# Troubleshooting

Start here:

```bash
./run doctor
```

It checks things in dependency order and reports the first thing that would stop
you, with the command to fix it. If `doctor` is clean but something still fails:

```bash
./run smoke
```

runs the whole pipeline on CPU with synthetic audio in a few minutes. If that
passes, the installation is fine and the problem is in your data or settings.

---

## Out of memory

The failure message prints this ladder automatically. Work down it — cheapest
first.

1. **`data.batch_size`**: halve it (8 → 4 → 2). Also lower
   `dataset.max_seconds` (14 → 10 → 8) and rebuild the dataset: long clips
   dominate peak memory, and transcription is cached so rebuilding is quick.
2. **`data.num_workers`**: 2 → 1 → 0. piper hardcodes
   `persistent_workers=True, prefetch_factor=4` whenever this is above 0, so each
   worker holds a prefetch queue — and on an APU that is the same physical memory
   as the GPU's.
3. **Close other GPU users.** On an APU the desktop compositor takes GTT memory.
   This genuinely helps there, unlike on a discrete card.
4. **`model.segment_size`**: 8192 → 4096. Must stay a multiple of `hop_length`.
   This changes what the discriminator sees, so it is a real trade rather than a
   free win.
5. **`trainer.precision: 16-mixed`.** Watch `loss_g` for divergence: piper trains
   a GAN with manual optimization, which is the least-tested combination for mixed
   precision. On RDNA2 there is no bf16 matrix path, so `bf16-mixed` buys nothing.
6. **`model.mos_metric: none`.** UTMOS runs a second model during validation.
7. **`trainer.check_val_every_n_epoch: 5`.** Validation synthesizes full
   utterances and is often the real peak, not training.
8. **`voice.quality: low`** (16 kHz), or `runtime.vendor: cpu` for a
   correctness-only run.

`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is set by default on ROCm and is
the biggest single anti-fragmentation win; do not remove it.

## "Zero batches", or an epoch that ends immediately

`batch_size` is larger than the training split. The dataloader drops the last
partial batch, so a split of 17 with a batch size of 20 gives no batches at all,
and Lightning's error names something else.

The preflight refuses this before launching and tells you the maximum. The dataset
report prints it too. `./run monitor` shows the split.

## Training starts, then the process dies with no Python traceback

On ROCm this is almost always an unsupported GPU architecture. `torch.cuda.is_available()`
returns True for architectures the build has no kernels for, and the runtime aborts
at the first real kernel launch — often `SIGABRT`, sometimes
`HSA_STATUS_ERROR_INVALID_ISA` or `Memory access fault`.

```bash
./run doctor
```

If the matmul check fails, see [GPU_SETUP.md](GPU_SETUP.md). On a **BC-250
(gfx1013)** neither of the usual moves helps: `HSA_OVERRIDE_GFX_VERSION` makes
things worse, and no stock ROCm wheel of any version ships gfx1013 kernels, so
re-downloading torch from another index costs gigabytes and fails identically.
See [BC250.md](BC250.md) instead. On any other card, try a different ROCm build:

```bash
./run setup --force-step torch --force-step constraint --torch-index https://download.pytorch.org/whl/rocm6.3 --torch-spec torch==2.6.0
```

## "invalid device function", or a fault after 20–40 iterations

The GPU passes the matmul check and then fails once training starts. This means
the torch build has no compiled kernels for your GPU architecture — matmul goes
through rocBLAS, but the elementwise and convolution kernels autograd needs are
missing.

`./run doctor` reproduces this deliberately with a 60-step autograd loop, so you
can confirm it in ten seconds rather than an hour.

On a BC-250 this is the documented state of the board and is reported as a
warning rather than an error; see [BC250.md](BC250.md) for the options
(prepare the dataset there and train elsewhere, or train on CPU). On any other
GPU, try a different ROCm index — see [GPU_SETUP.md](GPU_SETUP.md).

## The GPU worked, and now it does not

Something replaced the torch wheel. The usual culprit is a `pip install` that
resolved `torch>=2,<3` against PyPI and got the CUDA build.

```bash
./run doctor
```

Look at the `torch` line: `torch.version.hip` must be non-`None` on AMD. To
restore:

```bash
./run setup --force-step torch --force-step constraint
```

Setup guards against this by pinning the exact local version in
`.state/torch-constraint.txt` and passing it as `PIP_CONSTRAINT` to every later
pip call, so if you install packages by hand, do the same.

## The espeak-ng build fails

`piper1-gpl` builds espeak-ng from source as a CMake `ExternalProject`, which
needs three things:

- **CMake 3.26 or newer.** Remove an older system cmake from `PATH` and pip will
  fetch a wheel.
- **A C/C++ toolchain.** `base-devel` on Arch, `build-essential` on Debian.
- **Network access at build time.** pip's build runs in isolation and does not
  inherit every proxy variable.

The installer maps these to explicit messages. The full build log is in
`.state/setup.log`.

## `ModuleNotFoundError: No module named 'skbuild'`

From the `build_ext` step. `skbuild` is the **import** name of the PyPI package
`scikit-build`, so no distro package provides it, and a venv could not see a
system package if one did.

```bash
./run setup --force-step build_ext
```

That step installs `[build].requires` from `pins.toml` into `.venv` before
running upstream's `setup.py`. The gap it closes: `piper_install` builds through
pip, which creates an isolated environment from upstream's
`build-system.requires` and then discards it, while `build_ext` runs
`setup.py build_ext --inplace` directly with the venv interpreter — no isolated
environment, no `skbuild`.

If you install it by hand, pass the constraint like every other pip call:

```bash
PIP_CONSTRAINT=.state/torch-constraint.txt .venv/bin/python -m pip install scikit-build cmake ninja
```

## "monotonic_align" is missing

The VITS alignment kernel is a Cython extension built by a separate script, and a
successful script exit is not proof it is importable.

```bash
./run setup --force-step monotonic_align
```

The script activates `piper1-gpl/.venv` if that exists — ours is at the repo root,
so it is invoked with our venv's `bin` prepended to `PATH`.

## espeak produces no phonemes

```
espeak-ng cannot phonemize with voice 'en_US'
```

`voice.espeak_voice` needs an espeak-ng **voice name** (`en-us`, `en-gb`, `de`,
`fr`), not a locale tag. `voice.language` is the locale tag and is used only for
the exported filename. There is no `espeak-ng` binary to test with — piper embeds
the library — so `./run doctor` validates by phonemizing a sentence through it.

## Whisper is very slow, or uses the CPU

Check `whisper.device`. `auto` runs a real GPU operation before deciding, so if it
picked CPU, torch cannot use your GPU — see `./run doctor`.

On ROCm the device is still spelled `cuda`; that is what the ROCm build of torch
calls itself. Setting `whisper.device: cuda` on an AMD box is correct.

`turbo` is a good speed/accuracy trade. `large-v3` is several times slower.

## Transcripts are wrong

The most common quality problem, and the one worth spending time on.

- Read `data/<voice>/report.md`. The "clips worth listening to" table ranks by
  characters-per-second, which is what catches misalignment.
- Check `rejected.csv` — if a lot was rejected for the rate bounds, the
  segmentation is probably fighting the audio.
- Set `whisper.initial_prompt` with names and jargon it keeps getting wrong.
- Try `whisper.model: large-v3`.
- Fix individual lines by editing `metadata.csv` directly; `./run dataset` offers
  to open it in `$EDITOR`.

Do not reorder or renumber rows — piper's cache ids embed the row number.

## "The utterance cache was built with different settings"

Expected after editing `metadata.csv` or changing anything that affects the cached
tensors. piper's cache ids embed the row number and the transcript text, so
changing one line orphans the entries after it.

- **keep** — piper reuses what matches and recomputes the rest. Safe; stale
  entries just waste disk.
- **wipe** — delete and preprocess from scratch. Slower but tidy.
- **abort** — change nothing.

## "the cache is partial"

An interrupted `prepare_data`. piper will fill the gaps on this run.

Worth knowing why we check: upstream's own guard for missing cache files is
`if not <Path>:`, and a `Path` object is always truthy, so it never fires — a
partial cache passes validation and then fails inside the dataloader. Our
completeness check exists to give you the real message instead.

## Disk filling up

A run keeps the top five checkpoints by `val_mel`, the top five by `val_mos`, and
`last.ckpt` — roughly 10 GB at ~0.9 GB each.

```bash
./run monitor --prune
```

protects the best of each metric plus `last.ckpt` and asks before deleting. To
keep fewer from the start, lower `trainer.checkpoint_save_top_k`.

The decoded-audio cache under `data/<voice>/.cache/` is about 317 MB per hour of
source audio and is safe to delete; it will be regenerated.

## `val_mos` never appears

`model.mos_metric` is `none`, or UTMOS could not be fetched from `torch.hub`.
Neither affects training: upstream wraps the download in `try/except` and disables
the metric gracefully. Checkpoint selection by `val_mel` and `last.ckpt` are
unaffected.

## Ctrl-C — did I lose the run?

No. Ctrl-C is delivered to piper and we wait for it to exit, so Lightning writes
`last.ckpt` on the way out.

```bash
./run resume
```

It refuses if the profile no longer matches the configuration the run started with,
shows the diff, and `--force` overrides.

## Setup stopped partway

Re-run it. Completed steps are recorded in `.state/setup.json` and skipped:

```bash
./run setup
```

To redo a specific step: `./run setup --force-step <name>` (`--force-step all`
for everything). Step names are listed in the failure message.

## Python version complaints

The tool needs 3.11+ (it reads `pins.toml` with `tomllib`). It prefers 3.13, 3.12
or 3.11 for the venv and warns on 3.14+, because `numba` — which `openai-whisper`
needs — tends to lag new CPython releases. On rolling distributions like CachyOS
this is a real risk; install an older interpreter alongside if setup warns.

## After bumping the piper1-gpl pin

Run the checklist in [UPSTREAM_NOTES.md](UPSTREAM_NOTES.md). The flag mapping in
`train/argmap.py` is verified against a specific tag, and the unit tests pin the
exact configuration emitted — so if the contract changed, `python -m unittest
discover -s tests -t .` tells you before a run does.
