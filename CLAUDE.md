# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A terminal tool that takes **one long audio recording** to a trained, exported
[Piper](https://github.com/OHF-Voice/piper1-gpl) TTS voice. It wraps upstream
piper1-gpl; it does not reimplement any of it.

The core is **vendor-neutral and platform-neutral**: CUDA, ROCm and CPU are all
first-class, and it runs on Linux and Windows. Do not reintroduce AMD-only or
POSIX-only assumptions into it. There is no per-board special-casing any more —
that machinery (`hardware.py`, `runtime.hardware`, `scripts/bc250/`) existed for
one board and has been removed. Do not reintroduce it for a single device;
`runtime.env` in the profile is the escape hatch.

**Platform facts that constrain the code:**

- **ROCm has two deliveries, not one.** Linux gets an index
  (`download.pytorch.org`); native Windows gets fixed wheel URLs from AMD's
  `repo.radeon.com`, under `[torch.rocm.windows]` in `pins.toml`. That path is
  **cp312-only**, needs the ROCm runtime wheels installed *before* torch, and
  has a shorter supported-GPU list. `TorchPin.from_urls` is the discriminator;
  `env.rocm_windows_note()` is the single place the requirements are worded.
- Because of that, "newest supported interpreter" is the wrong rule when a pin
  carries `requires_python` — see `install._pick_exact_python`. Getting this
  wrong produces a bare "no matching distribution" at the torch step, which
  names neither the cause nor the fix.
- Venv executables live in `Scripts/*.exe` on Windows, `bin/*` elsewhere.
  `paths.venv_bin()` is the only place that spells this; never hardcode either.
- **`*.cmd` files must be CRLF; `run`, `setup` and `*.sh` must be LF.** Both
  failures are opaque: cmd.exe eats the first character of every LF-terminated
  line, and `sh` reports `bad interpreter: /bin/sh^M`. `.gitattributes` pins
  both and `tests/test_platform.py` checks them.
- PowerShell resolves `./run` to `run.cmd`, ignoring the extensionless POSIX
  file, so `./run ...` in docs and help strings is correct on both platforms.
  cmd.exe users type `run`.
- In batch, `shift` also shifts `%0`, so capture `%~dp0` before parsing args.

## Commands

```bash
./run                  # interactive menu
./run setup            # install torch for the detected GPU, clone+build piper1-gpl, install whisper
./run doctor           # diagnostics; non-zero exit when something is broken
./run dataset          # one recording -> wavs/ + metadata.csv + report.md
./run checkpoints      # browse/download pretrained checkpoints from HuggingFace
./run train            # walk settings, preflight, launch `piper.train fit`
./run resume           # continue from the newest last.ckpt
./run monitor          # tensorboard / metric table / prune checkpoints
./run preview          # synthesize from a checkpoint via infer_torch
./run export           # ONNX + voice config, then verify by synthesizing
./run smoke            # end-to-end CPU self-test on synthetic audio
```

On Windows the same commands run through `run.cmd`; PowerShell resolves `./run`
to it, and cmd.exe users drop the `./`.

`PIPERTRAINER_ENV=<name>` selects `.venv-<name>` and `.state-<name>` instead of
`.venv`/`.state`, so one clone can hold more than one installed environment —
the case that matters is a checkout on a drive shared between Windows and WSL2.
Unset — the normal case — keeps the historic layout exactly. The name is
*refused* rather than sanitised, by `run` in shell and `paths.env_name_problem()`
for `run.cmd`, so both always agree on the directory.

Tests (stdlib `unittest`, no venv, no GPU, no audio, <1s):

```bash
python -m unittest discover -s tests -t .
```

One module: `python -m unittest tests.test_argmap -v`

Acceptance gate: `scripts/smoke_test.sh` (or `./run smoke`). It runs the unit
tests, builds a synthetic dataset, trains one CPU epoch, exports, and runs four
negative tests.

**If you change a command, update this file.**

## Architecture

Package `pipertrainer` under `src/`. `./run` is a POSIX shim that prefers
`.venv/bin/python` and falls back to a system interpreter so a bare clone can
bootstrap.

| Module | Role |
| --- | --- |
| `profile.py` | **The schema.** One dataclass tree; every field carries a `Spec` (help/choices/bounds). The wizard renders prompts from it and the YAML writer emits the help as comments, so they cannot drift. Adding a setting means adding one field. |
| `train/argmap.py` | **Highest-risk module.** Pure function: profile → `lightning.yaml` + equivalent argv. Owns the link-argument table, the forbidden/blocked key lists, and every pre-launch invariant. No I/O, heavily tested. |
| `train/presets.py` | `medium`/`high`/`low` as explicit `model.*` dicts. Quality is *our* abstraction — upstream has no `--quality` flag. |
| `env.py` | Vendor detection and torch verification: a real matmul *and* a real autograd loop (`probe_training`), plus the `HSA_OVERRIDE_GFX_VERSION` retry for targets it actually helps. Also owns `.state/env.json`, the persisted environment both entry points share. |
| `paths.py` | Filesystem layout **and** the platform split (`WINDOWS`, `VENV_BINDIR`, `EXE_SUFFIX`, `venv_bin`). Nothing else may spell `bin` or `.exe`. |
| `install.py` | Ordered, idempotent, resumable setup state machine. Order is load-bearing. |
| `train/launch.py` | Preflight (dataset facts, checkpoint architecture, espeak, cache) → `--print_config` gate → run, with a reproducibility snapshot per run. |
| `dataset/segment.py` | Groups Whisper *words* into utterances honouring min/target/max. The core dataset-quality logic. |
| `dataset/pipeline.py` | probe → decode → macrosplit → transcribe → segment → emit → report, with per-stage caching. |
| `tui.py` | Line-based `input()` prompts. No curses — must work over SSH, in tmux, and with piped stdin. |
| `run` / `run.cmd` | The two entry points. Both stay dumb: find an interpreter, set `PYTHONPATH`, hand off. Anything they both need belongs in Python rather than duplicated in two shell dialects — that is why `.state/env.json` replaced the old sourced `env.sh`. |

`pins.toml` is the single source of truth for the piper1-gpl tag+sha, the torch
index/version per vendor, the Whisper pin, and the interpreter search order
(`prefer` / `prefer_windows`, as argv lists because `py -3.13` is two words).

## Non-negotiables

- **Never write to or delete the user's source audio.** `data/<voice>/source/` is
  read-only. The predecessor script (`PiperTTS-Dataset-Creator`) deleted the
  original recording after splitting it; there is a regression test asserting we
  do not.
- **Never edit `piper1-gpl/`.** It is a pinned clone, gitignored. Workarounds for
  upstream bugs live on our side, documented in `docs/UPSTREAM_NOTES.md`.
- **Never `sudo` without confirming**, and always print the command first. The
  same holds for `winget`; the MSVC build tools are never installed
  automatically, only printed, because a wrong workload selection leaves a
  Visual Studio that looks installed and cannot compile.
- **Nothing is dropped silently.** Every rejected clip gets a reason in
  `rejected.csv`; every refusal explains what to do instead.
- **TUI stays stdlib-only.** PyYAML may be used *inside the venv* (it is
  transitive via lightning/jsonargparse) but `yamlio.py` carries a fallback
  parser for the pre-venv bootstrap path, and that fallback is tested.

## Upstream facts that constrain the code

Read `docs/UPSTREAM_NOTES.md` before touching `train/argmap.py`. The short version
— all verified against piper1-gpl v1.6.0, all counterintuitive:

- **Eight `link_arguments` pairs** (`train/__main__.py:50-57`). Setting a link
  *target* is a hard error, so `sample_rate` goes on `model` and `batch_size` on
  `data`, never the reverse. `argmap.FORBIDDEN` encodes this; do not "fix" it.
- **`automatic_optimization = False`**, so `trainer.gradient_clip_val` is
  rejected by Lightning, and **`model.grad_clip` is dead code** — it is accepted
  but never applied. Both are blocked. `accumulate_grad_batches` is likely inert
  too.
- **`torch.cuda.is_available()` lies on ROCm** for unsupported architectures; the
  process aborts at the first kernel. Always verify with a real matmul plus
  `synchronize()`, and treat "the probe printed nothing" as failure. A matmul is
  still not proof of *training*: missing elementwise/conv kernels and
  allocation-churn faults only show up under autograd, which is what
  `env.probe_training` exists for.
- **`openai-whisper` will replace a ROCm torch** — its metadata wants an unpinned
  torch plus CUDA-flavoured triton. Install `--no-deps` under `PIP_CONSTRAINT`,
  and install the vendor torch wheel *before* `piper1-gpl[train]`.
- **`build_monotonic_align.sh` is deliberately not called.** It is four commands
  wrapped in bash, and it activates `piper1-gpl/.venv` if that happens to exist,
  which would build against the wrong interpreter. `install.step_monotonic_align`
  does those four steps directly, invoking `python -m Cython.Build.Cythonize`
  rather than the `cythonize` console script, whose location is platform-specific.
- **`batch_size` > training split gives zero batches** (`drop_last=True`), with an
  error that names something else. Preflight refuses it.
- **Cache ids embed the row number and transcript text**
  (`vits/utils.py:59`), so `metadata.csv` must be written in stable sorted order
  and edits are fingerprinted.
- **Upstream's missing-cache-file guard never fires** (`vits/dataset.py:442`
  tests a `Path` for truthiness), so we check cache completeness ourselves.
- **The voice config JSON is written by training, not export**, and its
  `hop_length` and `piper_version` are wrong. `train/export.py` corrects them,
  keeps the original, and prints a diff.
- **`infer_torch` reads JSON lines from stdin**, not plain text.
- **Clips under `segment_size` are zero-padded** with silence the model learns;
  hence the 1.0 s floor on `dataset.min_seconds`.

## Development environment

Linux and Windows are both targets, and the maintainer's machine is Windows.
Consequences:

- Write portable Python: `pathlib`, `os.pathsep`, `sys.platform` checks. Where a
  shell script is unavoidable, write both halves (`*.sh` **and** `*.cmd`) and
  respect the line-ending rule above.
- The pure modules (`profile`, `yamlio`, `argmap`, `presets`, `segment`,
  `textnorm`, `metadata`, `report`) are testable anywhere — that is deliberate,
  and it is where new logic should live. `tests/test_platform.py` covers the
  layout and entry-point facts without needing the other platform present.
- Anything touching torch, ffmpeg, or piper needs a real install to test. The
  CPU smoke test is the acceptance gate on either platform.
- `tui.py` degrades its Unicode glyphs to ASCII when stdout cannot encode them;
  keep new output going through `tui` rather than bare `print`.

## Licensing

GPL-3.0, matching Piper. Anything derived from or linked against it stays
GPL-3.0 compatible.
