# PiperTTSTraining

Train a [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice
from **one long audio recording**, on Linux or Windows, on an NVIDIA or AMD GPU
or on CPU.

You point it at a recording. It splits the audio at word boundaries, transcribes
it, writes a Piper-format dataset, reports what looks wrong, then configures and
runs training — with every setting available in a terminal wizard and saved to a
YAML profile you can edit and re-run.

The core is vendor-neutral: CUDA, ROCm and CPU are all first-class, and the GPU
is detected and *verified* during setup — not just reported, but proven with a
real matmul and a real autograd loop, because `torch.cuda.is_available()` lies
on unsupported ROCm targets.

AMD works on both platforms, by two different routes. On Linux, ROCm comes from
download.pytorch.org like any other wheel. On Windows, AMD publishes its own
ROCm build at fixed URLs on `repo.radeon.com`, which brings three requirements
setup checks for you: **Python 3.12 exactly** (the wheels are cp312-only), a
recent AMD graphics driver, and a GPU on
[AMD's supported list](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).
NVIDIA and CPU are the same story everywhere.

## Quickstart

```bash
git clone https://github.com/DJMalachite/PiperTTSTraining
cd PiperTTSTraining
./setup
```

On **Windows**, the same thing — `.\setup` in PowerShell, or `setup` in cmd.exe.
PowerShell resolves `./run` to `run.cmd`, so every `./run ...` command in this
documentation works there verbatim.

On **Linux**, if `./setup` reports "permission denied", the executable bit did
not survive the trip. Either fix it once:

```bash
chmod +x run setup scripts/*.sh
```

or just invoke it through the shell: `sh ./setup`.

Then:

```bash
./run doctor
```

```bash
./run
```

`./run` with no arguments opens a menu; every action is also a subcommand:

| Command | What it does |
| --- | --- |
| `./run setup` | Install torch for your GPU, clone and build piper1-gpl, install Whisper |
| `./run doctor` | Diagnose the environment; tells you the first thing to fix |
| `./run dataset` | One recording → `wavs/` + `metadata.csv` + a quality report |
| `./run checkpoints` | Browse and download a pretrained checkpoint to fine-tune from |
| `./run train` | Walk the training settings, run every preflight check, start |
| `./run resume` | Continue from the newest `last.ckpt` |
| `./run monitor` | TensorBoard (with audio samples), metrics, checkpoint housekeeping |
| `./run preview` | Synthesize a sentence from any checkpoint, no export needed |
| `./run export` | Produce `<lang>-<name>-<quality>.onnx` + `.onnx.json`, and verify it |
| `./run smoke` | Prove the whole pipeline on CPU with synthetic audio |

## What you need

- **Linux or Windows.** Training builds espeak-ng and two extension modules
  from source, so it needs a working C/C++ toolchain either way.
- **Python 3.11+**, `git`, `ffmpeg`, and a compiler. `./run setup` offers to
  install the system packages for `pacman`, `apt-get`, `dnf` or `zypper` on
  Linux, and via `winget` on Windows. The MSVC build tools are the one thing it
  will not install for you — it prints the exact command, because getting the
  workload selection wrong leaves a Visual Studio that looks installed and
  cannot compile.
- **About 20 GB of disk.** torch unpacks to ~4 GB, a pretrained checkpoint is
  ~0.9 GB, and a training run keeps around 10 GB of checkpoints by default.
- **A recording.** Anything ffmpeg can read, including video. 30 minutes of
  clean single-speaker speech is a realistic floor; one to two hours is
  comfortable. Your file is only ever read — never modified or deleted.

## How much audio, and how long?

Fine-tuning from a pretrained checkpoint is the difference between hours and
weeks, and it works even when the checkpoint is in a different language. `./run
checkpoints` downloads one for you. Training has no fixed end: piper's default
is to run until you stop it, because mel loss saturates long before the audio
stops improving. Listen to the samples in TensorBoard and stop when you are
happy.

## Profiles

Every setting lives in `profiles/<voice>.yaml`, generated with its own
documentation inline. A commented example ships with the repo:

- [`profiles/example-en_US-medium.yaml`](profiles/example-en_US-medium.yaml) —
  the defaults, annotated.
Copy it to `profiles/<yourvoice>.yaml`, or just run the wizard and let it write
the file.

## Documentation

| | |
| --- | --- |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | The six commands, no theory |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | NVIDIA / AMD / CPU, Linux and Windows, and how the GPU is verified |
| [docs/DATASET.md](docs/DATASET.md) | What good source audio is; every dataset setting |
| [docs/TRAINING.md](docs/TRAINING.md) | Profile → piper flag mapping; presets; fine-tuning |
| [docs/EXPORT.md](docs/EXPORT.md) | The two-file voice format and how to use it |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Out-of-memory ladder, build failures, cache questions |
| [docs/UPSTREAM_NOTES.md](docs/UPSTREAM_NOTES.md) | Verified piper1-gpl behaviour this repo depends on |

## What this is not

This is not a fork of Piper, and it does not reimplement any of it.
[`piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl) is cloned at a pinned
tag into `./piper1-gpl` during setup and never edited — workarounds for upstream
quirks live on this side of the boundary, documented in
[docs/UPSTREAM_NOTES.md](docs/UPSTREAM_NOTES.md). Bumping the pin in
[`pins.toml`](pins.toml) means re-running the checklist in that file.

It is also not a hosted or web interface. Everything is a terminal program with
line-based prompts and no curses, so it works over SSH, in tmux, and with piped
stdin on a headless machine.

## Related repositories

`Repos.md` lists the earlier iterations of this work.
[PiperTTS-Dataset-Creator](https://github.com/DJMalachite/PiperTTS-Dataset-Creator)
was the predecessor to `./run dataset`; this repo supersedes it and fixes several
of its behaviours, most importantly that it deleted your original recording after
splitting it. [DJMalachite/piper](https://github.com/DJMalachite/piper) was a
fork of the legacy `rhasspy/piper` adding ROCm support; it is no longer needed,
because ROCm support here is a matter of installing the right torch wheel.

## Licence

GPL-3.0, matching Piper.
