# Quickstart

Six commands, no theory. Everything here assumes Linux and one audio file.

## 1. Install

```bash
./setup
```

Detects your GPU, installs the matching torch wheel, clones and builds
`piper1-gpl` at a pinned tag, installs Whisper, and verifies the lot with a real
GPU matmul. It offers to install system packages first and shows you the exact
command before running it.

Takes 10–25 minutes, mostly building espeak-ng from source.

## 2. Check

```bash
./run doctor
```

Reports the first thing that would stop you, with the command to fix it. Exits
non-zero if something is actually broken, so you can gate a script on it.

## 3. Build a dataset

```bash
./run dataset
```

Point it at your recording. It walks you through every setting, then splits,
transcribes and writes:

```
data/<voice>/
  source/          your recording, never modified
  wavs/            000001.wav, 000002.wav, ...
  metadata.csv     000001.wav|The quick brown fox jumps over the lazy dog.
  report.md        clip statistics and anything that looks wrong
  rejected.csv     what was thrown away, and why
```

Read the report. Transcription is the weakest link in any automated pipeline, and
the wizard shows you the clips furthest from the norm — that two-minute skim is
the highest-value check available. You can edit `metadata.csv` directly from the
wizard.

## 4. Get a starting point

```bash
./run checkpoints
```

Browse `rhasspy/piper-checkpoints` and download one. Fine-tuning from a
pretrained checkpoint is the difference between hours and weeks, and it works
even across languages. Pick a **medium** quality one — that is the architecture
this repo defaults to.

Files are around 900 MB. Downloads resume.

## 5. Train

```bash
./run train
```

Walks the settings, suggests a batch size from your GPU's memory, runs every
preflight check, shows a summary, then starts. Ctrl-C once to stop — Lightning
writes `last.ckpt` on the way out, and `./run resume` picks it up.

Watch it:

```bash
./run monitor
```

TensorBoard, including synthesized audio for the held-back utterances every
validation epoch. That audio is the signal to judge by; the loss curves saturate
long before the voice stops improving. If you are on another machine, the command
prints the `ssh -L` line you need.

There is no fixed finish line. Stop when it sounds right.

## 6. Export

```bash
./run export
```

Produces two files that must stay together:

```
voices/<voice>/en_US-myvoice-medium.onnx
voices/<voice>/en_US-myvoice-medium.onnx.json
```

Then loads them back and synthesizes a sentence to prove they work. Use the
voice with any Piper install — copy both files over, keeping the names identical.

## If something goes wrong

```bash
./run smoke
```

Runs the whole pipeline on CPU with synthetic audio in a few minutes. If that
passes, the installation is fine and the problem is in your data or settings.

Then see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
