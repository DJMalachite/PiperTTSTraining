# Verified piper1-gpl behaviour

Everything this repo does to piper's argument surface is based on reading
`piper1-gpl` at the tag pinned in [`pins.toml`](../pins.toml) — currently
**v1.6.0** (`f04d52c5528ac7cf2d73757f57990ff490f75005`). Each entry below cites
`file:line` in that checkout and says what we do about it.

Several of these are behaviours you would otherwise discover hours into a
training run. That is why they are written down, and why
[`tests/test_argmap.py`](../tests/test_argmap.py) pins the emitted configuration:
if a pin bump changes the contract, the tests fail rather than the run.

> **Bumping the pin requires re-running the checklist at the end of this file.**

---

## 1. There is no separate preprocessing step

`src/piper/train/preprocess.py` **does not exist**. In the legacy
`rhasspy/piper` tree there was a `piper_train.preprocess` CLI; in piper1-gpl,
preprocessing is folded into `VitsDataModule.prepare_data()` and driven by
`--data.cache_dir`. Cached artefacts are skipped if present, so the cache is
incremental and resumable.

**We do:** one `fit` invocation. No preprocess menu item, and the cache directory
is treated as a first-class resumable artefact.

## 2. Eight linked arguments — setting the target is an error

`src/piper/train/__main__.py:50-57`:

```python
parser.link_arguments("data.batch_size", "model.batch_size")
parser.link_arguments("data.num_symbols", "model.num_symbols")
parser.link_arguments("model.num_speakers", "data.num_speakers")
parser.link_arguments("model.sample_rate", "data.sample_rate")
parser.link_arguments("model.filter_length", "data.filter_length")
parser.link_arguments("model.hop_length", "data.hop_length")
parser.link_arguments("model.win_length", "data.win_length")
parser.link_arguments("model.segment_size", "data.segment_size")
```

jsonargparse computes each target from its source. Setting a target yourself is a
hard error.

**We do:** `argmap.LINKS` and `argmap.FORBIDDEN` encode this. `sample_rate` etc.
are emitted on `model`, `batch_size` and `num_symbols` on `data`, and the profile
schema deliberately places `data.batch_size` under `data` so the two cannot drift.
A test asserts no link target is ever emitted, and `model.extra` plus any
user-supplied argv are checked against the same list.

## 3. Config files and `--print_config` both work

`LightningCLI` accepts `--config file.yaml`, and `--print_config` validates
without training.

**We do:** emit `runs/<voice>/lightning.yaml` and launch with `--config`, then
run `--print_config` first as a free type check. The equivalent flag-by-flag
command is written to the run log for reproducibility.

## 4. Manual optimization: gradient clipping cannot be used

`src/piper/train/vits/lightning.py:128` sets `self.automatic_optimization =
False`, so Lightning rejects `trainer.gradient_clip_val`.

Separately, **`model.grad_clip` is dead code**. It appears only in
`VitsModel.__init__` (`vits/lightning.py:83`). `clip_grad_value_` is defined at
`vits/commons.py:132` and never called; `training_step` does
`zero_grad → manual_backward → step` with no clipping.

**We do:** both are in `argmap.BLOCKED` with these reasons. `grad_clip` is hidden
from the wizard. `accumulate_grad_batches` is exposed but carries a warning that
Lightning does not apply accumulation under manual optimization.

The `gradient_clip_val` claim is the one item here originally inferred from
Lightning's semantics rather than executed, so the self-test
(`./run smoke --stage train`) runs a real `fit` with the flag and asserts a
non-zero exit. If it ever passes, the test says so and tells you to relax the
block.

## 5. Training runs forever by default, and keeps ten checkpoints

`src/piper/train/__main__.py` passes `trainer_defaults={"max_epochs": -1,
"callbacks": _DEFAULT_CALLBACKS}`, where `_DEFAULT_CALLBACKS`
(`__main__.py:24-45`) is exactly two `ModelCheckpoint`s: top-5 by `val_mel`
(min, `save_last=True`) and top-5 by `val_mos` (max, `save_last=False`).

Upstream's own comment explains why there is no early stopping: mel L1 saturates
early in VITS while the adversarial losses keep removing audible artifacts, so
an early stop on `val_mel` fires well before the audio is clean.

**`trainer.callbacks` does not replace `trainer_defaults` — it concatenates.**
This note previously claimed the opposite, and the claim was wrong.
`LightningCLI._instantiate_trainer` does:

```python
config[key].extend(callbacks)
if key in self.trainer_defaults:
    config[key] += self.trainer_defaults[key]
```

So naming a `ModelCheckpoint` in the config *adds a third one*, and naming one
whose init args match an upstream default is a hard failure, because two
stateful callbacks of the same type may not share a `state_key`:

```
RuntimeError: Found more than one stateful callback of type `ModelCheckpoint`
```

There is therefore no configuration that removes or retunes upstream's
checkpoints. This also means the old implementation of
`trainer.checkpoint_save_top_k` never worked: it produced four checkpoint
callbacks, two at the requested `save_top_k` and two still at upstream's five.

**We do:** surface `max_epochs: -1` as a note rather than a surprise, and reach
the live objects instead of trying to replace them.
`train/callbacks.py::CheckpointPolicy` is listed in `trainer.callbacks`, is
constructed alongside upstream's, and adjusts them in `setup` — long before the
first epoch ends. `./run monitor --prune` exists because ten checkpoints at
~0.9 GB is about 10 GB per run.

## 6. UTMOS failure is non-fatal, but *disabling* UTMOS is fatal

Two halves that have to be read together.

`src/piper/train/vits/mos.py:39-52` wraps
`torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` in
`try/except Exception` and sets `_disabled = True` on failure. It is loaded
lazily at first validation, not at startup. So a failed *download* really is
harmless.

Turning the metric off is not. `VitsModel.__init__` takes
`mos_metric: Optional[str] = "utmos"` and builds a predictor only when the value
is neither `None` nor `"none"`; `on_validation_epoch_end` logs `val_mos` only
when a predictor exists. Meanwhile `_DEFAULT_CALLBACKS` monitors `val_mos`
unconditionally (note 5). `ModelCheckpoint` raises
`MisconfigurationException` the first time its monitored key is missing from the
logged metrics — at the end of epoch 1, after real training work:

```
`ModelCheckpoint(monitor='val_mos')` could not find the monitored key in the
returned metrics: ['loss_g', ..., 'val_disc', 'epoch', 'step']
```

The two upstream defaults are consistent with each other, so this never fires
for a default online run whose hub fetch succeeds. It fires for anyone who sets
`mos_metric: none` — which includes every offline run — **and equally for an
online run whose fetch fails**, because
`MosPredictor.score()` returns `None` once `_disabled` is set, leaving
`mos_scores` empty and `val_mos` unlogged. The graceful degradation in `mos.py`
protects the metric, not the run.

**We do:** `argmap` treats the two as one decision. When `model.mos_metric`
resolves to `none`, the emitted `CheckpointPolicy` carries
`disable_monitors: [val_mos]`, which sets that checkpoint's `save_top_k` to 0.
Zero is the value that matters: `_save_topk_checkpoint` returns on it *before*
testing whether the monitored key is present. `val_mel` is untouched and keeps
`save_last=True`, so `last.ckpt` still exists for `./run resume` and export.
Regression tests in `tests/test_argmap.py::CheckpointCallbackTest` and
`::CheckpointPolicyTest`.

Not covered: an online run that loses the network at first validation still
dies at the end of epoch 1. Detecting that in advance would mean fetching UTMOS
during setup, which we do not do.

## 7. Cache ids embed the row number and the text

`src/piper/train/vits/utils.py:59`:

```python
cache_id = str(row_number) + speaker_id_str + "_" + sanitize_filename(text)
return cache_id[:max_length]   # max_length=50
```

So reordering rows, inserting a row, or editing one transcript orphans the cached
tensors from that point on.

**We do:** `metadata.csv` is always written in a stable sorted order with
zero-padded ids. `argmap.cache_fingerprint_inputs` hashes the metadata bytes
together with every setting that affects the cached tensors; a mismatch prompts
*keep / wipe / abort* with an explanation instead of silently re-preprocessing.
The `./run dataset` transcript editor warns against reordering.

## 8. Upstream bug: the missing-artefact guard never fires

`src/piper/train/vits/dataset.py:442`, `:451`, `:460`:

```python
phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
if not phoneme_ids_path:          # a Path object is always truthy
    _LOGGER.warning(...)
    continue
```

The intent was `.exists()`. As written, an interrupted `prepare_data` leaves a
partial cache that passes `setup()` and then fails inside the dataloader.

**We do:** count `*.phonemes.pt`, `*.audio.pt` and `*.spec.pt` against the usable
row count before launching, and report a partial cache as expected-after-an-
interruption rather than letting it surface as an opaque crash.

## 9. The split arithmetic, and the zero-batches trap

`src/piper/train/vits/dataset.py` `setup()`:

```python
valid_set_size = int(n * self.validation_split)
num_test = min(self.num_test_examples, max(0, n - valid_set_size - 1))
train_set_size = n - valid_set_size - num_test
```

`train_dataloader` uses `drop_last=True`. So a `batch_size` larger than the
training split yields **zero batches per epoch**, and Lightning's error says
nothing about the cause. Note the clamp on `num_test` means `train >= 1` for any
`validation_split` below 1.0.

**We do:** `argmap.split_sizes` mirrors this exactly, the dataset report prints
the maximum usable batch size, and `argmap.check_dataset_math` refuses to launch
with the number you should use instead.

## 10. Clips shorter than `segment_size` are padded with silence

`src/piper/train/vits/dataset.py:658`, in `UtteranceCollate`:

```python
max_audio_length = max(max_audio_length, self.segment_size)
```

At the default `segment_size=8192` and 22.05 kHz that is 0.372 s. Anything
shorter is zero-padded, and the model learns to produce that silence.

**We do:** `dataset.min_seconds` has a hard floor of 1.0 s in the schema, and
`argmap.check_clip_length` warns whenever the shortest clip is below
`segment_size / sample_rate`.

## 11. There is no `--quality` flag

`x-low` / `low` / `medium` / `high` survive only as a filename convention and as
directory names in the checkpoint repository. `vits/config.py` still contains
`low_quality()` and `high_quality()`, but nothing references them — dead code
carried over from `rhasspy/piper`. `VitsModel.__init__`'s defaults are identical
to legacy `low_quality()`, which is what legacy Piper shipped as *medium*.

**We do:** quality is our abstraction. `train/presets.py` expands it into raw
`model.*` values, `medium` matches upstream's defaults byte for byte, and a test
asserts that.

## 12. Tuple arguments accept both YAML sequences and string literals

`vits/lightning.py:102-121` coerces `resblock_kernel_sizes`,
`resblock_dilation_sizes`, `upsample_rates`, `upsample_kernel_sizes` and `betas`
with `ast.literal_eval` — but each call is guarded by `isinstance(..., str)`. A
YAML sequence passes straight through.

`vits/lightning.py:123-125` then enforces the invariant:

```python
expected_hop_length = reduce(operator.mul, self.hparams.upsample_rates, 1)
if expected_hop_length != hop_length:
    raise ValueError("Upsample rates do not match hop length")
```

`resblock` is compared as a **string** (`models.py:319`:
`modules.ResBlock1 if resblock == "1" else modules.ResBlock2`), so an integer
silently selects ResBlock2.

**We do:** emit native YAML sequences, which removes the quoting hazard
entirely; keep `resblock` quoted; and pre-empt the `ValueError` with a message
naming the product, the hop length, and a working alternative. Every preset is
tested against the invariant.

## 13. The voice config JSON is written by *training*, and two fields are wrong

`vits/dataset.py:168` builds `PiperConfig(...)` **without passing
`hop_length`**, so `config.py`'s `DEFAULT_HOP_LENGTH = 256` is always what lands
in the file. `piper_version` is the literal `"1.5.0"` while the package version
is 1.6.0. The file goes to `--data.config_path` during training;
`piper.train.export_onnx` does not produce it (its only flags are
`--checkpoint`, `--output-file`, `--debug`).

**We do:** `train/export.py` copies rather than moves, corrects `hop_length` to
the value actually trained at and `piper_version` to 1.6.0, keeps the untouched
original as `.onnx.json.from-training`, and prints the diff. A sample-rate
disagreement is treated as a hard error — that means the config belongs to a
different run — not a fixup.

## 13b. `export_onnx` needs a package upstream does not declare

`piper.train.export_onnx` calls `torch.onnx.export`. As of torch 2.9 the dynamo
exporter is the default path, and `torch/onnx/__init__.py:282` imports
`torch.onnx._internal.exporter._compat` on the way in, which imports
`onnxscript`. Nothing in piper1-gpl's dependencies or its `[train]` extra
mentions `onnxscript`, and `onnxruntime` — which piper *does* depend on, for
inference — does not provide it.

The result is a `ModuleNotFoundError: No module named 'onnxscript'` raised at the
worst possible moment: training has finished, the checkpoint is on disk, and the
export is the only step left.

**We do:** pin it in `pins.toml` under `[export]` and install it in its own
`export_deps` step, ordered after `constraint` so `PIP_CONSTRAINT` is protecting
the vendor torch. onnxscript has no torch dependency of its own (onnx, onnx_ir,
ml_dtypes, numpy, packaging, typing_extensions), so it cannot pull a CUDA build
into a ROCm environment. `./run doctor` and the setup verification both import
`torch.onnx` *and* `onnxscript`, so the gap is reported before training rather
than after.

## 13c. `export_onnx` needs the *TorchScript* exporter, not dynamo

`export_onnx.py` calls `torch.onnx.export` with no `dynamo` argument — on
`v1.6.0` and on `main` — and passes `dynamic_axes`, which is legacy-exporter
API. It was written against the old default.

torch 2.9 flipped that default to `dynamo=True`, routing the call through
`torch.export.export`, which VITS does not survive:

```
File "vits/transforms.py", line 174, in rational_quadratic_spline
    assert (discriminant >= 0).all(), discriminant
GuardOnDataDependentSymNode: Could not guard on data-dependent expression
Eq(u2, 1)
```

`rational_quadratic_spline` is called on `inputs[inside_interval_mask]`, so its
leading dimension is an *unbacked* symint — a size that depends on tensor
values, not shapes. `torch.export` must resolve the assert's `.all()` to a
concrete bool at trace time and cannot. It is not a checkpoint problem and not
a CPU/GPU problem: every Piper voice to date was exported through TorchScript,
where the assert simply evaluates.

**We do:** run the export through `train/export_shim.py`, which wraps
`torch.onnx.export` to force `dynamo=False` and then calls upstream's `main()`
with argv untouched. The wrapper is skipped on a torch too old to accept the
argument. `train/export.py::export_command` is the single place the entrypoint
is named, and `tests/test_export_shim.py` asserts we never point it back at
`piper.train.export_onnx`.

Delete the shim when upstream passes `dynamo` itself, or when VITS stops
tripping `torch.export`.

## 14. TensorBoard already logs listenable audio

`vits/lightning.py:394` calls `self.logger.experiment.add_audio(...)` from
`on_validation_epoch_end`.

**We do:** point at TensorBoard's AUDIO tab as the primary quality signal.
`./run preview` (`piper.train.infer_torch`) is the secondary path, for arbitrary
text.

## 15. `infer_torch` reads JSON lines, not plain text

`src/piper/train/infer_torch.py` iterates `sys.stdin` and does
`utt = json.loads(line)`, reading `utt["text"]` and an optional
`utt["speaker_id"]`. Output files are named after the stdin line index.

**We do:** `train/preview.py` sends `{"text": ...}` per line and prints the
index-to-sentence mapping, since `0.wav` is otherwise meaningless.

## 16. `build_monotonic_align.sh` prefers its own venv, and is bash

The script activates `piper1-gpl/.venv` **if it exists**, otherwise uses whatever
`python` is on `PATH`. It needs `cythonize`, which comes from the `[train]`
extra, and it emits into a nested `monotonic_align/monotonic_align/`. In full it
is four commands:

```sh
cd src/piper/train/vits/monotonic_align
mkdir -p monotonic_align
rm -f core.c
cythonize -i core.pyx
mv core*.so monotonic_align/
```

Two problems, not one. The venv preference would build against the wrong
interpreter whenever a stray `piper1-gpl/.venv` exists — ours lives at the repo
root. And it is bash, which is not available on Windows.

**We do:** not call it. `install.step_monotonic_align` performs those four steps
directly with our venv interpreter, invoking `python -m Cython.Build.Cythonize`
rather than the `cythonize` console script (which lands in `bin` or `Scripts`
depending on platform), and matching the built extension by stem so `.so` and
`.pyd` are both handled. Then it verifies with
`from piper.train.vits.monotonic_align import maximum_path` — a successful exit
is not proof the extension is importable.

Re-check this section when bumping the `[piper]` pin: it is the one place we
reimplement an upstream build step rather than calling it.

## 17. Dependency pins live in `setup.py`, not `pyproject.toml`

`pyproject.toml` contains only `build-system` and a black config. `setup.py` uses
`from skbuild import setup` and declares `python_requires=">=3.9"` with the
`[train]` extra: `torch>=2,<3`, `lightning>=2,<3`, `tensorboard>=2,<3`,
`tensorboardX>=2,<3`, `jsonargparse[signatures]>=4.27.7`, `onnx>=1,<2`,
`pysilero-vad>=2.1,<3`, `cython>=3,<4`, `librosa<1`.

`torch>=2,<3` is the important one: it is satisfied by *any* torch 2.x, including
whichever one PyPI serves by default.

`from skbuild import setup` matters for a second reason. `pip install -e .`
builds in isolation, so pip materialises `scikit-build` from
`build-system.requires` into a throwaway environment and discards it. Our
`build_ext` step runs `setup.py build_ext --inplace` *directly* with the venv
interpreter, where that environment does not exist, and the import fails:

```
ModuleNotFoundError: No module named 'skbuild'
```

The traceback is actively misleading — `skbuild` is the *import* name of the
PyPI package `scikit-build`, so searching for "skbuild" in a distro package
manager finds nothing, and a venv could not see a system package anyway.

**We do:** install `[build].requires` from `pins.toml` into the venv at the top
of `step_build_ext` (`scikit-build`, plus `cmake` and `ninja`, which skbuild
shells out to), and map this traceback to an explicit message in
`install._explain_build_failure`.

## 18. espeak-ng is built from source and embedded

`CMakeLists.txt` adds espeak-ng as an `ExternalProject` at `GIT_TAG 724808c`,
requires `cmake_minimum_required(VERSION 3.26)`, and needs network access at
build time. There is **no `espeak-ng` binary** afterwards — phonemization goes
through `piper.phonemize_espeak.EspeakPhonemizer`, whose signature is
`phonemize(voice, text)` (`phonemize_espeak.py:13`).

**We do:** never look for a system espeak-ng package, and validate
`voice.espeak_voice` by actually phonemizing a sentence through the embedded
library.

## 19. No ROCm anywhere upstream

Zero mentions of ROCm or HIP. The only vendor-specific code is in
`train/__main__.py`:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

(harmless no-ops elsewhere). `docs/TRAINING.md`'s only AMD reference is that
users have reported success with an RX 7600 and as little as 8 GB of VRAM.
`script/setup` offers `--torch-cpu` but has no ROCm option.

**We do:** treat the torch wheel as the entire portability story. See
[GPU_SETUP.md](GPU_SETUP.md).

---

## Checklist for bumping the pin

Do all of this when changing `[piper]` in `pins.toml`:

1. `./run setup --force-step clone --force-step piper_install --force-step monotonic_align --force-step build_ext`
2. `python -m unittest discover -s tests -t .` — `test_argmap.py` pins the exact
   emitted config, so a changed default fails here first.
3. Re-read `src/piper/train/__main__.py` and confirm the eight `link_arguments`
   pairs still match `argmap.LINKS`.
4. Re-read `VitsModel.__init__` and `VitsDataModule.__init__` and confirm no new
   required argument appeared and no default in `train/presets.py` changed.
5. Re-check items 4, 6, 8, 10 and 13 above — they are bugs or quirks that may
   have been fixed, in which case our workaround should be removed rather than
   left to rot.
6. `./run smoke` — the negative tests confirm the blocks still describe reality.
7. Update the tag, sha and `espeak_ng_tag` in `pins.toml`, and the version
   references in this file.
