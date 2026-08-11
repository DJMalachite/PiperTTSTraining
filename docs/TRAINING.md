# Training

## The profile is the interface

Every setting lives in `profiles/<voice>.yaml`, written with its own
documentation inline. The wizard (`./run train`) generates its prompts from the
same schema, so the file and the wizard cannot disagree.

From the profile, `train/argmap.py` builds `runs/<voice>/lightning.yaml` and runs:

```bash
.venv/bin/python -m piper.train fit --config runs/<voice>/lightning.yaml [--ckpt_path ...]
```

A config file rather than sixty flags, because it lets architecture values go out
as native YAML sequences and gives `--print_config` as a free validation gate.
The equivalent flag-by-flag command is written to
`runs/<voice>/logs/run-*/equivalent-flags.txt` if you want to reproduce a run by
hand.

## Profile → flag mapping

Most settings map one-to-one. The exceptions are where it gets interesting.

| Profile | Emitted as | Notes |
| --- | --- | --- |
| `voice.name` | `--data.voice_name` | |
| `voice.espeak_voice` | `--data.espeak_voice` | Validated by actually phonemizing a sentence |
| `voice.quality` | *expanded* | Not an upstream flag — see presets below |
| `audio.sample_rate` | `--model.sample_rate` | **model** side: it is a link source |
| `data.batch_size` | `--data.batch_size` | **data** side: it is a link source |
| `data.num_symbols` | `--data.num_symbols` | data side, likewise |
| `model.num_speakers` | `--model.num_speakers` | model side, likewise |
| `model.segment_size` | `--model.segment_size` | model side, likewise |
| `data.*` (rest) | `--data.*` | |
| `model.*` (rest) | `--model.*` | |
| `trainer.*` | `--trainer.*` | |
| `trainer.checkpoint_save_top_k` | `--trainer.callbacks` | Replicates both upstream `ModelCheckpoint`s |
| `finetune.mode` + `checkpoint` | see below | Three different mechanisms |
| `model.extra` | `--model.*` | Raw passthrough, checked against the forbidden list |

### Never settable

piper declares eight `link_arguments` pairs, and jsonargparse treats setting a
*target* as a hard error. These are refused before anything runs:

`data.sample_rate`, `data.filter_length`, `data.hop_length`, `data.win_length`,
`data.segment_size`, `data.num_speakers`, `model.batch_size`,
`model.num_symbols`.

Also blocked, with reasons:

| Flag | Why |
| --- | --- |
| `trainer.gradient_clip_val` | `VitsModel` uses manual optimization; Lightning raises |
| `trainer.gradient_clip_algorithm` | Same |
| `model.grad_clip` | Accepted by `__init__` but **never applied** in v1.6.0 |

`trainer.accumulate_grad_batches` is allowed but warns: Lightning does not apply
accumulation under manual optimization. Raise `data.batch_size` instead.

## Quality presets

There is no `--quality` flag upstream; the names survive only as a filename
convention. `train/presets.py` expands `voice.quality` into raw `model.*` values.

| | medium (default) | high | low |
| --- | --- | --- | --- |
| sample rate | 22050 | 22050 | **16000** |
| `resblock` | `"2"` | `"1"` | `"2"` |
| `resblock_kernel_sizes` | `[3, 5, 7]` | `[3, 7, 11]` | `[3, 5, 7]` |
| `resblock_dilation_sizes` | `[[1,2],[2,6],[3,12]]` | `[[1,3,5],[1,3,5],[1,3,5]]` | as medium |
| `upsample_rates` | `[8, 8, 4]` | `[8, 8, 2, 2]` | `[8, 8, 4]` |
| `upsample_initial_channel` | 256 | **512** | 256 |
| `upsample_kernel_sizes` | `[16, 16, 8]` | `[16, 16, 4, 4]` | `[16, 16, 8]` |
| `hop_length` | 256 | 256 | 256 |

**Use medium unless you have a specific reason not to.** It matches
`VitsModel`'s defaults byte for byte, which means it is the only preset that
matches the pretrained checkpoints, which means it is the only one you can
fine-tune from with `--ckpt_path`.

`high` roughly doubles the vocoder width and adds an upsample stage: better
high-frequency detail, noticeably slower, more memory, and no pretrained
checkpoint to start from. `low` is medium's architecture at 16 kHz — cheapest and
audibly duller.

Every preset satisfies `prod(upsample_rates) == hop_length`, which upstream
enforces with a bare `ValueError`. If you override these through `model.extra`,
that invariant and four others are checked before launch with a message naming a
working alternative.

## Fine-tuning

Upstream recommends it even across languages. Three mechanisms, and picking the
wrong one wastes a run:

| Mode | Flag | When |
| --- | --- | --- |
| `ckpt_path` | `--ckpt_path` | **Default.** Strict resume. Needs a matching architecture — in practice, a medium checkpoint. |
| `vocoder_warmstart` | `--model.vocoder_warmstart_ckpt` | Copies the vocoder but not the phoneme embedding, so a different phoneme inventory or language is fine. |
| `warmstart` | `--model.warmstart_ckpt` | Copies every matching-shape parameter, fresh optimizer. **Required** when `model.use_mrd` is on, because `--ckpt_path` strict-loads and fails on the extra keys. |
| `none` | — | From scratch. Far slower. |

`./run checkpoints` browses `rhasspy/piper-checkpoints`, shows each file's size,
and fetches the sibling `config.json` first so you can confirm the sample rate and
phoneme count before committing to a ~900 MB download. Filenames are not uniform
(`epoch=6679-step=1554200.ckpt`, `last.ckpt`, `bryce-3499.ckpt` all occur), so the
listing always comes from the API.

Before launching, the checkpoint's stored `hyper_parameters` are compared against
the configured architecture. A mismatch under `ckpt_path` is refused, with a
recommendation of `warmstart` or — if the phoneme count differs —
`vocoder_warmstart`.

## Preflight checks

All of these run before the subprocess starts, none need the GPU:

1. `prod(upsample_rates) == hop_length`.
2. `upsample_kernel_sizes` parallel to `upsample_rates`; each kernel ≥ its
   stride; the difference even.
3. `resblock_kernel_sizes` parallel to `resblock_dilation_sizes`.
4. `segment_size` a multiple of `hop_length`.
5. Warn if the shortest clip is under `segment_size / sample_rate` — shorter clips
   get zero-padded and the model learns the silence.
6. The training split has at least one utterance.
7. `batch_size` ≤ the training split. The dataloader drops the last partial
   batch, so a larger value gives **zero batches** and an error that names
   something else entirely.
8. The checkpoint's architecture matches, for strict loads.
9. `use_mrd` forbids `ckpt_path`.
10. Offline mode: `mos_metric` off, and the checkpoint must already be local.
11. `espeak_voice` actually phonemizes.
12. The utterance cache fingerprint still matches.
13. The utterance cache is complete.
14. `--print_config`, as a type check.

## How long, and when to stop

`trainer.max_epochs` defaults to `-1` — upstream's default. Training runs until
you stop it, and that is correct for VITS: mel loss saturates long before the
adversarial losses finish removing audible artifacts. Upstream deliberately does
not early-stop on `val_mel` for exactly this reason.

So judge by listening. `./run monitor` serves TensorBoard, whose AUDIO tab already
contains synthesized test utterances from every validation epoch. `./run preview`
synthesizes arbitrary text from any checkpoint without exporting.

Rough expectations when fine-tuning from a medium checkpoint with an hour of clean
audio: intelligible within a few epochs, recognisably the target speaker within
tens, and continuing to improve for hundreds. From scratch, multiply by a lot.

## Checkpoints and disk

Upstream keeps the top five by `val_mel`, the top five by `val_mos`, and
`last.ckpt` — about 10 GB per run at ~0.9 GB each. Lower
`trainer.checkpoint_save_top_k`, or reclaim space with:

```bash
./run monitor --prune
```

which protects the best-by-each-metric and `last.ckpt` and asks before deleting
anything.

`val_mos` comes from UTMOS, fetched from `torch.hub` at first validation. If it
cannot be downloaded, upstream disables it gracefully and `val_mos` is simply
never logged — training is unaffected. Offline mode sets `model.mos_metric: none`
to skip the timeout.

## Resuming

```bash
./run resume
```

Finds the newest `last.ckpt`, and refuses if the profile no longer matches the
`lightning.yaml` the run started with — a changed architecture breaks a strict
load, and changed cache settings invalidate the utterance cache. It shows the diff
and `--force` overrides.

Ctrl-C during training sends SIGINT to piper and waits for it, so Lightning
flushes `last.ckpt` before exiting.

## Reproducibility

Each run writes `runs/<voice>/logs/run-<timestamp>/` containing `argv.txt`,
`equivalent-flags.txt`, `env.txt`, `lightning.yaml`, `profile.yaml`, `git.txt`
(both repos' commits) and `pip-freeze.txt`. Two runs can be diffed directly.
