# Export

## A voice is two files

```
voices/<voice>/en_US-myvoice-medium.onnx
voices/<voice>/en_US-myvoice-medium.onnx.json
```

They must keep matching names and travel together. The naming convention
`<language>-<name>-<quality>` is what other Piper tooling expects, and `./run
export` builds it from `voice.language`, `voice.name` and `voice.quality` — so a
`high` model is not mislabelled `medium`.

**The two files come from different places.** The ONNX comes from
`piper.train.export_onnx`, whose only flags are `--checkpoint`, `--output-file`
and `--debug`. The JSON comes from **training**: `VitsDataModule.prepare_data()`
writes it to `--data.config_path`. Export does not produce it, which surprises
people — if it is missing, training never got far enough to prepare data.

## Two fields need correcting

Verified in piper1-gpl v1.6.0:

- **`hop_length` is always 256.** `vits/dataset.py` builds `PiperConfig(...)`
  without passing its own `hop_length`, so `config.py`'s `DEFAULT_HOP_LENGTH`
  wins. Harmless at the default; wrong for any other hop length.
- **`piper_version` is the literal `"1.5.0"`** while the package is 1.6.0.

So `./run export` copies rather than moves, corrects both, optionally writes your
preferred inference scales, keeps the untouched original as
`<name>.onnx.json.from-training`, and prints the diff. Nothing is rewritten
silently.

A sample-rate disagreement between the config and the checkpoint is a **hard
error**, not a fixup: it means the config belongs to a different run.

## Choosing a checkpoint

```bash
./run export
```

lists the candidates with their metrics, read straight out of Lightning's
filenames (`epoch={epoch}-val_mel={val_mel:.4f}.ckpt`, with
`auto_insert_metric_name=False`), so ranking them costs no torch load.

| Choice | Meaning |
| --- | --- |
| best by `val_mel` | Lowest mel L1 — reconstruction accuracy. The default. |
| best by `val_mos` | Highest UTMOS — perceptual quality. Often sounds better. |
| `last.ckpt` | Most recent. |

Neither metric beats listening. `val_mel` saturates long before the adversarial
losses finish cleaning up the audio — which is exactly why upstream does not
early-stop on it. Use `./run preview` to synthesize the same sentence from two or
three candidates and pick by ear.

`val_mos` is only present if `model.mos_metric` was `utmos` *and* UTMOS could be
fetched from `torch.hub`. If it is missing everywhere, that is why.

## Inference scales

Baked into the JSON as defaults, and adjustable per synthesis afterwards:

| Setting | Default | Effect |
| --- | --- | --- |
| `export.noise_scale` | 0.667 | Variation in timbre. Lower is flatter and more stable |
| `export.length_scale` | 1.0 | Speaking rate. Above 1.0 is slower |
| `export.noise_w` | 0.8 | Variation in phoneme duration |

Try them with `./run preview --noise-scale 0.5 --length-scale 1.1` before
committing.

## Verification

After export, the pair is loaded back with `PiperVoice.load(...)` on CPU via
onnxruntime and used to synthesize one sentence to `verify.wav`. The frame count
and sample rate are reported, and a failure is a non-zero exit — so an export that
"succeeded" but produces an unusable voice cannot pass quietly.

Skip it with `--no-verify` if you are exporting many checkpoints in a loop.

## Using the voice

On the same machine:

```bash
.venv/bin/python -m piper -m voices/myvoice/en_US-myvoice-medium.onnx -f out.wav -- "Hello there."
```

Anywhere else: copy **both** files, keeping the names identical, into whatever
Piper install you are using — Home Assistant, `pip install piper-tts`, or your own
code. The JSON is found by appending `.json` to the model path, so a rename that
breaks the pairing breaks the voice.

The voice needs no GPU to run. Inference is CPU-friendly by design; that is Piper's
whole point.

## Exporting mid-training

Perfectly safe. Checkpoints are complete, and export does not touch the run
directory. You can export a checkpoint, listen, and carry on training from
`last.ckpt` with `./run resume`.
