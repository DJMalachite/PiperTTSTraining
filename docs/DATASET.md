# Datasets

## What good source audio looks like

The dataset is the ceiling on voice quality. In rough order of importance:

- **One speaker.** There is no diarization here. Two voices in the recording
  become one confused model.
- **Clean.** Background music, traffic, hum and reverb are all reproduced
  faithfully by the vocoder, because to the model they are part of the voice.
- **Consistent.** Same microphone, same room, same distance. Changing any of them
  mid-recording teaches the model that the speaker sometimes sounds different.
- **Enough of it.** 30 minutes is a realistic floor when fine-tuning from a
  pretrained checkpoint; one to two hours is comfortable; more is better. Under
  about 10 minutes, expect artifacts regardless of settings.
- **Natural.** Read prose beats a word list. The model learns prosody from
  sentences.

Format does not matter — anything ffmpeg can read, including video. Sample rate
does not matter either; it is resampled once during decoding.

## What the pipeline does

```
probe → decode → macrosplit → transcribe → segment → emit → report
```

**probe** — `ffprobe` for duration, channels, codec. A file with no audio stream
is refused.

**decode** — one `ffmpeg` pass to float32 mono at `audio.sample_rate`, applying
the high-pass and `loudnorm` filters if enabled, written to a cache file that is
then memory-mapped. Materialising it costs ~317 MB per hour of audio and means
re-running with different clip lengths does not decode again.

**macrosplit** — break the recording at long silences into chunks of at most
`dataset.macro_max_seconds` (default 10 minutes). Two reasons: Whisper's
timestamps drift over hours of audio and it becomes prone to repetition loops;
and these chunks are the unit of work that gets cached, so an interrupted run
resumes.

The silence threshold is estimated from the recording's own noise floor unless
`dataset.silence_dbfs` is set. A fixed value — the predecessor script hardcoded
−40 dBFS — is wrong for both a quiet studio capture and a noisy phone recording.

**transcribe** — `openai-whisper` with `word_timestamps=True`, cached as one JSON
per chunk with timestamps rebased to absolute time.

openai-whisper rather than faster-whisper because it runs on torch: the same
install that trains the model also transcribes, on ROCm and CUDA alike.
faster-whisper is quicker but CTranslate2 has no ROCm build, so on an AMD board
it would silently fall back to CPU.

**segment** — group *words* into utterances. This is where quality is won.

**emit** — cut, normalise, write 16-bit mono WAVs; clean each transcript and
reject the ones that fail.

**report** — statistics, findings, and the split arithmetic.

## Why group words instead of splitting on silence

Splitting on silence gives clips of arbitrary length whose transcripts are
produced separately. A clip that clips a word mid-syllable gets a transcript
containing that whole word, and the model learns an alignment that does not
exist. Grouping words means each clip's boundaries and its text come from the
same alignment, and the length bounds are enforced rather than emergent.

The `align` strategy accumulates words and closes a clip on:

- terminal punctuation (`. ! ? … : ;`) once past `min_seconds`, or
- a pause of at least `boundary_gap` once past `min_seconds`, or
- a comma once past `target_seconds`, or
- `max_seconds`, as a hard ceiling.

Groups over the ceiling are split recursively at their widest internal pause,
preferring the middle so the halves stay balanced. Groups under the floor merge
into a neighbour, or are dropped with a reason. A single "word" longer than
`max_seconds` is a Whisper timestamp glitch and is dropped.

Cuts land at the midpoint of the surrounding pause, clamped by `pad_before` /
`pad_after` so neighbouring clips can never contain each other's speech, and
snapped to the nearest zero crossing to avoid a click.

The `vad` strategy is the fallback: segment on silence with `pysilero-vad`, then
transcribe each clip separately. Use it when Whisper's word timestamps are
unreliable — very noisy audio, or a language it handles poorly.

## Every setting

### Segmentation

| Setting | Default | Effect |
| --- | --- | --- |
| `strategy` | `align` | `align` groups Whisper words; `vad` segments on silence first |
| `min_seconds` | 1.0 | Shortest clip. **Do not go below 1.0** — see below |
| `target_seconds` | 8.0 | Preferred length; clips close at the first boundary past it |
| `max_seconds` | 14.0 | Hard ceiling. Lower this first when you hit OOM |
| `boundary_gap` | 0.35 | Silence counting as an utterance boundary |
| `pad_before` / `pad_after` | 0.15 | Audio kept around the speech |
| `snap_zero_crossing` | yes | Avoid clicks at cut points |
| `macro_silence_seconds` | 1.5 | Silence used for the coarse first pass |
| `macro_max_seconds` | 600 | Coarse chunk ceiling |
| `silence_dbfs` | 0 (auto) | Silence threshold; 0 estimates from the noise floor |
| `id_prefix` | *(empty)* | Prefix for clip filenames |
| `dry_run` | no | Analyse and report without writing WAVs |

**On the 1 second floor.** `UtteranceCollate` pads every batch up to at least
`segment_size` — 0.372 s at the default 8192 and 22.05 kHz. Clips shorter than
that are zero-padded, and the model learns to produce that silence. 1.0 s leaves
comfortable headroom, which is why the schema will not accept less.

**On `max_seconds` and memory.** Long clips dominate peak GPU memory, so this is
the second lever after `batch_size` when training runs out of memory. Lowering it
requires rebuilding the dataset, but transcription is cached so it is quick.

### Whisper

| Setting | Default | Effect |
| --- | --- | --- |
| `whisper.model` | `turbo` | `turbo` is the best speed/accuracy trade for English; `large-v3` is more accurate and much slower |
| `whisper.device` | `auto` | GPU when torch can really use one. On ROCm the device is still spelled `cuda` |
| `whisper.language` | `en` | Set it explicitly; detection is less reliable on long audio |
| `whisper.initial_prompt` | *(empty)* | Bias vocabulary — names, jargon, spelling conventions it keeps getting wrong |
| `whisper.condition_on_previous_text` | no | Improves fluency but causes repetition loops on long audio |
| `whisper.temperature` | `0.0 … 1.0` | Fallback ladder when a segment fails quality thresholds |
| `whisper.beam_size` | 5 | Higher is slower, marginally better |
| `whisper.fp16` | `auto` | On GPU only — fp16 on CPU warns and falls back |

### Audio

| Setting | Default | Effect |
| --- | --- | --- |
| `audio.sample_rate` | 22050 | Output rate and the rate the model trains at. 16000 for the `low` preset |
| `audio.normalize` | `peak` | `peak`, `loudnorm`, or `none` |
| `audio.peak_dbfs` | −1.0 | Target for `peak` |
| `audio.highpass_hz` | 0 | 60–80 removes rumble and handling noise |

**Normalisation is applied to the whole recording, not per clip.** Per-clip peak
normalisation flattens the natural loudness difference between a statement and an
aside, which is prosody worth keeping.

### Text

| Setting | Default | Effect |
| --- | --- | --- |
| `text.ensure_terminal_punctuation` | yes | Append a full stop; helps sentence-final prosody |
| `text.drop_bracketed` | yes | Remove `[Music]`, `(laughs)` — no audio counterpart to learn |
| `text.normalize_quotes` | yes | Fold curly quotes and dashes to ASCII |
| `text.min_chars` | 2 | Reject shorter transcripts |
| `text.cps_min` / `cps_max` | 3 / 30 | Characters-per-second bounds; see below |

**On characters per second.** This is the misalignment detector, and it catches
what nothing else can. Too few characters for the duration means the transcript
is missing words. Too many means it carries text belonging to neighbouring audio.
Both are invisible from the audio and both are poison for training. Rejects go to
`rejected.csv` with the reason.

Repetition loops — Whisper emitting "thank you" twenty times — are detected
separately and rejected.

## Output layout

```
data/<voice>/
  source/          your recording, read-only
  wavs/            000001.wav, 000002.wav, ...
  metadata.csv     000001.wav|The quick brown fox jumps over the lazy dog.
  rejected.csv     utt_id, start, end, reason, text
  report.md        statistics, findings, clips worth checking
  manifest.json    per-stage fingerprints
  .cache/          decoded audio and cached transcripts
```

`metadata.csv` is pipe-delimited with no header, written with Python's `csv`
module using the *default* dialect — which is exactly what piper reads with, so a
transcript containing a quote or a pipe round-trips correctly. (Writing it with an
f-string, as the predecessor did, silently corrupts those rows.)

Clip ids are bare filenames and `--data.audio_dir` points at `wavs/` directly.
The older `wavs/000001.wav|text` form forces `audio_dir` to be the *parent*
directory, which is a reliable source of confusion.

## Reading the report

The summary gives clip count, total speech, yield as a fraction of the recording,
the length distribution, and the split arithmetic including **the maximum usable
batch size**. That last number prevents a run failing an hour later with zero
batches.

Findings are ranked. Errors mean training will fail or produce nothing useful;
warnings mean look at it.

The "clips worth listening to" table lists the ones furthest from the norm by
characters-per-second. Skimming five of those is the cheapest quality check
available, and the wizard makes you acknowledge it. You can edit `metadata.csv`
from the wizard with `$EDITOR`.

## Re-running

Stages cache, so iterating is cheap:

- Changing clip lengths re-cuts from the cached transcripts — seconds.
- Changing Whisper settings re-transcribes — the slow one.
- Changing the sample rate or filters re-decodes.

Force a stage with `./run dataset --force-stage transcribe` (also `decode`,
`macrosplit`, `segment`, `emit`, or `all`).

**Editing `metadata.csv` invalidates piper's utterance cache.** Its cache ids
embed the row number and the transcript text, so changing one line orphans the
entries after it. The next `./run train` notices via a fingerprint and offers to
keep the cache (piper recomputes what is missing) or wipe it. Keeping is safe;
just do not reorder or renumber rows.
