"""Training/dataset profile schema.

One dataclass tree defines every knob exactly once. The interactive wizard, the
on-disk YAML, and ``train/argmap.py`` all read it from here, so they cannot
drift apart: adding a setting means adding one field.

Each field carries a ``Spec`` in its dataclass metadata (help text, choices,
bounds). The wizard renders prompts from it and the YAML writer emits the help
as comments, which is why the generated profile is self-documenting.

Two deliberate schema choices worth knowing:

* ``data.batch_size`` lives under ``data`` — not ``model`` — because
  piper1-gpl links ``data.batch_size -> model.batch_size`` and setting the
  target side of a jsonargparse link is an error. The profile mirrors the side
  the flag is actually emitted on.
* ``audio.sample_rate`` is the single source of truth for sample rate; it is
  emitted as ``--model.sample_rate`` (also a link source) and used for the
  dataset's WAV output.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import yamlio
from .paths import ACTIVE_PROFILE, PROFILES_DIR, profile_path, slug

SCHEMA_VERSION = 1

#: Kept in sync with ``hardware.NAMES``; duplicated as a literal so that
#: importing the schema never pulls in the hardware-probing machinery.
HARDWARE_NAMES = ("auto", "generic", "bc250")

QUALITIES = ("medium", "high", "low")
STRATEGIES = ("align", "vad")
VENDORS = ("rocm", "cuda", "cpu")
FINETUNE_MODES = ("none", "ckpt_path", "warmstart", "vocoder_warmstart")
PHONEME_TYPES = ("espeak", "text", "pinyin", "hebrew")
DATASET_TYPES = ("text", "phoneme_ids")
NORMALIZERS = ("peak", "loudnorm", "none")
PRECISIONS = ("32-true", "bf16-mixed", "16-mixed", "64-true")
WHISPER_MODELS = (
    "tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo",
)


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    """Presentation and validation metadata for one field."""

    help: str
    kind: str = "str"  # str|int|float|bool|path|list|dict|choice
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    advanced: bool = False
    label: str | None = None


def spec(default: Any, help: str, **kwargs: Any) -> Any:
    return field(default=default, metadata={"spec": Spec(help=help, **kwargs)})


def spec_factory(factory: Any, help: str, **kwargs: Any) -> Any:
    return field(
        default_factory=factory, metadata={"spec": Spec(help=help, **kwargs)}
    )


def nested(factory: Any) -> Any:
    return field(default_factory=factory)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass
class VoiceCfg:
    name: str = spec(
        "myvoice",
        "Short name for this voice. Used for directory names and the exported "
        "filename, so keep it lowercase and hyphen-free.",
    )
    language: str = spec(
        "en_US",
        "Locale tag used only for the exported filename, e.g. en_US-myvoice-"
        "medium.onnx. Does not affect phonemization.",
    )
    espeak_voice: str = spec(
        "en-us",
        "espeak-ng voice used to turn text into phonemes. This DOES affect the "
        "model. Run './run doctor' to verify it phonemizes.",
    )
    quality: str = spec(
        "medium",
        "Architecture preset. 'medium' matches every pretrained checkpoint and "
        "is the only one you can fine-tune from without extra work. 'high' is "
        "bigger and slower; 'low' is medium's architecture at 16 kHz.",
        kind="choice",
        choices=QUALITIES,
    )


@dataclass
class WhisperCfg:
    model: str = spec(
        "turbo",
        "Whisper model size. 'turbo' is the best speed/accuracy trade for "
        "English; 'large-v3' is more accurate and much slower.",
        kind="choice",
        choices=WHISPER_MODELS,
    )
    device: str = spec(
        "auto",
        "auto | cuda | cpu. 'auto' uses the GPU when torch reports one. On "
        "ROCm, 'cuda' is correct — that is what the ROCm build calls itself.",
        kind="choice",
        choices=("auto", "cuda", "cpu"),
    )
    language: str = spec(
        "en",
        "Source language code, or 'auto' to detect. Setting it explicitly is "
        "more reliable than detection on long recordings.",
    )
    initial_prompt: str = spec(
        "",
        "Optional text biasing Whisper's vocabulary — useful for names, jargon "
        "and spelling conventions it keeps getting wrong.",
    )
    condition_on_previous_text: bool = spec(
        False,
        "Feeding prior text back into the model improves fluency but causes "
        "repetition loops on long audio. Off is the safer default.",
        kind="bool",
    )
    temperature: list[float] = spec_factory(
        lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "Temperature fallback ladder. Whisper retries with the next value when "
        "a segment fails its quality thresholds.",
        kind="list",
        advanced=True,
    )
    beam_size: int = spec(
        5,
        "Beam width for decoding. Higher is slower and marginally better.",
        kind="int",
        minimum=1,
        maximum=20,
        advanced=True,
    )
    fp16: str = spec(
        "auto",
        "Half-precision inference. 'auto' enables it only on GPU — fp16 on CPU "
        "warns and falls back.",
        kind="choice",
        choices=("auto", "true", "false"),
        advanced=True,
    )


@dataclass
class TextCfg:
    ensure_terminal_punctuation: bool = spec(
        True,
        "Append a full stop when a transcript has no terminal punctuation. "
        "Helps the model learn sentence-final prosody.",
        kind="bool",
    )
    drop_bracketed: bool = spec(
        True,
        "Remove Whisper's non-speech annotations such as [Music] or (laughs), "
        "which have no audio counterpart the model could learn.",
        kind="bool",
    )
    normalize_quotes: bool = spec(
        True,
        "Fold curly quotes and en/em dashes to ASCII. espeak-ng handles both, "
        "but consistency reduces the phoneme inventory.",
        kind="bool",
    )
    min_chars: int = spec(
        2,
        "Reject transcripts shorter than this many characters.",
        kind="int",
        minimum=1,
    )
    cps_min: float = spec(
        3.0,
        "Reject clips below this many characters per second — usually a clip "
        "whose transcript is missing words.",
        kind="float",
        minimum=0.0,
        unit="chars/s",
    )
    cps_max: float = spec(
        30.0,
        "Reject clips above this many characters per second — usually a clip "
        "carrying text that belongs to neighbouring audio.",
        kind="float",
        minimum=1.0,
        unit="chars/s",
    )


@dataclass
class DatasetCfg:
    input_path: str = spec(
        "",
        "The single audio or video file to build the dataset from. Anything "
        "ffmpeg can read. This file is only ever read, never modified.",
        kind="path",
    )
    strategy: str = spec(
        "align",
        "'align' cuts clips at Whisper word boundaries so each transcript "
        "matches its audio exactly — best quality. 'vad' segments on silence "
        "first and transcribes each clip, for audio Whisper struggles with.",
        kind="choice",
        choices=STRATEGIES,
    )
    min_seconds: float = spec(
        1.0,
        "Shortest clip to keep. Must stay at or above 1.0 s: piper pads every "
        "clip up to segment_size (0.372 s at 22.05 kHz) with silence, and "
        "short clips teach the model to produce silence.",
        kind="float",
        minimum=1.0,
        maximum=10.0,
        unit="s",
    )
    target_seconds: float = spec(
        8.0,
        "Preferred clip length. Clips are closed at the first sentence or "
        "pause boundary after this point.",
        kind="float",
        minimum=2.0,
        maximum=20.0,
        unit="s",
    )
    max_seconds: float = spec(
        14.0,
        "Hard ceiling. Longer groups are split recursively at their widest "
        "internal pause. Long clips dominate peak GPU memory — lower this "
        "first when you hit OOM.",
        kind="float",
        minimum=3.0,
        maximum=30.0,
        unit="s",
    )
    boundary_gap: float = spec(
        0.35,
        "A silence at least this long counts as an utterance boundary.",
        kind="float",
        minimum=0.05,
        maximum=2.0,
        unit="s",
    )
    pad_before: float = spec(
        0.15,
        "Audio kept before the first word. piper trims silence itself with "
        "Silero VAD, so this only needs to avoid clipping the attack.",
        kind="float",
        minimum=0.0,
        maximum=1.0,
        unit="s",
    )
    pad_after: float = spec(
        0.15,
        "Audio kept after the last word.",
        kind="float",
        minimum=0.0,
        maximum=1.0,
        unit="s",
    )
    snap_zero_crossing: bool = spec(
        True,
        "Move cut points to the nearest zero crossing so clips do not start or "
        "end with a click.",
        kind="bool",
        advanced=True,
    )
    macro_silence_seconds: float = spec(
        1.5,
        "Silence length used for the first coarse pass, which breaks the "
        "recording into resumable chunks before transcription.",
        kind="float",
        minimum=0.3,
        unit="s",
        advanced=True,
    )
    macro_max_seconds: float = spec(
        600.0,
        "Maximum length of a coarse chunk. Keeping Whisper's context bounded "
        "stops timestamp drift on multi-hour recordings.",
        kind="float",
        minimum=30.0,
        unit="s",
        advanced=True,
    )
    silence_dbfs: float = spec(
        0.0,
        "Level below which audio counts as silence. 0 means estimate it from "
        "the recording's own noise floor, which is usually better than guessing "
        "a fixed value.",
        kind="float",
        minimum=-90.0,
        maximum=0.0,
        unit="dBFS",
        advanced=True,
    )
    id_prefix: str = spec(
        "",
        "Optional prefix for clip filenames, e.g. 'sess1-' when merging "
        "several sessions by hand later.",
        advanced=True,
    )
    dry_run: bool = spec(
        False,
        "Analyse and report without writing any WAV files. Useful for tuning "
        "the segmentation settings on a long recording.",
        kind="bool",
    )
    whisper: WhisperCfg = nested(WhisperCfg)
    text: TextCfg = nested(TextCfg)


@dataclass
class AudioCfg:
    sample_rate: int = spec(
        22050,
        "Output sample rate, and the rate the model is trained at. 22050 for "
        "medium/high, 16000 for the low preset.",
        kind="int",
        choices=(16000, 22050),
        unit="Hz",
    )
    normalize: str = spec(
        "peak",
        "'peak' scales each clip to peak_dbfs, 'loudnorm' targets perceptual "
        "loudness (slower, better for uneven recordings), 'none' leaves levels "
        "untouched.",
        kind="choice",
        choices=NORMALIZERS,
    )
    peak_dbfs: float = spec(
        -1.0,
        "Peak target for 'peak' normalization.",
        kind="float",
        minimum=-12.0,
        maximum=0.0,
        unit="dBFS",
    )
    highpass_hz: int = spec(
        0,
        "High-pass filter cutoff, 0 to disable. 60-80 Hz removes rumble and "
        "handling noise that the vocoder would otherwise reproduce.",
        kind="int",
        minimum=0,
        maximum=300,
        unit="Hz",
        advanced=True,
    )


@dataclass
class ModelCfg:
    """Model-side overrides. Architecture comes from the quality preset."""

    learning_rate: float = spec(
        2e-4, "Generator learning rate.", kind="float", minimum=1e-6
    )
    learning_rate_d: float = spec(
        1e-4, "Discriminator learning rate.", kind="float", minimum=1e-6
    )
    lr_decay: float = spec(
        0.999875, "Per-epoch generator LR decay.", kind="float", advanced=True
    )
    lr_decay_d: float = spec(
        0.9999, "Per-epoch discriminator LR decay.", kind="float", advanced=True
    )
    warmup_epochs: int = spec(
        0,
        "Epochs before LR decay begins.",
        kind="int",
        minimum=0,
        advanced=True,
    )
    c_mel: int = spec(
        45, "Weight of the mel reconstruction loss.", kind="int", advanced=True
    )
    c_kl: float = spec(
        1.0, "Weight of the KL divergence loss.", kind="float", advanced=True
    )
    segment_size: int = spec(
        8192,
        "Audio window the discriminator sees, in samples. Must be a multiple "
        "of hop_length. Halving it is the second-best OOM lever after "
        "batch_size, but changes what the discriminator learns.",
        kind="int",
        minimum=1024,
    )
    num_speakers: int = spec(
        1,
        "Set above 1 only for a multi-speaker metadata.csv (wav|speaker|text). "
        "This pipeline produces single-speaker datasets.",
        kind="int",
        minimum=1,
        advanced=True,
    )
    use_sdp: bool = spec(
        True,
        "Stochastic duration predictor. Leave on; off yields flatter prosody.",
        kind="bool",
        advanced=True,
    )
    use_mrd: bool = spec(
        False,
        "Add the multi-resolution STFT discriminator: better high-frequency "
        "detail at no inference cost, but it forces finetune mode 'warmstart' "
        "because --ckpt_path strict-loads and fails on the extra keys.",
        kind="bool",
        advanced=True,
    )
    mos_metric: str = spec(
        "utmos",
        "Perceptual quality metric logged during validation. 'utmos' "
        "downloads a model from torch.hub on first use; 'none' skips it, "
        "which is faster and required offline.",
        kind="choice",
        choices=("utmos", "none"),
    )
    extra: dict[str, Any] = spec_factory(
        dict,
        "Raw --model.* passthrough for anything not exposed above, e.g. "
        "{upsample_initial_channel: 384}. You own the consequences.",
        kind="dict",
        advanced=True,
    )


@dataclass
class DataCfg:
    batch_size: int = spec(
        16,
        "Utterances per step. The single biggest lever on GPU memory. Must not "
        "exceed the training split size or the loader yields zero batches.",
        kind="int",
        minimum=1,
    )
    validation_split: float = spec(
        0.1,
        "Fraction of clips held back for validation.",
        kind="float",
        minimum=0.0,
        maximum=0.5,
    )
    num_test_examples: int = spec(
        5,
        "Clips reserved for the audio samples logged to TensorBoard.",
        kind="int",
        minimum=0,
    )
    num_workers: int = spec(
        2,
        "Dataloader worker processes. piper forces persistent_workers and "
        "prefetch_factor=4 whenever this is above 0, so each worker costs real "
        "RAM — which an APU shares with the GPU. 0 is the smallest footprint.",
        kind="int",
        minimum=0,
        maximum=16,
    )
    trim_silence: bool = spec(
        True,
        "Let piper trim leading/trailing silence with Silero VAD during "
        "caching. Shorter clips mean less padding and less memory.",
        kind="bool",
    )
    keep_seconds_before_silence: float = spec(
        0.25,
        "Silence retained before speech when trimming.",
        kind="float",
        minimum=0.0,
        unit="s",
        advanced=True,
    )
    keep_seconds_after_silence: float = spec(
        0.25,
        "Silence retained after speech when trimming.",
        kind="float",
        minimum=0.0,
        unit="s",
        advanced=True,
    )
    phoneme_type: str = spec(
        "espeak",
        "'espeak' phonemizes the text column. 'text' treats the column as "
        "literal phonemes; 'pinyin' and 'hebrew' need extra upstream setup.",
        kind="choice",
        choices=PHONEME_TYPES,
        advanced=True,
    )
    dataset_type: str = spec(
        "text",
        "'text' or 'phoneme_ids'. Note upstream's TRAINING.md calls the flag "
        "--data.data_type; that is a doc typo, the real one is dataset_type.",
        kind="choice",
        choices=DATASET_TYPES,
        advanced=True,
    )
    phonemes_path: str = spec(
        "",
        "Optional JSON phoneme->id map copied into the voice config, for "
        "matching an existing model's phoneme inventory.",
        kind="path",
        advanced=True,
    )
    num_symbols: int = spec(
        256,
        "Size of the phoneme inventory. Leave at 256 unless you are supplying "
        "your own phoneme ids; it must match any checkpoint you fine-tune from.",
        kind="int",
        minimum=2,
        advanced=True,
    )


@dataclass
class TrainerCfg:
    accelerator: str = spec(
        "auto",
        "auto | gpu | cpu. 'gpu' is correct for both CUDA and ROCm.",
        kind="choice",
        choices=("auto", "gpu", "cpu"),
    )
    devices: str = spec(
        "auto",
        "Device count or list, e.g. 'auto', '1', '[0]'.",
        advanced=True,
    )
    precision: str = spec(
        "32-true",
        "32-true is the safe default: piper uses manual optimization with a "
        "GAN, and mixed precision is the least-tested combination. RDNA2 has "
        "no bf16 matrix path, so bf16 buys nothing there.",
        kind="choice",
        choices=PRECISIONS,
    )
    max_epochs: int = spec(
        -1,
        "-1 trains until you stop it, which is upstream's default and normal "
        "for VITS — mel loss saturates long before the audio stops improving.",
        kind="int",
        minimum=-1,
    )
    max_steps: int = spec(
        -1, "Optional step ceiling; -1 for none.", kind="int", minimum=-1,
        advanced=True,
    )
    check_val_every_n_epoch: int = spec(
        1,
        "Validation synthesizes full utterances and is often the real memory "
        "peak; raising this is a cheap OOM lever.",
        kind="int",
        minimum=1,
    )
    log_every_n_steps: int = spec(
        25, "TensorBoard scalar logging interval.", kind="int", minimum=1
    )
    accumulate_grad_batches: int = spec(
        1,
        "Almost certainly inert here: piper sets automatic_optimization=False, "
        "and Lightning does not apply accumulation for manual optimization. "
        "Raise batch_size instead.",
        kind="int",
        minimum=1,
        advanced=True,
    )
    enable_progress_bar: bool = spec(
        True,
        "Turn off for clean logs over SSH or in CI.",
        kind="bool",
        advanced=True,
    )
    checkpoint_save_top_k: int = spec(
        5,
        "Checkpoints kept per metric. Upstream keeps 5 by val_mel plus 5 by "
        "val_mos plus last — roughly 10 GB per run at ~0.9 GB each.",
        kind="int",
        minimum=1,
        maximum=20,
    )


@dataclass
class FinetuneCfg:
    mode: str = spec(
        "ckpt_path",
        "'ckpt_path' resumes from a checkpoint strictly and is what upstream "
        "recommends — it needs a matching architecture. 'vocoder_warmstart' "
        "copies only the vocoder, so the phoneme count may differ. "
        "'warmstart' copies every matching-shape parameter with a fresh "
        "optimizer, and is required when use_mrd is on. 'none' trains from "
        "scratch, which takes far longer.",
        kind="choice",
        choices=FINETUNE_MODES,
    )
    checkpoint: str = spec(
        "",
        "Path to a .ckpt. Use './run checkpoints' to browse and download one "
        "from HuggingFace.",
        kind="path",
    )


@dataclass
class RuntimeCfg:
    vendor: str = spec(
        "",
        "GPU vendor: rocm | cuda | cpu. Empty means autodetect. This decides "
        "which torch wheel gets installed and cannot be changed without "
        "reinstalling torch.",
        kind="choice",
        choices=("",) + VENDORS,
    )
    hardware: str = spec(
        "auto",
        "Hardware profile. 'generic' suits every officially supported GPU. "
        "'bc250' adds what the AMD BC-250 (gfx1013) needs: HSA_ENABLE_SDMA=0, "
        "conservative memory settings, and checks for the kernel and amdgpu "
        "module parameters that board requires. 'auto' picks from the detected "
        "GPU. See docs/BC250.md.",
        kind="choice",
        choices=HARDWARE_NAMES,
    )
    offline: bool = spec(
        False,
        "No network calls: no checkpoint downloads, HF_HUB_OFFLINE=1, and "
        "mos_metric forced to 'none' to skip a torch.hub timeout.",
        kind="bool",
    )
    low_vram: bool = spec(
        False,
        "Apply the small-GPU overlay (smaller batch, fewer workers, 32-true). "
        "Set automatically when a detected GPU has 16 GB or less.",
        kind="bool",
    )
    env: dict[str, str] = spec_factory(
        dict,
        "Extra environment variables for training and inference. The hardware "
        "profile fills in what your board needs; anything you add here wins "
        "over it.",
        kind="dict",
    )


@dataclass
class ExportCfg:
    output_dir: str = spec(
        "",
        "Where the .onnx and .onnx.json land. Empty means voices/<name>/.",
        kind="path",
    )
    filename: str = spec(
        "auto",
        "'auto' builds <language>-<name>-<quality> from the voice section, "
        "which is the naming other Piper tooling expects.",
    )
    noise_scale: float = spec(
        0.667,
        "Inference default written into the voice config: variation in timbre.",
        kind="float",
    )
    length_scale: float = spec(
        1.0,
        "Inference default: speaking rate. Above 1.0 is slower.",
        kind="float",
    )
    noise_w: float = spec(
        0.8,
        "Inference default: variation in phoneme duration.",
        kind="float",
    )


@dataclass
class Profile:
    schema: int = SCHEMA_VERSION
    voice: VoiceCfg = nested(VoiceCfg)
    dataset: DatasetCfg = nested(DatasetCfg)
    audio: AudioCfg = nested(AudioCfg)
    model: ModelCfg = nested(ModelCfg)
    data: DataCfg = nested(DataCfg)
    trainer: TrainerCfg = nested(TrainerCfg)
    finetune: FinetuneCfg = nested(FinetuneCfg)
    runtime: RuntimeCfg = nested(RuntimeCfg)
    export: ExportCfg = nested(ExportCfg)

    # -- convenience -------------------------------------------------------
    @property
    def slug(self) -> str:
        return slug(self.voice.name)

    @property
    def path(self) -> Path:
        return profile_path(self.voice.name)


SECTION_HELP = {
    "voice": "Identity of the voice being trained.",
    "dataset": "How one long recording becomes clips plus transcripts.",
    "audio": "Output audio format for the generated clips.",
    "model": "Model-side training overrides (architecture comes from "
    "voice.quality).",
    "data": "Datamodule settings: batching, splits, caching.",
    "trainer": "PyTorch Lightning trainer settings.",
    "finetune": "Starting point: a pretrained checkpoint, or from scratch.",
    "runtime": "Machine-specific settings. Not portable between boxes.",
    "export": "ONNX export and the inference defaults baked into the config.",
}


# --------------------------------------------------------------------------
# Walking the schema
# --------------------------------------------------------------------------


def iter_specs(obj: Any, trail: str = "") -> Iterator[tuple[str, Spec, Any]]:
    """Yield ``(dotted.path, Spec, current_value)`` for every leaf field."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        path = f"{trail}.{f.name}" if trail else f.name
        if dataclasses.is_dataclass(value):
            yield from iter_specs(value, path)
            continue
        field_spec = f.metadata.get("spec")
        if field_spec is not None:
            yield path, field_spec, value


def get_path(profile: Profile, dotted: str) -> Any:
    target: Any = profile
    for part in dotted.split("."):
        target = getattr(target, part)
    return target


def set_path(profile: Profile, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target: Any = profile
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def comment_map(profile: Profile | None = None) -> dict[str, str]:
    """Dotted path -> comment text, for the YAML writer."""
    reference = profile or Profile()
    comments = dict(SECTION_HELP)
    for path, field_spec, _ in iter_specs(reference):
        text = field_spec.help
        bits: list[str] = []
        if field_spec.choices:
            bits.append("one of: " + ", ".join(str(c) for c in field_spec.choices))
        if field_spec.minimum is not None or field_spec.maximum is not None:
            low = "" if field_spec.minimum is None else str(field_spec.minimum)
            high = "" if field_spec.maximum is None else str(field_spec.maximum)
            bits.append(f"range: {low}..{high}")
        if field_spec.unit:
            bits.append(f"unit: {field_spec.unit}")
        if bits:
            text = f"{text} ({'; '.join(bits)})"
        comments[path] = _wrap(text)
    return comments


def _wrap(text: str, width: int = 74) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > width:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + (1 if length else 0)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def to_dict(profile: Profile) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    return convert(profile)


_COERCERS = {
    "int": int,
    "float": float,
    "str": str,
    "path": str,
    "choice": lambda v: v,
    "list": list,
    "dict": dict,
}


def _coerce(value: Any, field_spec: Spec, path: str) -> Any:
    if value is None:
        if field_spec.kind in ("str", "path"):
            return ""
        return value
    if field_spec.kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "yes", "on", "1"):
            return True
        if text in ("false", "no", "off", "0"):
            return False
        raise ProfileError(f"{path}: expected a boolean, got {value!r}")
    coercer = _COERCERS.get(field_spec.kind, lambda v: v)
    try:
        return coercer(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{path}: cannot read {value!r} as {field_spec.kind}") from exc


def from_dict(data: dict[str, Any]) -> tuple[Profile, list[str]]:
    """Build a Profile, returning it plus a list of human-readable warnings."""
    if not isinstance(data, dict):
        raise ProfileError("profile must be a mapping at the top level")
    data = migrate(dict(data))
    profile = Profile()
    warnings: list[str] = []
    specs = {path: field_spec for path, field_spec, _ in iter_specs(profile)}

    def walk(node: Any, trail: str) -> None:
        if not isinstance(node, dict):
            warnings.append(f"ignoring {trail or '<root>'}: expected a mapping")
            return
        for key, value in node.items():
            path = f"{trail}.{key}" if trail else str(key)
            if path == "schema":
                continue
            if path in specs:
                set_path(profile, path, _coerce(value, specs[path], path))
            elif isinstance(value, dict) and any(
                p.startswith(path + ".") for p in specs
            ):
                walk(value, path)
            else:
                warnings.append(f"unknown setting {path!r} — ignored")

    walk(data, "")
    warnings += validate(profile)
    return profile, warnings


def validate(profile: Profile) -> list[str]:
    """Check values against their Spec bounds and choices.

    Warnings rather than errors: a hand-edited profile with one silly value
    should be visible, not fatal. The checks that would actually waste a
    training run live in ``train/argmap.py`` and do raise.
    """
    problems: list[str] = []
    for path, field_spec, value in iter_specs(profile):
        if value is None or isinstance(value, (dict, list)):
            continue
        if field_spec.choices and value not in field_spec.choices:
            rendered = ", ".join(repr(c) for c in field_spec.choices)
            problems.append(
                f"{path}: {value!r} is not one of {rendered} — expect this to "
                f"be rejected downstream"
            )
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if field_spec.minimum is not None and value < field_spec.minimum:
            problems.append(
                f"{path}: {value} is below the supported minimum "
                f"{field_spec.minimum}"
            )
        if field_spec.maximum is not None and value > field_spec.maximum:
            problems.append(
                f"{path}: {value} is above the supported maximum "
                f"{field_spec.maximum}"
            )
    return problems


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring an older profile up to SCHEMA_VERSION."""
    version = int(data.get("schema", SCHEMA_VERSION))
    if version > SCHEMA_VERSION:
        raise ProfileError(
            f"profile schema {version} is newer than this tool supports "
            f"({SCHEMA_VERSION}); update the repo"
        )
    # No migrations yet. Each future bump appends a block here and leaves the
    # earlier ones in place so old profiles keep working.
    data["schema"] = SCHEMA_VERSION
    return data


HEADER = """\
pipertrainer profile — safe to edit by hand.

Regenerate the comments at any time with:  ./run profile --refresh <name>
Every setting maps to a piper1-gpl flag; see docs/TRAINING.md for the table.
Settings under `runtime` describe this machine and are not portable.
"""


def save(profile: Profile, path: Path | None = None) -> Path:
    target = path or profile.path
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    yamlio.dump(
        to_dict(profile), target, comments=comment_map(profile), header=HEADER
    )
    return target


def load(path: Path) -> tuple[Profile, list[str]]:
    if not path.exists():
        raise ProfileError(f"no such profile: {path}")
    data = yamlio.load(path)
    if data is None:
        raise ProfileError(f"{path} is empty")
    return from_dict(data)


def load_by_name(name: str) -> tuple[Profile, list[str]]:
    return load(profile_path(name))


def list_profiles() -> list[Path]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(PROFILES_DIR.glob("*.yaml"))


def profile_names() -> list[str]:
    return [path.stem for path in list_profiles()]


# --------------------------------------------------------------------------
# Which profile the menu is currently pointed at
# --------------------------------------------------------------------------


def get_active() -> str | None:
    """Name of the profile the menu operates on, if one is selected."""
    if not ACTIVE_PROFILE.exists():
        return None
    name = ACTIVE_PROFILE.read_text(encoding="utf-8").strip()
    if not name or not profile_path(name).exists():
        return None
    return name


def set_active(name: str) -> None:
    ACTIVE_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE.write_text(slug(name) + "\n", encoding="utf-8", newline="\n")


def resolve_active(name: str | None = None) -> tuple[Profile, list[str]]:
    """Load ``name``, else the active profile, else the only one that exists."""
    if name:
        return load_by_name(name)
    active = get_active()
    if active:
        return load_by_name(active)
    names = profile_names()
    if len(names) == 1:
        return load_by_name(names[0])
    if not names:
        raise ProfileError(
            "no profiles yet — run './run dataset' or './run profile --new <name>'"
        )
    raise ProfileError(
        "several profiles exist; pass --profile <name> or select one in the menu: "
        + ", ".join(names)
    )
