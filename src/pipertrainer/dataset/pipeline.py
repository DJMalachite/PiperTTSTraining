"""Dataset pipeline orchestration.

One audio or video file in, ``wavs/`` + ``metadata.csv`` + ``report.md`` out:

    probe -> decode -> macrosplit -> transcribe -> segment -> emit -> report

The expensive stages cache. Decoding writes a raw float32 file keyed on the
source file and the decode settings; transcription caches one JSON of words per
macro segment. So an interrupted run resumes, and changing only the segmentation
bounds re-cuts clips without re-transcribing anything.

The source file is read and never modified. That is a deliberate correction: the
predecessor script deleted the user's original recording after splitting it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .. import profile as profile_mod
from .. import tui
from ..paths import VoicePaths
from . import ffmpeg, macrosplit, metadata as meta, report as report_mod, textnorm
from . import segment as seg
from .transcribe import Transcriber, WhisperRules

STAGES = ("decode", "macrosplit", "transcribe", "segment", "emit")
MANIFEST_VERSION = 1


class PipelineError(RuntimeError):
    pass


@dataclass
class DatasetResult:
    report: report_mod.Report
    rows: list[meta.Row] = field(default_factory=list)
    rejected: list[meta.Rejected] = field(default_factory=list)
    wrote_clips: bool = False
    dry_run: bool = False


def _whisper_rules(prof: profile_mod.Profile) -> WhisperRules:
    whisper = prof.dataset.whisper
    return WhisperRules(
        model=whisper.model,
        device=whisper.device,
        language=whisper.language,
        initial_prompt=whisper.initial_prompt,
        condition_on_previous_text=whisper.condition_on_previous_text,
        temperature=tuple(float(t) for t in whisper.temperature) or (0.0,),
        beam_size=int(whisper.beam_size),
        fp16=whisper.fp16,
    )


def _segment_rules(prof: profile_mod.Profile) -> seg.SegmentRules:
    dataset = prof.dataset
    return seg.SegmentRules(
        min_seconds=float(dataset.min_seconds),
        target_seconds=float(dataset.target_seconds),
        max_seconds=float(dataset.max_seconds),
        boundary_gap=float(dataset.boundary_gap),
        pad_before=float(dataset.pad_before),
        pad_after=float(dataset.pad_after),
    )


def _text_rules(prof: profile_mod.Profile) -> textnorm.TextRules:
    text = prof.dataset.text
    return textnorm.TextRules(
        ensure_terminal_punctuation=text.ensure_terminal_punctuation,
        drop_bracketed=text.drop_bracketed,
        normalize_quotes=text.normalize_quotes,
        min_chars=int(text.min_chars),
        cps_min=float(text.cps_min),
        cps_max=float(text.cps_max),
    )


def resolve_input(prof: profile_mod.Profile, paths: VoicePaths) -> Path:
    """Find the source recording, preferring an explicit path."""
    configured = prof.dataset.input_path.strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate
        # A bare filename is looked up in the voice's source directory.
        in_source = paths.source / candidate.name
        if in_source.exists():
            return in_source
        raise PipelineError(
            f"dataset.input_path points at {candidate}, which does not exist"
        )

    if paths.source.exists():
        candidates = sorted(
            item
            for item in paths.source.iterdir()
            if item.is_file() and not item.name.startswith(".")
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise PipelineError(
                f"{paths.source} holds {len(candidates)} files; set "
                f"dataset.input_path to the one you want: "
                + ", ".join(item.name for item in candidates[:5])
            )
    raise PipelineError(
        f"no source audio. Set dataset.input_path, or drop a single file into "
        f"{paths.source}/"
    )


def run(
    prof: profile_mod.Profile,
    *,
    force_stages: Sequence[str] = (),
    dry_run: bool = False,
    offline: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> DatasetResult:
    paths = VoicePaths(prof.voice.name)
    paths.ensure_dataset_dirs()
    log_path = paths.data_root / "dataset.log"

    force = set(force_stages)
    if "all" in force:
        force = set(STAGES)
    unknown = force - set(STAGES)
    if unknown:
        raise PipelineError(
            f"unknown stage(s): {', '.join(sorted(unknown))}. Valid: "
            + ", ".join(STAGES)
        )

    say = on_status or tui.info
    sample_rate = int(prof.audio.sample_rate)
    strategy = prof.dataset.strategy
    dry_run = dry_run or prof.dataset.dry_run

    # ---- probe -----------------------------------------------------------
    source = resolve_input(prof, paths)
    info = ffmpeg.probe(source)
    tui.ok(f"source: {source.name} — {info.pretty}")
    if info.duration < 60:
        tui.warn(
            f"the recording is only {macrosplit.format_duration(info.duration)}; "
            f"expect a very small dataset"
        )

    # ---- decode ----------------------------------------------------------
    loudnorm = prof.audio.normalize == "loudnorm"
    highpass = int(prof.audio.highpass_hz)
    key = ffmpeg.decode_key(source, sample_rate, highpass, loudnorm)
    raw_path = paths.decoded_cache / f"{key}.f32"

    if "decode" in force and raw_path.exists():
        raw_path.unlink()
    if raw_path.exists():
        tui.ok(f"decoded audio cached ({raw_path.stat().st_size / 1e6:.0f} MB)")
    else:
        say(f"decoding to {sample_rate} Hz mono float32 (one ffmpeg pass)")
        for old in paths.decoded_cache.glob("*.f32"):
            old.unlink()  # only ever one decode per voice
        ffmpeg.decode_to_raw(
            source,
            raw_path,
            sample_rate=sample_rate,
            highpass_hz=highpass,
            loudnorm=loudnorm,
            log_path=log_path,
        )
        tui.ok(f"decoded ({raw_path.stat().st_size / 1e6:.0f} MB cached)")

    samples = ffmpeg.open_samples(raw_path)
    total_duration = ffmpeg.samples_to_seconds(len(samples), sample_rate)

    gain = 1.0
    if prof.audio.normalize == "peak":
        gain = ffmpeg.peak_gain(samples, float(prof.audio.peak_dbfs))
        say(
            f"peak normalisation: applying {20 * math.log10(gain):+.1f} dB to "
            f"reach {prof.audio.peak_dbfs} dBFS across the whole recording"
        )

    # ---- levels and silence ---------------------------------------------
    say("measuring levels")
    levels = macrosplit.frame_levels(samples, sample_rate)
    silence = macrosplit.estimate_silence(levels, float(prof.dataset.silence_dbfs))
    tui.ok(silence.describe())

    speech = macrosplit.speech_spans(
        levels,
        silence.threshold_dbfs,
        min_silence=float(prof.dataset.boundary_gap),
        total_duration=total_duration,
    )
    coverage = macrosplit.coverage(speech, total_duration)
    tui.ok(f"speech covers {coverage * 100:.0f}% of the recording")

    # ---- macrosplit ------------------------------------------------------
    segments = macrosplit.split(
        levels,
        total_duration=total_duration,
        silence=silence,
        min_silence_seconds=float(prof.dataset.macro_silence_seconds),
        max_seconds=float(prof.dataset.macro_max_seconds),
    )
    if len(segments) > 1:
        tui.ok(
            f"split into {len(segments)} chunks for transcription "
            f"(max {macrosplit.format_duration(float(prof.dataset.macro_max_seconds))} each)"
        )

    # ---- transcribe + segment -------------------------------------------
    rules = _segment_rules(prof)
    text_rules = _text_rules(prof)

    if strategy == "align":
        utterances, dropped, language, whisper_label = _align_strategy(
            prof,
            paths,
            samples,
            sample_rate,
            segments,
            rules,
            total_duration,
            force=force,
            offline=offline,
            say=say,
        )
    elif strategy == "vad":
        utterances, dropped, language, whisper_label = [], [], "", ""
        result = seg.spans_to_utterances(speech, rules, total_duration)
        utterances, dropped = result.utterances, result.dropped
        whisper_label = f"{prof.dataset.whisper.model} (per clip)"
    else:
        raise PipelineError(f"unknown strategy {strategy!r}")

    tui.ok(f"{len(utterances)} candidate clips, {len(dropped)} dropped while grouping")

    # ---- emit ------------------------------------------------------------
    if "emit" in force and paths.wavs.exists():
        for old in paths.wavs.glob("*.wav"):
            old.unlink()

    rows, stats, rejected = _emit(
        prof,
        paths,
        samples,
        sample_rate,
        utterances,
        text_rules,
        gain=gain,
        dry_run=dry_run,
        offline=offline,
        strategy=strategy,
        say=say,
    )

    for item in dropped:
        rejected.append(
            meta.Rejected(
                utt_id="-",
                reason=item.reason,
                text=item.text,
                start=item.start,
                end=item.end,
            )
        )

    # ---- write outputs ---------------------------------------------------
    if not dry_run:
        written = meta.write(rows, paths.metadata_csv)
        tui.ok(f"wrote {written} rows to {paths.metadata_csv.name}")
    else:
        tui.warn("dry run: no WAV files or metadata.csv were written")
    meta.write_rejected(rejected, paths.rejected_csv)

    report = report_mod.build(
        rows,
        stats,
        rejected,
        voice=prof.voice.name,
        source=str(source),
        source_duration=total_duration,
        sample_rate=sample_rate,
        strategy=strategy,
        speech_coverage=coverage,
        silence_note=silence.describe(),
        whisper_model=whisper_label,
        language=language,
        validation_split=float(prof.data.validation_split),
        num_test_examples=int(prof.data.num_test_examples),
        batch_size=int(prof.data.batch_size),
        min_clip_floor=float(prof.dataset.min_seconds),
    )
    report.write(paths.report_md)

    _write_manifest(
        paths,
        prof,
        source=source,
        info=info,
        segments=len(segments),
        clips=len(rows),
        dry_run=dry_run,
    )

    return DatasetResult(
        report=report,
        rows=rows,
        rejected=rejected,
        wrote_clips=not dry_run,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def _align_strategy(
    prof: profile_mod.Profile,
    paths: VoicePaths,
    samples,
    sample_rate: int,
    segments: Sequence[macrosplit.Segment],
    rules: seg.SegmentRules,
    total_duration: float,
    *,
    force: set[str],
    offline: bool,
    say: Callable[[str], None],
) -> tuple[list[seg.Utterance], list[seg.Dropped], str, str]:
    from .transcribe import clear_cache

    if "transcribe" in force:
        removed = clear_cache(paths.asr_cache)
        if removed:
            tui.warn(f"cleared {removed} cached transcript(s)")

    transcriber = Transcriber(
        _whisper_rules(prof), paths.asr_cache, offline=offline
    )

    started = time.monotonic()
    cached_count = 0

    def progress(position, total, segment, piece):
        nonlocal cached_count
        if piece.from_cache:
            cached_count += 1
        elapsed = time.monotonic() - started
        rate = position / elapsed if elapsed > 0 else 0
        remaining = (total - position) / rate if rate > 0 else 0
        marker = "cached" if piece.from_cache else f"{len(piece.words)} words"
        tui.info(
            f"  [{position}/{total}] "
            f"{macrosplit.format_duration(segment.start)}"
            f"-{macrosplit.format_duration(segment.end)}: {marker}"
            + (
                f"  (about {macrosplit.format_duration(remaining)} left)"
                if remaining > 30 and not piece.from_cache
                else ""
            )
        )

    say(f"transcribing {len(segments)} chunk(s) with Whisper {prof.dataset.whisper.model}")
    transcript = transcriber.transcribe_all(
        segments, samples, sample_rate, on_progress=progress
    )
    if cached_count:
        tui.ok(f"{cached_count} of {len(segments)} chunk(s) came from the cache")
    if not transcript.words:
        raise PipelineError(
            "Whisper produced no words. Check that the recording contains "
            "speech, and that dataset.whisper.language matches it."
        )

    result = seg.group_words(transcript.words, rules, total_duration)
    label = f"{prof.dataset.whisper.model} on {transcriber.device}"
    return result.utterances, result.dropped, transcript.language, label


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def _emit(
    prof: profile_mod.Profile,
    paths: VoicePaths,
    samples,
    sample_rate: int,
    utterances: Sequence[seg.Utterance],
    text_rules: textnorm.TextRules,
    *,
    gain: float,
    dry_run: bool,
    offline: bool,
    strategy: str,
    say: Callable[[str], None],
) -> tuple[list[meta.Row], list[report_mod.ClipStat], list[meta.Rejected]]:
    rows: list[meta.Row] = []
    stats: list[report_mod.ClipStat] = []
    rejected: list[meta.Rejected] = []
    snap = bool(prof.dataset.snap_zero_crossing)
    prefix = prof.dataset.id_prefix
    window = max(1, sample_rate // 1000)  # 1 ms search for a zero crossing

    transcriber = None
    if strategy == "vad" and not dry_run:
        transcriber = Transcriber(
            _whisper_rules(prof), paths.asr_cache, offline=offline
        )

    say(f"writing {len(utterances)} clip(s)")
    index = 0
    for utterance in utterances:
        start_sample = ffmpeg.seconds_to_samples(utterance.start, sample_rate)
        end_sample = ffmpeg.seconds_to_samples(utterance.end, sample_rate)
        if snap:
            start_sample = ffmpeg.find_zero_crossing(samples, start_sample, window)
            end_sample = ffmpeg.find_zero_crossing(samples, end_sample, window)
        if end_sample <= start_sample:
            continue

        seconds = ffmpeg.samples_to_seconds(end_sample - start_sample, sample_rate)
        index += 1
        utt_id = meta.clip_id(index, prefix)

        # The align strategy already has text; the vad strategy needs the clip
        # written first so Whisper can read it.
        raw_text = utterance.text
        clip_path = paths.wavs / f"{utt_id}.wav"

        if not dry_run:
            ffmpeg.write_clip(
                samples,
                clip_path,
                sample_rate=sample_rate,
                start_sample=start_sample,
                end_sample=end_sample,
                gain=gain,
            )
        if transcriber is not None:
            raw_text = transcriber.transcribe_clip(clip_path)

        normalized = textnorm.normalize(raw_text, text_rules)
        if not normalized.ok:
            rejected.append(
                meta.Rejected(
                    utt_id=utt_id,
                    reason=normalized.reason,
                    text=raw_text,
                    start=utterance.start,
                    end=utterance.end,
                )
            )
            if not dry_run and clip_path.exists():
                clip_path.unlink()
            index -= 1
            continue

        rate_problem = textnorm.check_rate(normalized.text, seconds, text_rules)
        if rate_problem:
            rejected.append(
                meta.Rejected(
                    utt_id=utt_id,
                    reason=rate_problem,
                    text=normalized.text,
                    start=utterance.start,
                    end=utterance.end,
                )
            )
            if not dry_run and clip_path.exists():
                clip_path.unlink()
            index -= 1
            continue

        if textnorm.looks_repetitive(normalized.text):
            rejected.append(
                meta.Rejected(
                    utt_id=utt_id,
                    reason="looks like a Whisper repetition loop",
                    text=normalized.text,
                    start=utterance.start,
                    end=utterance.end,
                )
            )
            if not dry_run and clip_path.exists():
                clip_path.unlink()
            index -= 1
            continue

        window_samples = samples[start_sample:end_sample]
        rows.append(meta.Row(utt_id=f"{utt_id}.wav", text=normalized.text))
        stats.append(
            report_mod.ClipStat(
                utt_id=utt_id,
                seconds=seconds,
                chars=len(normalized.text),
                text=normalized.text,
                peak_dbfs=ffmpeg.peak_dbfs(window_samples) + 20 * math.log10(gain),
                clipped_samples=ffmpeg.clipped_sample_count(window_samples),
            )
        )

    # Renumber so ids are contiguous after rejections; the cache key includes
    # the row number, so gaps are harmless but confusing.
    if not dry_run:
        rows, stats = _renumber(rows, stats, paths, prefix)

    return rows, stats, rejected


def _renumber(
    rows: list[meta.Row],
    stats: list[report_mod.ClipStat],
    paths: VoicePaths,
    prefix: str,
) -> tuple[list[meta.Row], list[report_mod.ClipStat]]:
    new_rows: list[meta.Row] = []
    new_stats: list[report_mod.ClipStat] = []
    for position, (row, stat) in enumerate(zip(rows, stats), start=1):
        target_id = meta.clip_id(position, prefix)
        if stat.utt_id != target_id:
            source = paths.wavs / f"{stat.utt_id}.wav"
            destination = paths.wavs / f"{target_id}.wav"
            if source.exists():
                source.replace(destination)
        new_rows.append(meta.Row(utt_id=f"{target_id}.wav", text=row.text))
        new_stats.append(
            report_mod.ClipStat(
                utt_id=target_id,
                seconds=stat.seconds,
                chars=stat.chars,
                text=stat.text,
                peak_dbfs=stat.peak_dbfs,
                clipped_samples=stat.clipped_samples,
            )
        )
    return new_rows, new_stats


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _write_manifest(
    paths: VoicePaths,
    prof: profile_mod.Profile,
    *,
    source: Path,
    info: ffmpeg.AudioInfo,
    segments: int,
    clips: int,
    dry_run: bool,
) -> None:
    stat = source.stat()
    payload = {
        "version": MANIFEST_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": dry_run,
        "source": {
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "duration": info.duration,
            "codec": info.codec,
        },
        "audio": {
            "sample_rate": int(prof.audio.sample_rate),
            "normalize": prof.audio.normalize,
            "peak_dbfs": float(prof.audio.peak_dbfs),
            "highpass_hz": int(prof.audio.highpass_hz),
        },
        "segmentation": {
            "strategy": prof.dataset.strategy,
            "min_seconds": float(prof.dataset.min_seconds),
            "target_seconds": float(prof.dataset.target_seconds),
            "max_seconds": float(prof.dataset.max_seconds),
            "boundary_gap": float(prof.dataset.boundary_gap),
        },
        "whisper": _whisper_rules(prof).cache_key(),
        "counts": {"macro_segments": segments, "clips": clips},
    }
    paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def read_manifest(paths: VoicePaths) -> dict | None:
    if not paths.manifest_json.exists():
        return None
    try:
        return json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
