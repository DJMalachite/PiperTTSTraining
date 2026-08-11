"""Transcription with openai-whisper.

openai-whisper rather than faster-whisper because it runs on torch: the same
install that trains the model also transcribes, on ROCm and CUDA alike.
faster-whisper is quicker but CTranslate2 has no ROCm build, so on an AMD board
it would silently fall back to CPU.

Word timestamps are the point. The ``align`` segmentation strategy cuts clips at
word boundaries so each transcript describes exactly its own audio, which is the
failure mode that hurts VITS most.

Results are cached per macro segment as JSON with timestamps rebased to absolute
recording time, so an interrupted run resumes rather than re-transcribing hours
of audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .. import tui
from .macrosplit import Segment
from .segment import Word

CACHE_VERSION = 2


@dataclass
class WhisperRules:
    model: str = "turbo"
    device: str = "auto"
    language: str = "en"
    initial_prompt: str = ""
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    beam_size: int = 5
    fp16: str = "auto"

    def cache_key(self) -> dict[str, Any]:
        """The settings that change the transcript. Device is not one of them."""
        return {
            "version": CACHE_VERSION,
            "model": self.model,
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "temperature": list(self.temperature),
            "beam_size": self.beam_size,
        }


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    text: str = ""
    language: str = ""
    from_cache: bool = False


def resolve_device(requested: str) -> str:
    """'auto' -> cuda when torch can really use a GPU, else cpu.

    On ROCm the device is still spelled 'cuda' — that is what the ROCm build of
    torch calls itself.
    """
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        # is_available() is not proof on ROCm; a tiny real op is.
        probe = torch.zeros(8, device="cuda")
        _ = float((probe + 1).sum().item())
        torch.cuda.synchronize()
        return "cuda"
    except Exception:  # noqa: BLE001 - any failure means fall back to CPU
        return "cpu"


def resolve_fp16(setting: str, device: str) -> bool:
    if setting == "true":
        return True
    if setting == "false":
        return False
    # fp16 on CPU makes whisper warn and fall back; only enable it on GPU.
    return device == "cuda"


class Transcriber:
    """Loads the Whisper model once and transcribes segments against a cache."""

    def __init__(
        self,
        rules: WhisperRules,
        cache_dir: Path,
        *,
        offline: bool = False,
    ) -> None:
        self.rules = rules
        self.cache_dir = cache_dir
        self.offline = offline
        self.device = resolve_device(rules.device)
        self.fp16 = resolve_fp16(rules.fp16, self.device)
        self._model = None

    # -- model ------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Run './run setup "
                "--force-step whisper'."
            ) from exc

        if self.rules.model not in whisper.available_models():
            raise ValueError(
                f"unknown Whisper model {self.rules.model!r}. Available: "
                + ", ".join(whisper.available_models())
            )

        cached = Path.home() / ".cache" / "whisper" / f"{self.rules.model}.pt"
        if self.offline and not cached.exists():
            raise RuntimeError(
                f"offline mode: the Whisper model {self.rules.model!r} is not "
                f"cached at {cached}. Download it on a networked machine and "
                f"copy that file over."
            )

        tui.info(
            f"loading Whisper {self.rules.model} on {self.device}"
            + (" (fp16)" if self.fp16 else "")
        )
        self._model = whisper.load_model(self.rules.model, device=self.device)
        return self._model

    # -- cache ------------------------------------------------------------
    def _cache_path(self, segment: Segment) -> Path:
        return self.cache_dir / f"{segment.index:05d}.json"

    def _read_cache(self, segment: Segment) -> Transcript | None:
        path = self._cache_path(segment)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("key") != self.rules.cache_key():
            return None
        if abs(float(payload.get("start", -1)) - segment.start) > 0.01:
            return None
        words = [
            Word(text=item["t"], start=float(item["s"]), end=float(item["e"]))
            for item in payload.get("words", [])
        ]
        return Transcript(
            words=words,
            text=payload.get("text", ""),
            language=payload.get("language", ""),
            from_cache=True,
        )

    def _write_cache(self, segment: Segment, transcript: Transcript) -> None:
        path = self._cache_path(segment)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": self.rules.cache_key(),
            "start": segment.start,
            "end": segment.end,
            "language": transcript.language,
            "text": transcript.text,
            "words": [
                {"t": word.text, "s": round(word.start, 4), "e": round(word.end, 4)}
                for word in transcript.words
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n"
        )

    # -- transcription ----------------------------------------------------
    def transcribe_segment(
        self, segment: Segment, samples, sample_rate: int
    ) -> Transcript:
        cached = self._read_cache(segment)
        if cached is not None:
            return cached

        import numpy as np

        model = self._load()
        start_index = int(segment.start * sample_rate)
        end_index = min(len(samples), int(segment.end * sample_rate))
        audio = np.asarray(samples[start_index:end_index], dtype="f4")

        options: dict[str, Any] = {
            "word_timestamps": True,
            "verbose": None,
            "fp16": self.fp16,
            "condition_on_previous_text": self.rules.condition_on_previous_text,
            "temperature": tuple(self.rules.temperature),
            "beam_size": self.rules.beam_size,
        }
        if self.rules.language and self.rules.language != "auto":
            options["language"] = self.rules.language
        if self.rules.initial_prompt:
            options["initial_prompt"] = self.rules.initial_prompt

        result = model.transcribe(audio, **options)
        transcript = _to_transcript(result, offset=segment.start)
        self._write_cache(segment, transcript)
        return transcript

    def transcribe_all(
        self,
        segments: Sequence[Segment],
        samples,
        sample_rate: int,
        *,
        on_progress=None,
    ) -> Transcript:
        """Transcribe every segment and return one merged, absolute-time result."""
        merged = Transcript()
        languages: list[str] = []
        for position, segment in enumerate(segments, start=1):
            piece = self.transcribe_segment(segment, samples, sample_rate)
            merged.words.extend(piece.words)
            if piece.text:
                merged.text = (merged.text + " " + piece.text).strip()
            if piece.language:
                languages.append(piece.language)
            if on_progress:
                on_progress(position, len(segments), segment, piece)
        merged.language = languages[0] if languages else ""
        merged.words.sort(key=lambda word: word.start)
        return merged

    def transcribe_clip(self, path: Path) -> str:
        """Transcribe one finished clip. Used by the ``vad`` strategy."""
        model = self._load()
        options: dict[str, Any] = {
            "verbose": None,
            "fp16": self.fp16,
            "condition_on_previous_text": False,
            "temperature": tuple(self.rules.temperature),
            "beam_size": self.rules.beam_size,
        }
        if self.rules.language and self.rules.language != "auto":
            options["language"] = self.rules.language
        if self.rules.initial_prompt:
            options["initial_prompt"] = self.rules.initial_prompt
        result = model.transcribe(str(path), **options)
        return str(result.get("text", "")).strip()


def _to_transcript(result: dict[str, Any], offset: float) -> Transcript:
    """Flatten Whisper's segment/word structure, rebasing to absolute time."""
    words: list[Word] = []
    for segment in result.get("segments", []) or []:
        entries = segment.get("words") or []
        if entries:
            for entry in entries:
                text = str(entry.get("word", "")).strip()
                if not text:
                    continue
                start = float(entry.get("start", segment.get("start", 0.0)))
                end = float(entry.get("end", segment.get("end", start)))
                if end <= start:
                    continue
                words.append(Word(text=text, start=start + offset, end=end + offset))
        else:
            # word_timestamps can come back empty for a segment; keep the
            # segment itself so its text is not lost, treated as one long word.
            text = str(segment.get("text", "")).strip()
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            if text and end > start:
                words.append(Word(text=text, start=start + offset, end=end + offset))

    return Transcript(
        words=words,
        text=str(result.get("text", "")).strip(),
        language=str(result.get("language", "")),
    )


def clear_cache(cache_dir: Path) -> int:
    """Delete cached transcripts. Returns how many were removed."""
    if not cache_dir.exists():
        return 0
    removed = 0
    for path in cache_dir.glob("*.json"):
        path.unlink()
        removed += 1
    return removed
