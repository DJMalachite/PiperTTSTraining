"""Subprocess execution with logging.

Every external command goes through here so that three things are always true:

* Output is streamed to the console *and* teed to a log file, because training
  runs for hours and the interesting error is never the last line.
* On failure the tail of the output is reprinted, so the user does not have to
  go find the log.
* Ctrl-C is delivered to the child and then waited on rather than killing us
  immediately — Lightning writes ``last.ckpt`` on SIGINT, and throwing that
  away because we exited first would cost the user their run.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from . import tui


class CommandFailed(RuntimeError):
    def __init__(self, result: "Result") -> None:
        self.result = result
        super().__init__(
            f"command failed with exit code {result.returncode}: {result.pretty}"
        )


@dataclass
class Result:
    argv: list[str]
    returncode: int
    lines: list[str] = field(default_factory=list)
    log_path: Path | None = None
    duration: float = 0.0
    interrupted: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    @property
    def pretty(self) -> str:
        return describe(self.argv)


def describe(argv: Sequence[str | Path]) -> str:
    """Shell-quoted form of a command, for logs and error messages."""
    return " ".join(shlex.quote(str(part)) for part in argv)


def _open_log(log_path: Path | None):
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "a", encoding="utf-8", errors="replace", newline="\n")


def run(
    argv: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    check: bool = True,
    quiet: bool = False,
    tail: int = 40,
    echo: bool = True,
    keep_lines: int | None = None,
) -> Result:
    """Run a command, streaming its combined output.

    ``quiet`` suppresses live output but still records it (and still prints the
    tail on failure). ``keep_lines`` caps retained output for commands that emit
    a lot — ``None`` keeps everything.
    """
    argv = [str(part) for part in argv]
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    handle = _open_log(log_path)
    started = time.monotonic()
    lines: list[str] = []
    interrupted = False

    if echo and not quiet:
        tui.hint(f"$ {describe(argv)}")
    if handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {describe(argv)}\n")
        handle.flush()

    try:
        # No start_new_session: we want the terminal's SIGINT to reach the child
        # directly, which is what lets Lightning checkpoint on Ctrl-C.
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise CommandFailed(
            Result(argv=argv, returncode=127, lines=[str(exc)], log_path=log_path)
        ) from exc

    assert process.stdout is not None
    try:
        for raw in process.stdout:
            line = raw.rstrip("\n")
            if not quiet:
                print(line)
            if handle:
                handle.write(line + "\n")
            lines.append(line)
            if keep_lines is not None and len(lines) > keep_lines:
                del lines[: len(lines) - keep_lines]
        process.wait()
    except KeyboardInterrupt:
        interrupted = True
        tui.warn("interrupted — waiting for the child to shut down cleanly")
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            tui.warn("child did not exit; terminating")
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    finally:
        if handle:
            handle.flush()
            handle.close()

    result = Result(
        argv=argv,
        returncode=process.returncode if process.returncode is not None else 130,
        lines=lines,
        log_path=log_path,
        duration=time.monotonic() - started,
        interrupted=interrupted,
    )

    if not result.ok and not interrupted:
        if quiet and lines:
            tui.error(f"command failed: {result.pretty}")
            for line in lines[-tail:]:
                print(f"  {line}")
        if log_path:
            tui.hint(f"  full log: {log_path}")
        if check:
            raise CommandFailed(result)
    elif not result.ok and check and interrupted:
        raise KeyboardInterrupt()

    return result


def capture(
    argv: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    timeout: float | None = 60.0,
) -> Result:
    """Run a command for its output only. Never streams, never raises on I/O."""
    argv = [str(part) for part in argv]
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        return Result(argv=argv, returncode=127, lines=[f"{argv[0]}: not found"])
    except subprocess.TimeoutExpired:
        return Result(argv=argv, returncode=124, lines=["timed out"])
    result = Result(
        argv=argv,
        returncode=completed.returncode,
        lines=(completed.stdout or "").splitlines(),
    )
    if check and not result.ok:
        raise CommandFailed(result)
    return result


def which(name: str) -> str | None:
    from shutil import which as _which

    return _which(name)


def tee_note(log_path: Path, text: str) -> None:
    """Append a plain note to a run log (used to record argv, env, versions)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip("\n") + "\n")


def stream_to_stderr(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()
