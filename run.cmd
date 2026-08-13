@echo off
rem Windows entry point for everything in this repo. The POSIX twin is ./run.
rem
rem   run              interactive menu
rem   run setup        install toolchain + piper1-gpl
rem   run doctor       diagnose the environment
rem   run --help       full command list
rem
rem PowerShell resolves `./run` and `.\run` to this file, so instructions
rem written as ./run are correct there. In cmd.exe, type `run`.
rem
rem Prefers the project venv; falls back to a system interpreter only so that a
rem fresh clone can bootstrap itself. Anything past `setup` needs the venv.
rem
rem Written as a flat sequence of gotos rather than a loop over versions: batch
rem expands %ERRORLEVEL% when a block is *parsed*, so reading an exit code
rem inside a for loop silently reports the wrong one.
@setlocal EnableExtensions
@cd /d "%~dp0"

rem PIPERTRAINER_ENV names a second installed environment: .venv-<name> and
rem .state-<name>. Not validated here — batch cannot express the rule cheaply,
rem so paths.env_name_problem() rejects a bad name with the same message the
rem POSIX ./run prints. An invalid name simply misses the venv below and falls
rem through to the interpreter that reports it.
set "SUFFIX="
if defined PIPERTRAINER_ENV set "SUFFIX=-%PIPERTRAINER_ENV%"

rem The environment setup discovered (e.g. HSA_OVERRIDE_GFX_VERSION) lives in
rem .state/env.json and is applied by pipertrainer.__main__.

if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0src"
)
rem Unbuffered so streamed subprocess output interleaves correctly in a log.
set "PYTHONUNBUFFERED=1"

set "VENVPY=%~dp0.venv%SUFFIX%\Scripts\python.exe"
if exist "%VENVPY%" goto :run

rem No venv yet: find a bootstrap interpreter. The py launcher is the only
rem reliable way to ask for a specific version on Windows; plain `python` is
rem tried last because it may be the Microsoft Store stub.
set "VENVPY=py"
py -3.13 -c "" >nul 2>&1 && set "PYARGS=-3.13" && goto :run
py -3.12 -c "" >nul 2>&1 && set "PYARGS=-3.12" && goto :run
py -3.11 -c "" >nul 2>&1 && set "PYARGS=-3.11" && goto :run
py -3 -c "" >nul 2>&1 && set "PYARGS=-3" && goto :run

set "VENVPY=python"
set "PYARGS="
python -c "" >nul 2>&1 && goto :run

echo error: no python interpreter found ^(need 3.11+^) 1>&2
echo        install one from python.org, or: winget install Python.Python.3.13 1>&2
exit /b 1

:run
"%VENVPY%" %PYARGS% -m pipertrainer %*
exit /b %ERRORLEVEL%
