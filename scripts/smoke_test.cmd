@echo off
rem Acceptance gate on Windows. The POSIX twin is scripts/smoke_test.sh.
rem Must stay CRLF — see .gitattributes.
rem
rem   scripts\smoke_test.cmd              every tier
rem   scripts\smoke_test.cmd unit         unit tests only (no venv needed)
rem   scripts\smoke_test.cmd --keep       leave the _smoke voice behind
rem
rem Tiers: unit (pure functions) -> dataset (synthetic clips) -> train (one CPU
rem epoch) -> export (ONNX plus a synthesized sentence).
@setlocal EnableExtensions
@cd /d "%~dp0.."

rem Captured before the parse loop on purpose: `shift` also shifts %0, so
rem %~dp0 stops referring to this script the moment an argument is consumed.
set "RUN=%~dp0..\run.cmd"

set "STAGE=all"
set "KEEP="

:parse
if "%~1"=="" goto :done
if /i "%~1"=="--keep" (set "KEEP=--keep") else ^
if /i "%~1"=="unit" (set "STAGE=unit") else ^
if /i "%~1"=="dataset" (set "STAGE=dataset") else ^
if /i "%~1"=="train" (set "STAGE=train") else ^
if /i "%~1"=="export" (set "STAGE=export") else ^
if /i "%~1"=="all" (set "STAGE=all") else (
    echo unknown argument: %~1 1>&2
    exit /b 2
)
shift
goto :parse

:done
echo running self-test ^(stage: %STAGE%^)
call "%RUN%" smoke --stage %STAGE% %KEEP%
exit /b %ERRORLEVEL%
