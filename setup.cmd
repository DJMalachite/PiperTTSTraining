@echo off
rem Convenience alias so a fresh clone has one obvious command: setup
rem The POSIX twin is ./setup. Must stay CRLF — see .gitattributes.
@setlocal EnableExtensions
"%~dp0run.cmd" setup %*
exit /b %ERRORLEVEL%
