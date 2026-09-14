@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  call 00_setup\01_setup_windows.bat
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation >nul
.venv\Scripts\python.exe -m pytest 04_tests -v --cov=prompt_benchmark --cov-report=term-missing
exit /b %errorlevel%
