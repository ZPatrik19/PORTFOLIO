@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" call 00_setup\01_setup_windows.bat
if errorlevel 1 exit /b 1
where ollama >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Ollama executable was not found. Install Ollama and pull a model first.
  echo Example: ollama pull llama3.2:3b
  exit /b 1
)
.venv\Scripts\python.exe 05_scripts\03_check_provider.py --provider ollama || exit /b 1
.venv\Scripts\python.exe 05_scripts\00_run_pipeline.py --provider ollama --source mock --profile pilot --strategy all --force
endlocal
