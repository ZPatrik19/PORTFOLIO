@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call 00_setup\01_setup_windows.bat
  if errorlevel 1 exit /b 1
) else (
  .venv\Scripts\python.exe 00_setup\00_dependency_manager.py --requirements requirements-dev.txt
  if errorlevel 1 exit /b 1
  .venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation >nul
  if errorlevel 1 exit /b 1
)

if not exist ".env" (
  echo [INFO] .env is optional for Mock/Ollama. Cloud providers can be configured in the UI.
)

echo [START] http://localhost:8501
.venv\Scripts\python.exe -m streamlit run 05_scripts\11_ui_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=false
endlocal
