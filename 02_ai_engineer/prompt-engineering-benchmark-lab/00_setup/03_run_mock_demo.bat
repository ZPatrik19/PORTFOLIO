@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" call 00_setup\01_setup_windows.bat
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe 05_scripts\00_run_pipeline.py --provider mock --source mock --profile standard --strategy all --force
if errorlevel 1 exit /b 1
echo Standard 120-sample mock demo complete. Open run_project.bat to inspect results.
endlocal
