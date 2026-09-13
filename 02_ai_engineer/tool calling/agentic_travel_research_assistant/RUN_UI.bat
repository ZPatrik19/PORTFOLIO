@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run SETUP_AND_START_UI.bat once first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run "08_ui\app.py"
endlocal
