@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.14 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

echo.
echo Checking installed requirements...
python 00_setup\06_check_dependencies.py
if errorlevel 1 (
  echo.
  echo Missing or incompatible packages detected. Installing project requirements once...
  python -m pip install -e ".[dev]"
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
  type nul > ".venv\.project_editable_installed"
) else (
  echo Requirements already satisfied. Skipping dependency installation and upgrades.
  if not exist ".venv\.project_editable_installed" (
    echo Registering the local project package without touching dependencies...
    python -m pip install -e . --no-deps --no-build-isolation
    if not errorlevel 1 type nul > ".venv\.project_editable_installed"
  ) else (
    echo Local project package already registered. Skipping editable reinstall.
  )
)

echo.
python 00_setup\02_setup_check.py --quick
if errorlevel 1 (
  pause
  exit /b 1
)

echo.
echo Checking whether ML router training is actually needed...
python 00_setup\07_prepare_if_needed.py
if errorlevel 1 (
  pause
  exit /b 1
)

echo.
echo Starting Streamlit UI...
python -m streamlit run "08_ui\app.py"
endlocal
