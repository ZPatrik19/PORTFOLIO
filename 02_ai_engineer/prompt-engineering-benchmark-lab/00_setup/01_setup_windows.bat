@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

set "PYTHON_EXE="
set "PYTHON_ARGS="

echo ============================================================
echo  Prompt Engineering Benchmark Lab - Windows Setup
echo ============================================================
echo.
echo This bootstrap is idempotent and does NOT run a full benchmark.
echo.

call :find_python
if not defined PYTHON_EXE call :install_python
if not defined PYTHON_EXE (
  echo [ERROR] Python 3.10-3.14 was not found. Install a supported Python version and rerun.
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,15) else 1)" >nul 2>nul
  if errorlevel 1 rmdir /s /q .venv
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/7] Creating .venv...
  %PYTHON_EXE% %PYTHON_ARGS% -m venv .venv || goto :failed
) else (
  echo [1/7] Compatible .venv found.
)
set "VENV_PY=.venv\Scripts\python.exe"


echo [2/7] Bootstrapping pip/packaging...
%VENV_PY% -m pip --version >nul 2>nul || %VENV_PY% -m ensurepip --upgrade || goto :failed
%VENV_PY% -c "import packaging" >nul 2>nul || %VENV_PY% -m pip install "packaging>=24.0" || goto :failed


echo [3/7] Checking runtime + development dependencies...
%VENV_PY% 00_setup\00_dependency_manager.py --requirements requirements-dev.txt || goto :failed


echo [4/7] Installing the project package (editable)...
%VENV_PY% -m pip install -e . --no-deps --no-build-isolation || goto :failed
%VENV_PY% -c "import prompt_benchmark, pandas, sklearn, streamlit, plotly; print('      Imports: OK')" || goto :failed
%VENV_PY% -m compileall -q 03_src 05_scripts || goto :failed


echo [5/7] Checking benchmark data...
%VENV_PY% -c "from prompt_benchmark.paths import PATHS; import pandas as pd, sys; p=PATHS.processed_data/'benchmark.csv'; ok=p.exists(); f=pd.read_csv(p) if ok else pd.DataFrame(); required={'sample_id','text','true_label','case_type','difficulty','scenario_id'}; sys.exit(0 if ok and len(f)>=6000 and required.issubset(f.columns) else 1)" >nul 2>nul
if errorlevel 1 (
  echo       Preparing the local challenge dataset...
  %VENV_PY% 05_scripts\01_generate_mock_data.py || goto :failed
  %VENV_PY% 05_scripts\02_prepare_data.py --source mock || goto :failed
) else (
  echo       Existing compatible dataset found.
)


echo [6/7] Generating conceptual diagrams and running smoke test...
%VENV_PY% 05_scripts\08_generate_diagrams.py || goto :failed
%VENV_PY% 05_scripts\09_run_smoke_test.py || goto :failed


echo [7/7] Running fast unit/integration test suite...
%VENV_PY% -m pytest 04_tests -q || goto :failed


echo.
echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo UI: run_project.bat  ^(or legacy alias RUN_UI.bat^)
echo Tests: run_tests.bat
echo Full workflow: python 05_scripts\00_run_pipeline.py --provider mock --profile standard
echo.
exit /b 0

:find_python
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,15) else 1)" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,15) else 1)" >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=python"
    exit /b 0
  )
)
exit /b 0

:install_python
where winget >nul 2>nul
if errorlevel 1 exit /b 0
echo [Python] Installing Python 3.12 with winget...
winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 0
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
  exit /b 0
)
where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_EXE=py"
  set "PYTHON_ARGS=-3.12"
)
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Review the error above; project files were not deleted.
pause
exit /b 1
