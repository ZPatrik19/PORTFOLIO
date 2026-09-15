@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Technical Knowledge Intelligence
set "PY=.venv\Scripts\python.exe"
set "CMD=%~1"
set "ARG=%~2"
if "%CMD%"=="" set "CMD=run"

if /I "%CMD%"=="setup" goto setup
if /I "%CMD%"=="setup-run" goto setup_run
if /I "%CMD%"=="run" goto run
if /I "%CMD%"=="public" goto public
if /I "%CMD%"=="ingest" goto ingest
if /I "%CMD%"=="index" goto ingest
if /I "%CMD%"=="index-variants" goto index_variants
if /I "%CMD%"=="list-indexes" goto list_indexes
if /I "%CMD%"=="benchmark" goto benchmark
if /I "%CMD%"=="evaluate" goto evaluate
if /I "%CMD%"=="failure" goto failure
if /I "%CMD%"=="feedback" goto feedback
if /I "%CMD%"=="figures" goto figures
if /I "%CMD%"=="test" goto test
if /I "%CMD%"=="quality" goto quality
if /I "%CMD%"=="api" goto api
if /I "%CMD%"=="ui" goto ui
if /I "%CMD%"=="all" goto all

echo [ERROR] Unknown command: %CMD%
echo.
echo Usage:
echo   run_project.bat [run^|setup^|setup-run^|ingest^|benchmark^|test [profile]^|quality^|api^|ui^|all]
exit /b 2

:ensure_python
if exist "%PY%" exit /b 0

echo [INFO] Creating .venv...
py -3.12 -m venv .venv 2>nul || py -3.13 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || py -3.14 -m venv .venv 2>nul || py -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 (
  echo [ERROR] Could not create .venv. Install Python 3.10+ and ensure py/python is on PATH.
  exit /b 1
)
if not exist "%PY%" (
  echo [ERROR] .venv was created but Python is missing.
  exit /b 1
)
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 "%PY%" -m ensurepip --upgrade
if errorlevel 1 exit /b 1
exit /b 0

:install_environment
call :ensure_python || exit /b 1

echo [1/2] Project runtime dependencies...
"%PY%" -m pip install -e . --no-build-isolation
if errorlevel 1 (
  echo [ERROR] Project installation failed.
  exit /b 1
)

echo [2/2] Environment validation...
"%PY%" 00_setup\01_check_environment.py
if errorlevel 1 exit /b 1
"%PY%" -c "import tkip; print('[OK] tkip', tkip.__version__)"
exit /b %errorlevel%

:ensure_env
call :ensure_python || exit /b 1
"%PY%" -c "import tkip, streamlit, fastapi" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Environment is incomplete; repairing it...
  call :install_environment || exit /b 1
)
exit /b 0


:ensure_dev_environment
call :ensure_env || exit /b 1
"%PY%" -c "import pytest, pytest_cov, ruff, mypy, httpx" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing developer and test tooling...
  "%PY%" -m pip install -e ".[dev]" --no-build-isolation
  if errorlevel 1 (
    echo [ERROR] Developer/test dependency installation failed.
    exit /b 1
  )
)
exit /b 0

:setup
call :install_environment || exit /b 1
"%PY%" 00_setup\02_check_gemini_api.py
if errorlevel 1 echo [INFO] Gemini validation is optional; local retrieval remains available.
"%PY%" 00_setup\03_check_optional_services.py
exit /b 0

:setup_run
call :setup || exit /b 1
goto run

:run
call :ensure_env || exit /b 1
if not exist "01_data\indexes\numpy\chunks.json" call :ingest || exit /b 1
echo [INFO] Starting Streamlit UI...
"%PY%" 05_scripts\launch_app.py --ui-only
set "RUN_RC=%ERRORLEVEL%"
if not "%RUN_RC%"=="0" (
  echo.
  echo [ERROR] The application did not start correctly.
  echo Review the error above. This window will remain open.
  pause
)
exit /b %RUN_RC%

:public
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli download-public
exit /b %errorlevel%

:ingest
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli ingest
exit /b %errorlevel%

:index_variants
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli index-variants --strategies fixed recursive semantic
exit /b %errorlevel%

:list_indexes
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli list-indexes
exit /b %errorlevel%

:benchmark
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli benchmark
exit /b %errorlevel%

:evaluate
call :ensure_env || exit /b 1
"%PY%" 03_pipeline\run_generation_evaluation.py
exit /b %errorlevel%

:failure
call :ensure_env || exit /b 1
"%PY%" 03_pipeline\generate_failure_report.py
exit /b %errorlevel%

:feedback
call :ensure_env || exit /b 1
"%PY%" 03_pipeline\export_feedback_regression.py
exit /b %errorlevel%

:figures
call :ensure_env || exit /b 1
"%PY%" -m tkip.cli figures
exit /b %errorlevel%

:test
call :ensure_dev_environment || exit /b 1
set "PROFILE=%ARG%"
if "%PROFILE%"=="" set "PROFILE=offline"
if /I "%PROFILE%"=="live" echo [WARNING] Live tests consume real Gemini quota.
echo [INFO] Test profile: %PROFILE%
"%PY%" 06_tests\run_suite.py "%PROFILE%"
exit /b %errorlevel%

:quality
call :ensure_dev_environment || exit /b 1
"%PY%" -m ruff check . || exit /b 1
"%PY%" -m ruff format --check . || exit /b 1
"%PY%" -m mypy 03_pipeline\tkip || exit /b 1
"%PY%" 06_tests\run_suite.py offline
exit /b %errorlevel%

:api
call :ensure_env || exit /b 1
"%PY%" -m uvicorn 04_api.main:app --host 0.0.0.0 --port 8000
exit /b %errorlevel%

:ui
call :ensure_env || exit /b 1
"%PY%" -m streamlit run 05_ui\app.py --server.port 8501
exit /b %errorlevel%

:all
call :setup || exit /b 1
if not exist "01_data\reference_docs\demo_python.md" call :public || exit /b 1
call :ingest || exit /b 1
call :test || exit /b 1
call :benchmark || exit /b 1
call :evaluate || exit /b 1
call :failure || exit /b 1
call :feedback || exit /b 1
call :figures || exit /b 1
goto run
