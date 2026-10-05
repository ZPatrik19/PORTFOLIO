@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
setlocal EnableExtensions
cd /d "%~dp0"

set "PROJECT_DIR=%CD%"
set "PY=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [INFO] A .venv nem talalhato. SETUP.bat indul...
  call "%PROJECT_DIR%\SETUP.bat"
  if errorlevel 1 goto :fatal_setup
)

if not exist "%PY%" goto :fatal_setup

if "%~1"=="" (
  "%PY%" "%PROJECT_DIR%\scripts\infrastructure_cli.py" menu balanced
  set "RC=%ERRORLEVEL%"
  echo.
  echo Infrastructure menu bezarult. Exit code: %RC%
  echo A reszletes logok itt vannak:
  echo   %PROJECT_DIR%\logs\infrastructure
  echo.
  pause
  exit /b %RC%
)

"%PY%" "%PROJECT_DIR%\scripts\infrastructure_cli.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [HIBA] Infrastructure muvelet sikertelen. Exit code: %RC%
  echo Reszletes log:
  echo   %PROJECT_DIR%\logs\infrastructure
)
exit /b %RC%

:fatal_setup
echo.
echo [HIBA] A Python 3.14 kornyezet nem keszult el.
echo Futtasd kulon a SETUP.bat fajlt, majd probald ujra.
echo.
pause
exit /b 1
