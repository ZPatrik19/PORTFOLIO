@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  call 00_setup\01_setup_windows.bat
  exit /b %errorlevel%
)
.venv\Scripts\python.exe -c "import packaging" >nul 2>nul || .venv\Scripts\python.exe -m pip install "packaging>=24.0"
.venv\Scripts\python.exe 00_setup\00_dependency_manager.py --requirements requirements-dev.txt || exit /b 1
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation || exit /b 1
.venv\Scripts\python.exe -m pytest 04_tests -q
endlocal
