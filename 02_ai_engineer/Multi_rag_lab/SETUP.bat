@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ============================================================================
rem Multi-RAG Engineering Lab - IDEMPOTENS Windows setup
rem Python 3.14 project runtime
rem - Meglevo kompatibilis .venv-et NEM torli.
rem - Fuggoseget csak uj venv / config valtozas / force eseten frissit.
rem - Infrastructure kulon lepes: INFRASTRUCTURE.bat setup
rem ============================================================================

set "PROJECT_DIR=%CD%"
set "STORAGE_ROOT=%PROJECT_DIR%"
set "VENV_DIR=%STORAGE_ROOT%\.venv"
set "CACHE_ROOT=%STORAGE_ROOT%\.cache"
set "PIP_CACHE_DIR=%CACHE_ROOT%\pip"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "HF_HUB_CACHE=%CACHE_ROOT%\huggingface\hub"
set "HF_HUB_DOWNLOAD_TIMEOUT=120"
set "HF_HUB_ETAG_TIMEOUT=30"
set "HF_XET_NUM_CONCURRENT_RANGE_GETS=4"
set "TRANSFORMERS_CACHE=%CACHE_ROOT%\huggingface\transformers"
set "SENTENCE_TRANSFORMERS_HOME=%CACHE_ROOT%\sentence-transformers"
set "TORCH_HOME=%CACHE_ROOT%\torch"
set "XDG_CACHE_HOME=%CACHE_ROOT%\xdg"
set "OLLAMA_MODELS=%PROJECT_DIR%\infrastructure\llm\models"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "FORCE_SETUP=0"
if /I "%~1"=="force" set "FORCE_SETUP=1"

cls
echo ==============================================================================
echo   Multi-RAG Engineering Lab - Python 3.14 SETUP
echo ==============================================================================
echo.
echo Projekt              : %PROJECT_DIR%
echo Virtual environment  : %VENV_DIR%
echo Cache                : %CACHE_ROOT%
echo Python               : 3.14 x64
echo.

for %%D in ("%CACHE_ROOT%" "%PIP_CACHE_DIR%" "%HF_HOME%" "%SENTENCE_TRANSFORMERS_HOME%" "%TORCH_HOME%" "%OLLAMA_MODELS%" "data\raw" "data\processed" "artifacts\indexes" "artifacts\evaluations" "artifacts\experiments" "logs") do (
  if not exist "%%~D" mkdir "%%~D"
)

if not exist ".env" if exist ".env.example" copy /Y ".env.example" ".env" >nul

echo [1/9] Python 3.14 virtual environment ellenorzese...
set "NEW_VENV=0"
if exist "%VENV_DIR%\Scripts\python.exe" (
  "%VENV_DIR%\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)" >nul 2>&1
  if not errorlevel 1 (
    echo       Meglevo Python 3.14 .venv hasznalhato - NEM hozzuk letre ujra.
    "%VENV_DIR%\Scripts\python.exe" --version
    goto :venv_ready
  )
  echo       A meglevo .venv nem Python 3.14 - uj kornyezet keszul.
  rmdir /s /q "%VENV_DIR%"
)

set "NEW_VENV=1"
call :create_venv "py -3.14" "3.14" && goto :venv_ready
call :create_venv "python" "3.14" && goto :venv_ready

echo.
echo [HIBA] Python 3.14 x64 nem talalhato.
echo Telepits Python 3.14-et, majd futtasd ujra a SETUP.bat fajlt.
echo A Python Launcher eseten ennek mukodnie kell: py -3.14 --version
echo.
pause
exit /b 1

:venv_ready
echo.
echo [2/9] Dependency fingerprint ellenorzese...
set "NEED_INSTALL=%NEW_VENV%"
if "%FORCE_SETUP%"=="1" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="0" (
  "%VENV_DIR%\Scripts\python.exe" scripts\setup_state.py check >nul 2>&1
  if errorlevel 10 set "NEED_INSTALL=1"
)
if "%NEED_INSTALL%"=="0" (
  echo       A projekt dependency/config allapota nem valtozott.
) else (
  echo       Uj vagy megvaltozott kornyezet - telepites/frissites indul.
)
echo.

if "%NEED_INSTALL%"=="1" (
  echo [3/9] pip / setuptools / wheel frissitese...
  "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
  if errorlevel 1 goto :fatal
  echo.
  echo [4/9] Projekt + dev + FAISS CPU fuggosegek telepitese...
  "%VENV_DIR%\Scripts\python.exe" -m pip install -e ".[dev,cpu]"
  if errorlevel 1 goto :fatal
) else (
  echo [3/9] Build tooling frissites kihagyva.
  echo [4/9] Dependency install kihagyva.
)
echo.

echo [5/9] Hugging Face embedding es reranker modellek elotoltese...
"%VENV_DIR%\Scripts\python.exe" scripts\prepare_runtime_assets.py --retries 5 --verify
if errorlevel 1 goto :fatal
echo.

echo [6/9] Kritikus importok ellenorzese...
"%VENV_DIR%\Scripts\python.exe" -c "import streamlit, plotly, requests, bs4, torch, sentence_transformers, faiss; print('       Streamlit / Torch / SentenceTransformers / FAISS: OK')"
if errorlevel 1 (
  echo       Egy kritikus csomag hianyzik - egyszeri javito install indul.
  "%VENV_DIR%\Scripts\python.exe" -m pip install -e ".[dev,cpu]"
  if errorlevel 1 goto :fatal
)
echo.

echo [7/9] CPU-kompatibilis regresszios tesztek...
"%VENV_DIR%\Scripts\python.exe" -m pytest -q -m "not gpu"
if errorlevel 1 (
  echo.
  echo [HIBA] Van sikertelen CPU-kompatibilis teszt.
  echo A projektet nem jeloljuk kesznek, amig ez nincs rendben.
  goto :fatal
)

echo.
echo [8/9] NVIDIA / CUDA gyorsitas ellenorzese...
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>&1 && set "HAS_NVIDIA=1"
if exist "%SystemRoot%\System32\nvidia-smi.exe" set "HAS_NVIDIA=1"
if "%HAS_NVIDIA%"=="0" (
  echo       NVIDIA GPU/driver nem lathato - CPU runtime marad aktiv.
) else (
  "%VENV_DIR%\Scripts\python.exe" scripts\infrastructure_cli.py cuda --install-cuda yes
  if errorlevel 1 (
    echo.
    echo [HIBA] NVIDIA GPU lathato, de a CUDA-s PyTorch runtime nem aktivalhato.
    echo Futtasd: INFRASTRUCTURE.bat majd valaszd a [6] Csak CUDA javitas menupontot.
    goto :fatal
  )
)
echo.

echo [9/9] Setup state mentese...
"%VENV_DIR%\Scripts\python.exe" scripts\setup_state.py write >nul
if errorlevel 1 goto :fatal

echo.
echo ==============================================================================
echo   SETUP KESZ - Python 3.14 virtual environment mukodik
echo ==============================================================================
echo Elso telepitesnel kovetkezo lepes:
echo   INFRASTRUCTURE.bat setup
echo.
echo Napi hasznalathoz:
echo   RUN.bat
echo.
echo Benchmark / evaluation:
echo   EVALUATION.bat
echo.
echo A gyokerben szandekosan csak negy felhasznaloi BAT van:
echo   SETUP.bat
echo   INFRASTRUCTURE.bat
echo   RUN.bat
echo   EVALUATION.bat
echo.
pause
exit /b 0

:create_venv
set "CANDIDATE=%~1"
set "LABEL=%~2"
%CANDIDATE% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)" >nul 2>&1 || exit /b 1
echo       Python %LABEL% megtalalva - uj .venv letrehozasa...
%CANDIDATE% -m venv "%VENV_DIR%" || exit /b 1
exit /b 0

:fatal
echo.
echo [HIBA] A setup egyik kotelezo lepese sikertelen.
echo A konzol nyitva marad, hogy a hiba olvashato legyen.
echo.
pause
exit /b 1
