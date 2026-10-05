@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
setlocal EnableExtensions
cd /d "%~dp0"

set "PROJECT_DIR=%CD%"
set "PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=balanced"

if not exist "%PY%" (
  echo [INFO] A .venv meg nincs kesz. SETUP.bat indul...
  call "%PROJECT_DIR%\SETUP.bat"
  if errorlevel 1 goto :fatal
)
if not exist "%PY%" goto :fatal

"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,14) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [INFO] A .venv nem Python 3.14. SETUP.bat ujraepiti...
  call "%PROJECT_DIR%\SETUP.bat" force
  if errorlevel 1 goto :fatal
)

"%PY%" scripts\setup_state.py check >nul 2>&1
if errorlevel 10 (
  echo [INFO] A dependency/config fingerprint megvaltozott. SETUP frissites indul...
  call "%PROJECT_DIR%\SETUP.bat"
  if errorlevel 1 goto :fatal
)

if not exist ".env" if exist ".env.example" copy /Y ".env.example" ".env" >nul

set "CACHE_ROOT=%PROJECT_DIR%\.cache"
set "PIP_CACHE_DIR=%CACHE_ROOT%\pip"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "HF_HUB_CACHE=%CACHE_ROOT%\huggingface\hub"
set "HF_HUB_DOWNLOAD_TIMEOUT=120"
set "HF_HUB_ETAG_TIMEOUT=30"
set "HF_XET_NUM_CONCURRENT_RANGE_GETS=4"
set "SENTENCE_TRANSFORMERS_HOME=%CACHE_ROOT%\sentence-transformers"
set "TORCH_HOME=%CACHE_ROOT%\torch"
set "OLLAMA_MODELS=%PROJECT_DIR%\infrastructure\llm\models"

cls
echo ==============================================================================
echo   MULTI-RAG ENGINEERING LAB - Python 3.14
echo ==============================================================================
echo Profil: %PROFILE%
echo.
echo [1/2] Infrastruktur ellenorzese / inditasa...
call "%PROJECT_DIR%\INFRASTRUCTURE.bat" run "%PROFILE%"
if errorlevel 1 (
  echo.
  echo [HIBA] Az infrastruktura nem keszult el.
  echo A teljes log itt van:
  echo   %PROJECT_DIR%\logs\infrastructure
  echo.
  echo Nyisd meg az INFRASTRUCTURE.bat fajlt es valaszd a diagnosztikat.
  echo.
  pause
  exit /b 1
)

set "LLM_PROVIDER=ollama"
set "OLLAMA_PROFILE=%PROFILE%"
set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
set "VECTOR_DEVICE=cpu"
set "EMBEDDING_DEVICE=auto"
set "RERANKER_DEVICE=auto"
set "HF_HUB_OFFLINE=1"

echo.
echo [2/2] Streamlit inditasa...
start "Multi-RAG browser opener" /B "%PY%" "%PROJECT_DIR%\scripts\open_when_ready.py" http://localhost:8501
"%PY%" -m streamlit run "%PROJECT_DIR%\ui\app.py" --server.address=127.0.0.1 --server.port=8501 --server.headless=true
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [HIBA] A Streamlit leallt. Exit code: %RC%
  echo.
  pause
)
exit /b %RC%

:fatal
echo.
echo [HIBA] A RUN.bat nem tudja hasznalni a Python 3.14 virtual environmentet.
echo Futtasd kulon: SETUP.bat
echo.
pause
exit /b 1
