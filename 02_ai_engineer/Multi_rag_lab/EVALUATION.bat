@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
setlocal EnableExtensions
cd /d "%~dp0"

set "PROJECT_DIR=%CD%"
set "PY=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [INFO] A Python 3.14 kornyezet meg nincs kesz. SETUP.bat indul...
  call "%PROJECT_DIR%\SETUP.bat"
  if errorlevel 1 exit /b 1
)

:menu
cls
echo ==============================================================================
echo   MULTI-RAG EVALUATION / BENCHMARK
echo ==============================================================================
echo.
echo  [1] 80 kerdeses forrasolt evaluation dataset ujraepitese
echo  [2] QUICK retrieval benchmark - 20 kerdes
echo  [3] FULL retrieval benchmark - mind a 80 kerdes
echo  [4] QUICK RAG benchmark - 20 kerdes, Dummy LLM
echo  [5] RAG benchmark - 20 kerdes, Ollama/Qwen
echo  [6] QUICK teljes evaluation - retrieval + RAG
echo  [7] Experiment Registry osszefoglalo
echo  [8] Measurement suite - latency + robustness
echo  [0] Vissza
echo.
set /p "CHOICE=Valassz: "

if "%CHOICE%"=="1" goto :dataset
if "%CHOICE%"=="2" goto :retrieval_quick
if "%CHOICE%"=="3" goto :retrieval_full
if "%CHOICE%"=="4" goto :rag_dummy
if "%CHOICE%"=="5" goto :rag_ollama
if "%CHOICE%"=="6" goto :all_quick
if "%CHOICE%"=="7" goto :registry
if "%CHOICE%"=="8" goto :measurement
if "%CHOICE%"=="0" exit /b 0
goto :menu

:dataset
"%PY%" scripts\build_medical_eval_dataset.py --questions 80
pause
goto :menu

:retrieval_quick
"%PY%" scripts\evaluate_medical_rag.py --mode retrieval --questions 20
pause
goto :menu

:retrieval_full
"%PY%" scripts\evaluate_medical_rag.py --mode retrieval --questions 0
pause
goto :menu

:rag_dummy
"%PY%" scripts\evaluate_medical_rag.py --mode rag --questions 20 --llm-provider dummy
pause
goto :menu

:rag_ollama
echo [INFO] Ollama/Qwen infrastruktura ellenorzese...
call INFRASTRUCTURE.bat run balanced
if errorlevel 1 (
  echo [HIBA] Az Ollama/Qwen runtime nem kesz.
  pause
  goto :menu
)
"%PY%" scripts\evaluate_medical_rag.py --mode rag --questions 20 --llm-provider ollama
pause
goto :menu

:all_quick
"%PY%" scripts\evaluate_medical_rag.py --mode all --questions 20 --llm-provider dummy
pause
goto :menu

:registry
"%PY%" scripts\experiment_registry.py summary
pause
goto :menu

:measurement
"%PY%" scripts\run_measurement_suite.py --repeats 5 --warmup 3 --concurrency 1 2 4 8
pause
goto :menu
