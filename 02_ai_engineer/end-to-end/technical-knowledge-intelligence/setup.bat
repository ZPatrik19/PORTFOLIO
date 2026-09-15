@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Technical Knowledge Intelligence - First Setup

echo ================================================================
echo  TECHNICAL KNOWLEDGE INTELLIGENCE - FIRST SETUP
echo ================================================================
echo.
echo This script prepares the local environment and starts the app.
echo After the first successful setup, use run_project.bat for daily use.
echo.

call "%~dp0run_project.bat" setup
if errorlevel 1 goto failed

echo.
echo [INFO] Setup finished. Starting the application...
call "%~dp0run_project.bat" run
if errorlevel 1 goto failed

exit /b 0

:failed
echo.
echo ================================================================
echo  SETUP / START FAILED
echo ================================================================
echo The window is kept open so the error above can be read.
echo Copy the error text if you need help diagnosing it.
echo.
pause
exit /b 1
