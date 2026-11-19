@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "PYTHONPATH="
set "ENABLE_APSCHEDULER=false"
"%~dp0.venv313\Scripts\python.exe" app\deploy.py
exit /b %ERRORLEVEL%
