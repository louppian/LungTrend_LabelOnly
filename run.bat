@echo off
cd /d "%~dp0"

REM Find torch_ev env python directly - no "conda activate" needed.
set "ENVPY="
for %%p in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if exist "%%~p\envs\torch_ev\python.exe" set "ENVPY=%%~p\envs\torch_ev\python.exe"
)
if not defined ENVPY (
    echo [ERROR] torch_ev environment not found.
    echo         Create it first in Anaconda Prompt:
    echo             conda create -n torch_ev python=3.10 -y
    pause
    exit /b 1
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860 "') do taskkill /PID %%a /F >nul 2>&1
"%ENVPY%" app.py
pause
