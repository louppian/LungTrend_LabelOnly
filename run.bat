@echo off
setlocal

cd /d "%~dp0"

set "ENV_NAME=torch_ev"

call :find_env_python
if not defined ENVPY (
    call :find_conda
    if defined CONDA_CMD (
        call :find_env_python_from_conda
        if not defined ENVPY call :find_env_python_by_conda_run
    )
)

if not defined ENVPY (
    echo [ERROR] %ENV_NAME% environment not found.
    echo         Run install.bat first.
    echo         If Anaconda/Miniconda is installed in a custom location, open Anaconda Prompt and run:
    echo             conda info --envs
    pause
    exit /b 1
)

echo Using: %ENVPY%

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860 "') do taskkill /PID %%a /F >nul 2>&1
"%ENVPY%" app.py
pause
exit /b 0

:find_env_python
set "ENVPY="
for %%p in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if exist "%%~p\envs\%ENV_NAME%\python.exe" set "ENVPY=%%~p\envs\%ENV_NAME%\python.exe"
)
exit /b 0

:find_conda
set "CONDA_CMD="
for %%p in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if exist "%%~p\condabin\conda.bat" set "CONDA_CMD=%%~p\condabin\conda.bat"
    if exist "%%~p\Scripts\conda.exe" set "CONDA_CMD=%%~p\Scripts\conda.exe"
)
for /f "delims=" %%c in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%c"
for /f "delims=" %%c in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%c"
exit /b 0

:find_env_python_from_conda
for /f "tokens=1,2,*" %%a in ('call "%CONDA_CMD%" env list 2^>nul ^| findstr /I /R /C:"^%ENV_NAME%[ *]"') do (
    if "%%~b"=="*" (
        if exist "%%~c\python.exe" set "ENVPY=%%~c\python.exe"
    ) else (
        if exist "%%~b\python.exe" set "ENVPY=%%~b\python.exe"
    )
)
exit /b 0

:find_env_python_by_conda_run
for /f "delims=" %%p in ('call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import sys; print(sys.executable)" 2^>nul') do (
    if exist "%%~p" set "ENVPY=%%~p"
)
exit /b 0
