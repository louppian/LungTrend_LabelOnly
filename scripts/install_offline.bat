@echo off
setlocal

set "ROOT=%~dp0.."
set "ENV_NAME=torch_ev"
set "PYTHON_VERSION=3.10"
set "APP_WHEEL_DIR=offline_packages\app"
set "TORCH_WHEEL_DIR=offline_packages\torch-cu118"
set "APP_REQ=requirements\offline-app.txt"
set "TORCH_REQ=requirements\offline-torch-cu118.txt"

cd /d "%ROOT%"

call :print_header
call :ensure_environment || goto :fail
call :install_cuda_torch || goto :fail
call :install_app_packages || goto :fail
call :check_cuda || goto :fail

echo.
echo Install complete. Run run.bat to start.
pause
exit /b 0

:print_header
echo ===== CXR GUI offline installer =====
echo Project: %CD%
echo Environment: %ENV_NAME% ^(Python %PYTHON_VERSION%^)
echo.
exit /b 0

:ensure_environment
call :find_env_python
if not defined ENVPY (
    call :find_conda
    if defined CONDA_CMD (
        call :find_env_python_from_conda
        if not defined ENVPY call :find_env_python_by_conda_run
    )
)
if defined ENVPY (
    echo [1/4] Found existing %ENV_NAME% environment.
    echo       Using: %ENVPY%
    echo.
    exit /b 0
)

echo [1/4] %ENV_NAME% environment not found.
echo       Looking for conda...
if not defined CONDA_CMD call :find_conda
if not defined CONDA_CMD (
    echo [ERROR] conda was not found.
    echo         Install Anaconda or Miniconda first, then run install.bat again.
    echo         The installer will create this environment automatically:
    echo             conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    exit /b 1
)

echo       Found conda: %CONDA_CMD%
echo       Creating %ENV_NAME% environment...
call "%CONDA_CMD%" create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if errorlevel 1 (
    echo [ERROR] Failed to create %ENV_NAME% environment.
    exit /b 1
)

call :find_env_python
if not defined ENVPY call :find_env_python_from_conda
if not defined ENVPY call :find_env_python_by_conda_run
if not defined ENVPY (
    echo [ERROR] %ENV_NAME% was created, but python.exe could not be found.
    exit /b 1
)

echo       Created: %ENVPY%
echo.
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

:install_cuda_torch
echo [2/4] Installing CUDA torch...
set "TORCH_WHEEL="
for %%f in ("%TORCH_WHEEL_DIR%\torch-*.whl") do if exist "%%~f" set "TORCH_WHEEL=%%~f"

if defined TORCH_WHEEL (
    "%ENVPY%" -m pip install --no-index --find-links "%TORCH_WHEEL_DIR%" -r "%TORCH_REQ%"
    if errorlevel 1 (
        echo [ERROR] CUDA torch install failed.
        exit /b 1
    )
    echo.
    exit /b 0
)

echo       No bundled torch wheel found in %TORCH_WHEEL_DIR%.
echo       Checking whether torch is already installed...
"%ENVPY%" -c "import torch" 2>nul
if errorlevel 1 (
    echo [ERROR] torch is not installed and %TORCH_WHEEL_DIR% has no torch wheel.
    echo         On the online PC: run scripts\dev_download_offline_packages.bat, then copy
    echo         the whole offline_packages folder here and run install.bat again.
    exit /b 1
)

echo       torch already installed - skipping torch step.
echo.
exit /b 0

:install_app_packages
echo [3/4] Installing app packages...
"%ENVPY%" -m pip install --no-index --find-links "%APP_WHEEL_DIR%" -r "%APP_REQ%"
if errorlevel 1 (
    echo [ERROR] App package install failed.
    exit /b 1
)
echo.
exit /b 0

:check_cuda
echo [4/4] Checking torch / CUDA...
"%ENVPY%" -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] torch import check failed.
    exit /b 1
)
echo       If CUDA shows False, torch is a CPU build - stop and investigate.
exit /b 0

:fail
echo.
echo Install failed. See the message above.
pause
exit /b 1
