@echo off
cd /d "%~dp0"

REM Find torch_ev env python directly - no "conda activate" needed.
REM (conda activate can kill the window instantly when run by double-click)
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
echo Using: %ENVPY%

REM 1) Install CUDA torch first (from bundled wheels, if present)
set "TORCH_WHEEL="
for %%f in (offline_packages_torch\torch-*.whl) do set "TORCH_WHEEL=%%f"
if defined TORCH_WHEEL (
    echo Installing CUDA torch from offline_packages_torch...
    "%ENVPY%" -m pip install --no-index --find-links offline_packages_torch torch==2.6.0+cu118 torchvision==0.21.0+cu118
    if errorlevel 1 (
        echo [ERROR] CUDA torch install failed. Aborting.
        pause
        exit /b 1
    )
) else (
    REM No bundled torch wheels - then torch must already be in the env.
    "%ENVPY%" -c "import torch" 2>nul
    if errorlevel 1 (
        echo [ERROR] torch is not installed and offline_packages_torch has no torch wheel.
        echo         On the online PC: run download_packages.bat, then copy the
        echo         offline_packages_torch folder here and run this file again.
        pause
        exit /b 1
    )
    echo torch already installed - skipping torch step.
)

REM 2) Install remaining packages
"%ENVPY%" -m pip install --no-index --find-links offline_packages -r requirements_extra.txt

echo.
echo ===== torch / CUDA check =====
"%ENVPY%" -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo If CUDA shows False, torch is a CPU build - stop and investigate.
echo.
echo Install complete. Run run.bat to start.
pause
