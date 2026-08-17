@echo off
cd /d "%~dp0"

REM Find torch_ev env python directly - no "conda activate" needed.
REM Using the env's own pip guarantees cp310 wheels are downloaded.
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

if not exist offline_packages mkdir offline_packages
"%ENVPY%" -m pip download -r requirements_extra.txt -d offline_packages

REM Remove CPU-build torch wheels pulled in as dependencies (e.g. by timm).
REM A CPU wheel here could silently replace the CUDA build on the target PC.
del /q "offline_packages\torch-*.whl" 2>nul
del /q "offline_packages\torchvision-*.whl" 2>nul

REM CUDA torch wheels for a fresh PC (about 2.7GB), kept in a separate folder.
if not exist offline_packages_torch mkdir offline_packages_torch
"%ENVPY%" -m pip download torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple -d offline_packages_torch

echo.
echo Download complete. Copy offline_packages and offline_packages_torch to target PC.
pause
