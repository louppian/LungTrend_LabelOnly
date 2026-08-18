@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

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
if not exist offline_packages\app mkdir offline_packages\app
"%ENVPY%" -m pip download -r requirements\offline-app.txt -d offline_packages\app

REM Remove CPU-build torch wheels pulled in as dependencies (e.g. by timm).
REM A CPU wheel here could silently replace the CUDA build on the target PC.
del /q "offline_packages\app\torch-*.whl" 2>nul
del /q "offline_packages\app\torchvision-*.whl" 2>nul

REM CUDA torch wheels for a fresh PC (about 2.7GB), kept in a separate folder.
if not exist offline_packages\torch-cu118 mkdir offline_packages\torch-cu118
"%ENVPY%" -m pip download -r requirements\offline-torch-cu118.txt --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple -d offline_packages\torch-cu118

echo.
echo Download complete. Copy the whole offline_packages folder to the target PC.
pause
