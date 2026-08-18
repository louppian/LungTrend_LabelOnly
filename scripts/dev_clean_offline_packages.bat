@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo Removing accidental CPU torch wheels from offline_packages\app...
del /q "offline_packages\app\torch-*.whl" 2>nul
del /q "offline_packages\app\torchvision-*.whl" 2>nul
echo Done.
pause
