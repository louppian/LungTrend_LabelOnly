@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist offline_packages\app (
    echo [ERROR] offline_packages\app not found.
    pause
    exit /b 1
)
if not exist offline_packages\torch-cu118 (
    echo [ERROR] offline_packages\torch-cu118 not found.
    pause
    exit /b 1
)
if exist offline_packages_torch (
    echo [ERROR] offline_packages_torch already exists. Aborting.
    pause
    exit /b 1
)
if exist offline_packages_legacy_tmp (
    echo [ERROR] offline_packages_legacy_tmp already exists. Aborting.
    pause
    exit /b 1
)

move offline_packages\app offline_packages_legacy_tmp
move offline_packages\torch-cu118 offline_packages_torch
rmdir offline_packages
move offline_packages_legacy_tmp offline_packages

echo Legacy offline package layout restored.
pause
