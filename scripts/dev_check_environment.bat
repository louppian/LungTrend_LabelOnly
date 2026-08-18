@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"

set "ENVPY="
for %%p in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if exist "%%~p\envs\torch_ev\python.exe" set "ENVPY=%%~p\envs\torch_ev\python.exe"
)
if not defined ENVPY (
    echo [ERROR] torch_ev environment not found.
    pause
    exit /b 1
)

echo Using: %ENVPY%
echo.
echo ===== Python packages =====
"%ENVPY%" -c "import torch, gradio, cv2, pandas, pydicom; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('gradio:', gradio.__version__); print('cv2:', cv2.__version__)"
echo.
echo ===== Model weights =====
if exist assets\weights\DORGAFull_posthoc.pth (echo OK DORGAFull_posthoc.pth) else echo MISSING DORGAFull_posthoc.pth
if exist assets\weights\finetuned_9601.pt (echo OK finetuned_9601.pt) else echo MISSING finetuned_9601.pt
if exist assets\weights\stn_weights.pth (echo OK stn_weights.pth) else echo MISSING stn_weights.pth
pause
