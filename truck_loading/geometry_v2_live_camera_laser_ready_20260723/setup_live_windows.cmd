@echo off
setlocal

set "PYTHON_EXE="
if defined PALLET_POSE_PYTHON if exist "%PALLET_POSE_PYTHON%" set "PYTHON_EXE=%PALLET_POSE_PYTHON%"
if not defined PYTHON_EXE if exist "C:\Users\DELL\anaconda3\envs\pallet-pose\python.exe" set "PYTHON_EXE=C:\Users\DELL\anaconda3\envs\pallet-pose\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not defined PYTHON_EXE (
    echo Python was not found. Install 64-bit Python 3.10 first.
    pause
    exit /b 1
)

echo Using Python:
echo %PYTHON_EXE%

"%PYTHON_EXE%" -c "import torch,torchvision; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch CUDA 12.1...
    "%PYTHON_EXE%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 goto :failed
)

"%PYTHON_EXE%" -m pip install -r "%~dp0requirements-live.txt"
if errorlevel 1 goto :failed

echo.
echo Setup complete. Start run_live_windows.cmd.
exit /b 0

:failed
echo.
echo Setup failed.
pause
exit /b 1
