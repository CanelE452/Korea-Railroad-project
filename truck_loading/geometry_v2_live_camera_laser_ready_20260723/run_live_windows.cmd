@echo off
setlocal

set "PYTHON_EXE="
set "APP=%~dp0live_geometry_v2.py"

if defined PALLET_POSE_PYTHON if exist "%PALLET_POSE_PYTHON%" set "PYTHON_EXE=%PALLET_POSE_PYTHON%"
if not defined PYTHON_EXE if exist "%~dp0runtime\python.exe" set "PYTHON_EXE=%~dp0runtime\python.exe"
if not defined PYTHON_EXE if exist "C:\Users\DELL\anaconda3\envs\pallet-pose\python.exe" set "PYTHON_EXE=C:\Users\DELL\anaconda3\envs\pallet-pose\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"

if not defined PYTHON_EXE (
    echo Compatible Python was not found.
    echo Run setup_live_windows.cmd first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import torch,torchvision,cv2,serial,yacs,skimage" >nul 2>&1
if errorlevel 1 (
    echo Required Python packages are missing from:
    echo %PYTHON_EXE%
    echo Run setup_live_windows.cmd first.
    pause
    exit /b 1
)

set "PYTHONDONTWRITEBYTECODE=1"
"%PYTHON_EXE%" "%APP%" --camera-id 1 --laser-port COM6 --laser-baud 115200 %*
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo Live inference exited with code %APP_EXIT%.
    pause
)
exit /b %APP_EXIT%
