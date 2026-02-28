@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title EPUB and SRT Translator - Installer
color 0A

echo.
echo  ============================================================
echo    EPUB and SRT Translator - Installer
echo  ============================================================
echo.

:: ============================================================
:: PYTHON CHECK
:: ============================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_NUM=%%v
echo  [OK] Python !PY_NUM!
echo.

:: ============================================================
:: VENV
:: ============================================================

set VENV_DIR=venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo  Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: ============================================================
:: GPU DETECTION
:: ============================================================

echo  Detecting NVIDIA GPU...

set GPU_FOUND=0
set DRIVER_MAJOR=0

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1

    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv 2^>nul ^| findstr /R "^[0-9]"') do (
        set DRIVER_MAJOR=%%a
        goto DRIVER_DONE
    )
)

:DRIVER_DONE

echo.

if !GPU_FOUND! EQU 1 (
    echo  NVIDIA GPU detected.
    echo  Driver major version: !DRIVER_MAJOR!
    echo  Recommended: GPU (CUDA)
) else (
    echo  No NVIDIA GPU detected.
    echo  Recommended: CPU
)

echo.
echo  Choose installation type:
echo.
echo   [1] CPU (works everywhere)
echo   [2] GPU (NVIDIA CUDA)
echo.

:CHOICE
set USER_CHOICE=
set /p USER_CHOICE= Enter choice [1/2]: 

if "%USER_CHOICE%"=="1" goto CPU_MODE
if "%USER_CHOICE%"=="2" goto GPU_MODE

echo Invalid choice.
goto CHOICE

:: ============================================================
:: CPU MODE
:: ============================================================

:CPU_MODE
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
goto INSTALL

:: ============================================================
:: GPU MODE
:: ============================================================

:GPU_MODE

if !GPU_FOUND! EQU 0 (
    echo.
    echo  [WARNING] No NVIDIA GPU detected.
    echo  Installing CPU version instead.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    goto INSTALL
)

if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    goto INSTALL
)

if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    goto INSTALL
)

if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    goto INSTALL
)

echo Driver too old — installing CPU.
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU fallback

:: ============================================================
:: INSTALL
:: ============================================================

:INSTALL

echo.
echo  Selected: %TORCH_VARIANT%
echo.

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo import sys > _filter.py
echo lines=open('requirements.txt').readlines() >> _filter.py
echo out=[l for l in lines if not l.strip().lower().startswith('torch')] >> _filter.py
echo open('_req.txt','w').writelines(out) >> _filter.py

"%VENV_DIR%\Scripts\python.exe" _filter.py
del _filter.py

"%VENV_DIR%\Scripts\python.exe" -m pip install -r _req.txt
del _req.txt

if errorlevel 1 (
    echo  Dependency install failed.
    pause
    exit /b 1
)

echo.
echo  Installing PyTorch...
echo  Index: %TORCH_INDEX_URL%
echo.

"%VENV_DIR%\Scripts\python.exe" -m pip install torch --index-url %TORCH_INDEX_URL%

if errorlevel 1 (
    echo  PyTorch install failed.
    pause
    exit /b 1
)

echo.
echo  Verifying installation...
echo.

echo import torch > _verify.py
echo print("Torch:", torch.__version__) >> _verify.py
echo print("CUDA available:", torch.cuda.is_available()) >> _verify.py
echo if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0)) >> _verify.py

"%VENV_DIR%\Scripts\python.exe" _verify.py
del _verify.py

echo.
echo  ============================================================
echo    INSTALL COMPLETE - Now you can run launcher.vbs
echo  ============================================================
echo    PyTorch variant : %TORCH_VARIANT%
echo.
pause