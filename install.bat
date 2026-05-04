@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

chcp 65001 >nul
title EPUB and SRT Translator - Bulletproof Installer
color 0A

echo.
echo ============================================================
echo EPUB and SRT Translator - Bulletproof Installation
echo ============================================================
echo.

:: ============================================================
:: PYTHON CHECK (3.10+)
:: ============================================================
echo [1/6] Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('py -3 --version 2^>^&1') do set PY_NUM=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_NUM!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if !PY_MAJOR! LSS 3 (
    echo [ERROR] Python !PY_NUM! is not supported. Python 3.10 or newer is required.
    pause
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo [ERROR] Python !PY_NUM! is not supported. Python 3.10 or newer is required.
    pause
    exit /b 1
)

echo [OK] Python !PY_NUM!
echo.

:: ============================================================
:: CLEAN & CREATE VENV
:: ============================================================
set "VENV_DIR=venv"
echo [2/6] Creating clean virtual environment...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

set "VENV_PY=%~dp0%VENV_DIR%\Scripts\python.exe"
echo.

:: ============================================================
:: GPU DETECTION
:: ============================================================
echo [3/6] Detecting NVIDIA GPU...
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

if !GPU_FOUND! EQU 1 (
    echo [OK] NVIDIA GPU detected.
    echo Driver major version: !DRIVER_MAJOR!
) else (
    echo [INFO] No NVIDIA GPU detected.
)
echo.

:: ============================================================
:: USER CHOICE
:: ============================================================
echo Choose installation type:
echo.
echo [1] CPU only (works everywhere)
echo [2] GPU (NVIDIA CUDA - recommended if you have GPU)
echo.
:CHOICE
set USER_CHOICE=
set /p USER_CHOICE=Enter choice [1/2]: 
if "%USER_CHOICE%"=="1" goto CPU_MODE
if "%USER_CHOICE%"=="2" goto GPU_MODE
echo Invalid choice. Try again.
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
    echo [WARNING] No NVIDIA GPU detected. Falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    goto INSTALL
)
if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
) else if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
) else if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
) else (
    echo [WARNING] Driver too old. Falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
)
goto INSTALL

:: ============================================================
:: INSTALL
:: ============================================================
:INSTALL
echo.
echo [4/6] Selected: !TORCH_VARIANT!
echo Index URL: !TORCH_INDEX_URL!
echo.

echo [5/6] Installing dependencies (without torch)...
"%VENV_PY%" -m pip install --upgrade pip --quiet

echo import sys > _filter.py
echo lines=open('requirements.txt').readlines() >> _filter.py
echo out=[l for l in lines if not l.strip().lower().startswith('torch')] >> _filter.py
echo open('_req.txt','w').writelines(out) >> _filter.py
"%VENV_PY%" _filter.py
del _filter.py 2>nul

"%VENV_PY%" -m pip install -r _req.txt
del _req.txt 2>nul
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo [OK] Base dependencies installed.

echo.
echo [6/6] Installing PyTorch...
"%VENV_PY%" -m pip install torch --index-url !TORCH_INDEX_URL!
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

:: ============================================================
:: VERIFICATION
:: ============================================================
echo.
echo Verifying installation...
echo.
echo import torch > _verify.py
echo print("Torch version :", torch.__version__) >> _verify.py
echo print("CUDA available:", torch.cuda.is_available()) >> _verify.py
echo if torch.cuda.is_available(): >> _verify.py
echo     print("GPU detected  :", torch.cuda.get_device_name(0)) >> _verify.py
"%VENV_PY%" _verify.py
del _verify.py 2>nul

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo ============================================================
echo PyTorch variant: !TORCH_VARIANT!
echo.
echo You can now run the application using launcher.vbs
echo.
pause
