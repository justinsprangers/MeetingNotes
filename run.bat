@echo off
rem Starts Meeting Notes. The first run sets up a private Python environment (a few minutes).
cd /d "%~dp0"
if not exist ".venv\setup-done" (
    echo First run: installing what Meeting Notes needs. This takes a few minutes...
    if not exist ".venv" (
        python -m venv .venv
        if errorlevel 1 (
            echo.
            echo Python 3.10 or newer is required. Get it from https://www.python.org/downloads/
            echo and tick "Add python.exe to PATH" during install.
            pause
            exit /b 1
        )
    )
    ".venv\Scripts\python" -m pip install --upgrade pip
    ".venv\Scripts\python" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Setup failed. Check your internet connection and run this again.
        pause
        exit /b 1
    )
    rem NVIDIA GPU found: add CUDA libraries for much faster, more accurate transcription.
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo NVIDIA GPU found. Installing GPU support, about 1 GB...
        ".venv\Scripts\python" -m pip install -r requirements-gpu.txt
    )
    echo done> ".venv\setup-done"
)
".venv\Scripts\python" app.py
pause
