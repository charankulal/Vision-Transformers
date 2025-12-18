@echo off
REM Helper script for Windows users to upload to Kaggle

echo ===============================================================
echo Vision Transformer - Kaggle Upload (Windows)
echo ===============================================================
echo.

REM Check if username is provided
if "%1"=="" (
    echo Usage: upload_to_kaggle.bat YOUR_KAGGLE_USERNAME [--update]
    echo.
    echo Examples:
    echo   upload_to_kaggle.bat john_doe
    echo   upload_to_kaggle.bat john_doe --update
    echo.
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    exit /b 1
)

REM Run the Python script
if "%2"=="--update" (
    python upload_to_kaggle.py --username %1 --update
) else (
    python upload_to_kaggle.py --username %1
)
