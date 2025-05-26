@echo off
echo ===============================================
echo Qwen 2.5VL OCR - Quick Start
echo ===============================================

cd /d "%~dp0"

if not exist ".venv\" (
    echo Virtual environment not found!
    echo Please run setup.py first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Available commands:
echo 1. Demo test (test model loading)
echo 2. Quick setup (create folders)
echo 3. Run OCR on input folder
echo 4. Help
echo.

set /p choice="Choose option (1-4): "

if "%choice%"=="1" (
    echo Running demo test...
    python demo_test.py
) else if "%choice%"=="2" (
    echo Running quick setup...
    python quick_setup.py
) else if "%choice%"=="3" (
    echo Running OCR batch processing...
    python batch_ocr.py input output
) else if "%choice%"=="4" (
    echo.
    echo Usage instructions:
    echo - Place your .tif images in the input folders
    echo - Run option 3 to process all images
    echo - Check output folder for results
    echo.
    python batch_ocr.py --help
) else (
    echo Invalid option selected.
)

echo.
pause
