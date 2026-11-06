@echo off
setlocal
cd /d "%~dp0"
title AutoClipper

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo  FFmpeg is not installed. Install it by running this in a terminal:
  echo.
  echo      winget install Gyan.FFmpeg
  echo.
  echo  Then close this window and double-click start.bat again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo  and tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

if not exist ".installed" (
  echo Installing dependencies. This takes a few minutes the first time...
  python -m pip install --upgrade pip -q
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Install failed. Scroll up to see the error.
    pause
    exit /b 1
  )
  echo done> .installed
) else (
  echo Checking for yt-dlp updates...
  python -m pip install -U "yt-dlp[default]" -q
)

if not exist ".env" copy ".env.example" ".env" >nul

where deno >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Tip: YouTube links work best with Deno installed: winget install DenoLand.Deno
  echo.
)

echo.
echo  AutoClipper is running at http://localhost:8000
echo  Keep this window open while you use it. Close it to stop.
echo.
start "" http://localhost:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
