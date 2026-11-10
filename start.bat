@echo off
REM ============================================================
REM  TA-LNN / TALNet - one-click launcher (Windows)
REM  Double-click this file from inside the 'talnet' folder.
REM ============================================================
cd /d "%~dp0"

echo.
echo [TA-LNN] Installing dependencies (first run only)...
python -m pip install -r requirements.txt

echo.
echo [TA-LNN] Starting server at http://localhost:8000
echo [TA-LNN] Leave this window open. Press Ctrl+C to stop.
echo.
start "" http://localhost:8000
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

pause
