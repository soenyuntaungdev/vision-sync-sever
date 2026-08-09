@echo off
title VisionSync AI Backend
echo ===================================================
echo   Starting VisionSync FastAPI Backend Server
echo ===================================================
echo.

cd /d "%~dp0"
python -m pip install -r requirements.txt
python main.py

pause
