@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
echo Starting JP Manga Translator v12.6 on http://127.0.0.1:7860
py -3.11 app.py
pause
