@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo ================================================
echo JP Manga Translator v12.1.3 - PP-OCRv6 manga model
echo ================================================
py -3.11 download_ppocr_manga_models.py
if errorlevel 1 echo MODEL DOWNLOAD FAILED.
pause
endlocal
