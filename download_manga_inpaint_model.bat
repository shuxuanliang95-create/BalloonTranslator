@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

echo ================================================
echo JP Manga Translator v12.1.4 - Manga AI model
echo Anime-Manga Big-LaMa, about 206 MB
echo Trying GitHub, Hugging Face, then HF mirror.
echo ================================================
py -3.11 download_manga_inpaint_model.py
if errorlevel 1 (
  echo.
  echo MODEL DOWNLOAD FAILED.
  echo Run this file again after checking network access.
)
pause
endlocal
