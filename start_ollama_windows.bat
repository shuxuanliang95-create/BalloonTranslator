@echo off
cd /d "%~dp0"
setlocal
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama was not found in PATH.
  echo Please install Ollama first.
  pause
  exit /b 1
)
ollama list | findstr /i "hy-mt2-7b" >nul
if errorlevel 1 (
  echo WARNING: hy-mt2-7b was not found in Ollama.
  echo The web app can still start, but translation will fail until the model exists.
)
if exist "C:\Python311\python.exe" (
  "C:\Python311\python.exe" app.py
) else (
  py -3.11 app.py
)
pause
