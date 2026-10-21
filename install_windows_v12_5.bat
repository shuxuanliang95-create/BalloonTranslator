@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
echo ========================================
echo JP Manga Translator v12.5 dependencies
echo ========================================
py -3.11 -m pip install -U flask requests pillow numpy opencv-python onnxruntime pyclipper huggingface_hub py7zr rarfile
if errorlevel 1 (
  echo.
  echo 安装失败，请检查 Python 3.11 / pip。
  pause
  exit /b 1
)
echo.
echo 依赖安装完成。
echo 首次使用仍需确保 PP-OCRv6_manga 和 Anime/Manga LaMa 模型已经准备好。
pause
