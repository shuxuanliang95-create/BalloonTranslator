# INSTALL.md — BalloonTranslator v12.6 安装与测试指南

本文根据仓库当前的 v12.6 README、启动/安装脚本以及 Python 源码整理。主要面向 Windows 用户，使用 **Ollama + Hy-MT2-7B 本地翻译模型**。首次安装需要下载 Python 依赖和两个独立的视觉模型；模型文件不会因为克隆 GitHub 仓库而自动下载。

> **注意：** 本文记录的是代码和脚本显示的要求，不代表已在所有 Windows 电脑上完成实机测试。不同显卡、驱动和网络环境可能需要额外调整。

## 1. 本地 AI 组件概览

| 功能 | 当前项目使用的组件 | 位置/要求 |
|---|---|---|
| 日文文字识别（OCR） | PP-OCRv6_manga ONNX 模型 | `models/PP-OCRv6_manga/` |
| 日文转中文 | Ollama 中的 `hy-mt2-7b` 模型 | 本机 Ollama 服务，默认地址通常为 `http://127.0.0.1:11434` |
| 擦除原文字、修复背景 | Anime/Manga LaMa TorchScript 模型 | `models/AnimeMangaInpainting/anime-manga-big-lama.pt` |
| Web 界面与任务流程 | Python + Flask | 默认打开 `http://127.0.0.1:7860` |

这些组件是分开的：Ollama 负责翻译；OCR 和擦字修复模型由项目代码在本地调用。**仅安装 Ollama 不会自动准备 OCR 和 LaMa 模型。**

## 2. 基本要求

- Windows 电脑。
- Python 3.11 可通过命令 `py -3.11` 调用。
- 网络连接，用于安装 Python 依赖和下载模型文件。
- Ollama 已安装并能在命令行运行。
- 足够的磁盘空间存放依赖和模型；模型大小会因文件版本而异。
- GPU 并非所有步骤都必需：OCR 当前使用 ONNX Runtime 的 CPU provider；LaMa 代码会在 PyTorch CUDA 可用时使用 GPU，否则回退 CPU。Hy-MT2-7B 的速度和可运行性取决于 Ollama、模型量化方式、内存/显存及机器配置。

## 3. 获取项目

可以在 GitHub 仓库页面点击 **Code → Download ZIP**，解压到一个容易找到的目录，例如：

`C:\\BalloonTranslator`

也可以使用 Git：

```powershell
git clone https://github.com/shuxuanliang95-create/BalloonTranslator.git
cd BalloonTranslator
```

不要将项目放在权限受限的系统目录。LaMa 代码会将模型临时复制到系统临时目录下的 ASCII 路径，以规避部分 Windows/JIT 路径问题。

## 4. 安装 Python 依赖

在项目文件夹中双击：

`install_windows_v12_6.bat`

脚本会使用 `py -3.11` 安装 Flask、Requests、Pillow、NumPy、OpenCV、ONNX Runtime、PyClipper、Hugging Face Hub、py7zr 和 rarfile 等依赖。

也可以在项目目录打开命令提示符手动运行：

```powershell
py -3.11 --version
py -3.11 -m pip install -U flask requests pillow numpy opencv-python onnxruntime pyclipper huggingface_hub py7zr rarfile
```

### 关于 PyTorch

`manga_inpaint.py` 在加载 LaMa 擦字模型时需要 PyTorch；但当前 `install_windows_v12_6.bat` 中的 pip 安装命令没有安装 `torch`。如果首次运行时出现“缺少 PyTorch”或相关导入错误，需要根据电脑的 Windows、Python 版本和 CUDA/显卡情况，从 PyTorch 官方安装说明选择匹配的安装命令。不要直接照搬不匹配的 CUDA 命令。

PyTorch 官方安装说明：https://pytorch.org/get-started/locally/

## 5. 安装 Ollama 与翻译模型

1. 安装 Ollama：https://ollama.com/download
2. 打开命令提示符，确认命令可用：

```powershell
ollama --version
ollama list
```

3. 确认模型列表中存在项目预期的模型名 `hy-mt2-7b`。项目的 `README_v12.6.txt` 和启动脚本要求该模型已经在 Ollama 中准备好。
4. 如果你使用的模型仓库/模型包提供了可由 Ollama 使用的名称和安装步骤，请按该模型的官方说明导入或拉取，并确认 `ollama list` 显示的名称与 `hy-mt2-7b` 相符。**不要假定任意同名模型都来自可信来源。**

检查 Ollama 本地 API 是否可访问：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

项目会请求该本地地址检查模型。通常 Ollama 应保持运行；若本机 Ollama 服务没有启动，翻译将失败。

可选：项目提供 `start_ollama_windows.bat`，它会检查 Ollama 命令和模型名，然后启动 Web 应用。

## 6. 下载 OCR 模型

在项目目录双击：

`download_ppocr_manga_models.bat`

脚本会下载以下文件到 `models/PP-OCRv6_manga/`：

- `det/manga_det_v0.2.onnx`
- `rec/manga_rec_v0.2.onnx`
- `ppocrv6_dict.txt`

下载源配置在 `download_ppocr_manga_models.py` 中，当前会先尝试 Hugging Face 镜像，再尝试官方 Hugging Face 地址。

完成后检查上述三个文件是否存在。若下载失败，保留完整错误信息并检查网络连接后再重试。OCR 代码使用 ONNX Runtime，当前默认是 CPU provider；如果指定的 provider 不可用，代码会回退到 CPU。

## 7. 下载 LaMa 擦字修复模型

在项目目录双击：

`download_manga_inpaint_model.bat`

脚本会尝试从脚本列出的下载源获取 `anime-manga-big-lama.pt`，并放入：

`models/AnimeMangaInpainting/anime-manga-big-lama.pt`

下载脚本要求文件大于约 100 MB；运行时模型检查也要求模型文件大于 10 MB。下载完成后请确认文件存在且大小合理。不要把未下载完成的 `.part` 文件当作模型使用。

LaMa 需要 PyTorch。代码在 CUDA 可用时尝试使用 GPU，否则使用 CPU；CPU 运行可能明显较慢。

## 8. 启动程序

推荐在项目目录双击：

`start_windows_v12_6.bat`

脚本会运行：

```powershell
py -3.11 app.py
```

启动后，在同一台电脑的浏览器打开：

http://127.0.0.1:7860

如果端口已被占用或启动失败，请记录命令行中的完整错误信息。不要直接关闭窗口后就认定启动成功。

## 9. 第一次测试流程

1. 确认 Ollama 正在运行，且 `ollama list` 中有 `hy-mt2-7b`。
2. 确认 OCR 模型三个文件都已下载。
3. 确认 LaMa 模型文件已下载，且 PyTorch 可导入。
4. 启动 Web 应用并打开本地页面。
5. 选择一张自己有权处理的测试漫画图片，或项目支持的 ZIP/CBZ/7Z/RAR 压缩包。
6. 先执行 OCR；检查识别框和识别文本，必要时手动修改、合并或调整框。
7. 确认/锁定识别框后，再开始翻译和生成结果。
8. 检查中文翻译、文字大小与描边、背景修复、横排/竖排布局，以及输出文件是否正常。

建议首次只使用一张普通测试图片，确认完整流程正常后，再尝试批量处理。

## 10. 常见问题

### `py -3.11` 找不到命令
安装 Python 3.11，并确认 Python Launcher 已安装。可运行 `py -0p` 查看已安装版本。

### 安装 `torch` 或 `torchvision` 失败
先记录完整报错、Python 版本、Windows 版本和显卡/CUDA 信息。不要随意混装不同版本。根据实际设备选择 PyTorch 官方安装命令。

### OCR 提示模型文件缺失
重新运行 `download_ppocr_manga_models.bat`，确认检测模型、识别模型和字典文件都在 `models/PP-OCRv6_manga/` 下。

### 提示 LaMa 模型未下载
重新运行 `download_manga_inpaint_model.bat`，确认模型位于 `models/AnimeMangaInpainting/anime-manga-big-lama.pt` 且文件大小正常。

### Ollama / Hy-MT2 翻译失败
- 运行 `ollama list`，确认模型名存在。
- 在浏览器或 PowerShell 中检查 `http://127.0.0.1:11434/api/tags`。
- 确认 Ollama 服务正在运行，且机器有足够内存/显存。
- 检查启动日志中关于 Ollama HTTP 错误或模型不存在的提示。

### 页面无法打开
确认命令行显示应用已启动，并访问 `http://127.0.0.1:7860`。若端口冲突，记录完整日志后再排查。

### RAR 文件无法处理
项目依赖 `rarfile`；某些 RAR 文件还可能需要系统中额外安装兼容的解压程序。若只有 RAR 失败，请同时提供压缩包类型和错误日志；可先用 ZIP/CBZ 测试基本流程。

## 11. 报告问题时请提供

- BalloonTranslator 版本和 Git commit（如适用）。
- Windows 版本、Python 版本。
- `ollama --version` 和 `ollama list` 的相关输出（可隐藏无关模型名）。
- 失败步骤和完整错误日志。
- 若是图像质量问题，请提供经授权且不含私人内容的原图与结果图。

**请勿提交 API 密钥、访问令牌、密码、个人目录中的敏感信息或未经授权公开的漫画素材。** 日志和截图公开前请先检查并脱敏。

## 12. 已知文档/脚本注意事项

- 当前 v12.6 安装脚本没有显式安装 PyTorch；LaMa 模型运行时需要它。
- `README_v12.6.txt` 中的启动说明提到先确认两个视觉模型目录存在；这两个模型分别由独立下载脚本准备。
- 本指南没有替用户实际运行安装、下载模型或完整漫画处理流程。请在目标电脑上验证后，再将测试结果标记为通过。
