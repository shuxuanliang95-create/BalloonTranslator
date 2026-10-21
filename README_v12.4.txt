JP Manga Translator v12.4.2 - 本地 AI 擦字版

本版本基于 v12 漫画排版重构版。

核心流程：
- OCR：PP-OCRv6_manga v0.2，本地 CPU。
- 翻译：Ollama + Hy-MT2-7B，本地 RTX 3070。
- 擦字：Anime/Manga 专用 LaMa TorchScript，本地 AI。
- 排版：v12 的气泡整泡排版 + 底部正文面板整块排版。

v12.4.2 的主要变化：
- 不再使用 OpenCV Telea/NS 作为默认擦字器。
- OCR 产生的文字 mask 会交给本地 Anime/Manga LaMa 重建背景。
- 只把 AI 输出用于 mask 区域，mask 外的原始漫画保持不变，减少人物/衣服/背景被 AI 改坏。
- 气泡和正文都使用同一个本地 AI 擦字模型。
- 如果没有下载 AI 擦字模型，翻译页会直接提示先下载，不会偷偷使用传统擦字替代。

首次安装：
1) 运行 install_windows_v12_1.bat
2) 运行 download_ppocr_manga_models.bat（如果 v11 已经下载好，可跳过）
3) 运行 download_manga_inpaint_model.bat
4) 确认 Ollama 已经有 hy-mt2-7b
5) 运行 start_ollama_windows.bat
6) 打开 http://127.0.0.1:7860
7) 先用“v12.4.2 本地 AI 擦字单页测试”测试一张图片，再跑整本 CBZ。

AI 擦字模型：
Sanster/AnimeMangaInpainting（Anime/Manga Big-LaMa TorchScript）
约 200 MB，TorchScript，专门用于 anime/manga inpainting。
模型卡：https://huggingface.co/Sanster/AnimeMangaInpainting（Anime/Manga Big-LaMa TorchScript）

下载器会优先尝试 hf-mirror.com，再尝试官方 Hugging Face。
hf-mirror.com 是第三方镜像；如果你不想使用镜像，可把 download_manga_inpaint_model.py 中的 endpoint 顺序改为只使用官方源。

GPU：
- 如果 PyTorch 检测到 CUDA，会使用 NVIDIA GPU。
- 否则自动使用 CPU，但 AI 擦字会明显变慢。
- RTX 3070 建议使用 CUDA 版 PyTorch。

PP-OCRv6_manga：
https://huggingface.co/Kellenok/PP-OCRv6_manga


v12.4.2 修复：
- 修复多个对白/段落被合并成一个超大框的问题。
- 页面下半部不再自动视为一个正文大面板。
- 只有检测到真实浅色气泡边界的区域才会作为气泡处理。
- 正文只按相邻、同一水平带的文字栏局部分组。
- 修复本地 AI 擦字模型文件路径；默认使用 anime-manga-big-lama.pt，并兼容旧文件名。
- AI 擦字仍为本地 Anime/Manga Big-LaMa，不改翻译/OCR流程。


v12.4.3 changes:
- Bubble grouping is now based on local light/white bubble boundaries, not page Y position.
- Bottom speech bubbles remain independent; the whole bottom panel is no longer forced into one box.
- Body columns are only grouped in small clusters (max 3 columns / max 38% page width).
- Manga Big-LaMa downloader uses GitHub + Hugging Face + HF mirror fallback.
- Windows TorchScript loading copies the model to an ASCII-only temp path to avoid non-ASCII path failures.


修复：补回正文翻译分段函数 _split_body_para_item，解决“name '_split_body_para_item' is not defined”。


v12.4 changes:
- 重写 OCR 逻辑区域分组：不再用大范围亮色连通域推断气泡，避免人物/衣服/整页背景被误合并。
- 同一气泡仅基于 OCR 框的方向、距离、重叠与连接带进行保守合并。
- 真正的密集正文才合并为 body_para；保留原始 member_boxes 供擦字。
- 修复 Anime/Manga LaMa AI 擦字在缩放后 mask 尺寸不一致导致的 NumPy broadcast 错误。
- 补齐正文/横排排版函数，增加单页翻译第一条失败原因显示。
- 新增 preflight_v12_2.py，可检查内部函数漏打包问题。
