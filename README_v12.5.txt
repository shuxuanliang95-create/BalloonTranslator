JP Manga Translator v12.5

本版重点：
1. 批量输入 ZIP / CBZ / 7Z / RAR。
2. 一键后台逐页翻译，网页实时显示总页数、已完成、失败/部分失败、当前页文件名、当前阶段和右侧当前完成图片预览。
3. 每完成一页就把处理后的图片写入最终 ZIP/CBZ；单页失败时保留原图并继续下一页。
4. 输出保持原图片扩展名：JPG/JPEG、PNG、WEBP；7Z/RAR 输入默认输出 ZIP。
5. 中文字体优先使用 Windows 的 Meiryo / MS Gothic，并根据原文区域墨迹密度自动选择常规/粗体；中文带细描边。
6. 每页完成后轻度增强对比度和锐度，使字幕与网点更清楚。
7. OCR / 区域分组 / AI 擦字继续沿用上一版已经验证的 v12.4 逻辑。

使用：
1. 运行 install_windows_v12_5.bat（第一次用或依赖缺失时）。
2. 确认 Ollama 中已有 hy-mt2-7b。
3. 确认 models/PP-OCRv6_manga 和 models/AnimeMangaInpainting 已经存在。
4. 启动原来的程序方式即可（项目主程序仍是 app.py，端口 7860）。
5. 先做 OCR 预览；确认区域后，第二步选择整本 ZIP/CBZ/7Z/RAR，点击“一键开始批量翻译并生成 ZIP”。
6. 批量过程中右侧预览会按页更新；完成后点击“下载完成的 ZIP / CBZ”。

注意：RAR 解压依赖 rarfile，并且某些 Windows 环境还需要本机可用的 unrar/bsdtar 后端；ZIP/CBZ/7Z 不需要额外解压程序。
