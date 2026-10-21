# -*- coding: utf-8 -*-
from pathlib import Path
import os, sys, time
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent / 'models' / 'PP-OCRv6_manga'
REPO = 'Kellenok/PP-OCRv6_manga'
FILES = [
    'det/manga_det_v0.2.onnx',
    'rec/manga_rec_v0.2.onnx',
    'ppocrv6_dict.txt',
]

# The previous installer could appear frozen when the official HF endpoint/Xet
# connection was unreachable. Try the mirror first, then official HF.
ENDPOINTS = [
    'https://hf-mirror.com',
    'https://huggingface.co',
]


def ok(path):
    return path.exists() and path.stat().st_size > 1000

for f in FILES:
    local = ROOT / f
    if ok(local):
        print(f'[已存在] {f}  ({local.stat().st_size/1024/1024:.2f} MB)')
        continue
    local.parent.mkdir(parents=True, exist_ok=True)
    last = None
    done = False
    for ep in ENDPOINTS:
        print('\n' + '='*70)
        print(f'下载: {f}')
        print(f'来源: {ep}')
        print('如果 30 秒没有任何变化，会自动换下一个下载源。')
        print('='*70)
        try:
            t0=time.time()
            # tqdm from huggingface_hub prints byte progress when available.
            got = hf_hub_download(
                repo_id=REPO,
                filename=f,
                local_dir=str(ROOT),
                endpoint=ep,
                etag_timeout=10,
                force_download=False,
            )
            if ok(local):
                print(f'[完成] {f} -> {local}')
                print(f'大小: {local.stat().st_size/1024/1024:.2f} MB, 用时 {time.time()-t0:.1f}s')
                done=True
                break
            else:
                raise RuntimeError('下载后文件不存在或大小异常')
        except Exception as e:
            last=e
            print(f'[失败] {ep}: {type(e).__name__}: {e}')
    if not done:
        print('\n下载失败。你可以把上面的错误截图给我。')
        sys.exit(2)

print('\nPP-OCRv6_manga 模型全部准备完成。')
print('现在可以运行 start_ollama_windows.bat。')
