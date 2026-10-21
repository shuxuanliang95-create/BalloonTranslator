import os, time, threading
import numpy as np
from PIL import Image

_MODEL = None
_DEVICE = None
_LOCK = threading.Lock()

MODEL_REPO = os.environ.get('MANGA_INPAINT_REPO', 'TareHimself/AnimeMangaInpainting-torchscript')
MODEL_FILE = os.environ.get('MANGA_INPAINT_FILE', 'anime-manga-big-lama.pt')
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, 'models', 'AnimeMangaInpainting')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, 'anime_manga_lama.pt')


def model_path():
    return MODEL_PATH


def is_ready():
    return (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000) or (os.path.exists(LEGACY_MODEL_PATH) and os.path.getsize(LEGACY_MODEL_PATH) > 10_000_000)


def load_model():
    global _MODEL, _DEVICE
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000_000 else LEGACY_MODEL_PATH
        if not (os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000):
            raise RuntimeError('本地 Manga AI 擦字模型尚未下载，请先运行 download_manga_inpaint_model.bat')
        try:
            import torch
        except Exception as e:
            raise RuntimeError('缺少 PyTorch。请先运行 install_windows_v12_1.bat。') from e
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Windows TorchScript/JIT can fail on non-ASCII paths. Copy the model to an ASCII-only
        # temporary path before loading, then keep the model in memory.
        import tempfile, shutil
        ascii_dir = os.path.join(tempfile.gettempdir(), 'jp_manga_models')
        os.makedirs(ascii_dir, exist_ok=True)
        ascii_path = os.path.join(ascii_dir, 'anime_manga_big_lama.pt')
        if os.path.abspath(model_path) != os.path.abspath(ascii_path):
            if (not os.path.exists(ascii_path) or os.path.getsize(ascii_path) != os.path.getsize(model_path)):
                shutil.copyfile(model_path, ascii_path)
        model = torch.jit.load(ascii_path, map_location=device).eval()
        if device.type == 'cuda':
            model = model.to(device)
        _MODEL, _DEVICE = model, device
        return _MODEL, _DEVICE


def inpaint(image: Image.Image, mask: np.ndarray, max_side=1280) -> Image.Image:
    """AI inpaint. mask: uint8 0 preserve / 255 replace."""
    import torch
    if mask.ndim != 2:
        raise ValueError('mask must be HxW')
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(f'mask size {mask.shape} != image {(h,w)}')
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image.convert('RGB').copy()

    # Keep enough context around text. Model works best when the surrounding drawing is visible.
    pad = max(24, int(min(w, h) * 0.035))
    x1=max(0,int(xs.min())-pad); y1=max(0,int(ys.min())-pad)
    x2=min(w,int(xs.max())+pad+1); y2=min(h,int(ys.max())+pad+1)
    crop=rgb[y1:y2,x1:x2]
    cmask=mask[y1:y2,x1:x2]
    ch,cw=crop.shape[:2]
    scale=min(1.0, float(max_side)/max(ch,cw))
    nw=max(8,int(round(cw*scale/8)*8)); nh=max(8,int(round(ch*scale/8)*8))
    if nw != cw or nh != ch:
        crop_i=Image.fromarray(crop).resize((nw,nh),Image.Resampling.LANCZOS)
        mask_i=Image.fromarray(cmask).resize((nw,nh),Image.Resampling.NEAREST)
    else:
        crop_i=Image.fromarray(crop); mask_i=Image.fromarray(cmask)
    arr=np.asarray(crop_i).astype(np.float32)/255.0
    marr=(np.asarray(mask_i).astype(np.float32)/255.0)
    # TorchScript model expects RGB + binary mask, NCHW.
    inp=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    m=torch.from_numpy(marr).unsqueeze(0).unsqueeze(0)
    model,device=load_model()
    inp=inp.to(device); m=m.to(device)
    with torch.inference_mode():
        out=model(inp,m).clamp(0,1)
    out_np=(out[0].permute(1,2,0).detach().float().cpu().numpy()*255.0).round().astype(np.uint8)
    out_i=Image.fromarray(out_np).resize((cw,ch),Image.Resampling.LANCZOS)
    # 关键修复：mask_i 是缩放后的尺寸，而输出/原图已经恢复为 cw×ch。
    # 先把 mask 恢复到原始 crop 尺寸，再进行像素融合，避免 (nh,nw,1) 与 (ch,cw,3) 广播错误。
    blend_mask=np.asarray(mask_i.resize((cw,ch),Image.Resampling.NEAREST)).astype(np.uint8)
    out_full=np.asarray(out_i.convert('RGB'))
    base=rgb[y1:y2,x1:x2].copy()
    blend_mask=(blend_mask>32)[...,None]
    merged=np.where(blend_mask,out_full,base).astype(np.uint8)
    result=Image.fromarray(rgb.copy())
    result.paste(Image.fromarray(merged),(x1,y1))
    return result
