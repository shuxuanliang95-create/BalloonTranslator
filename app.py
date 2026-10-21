# -*- coding: utf-8 -*-
import os, io, json, zipfile, shutil, re, math, time, traceback, threading, uuid, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_file
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from PIL import ImageFilter
from manga_inpaint import inpaint as ai_inpaint, is_ready as ai_inpaint_ready
import numpy as np
import cv2
from ppocr_manga import PPOMangaOCR

try:
    import py7zr
except Exception:
    py7zr = None
try:
    import rarfile
except Exception:
    rarfile = None

APP = Flask(__name__)
WORK = Path(__file__).resolve().parent / "work"; WORK.mkdir(exist_ok=True)
PP_OCR = None
TASKS = {}
REVIEWS = {}
TASK_LOCK = threading.Lock()
IMAGE_EXTS={".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
FONT_SCALE=1.72

def _fs(n):
    return max(9, int(round(float(n) * FONT_SCALE)))


def get_pp_ocr():
    global PP_OCR
    if PP_OCR is None:
        PP_OCR=PPOMangaOCR()
    return PP_OCR

def box_iou(a,b):
    if len(a)!=4 or len(b)!=4:
        return 0.0
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1:return 0.0
    inter=(ix2-ix1)*(iy2-iy1); aa=max(1,(ax2-ax1)*(ay2-ay1)); bb=max(1,(bx2-bx1)*(by2-by1))
    return inter/max(1,aa+bb-inter)

def clip_box(b,w,h,pad=0):
    if len(b)!=4: return None
    x1,y1,x2,y2=map(int,b)
    x1=max(0,min(w,x1-pad)); y1=max(0,min(h,y1-pad)); x2=max(0,min(w,x2+pad)); y2=max(0,min(h,y2+pad))
    return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None

def normalize_boxes(boxes,w,h,pad=0):
    out=[]
    for b in boxes:
        if not isinstance(b,(tuple,list)) or len(b)!=4: continue
        b=clip_box(b,w,h,pad)
        if b: out.append(b)
    return out

def split_wide_box(b,img):
    x1,y1,x2,y2=map(int,b)
    crop=np.array(img.crop((x1,y1,x2,y2)).convert('L'))
    if crop.size==0:return [b]
    width=x2-x1; height=y2-y1
    # Wide region: look for low-ink vertical gutters and split into columns.
    if width<80 or width<height*0.75:return [b]
    ink=(crop<190).astype(np.uint8); proj=ink.sum(axis=0)
    smooth=np.convolve(proj,np.ones(7)/7,mode='same')
    low=np.percentile(smooth,30); min_gap=max(10,int(width*.025)); cuts=[]
    i=1
    while i<width-1:
        if smooth[i]<=low:
            j=i
            while j<width and smooth[j]<=low:j+=1
            if j-i>=min_gap: cuts.append((i+j)//2)
            i=j
        else:i+=1
    groups=[]; start=0
    for c in cuts:
        if c-start>=14: groups.append((start,c)); start=c
    if width-start>=14: groups.append((start,width))
    good=[g for g in groups if g[1]-g[0]>=14]
    if 2<=len(good)<=16:
        out=[]
        for a,c in good:
            out.append((x1+a, y1, x1+c, y2))
        return out
    return [b]

def detect_bottom_panel_top(img):
    """自动定位漫画下方白底密集正文的起始 Y。"""
    a=np.array(img.convert("L"))
    h,w=a.shape
    row=a.mean(axis=1)
    smooth=np.convolve(row,np.ones(11)/11,mode="same")
    lo=max(int(h*.40),1)
    hi=min(int(h*.72),h-2)
    diff=np.diff(smooth[lo:hi])
    if len(diff):
        idx=lo+int(np.argmax(diff))
        jump=float(diff[idx-lo])
        if jump>18:
            return max(int(h*.48),min(int(h*.62),idx+8))
    return int(h*.54)

def dense_vertical_candidates(img):
    """检测底部白/灰底密集竖排正文：按“文字列”生成框，而不是按单字生成几十个碎框。"""
    a=np.array(img.convert("L"))
    h,w=a.shape
    y0=detect_bottom_panel_top(img)
    y_start=min(h-20,y0+10)
    y_end=max(y_start+50,h-12)
    xpad=max(35,int(w*.045))
    roi=a[y_start:y_end,xpad:w-xpad]
    if roi.size==0:return []

    # 正文纸面通常约 170~200 灰度，字符明显更暗。
    mask=(roi<160).astype(np.uint8)*255
    # 只沿竖向连接相邻字符，避免横向把相邻列粘成一大片。
    close=cv2.morphologyEx(
        mask,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,15))
    )
    n,lab,stats,cent=cv2.connectedComponentsWithStats(close,8)
    comps=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if 8<=bw<=35 and bh>=70 and area>=150:
            comps.append((int(x+xpad),int(y+y_start),int(x+xpad+bw),int(y+y_start+bh)))

    # 同一竖排列可能因空行被分成数段，按 X 中心重新合并。
    comps.sort(key=lambda b:(b[0]+b[2])/2)
    groups=[]
    for b in comps:
        cx=(b[0]+b[2])/2
        placed=False
        for g in groups:
            gc=sum((q[0]+q[2])/2 for q in g)/len(g)
            if abs(cx-gc)<=10:
                g.append(b);placed=True;break
        if not placed:groups.append([b])

    boxes=[]
    for g in groups:
        x1=min(q[0] for q in g)-5; x2=max(q[2] for q in g)+5
        yy1=min(q[1] for q in g)-6; yy2=max(q[3] for q in g)+6
        if 12<=x2-x1<=55 and yy2-yy1>=120:
            boxes.append((max(xpad,int(x1)),max(y0,int(yy1)),min(w-xpad,int(x2)),min(h-2,int(yy2))))

    # 去重并按 X 从左到右返回，后续阅读排序会改成右到左。
    out=[]
    for b in sorted(boxes,key=lambda z:z[0]):
        if not any(box_iou(b,k)>.65 for k in out):out.append(b)
    return out


def _text_region_stats(img,b):
    """计算一个候选框是否更像“文字在浅色背景上”，而不是人物/衣服纹理。"""
    x1,y1,x2,y2=map(int,b)
    pad=3
    x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(img.width,x2+pad);y2=min(img.height,y2+pad)
    crop=np.array(img.crop((x1,y1,x2,y2)).convert("RGB"))
    if crop.size==0:return 0,0,0
    g=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    light=float(np.mean(g>180)); dark=float(np.mean(g<115)); sat=float(np.mean(hsv[...,1]))
    return light,dark,sat


def _looks_like_upper_text(img,b):
    """过滤上半页明显落在人物/衣服上的误识别框。"""
    x1,y1,x2,y2=b
    bw=x2-x1;bh=y2-y1
    if bw<8 or bh<10:return False
    light,dark,sat=_text_region_stats(img,b)
    # 气泡文字：大部分背景偏亮、少量深色笔画、整体低饱和。
    if light>=0.38 and dark>=0.012 and sat<=68:
        return True
    return False

def tesseract_candidates(img):
    # Backward-compatible name: the v11 pipeline no longer uses Tesseract.
    return get_pp_ocr().detect(img)


def _merge_upper_vertical_text(boxes,panel_top,img_h):
    """
    将上半页中属于同一个竖排气泡/竖排文本列的碎框合并，
    避免“一个字一个红框”。
    """
    upper=[b for b in boxes if b[1] < panel_top-12]
    rest=[b for b in boxes if b[1] >= panel_top-12]
    upper=sorted(upper,key=lambda b:(b[0],b[1]))
    used=[False]*len(upper)
    merged=[]
    for i,a in enumerate(upper):
        if used[i]:continue
        group=[a];used[i]=True
        changed=True
        while changed:
            changed=False
            gx1=min(x[0] for x in group); gy1=min(x[1] for x in group)
            gx2=max(x[2] for x in group); gy2=max(x[3] for x in group)
            gc=(gx1+gx2)/2; gw=max(8,gx2-gx1)
            for j,b in enumerate(upper):
                if used[j]:continue
                bw=max(5,b[2]-b[0]); bh=max(5,b[3]-b[1])
                bc=(b[0]+b[2])/2
                xover=max(0,min(gx2,b[2])-max(gx1,b[0]))/max(1,min(gw,bw))
                ygap=max(0,b[1]-gy2)
                same_col=(
                    (xover>=.22 or abs(bc-gc)<=max(16,1.2*max(gw,bw)))
                    and ygap<=max(18,1.8*min(bw,bw))
                )
                union_w=max(gx2,b[2])-min(gx1,b[0])
                union_h=max(gy2,b[3])-min(gy1,b[1])
                if same_col and union_w<=170 and union_h<=300:
                    group.append(b);used[j]=True;changed=True
        bx=(min(x[0] for x in group),min(x[1] for x in group),
            max(x[2] for x in group),max(x[3] for x in group))
        merged.append(bx)
    return rest+merged

def split_tall_box(b,img):
    """把疑似跨越两个气泡的超高框按横向空白带拆开。"""
    x1,y1,x2,y2=b;w=x2-x1;h=y2-y1
    if h<320 or w<40:return [b]
    g=np.array(img.crop((x1,y1,x2,y2)).convert("L"))
    ink=(g<165).astype(np.uint8)
    row=ink.sum(axis=1).astype(float)
    sm=np.convolve(row,np.ones(9)/9,mode="same")
    low=np.percentile(sm,18)
    runs=[];st=None
    for i,v in enumerate(sm<=low):
        if v and st is None:st=i
        elif not v and st is not None:
            if i-st>=10:runs.append((st,i))
            st=None
    if st is not None:runs.append((st,len(sm)))
    cuts=[(a+c)//2 for a,c in runs if a>8 and c<len(sm)-8]
    if not cuts:return [b]
    groups=[];start=0
    for c in cuts:
        if c-start>=45:
            groups.append((start,c));start=c
    if h-start>=45:groups.append((start,h))
    if len(groups)<2 or len(groups)>5:return [b]
    out=[]
    for a,c in groups:
        yy1=max(y1,y1+a-4);yy2=min(y2,y1+c+4)
        if yy2-yy1>=55:out.append((x1,yy1,x2,yy2))
    return out or [b]

def repair_boxes(img,boxes):
    w,h=img.size
    panel_top=detect_bottom_panel_top(img)

    # 上半页：先过滤明显落在人物/衣服/背景纹理上的候选框。
    upper_raw=[b for b in boxes if b[1] < panel_top-15]
    upper=[b for b in upper_raw if _looks_like_upper_text(img,b)]
    bottom=dense_vertical_candidates(img)

    pre=[]
    for b in upper:
        pre.extend(split_tall_box(b,img))
    candidates=normalize_boxes(pre+bottom,w,h,5)
    candidates=sorted(candidates,key=lambda b:(b[1],b[0],-(b[2]-b[0])*(b[3]-b[1])))

    # 初次去重。
    dedup=[]
    for b in candidates:
        if any(box_iou(b,k)>.78 for k in dedup):continue
        dedup.append(b)

    # 上半页气泡中的碎竖排框合并；底部正文已经按列生成，不再拆碎。
    merged=_merge_upper_vertical_text(dedup,panel_top,h)

    expanded=[]
    for b in merged:
        if b[1] < panel_top-15:
            expanded.extend(split_wide_box(b,img))
        else:
            expanded.append(b)

    filt=[]
    for b in expanded:
        x1,y1,x2,y2=b;bw=x2-x1;bh=y2-y1;area=bw*bh
        if area<180 or area>w*h*.32:continue
        if bw>w*.70 and bh>h*.18:continue
        if y1 < panel_top-15 and not _looks_like_upper_text(img,b):continue
        filt.append(b)

    final=[]
    for b in sorted(filt,key=lambda z:(z[1],z[0],(z[2]-z[0])*(z[3]-z[1]))):
        dup=False
        for k in range(len(final)-1,-1,-1):
            ov=box_iou(b,final[k])
            if ov>.82:
                a1=(b[2]-b[0])*(b[3]-b[1]);a2=(final[k][2]-final[k][0])*(final[k][3]-final[k][1])
                if a1<a2:final[k]=b
                dup=True;break
        if not dup:final.append(b)

    # 正常情况下本页应只有少量气泡框 + 约几十个正文列。
    if len(final)>80:
        bottom_final=[b for b in final if b[1]>=panel_top-12]
        upper_final=[b for b in final if b[1]<panel_top-12]
        upper_final=sorted(upper_final,key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),b[1],b[0]))[:max(0,80-len(bottom_final))]
        final=upper_final+bottom_final[:80]
    return final

def _is_vertical(b):
    x1,y1,x2,y2=b
    return (y2-y1) > (x2-x1)*1.18

def order_manga_boxes(boxes):
    """稳定的漫画阅读顺序：竖排按列从右到左，列内从上到下；横排按上到下。"""
    bs=list(boxes)
    vertical=[b for b in bs if _is_vertical(b)]
    horizontal=[b for b in bs if not _is_vertical(b)]

    # 把 x 中心接近的竖排框归入同一列，避免检测框轻微倾斜导致顺序跳动。
    cols=[]
    for b in sorted(vertical, key=lambda z: -((z[0]+z[2])/2)):
        cx=(b[0]+b[2])/2
        bw=max(8,b[2]-b[0])
        placed=False
        for col in cols:
            ref=col["cx"]
            if abs(cx-ref) <= max(18,bw*0.75):
                col["boxes"].append(b)
                col["cx"]=(col["cx"]*(len(col["boxes"])-1)+cx)/len(col["boxes"])
                placed=True
                break
        if not placed:
            cols.append({"cx":cx,"boxes":[b]})

    ordered=[]
    for col in sorted(cols,key=lambda c:-c["cx"]):
        ordered.extend(sorted(col["boxes"],key=lambda b:(b[1],b[0])))

    # 横排文字通常按页面位置从上到下、同一行从左到右。
    ordered.extend(sorted(horizontal,key=lambda b:(b[1],b[0])))
    return ordered

def order_manga_items(items):
    """按文字框的漫画阅读顺序排列 OCR item，而不是把 dict 当成坐标框。"""
    pairs=[]
    for idx,it in enumerate(items):
        b=it.get("box") if isinstance(it,dict) else None
        if isinstance(b,(tuple,list)) and len(b)==4:
            pairs.append((idx,b,it))
    ordered_boxes=order_manga_boxes([b for _,b,_ in pairs])
    by_box={}
    for idx,b,it in pairs:
        by_box.setdefault(tuple(b),[]).append(it)
    out=[]
    for b in ordered_boxes:
        bucket=by_box.get(tuple(b),[])
        if bucket: out.append(bucket.pop(0))
    # 异常 item 不丢失，放到最后。
    used={id(x) for x in out}
    out.extend(it for _,_,it in pairs if id(it) not in used)
    return out

def _text_key(s):
    s=re.sub(r"\\s+","",str(s or ""))
    return s.strip()

def dedup_ocr_items(items):
    """二次去重：完全重复文本 + 高重叠框只保留一个，避免同一段重复送 AI。"""
    kept=[]
    seen_text={}
    for it in items:
        t=_text_key(it.get("text",""))
        b=it["box"]
        if not t or t.startswith("[OCR失败"):
            kept.append(it)
            continue
        if t in seen_text:
            old=seen_text[t]
            if box_iou(b,old["box"]) >= 0.25:
                continue
        replaced=False
        for j,old in enumerate(kept):
            if _text_key(old.get("text",""))==t and box_iou(b,old["box"])>=0.25:
                replaced=True
                break
        if not replaced:
            seen_text[t]=it
            kept.append(it)
    # 重新编号，保证阅读顺序与 AI id 一致。
    for i,it in enumerate(kept,1):
        it["id"]=i
    return kept

def merge_sentence_items(img, items):
    """把明显属于同一句话的碎框合成一个，隔开的两个气泡保持分开。"""
    if not items:
        return []
    W, H = img.size
    groups = [[dict(it)] for it in items]

    def box_of(g):
        return (
            min(q["box"][0] for q in g),
            min(q["box"][1] for q in g),
            max(q["box"][2] for q in g),
            max(q["box"][3] for q in g),
        )

    def can(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        ux1, uy1 = min(ax1, bx1), min(ay1, by1)
        ux2, uy2 = max(ax2, bx2), max(ay2, by2)
        uw, uh = ux2 - ux1, uy2 - uy1
        xover = max(0, min(ax2, bx2) - max(ax1, bx1)) / max(1, min(aw, bw))
        yover = max(0, min(ay2, by2) - max(ay1, by1)) / max(1, min(ah, bh))
        ygap = max(0, max(ay1, by1) - min(ay2, by2))
        xgap = max(0, max(ax1, bx1) - min(ax2, bx2))
        acx, bcx = (ax1 + ax2) / 2, (bx1 + bx2) / 2
        acy, bcy = (ay1 + ay2) / 2, (by1 + by2) / 2
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        if (ix * iy) / max(1, min(aw * ah, bw * bh)) >= 0.72 and uw < W * 0.45 and uh < H * 0.45:
            return True
        vertical = (
            xover >= 0.55
            and abs(acx - bcx) <= 0.42 * max(aw, bw)
            and ygap <= max(20, 0.45 * min(ah, bh))
            and uw <= max(aw, bw) * 1.35
            and uh <= max(ah, bh) * 2.6
            and uh <= H * 0.40
            and uw <= W * 0.28
        )
        horizontal = (
            yover >= 0.55
            and abs(acy - bcy) <= 0.42 * max(ah, bh)
            and xgap <= max(24, 0.55 * min(aw, bw))
            and uh <= max(ah, bh) * 1.35
            and uw <= max(aw, bw) * 2.8
            and uw <= W * 0.55
            and uh <= H * 0.22
        )
        return vertical or horizontal

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if can(box_of(groups[i]), box_of(groups[j])):
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    out = []
    for i, g in enumerate(groups, 1):
        box = [int(v) for v in box_of(g)]
        tall = box[3] - box[1] > (box[2] - box[0]) * 1.1
        if tall:
            g = sorted(g, key=lambda q: (-(q["box"][0] + q["box"][2]) / 2, q["box"][1]))
        else:
            g = sorted(g, key=lambda q: (q["box"][1], q["box"][0]))
        text = "".join(str(q.get("text") or "").strip() for q in g if str(q.get("text") or "").strip())
        out.append({"id": i, "box": box, "text": text})
    return out


def run_ocr(img,boxes=None,progress=None):
    """v11: one dedicated manga OCR model does both detection and recognition."""
    engine=get_pp_ocr()
    if boxes is None:
        return merge_sentence_items(img, engine.ocr_page(img,progress=progress))
    # When caller already has detector boxes, recognize those boxes only.
    items=[]; total=len(boxes)
    for idx,b in enumerate(boxes,1):
        x1,y1,x2,y2=map(int,b)
        pad=2
        cb=(max(0,x1-pad),max(0,y1-pad),min(img.width,x2+pad),min(img.height,y2+pad))
        try: text=engine.recognize(img.crop(cb)).strip()
        except Exception as e: text=f'[OCR失败: {e}]'
        if text or text.startswith('[OCR失败'):
            items.append({'id':idx,'box':[x1,y1,x2,y2],'text':text})
        if progress: progress(idx,total)
    return dedup_ocr_items(items) if items else []


def make_preview(img,items):
    out=img.copy();d=ImageDraw.Draw(out)
    for it in items:
        x1,y1,x2,y2=it['box'];d.rectangle((x1,y1,x2,y2),outline=(235,40,40),width=max(2,min(7,img.width//250)))
        d.rectangle((x1,y1,x1+40,y1+24),fill=(235,40,40));d.text((x1+5,y1+3),str(it['id']),fill='white')
    return out

def natural_key(name):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]

def safe_extract_zip(z, root):
    root=Path(root).resolve(); names=[]
    for info in z.infolist():
        n=info.filename.replace('\\','/')
        if n.endswith('/') or Path(n).suffix.lower() not in IMAGE_EXTS: continue
        dest=(root/n).resolve()
        if not str(dest).startswith(str(root)): continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with z.open(info) as src, open(dest,'wb') as dst: shutil.copyfileobj(src,dst)
        names.append(n)
    return sorted(names,key=natural_key)

def extract_archive_to_dir(src_path, out_dir):
    ext=Path(src_path).suffix.lower(); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    if ext in {'.zip','.cbz'}:
        with zipfile.ZipFile(src_path,'r') as z: return safe_extract_zip(z,out_dir)
    if ext=='.7z':
        if py7zr is None: raise RuntimeError('缺少 py7zr，请运行安装脚本')
        with py7zr.SevenZipFile(src_path,'r') as z: z.extractall(path=out_dir)
    elif ext=='.rar':
        if rarfile is None: raise RuntimeError('缺少 rarfile，请运行安装脚本')
        with rarfile.RarFile(src_path) as z:
            for info in z.infolist():
                n=info.filename.replace('\\','/')
                if info.isdir() or Path(n).suffix.lower() not in IMAGE_EXTS: continue
                dest=(out_dir/n).resolve()
                if not str(dest).startswith(str(out_dir.resolve())): continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src, open(dest,'wb') as dst: shutil.copyfileobj(src,dst)
    else:
        raise RuntimeError('支持的压缩格式：ZIP / CBZ / 7Z / RAR')
    return sorted([str(p.relative_to(out_dir)).replace('\\','/') for p in out_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS],key=natural_key)

def extract_first_image(upload):
    name=upload.filename or 'input'; ext=Path(name).suffix.lower()
    if ext in IMAGE_EXTS: return Image.open(upload.stream).convert('RGB'),name
    if ext in {'.zip','.cbz','.7z','.rar'}:
        with tempfile.TemporaryDirectory(prefix='jpmt_preview_') as td:
            tmp=Path(td)/Path(name).name; upload.stream.seek(0); tmp.write_bytes(upload.stream.read())
            names=extract_archive_to_dir(tmp,Path(td)/'pages')
            if not names: raise RuntimeError('压缩包里没有图片')
            return Image.open(Path(td)/'pages'/names[0]).convert('RGB'),names[0]
    raise RuntimeError('预览支持图片、ZIP、CBZ、7Z、RAR')

def safe_json_array(text):
    text=text.strip();text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I)
    start=text.find("[")
    if start<0:raise ValueError("AI没有返回JSON数组")
    depth=0;ins=False;esc=False;end=-1
    for i in range(start,len(text)):
        c=text[i]
        if ins:
            if esc:esc=False
            elif c=="\\":esc=True
            elif c=='"':ins=False
        else:
            if c=='"':ins=True
            elif c=="[":depth+=1
            elif c=="]":
                depth-=1
                if depth==0:end=i+1;break
    if end<0:raise ValueError("JSON数组被截断")
    s=text[start:end].replace("“",'"').replace("”",'"').replace("’","'")
    s=re.sub(r",\s*([}\]])",r"\1",s);data=json.loads(s)
    if not isinstance(data,list):raise ValueError("JSON根节点不是数组")
    return data

def _looks_like_refusal(text):
    """判断模型输出是否明显是拒答/错误提示，而不是译文。"""
    t=str(text or "").strip().lower()
    if not t:
        return True
    patterns=[
        "i can\'t help", "i cannot help", "i can’t help",
        "i\'m sorry, i can\'t", "i’m sorry, i can’t",
        "cannot assist", "can\'t assist", "can’t assist",
        "unable to comply", "policy", "安全政策", "无法协助",
        "不能协助", "无法提供", "不能提供",
    ]
    return any(x in t for x in patterns)

class AIRejectedError(RuntimeError):
    def __init__(self, message, code=None, status=None, raw=None):
        super().__init__(message)
        self.code=code
        self.status=status
        self.raw=raw


def _is_ollama_endpoint(endpoint):
    e=(endpoint or "").strip().lower().rstrip("/")
    return e.endswith("/api/chat") or "127.0.0.1:11434" in e or "localhost:11434" in e

def call_chat(endpoint,key,model,prompt,timeout,maxout):
    endpoint=(endpoint or "").strip()
    if _is_ollama_endpoint(endpoint):
        r=requests.post(endpoint.rstrip("/"),headers={"Content-Type":"application/json"},json={
            "model":model,
            "messages":[{"role":"user","content":prompt}],
            "stream":False,
            "keep_alive":"10m",
            "options":{
                "temperature":0.7,
                "top_p":0.6,
                "top_k":20,
                "repeat_penalty":1.05,
                "num_predict":int(maxout),
            },
        },timeout=timeout)
        if r.status_code>=400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:1600]}")
        data=r.json()
        content=((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(f"Ollama 未返回文本：{r.text[:800]}")
        pcount=int(data.get("prompt_eval_count") or 0)
        ccount=int(data.get("eval_count") or 0)
        return content,{"prompt_tokens":pcount,"completion_tokens":ccount,"total_tokens":pcount+ccount}

    url=endpoint.rstrip("/")+"/chat/completions"
    headers={"Content-Type":"application/json"}
    if key:
        headers["Authorization"]="Bearer "+key
    r=requests.post(url,headers=headers,json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":.2,"max_tokens":maxout},timeout=timeout)
    if r.status_code>=400:
        raw=r.text[:1200]; code=None
        try:
            obj=r.json(); code=str(obj.get("error",{}).get("code") or ""); msg=obj.get("error",{}).get("message") or raw
        except Exception:
            msg=raw
        if str(code)=="1301":
            raise AIRejectedError(f"服务商拒绝本次内容（1301）：{msg}",code=code,status=r.status_code,raw=raw)
        raise RuntimeError(f"HTTP {r.status_code}: {raw}")
    data=r.json();return data["choices"][0]["message"]["content"],data.get("usage",{})

def font_for(size, bold=False):
    """中文覆盖优先用黑体/雅黑，避免 Meiryo 缺简体字出现口口。"""
    root=Path(__file__).resolve().parent
    regular=[
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        str(root/"NotoSansSC.ttf"),
        str(root/"fonts"/"NotoSansSC.ttf"),
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    heavy=[
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        str(root/"fonts"/"NotoSansSC.ttf"),
        r"C:\Windows\Fonts\meiryob.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in (heavy if bold else regular):
        if os.path.exists(fp):
            for idx in (0,1,2):
                try:return ImageFont.truetype(fp,size=size,index=idx)
                except Exception:pass
            try:return ImageFont.truetype(fp,size=size)
            except Exception:pass
    return ImageFont.load_default()

def style_font(img, box, size, force_bold=False):
    try:
        x1,y1,x2,y2=map(int,box)
        crop=np.array(img.crop((max(0,x1),max(0,y1),min(img.width,x2),min(img.height,y2))).convert('L'))
        if crop.size:
            p25=float(np.percentile(crop,25)); p65=float(np.percentile(crop,65))
            ink=float(np.mean(crop < min(165,p65-12 if p65>30 else 145)))
            force_bold=force_bold or ink>0.115 or p25<70
    except Exception:
        pass
    return font_for(size,bold=force_bold)

def enhance_final_image(img):
    img=ImageEnhance.Contrast(img).enhance(1.10)
    img=ImageEnhance.Sharpness(img).enhance(1.08)
    return img

def _text_ink_mask(img,box):
    """生成局部文字笔画 mask：优先检测相对局部背景更暗的笔画，减少漏擦。"""
    x1,y1,x2,y2=map(int,box)
    crop=np.array(img.crop((x1,y1,x2,y2)).convert("RGB"))
    g=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)

    # 漫画气泡/正文通常是浅背景+深色字。用局部背景估计补足抗锯齿和灰色字。
    bg=cv2.GaussianBlur(g,(0,0),sigmaX=7)
    local_dark=(g.astype(np.int16) < (bg.astype(np.int16)-10))
    absolute_dark=(g<185)
    # 彩色文字也允许保留，但不把高亮彩色大面积背景全部算进去。
    color_ink=(hsv[...,1]>45) & (g<155)
    mask=local_dark | (absolute_dark & (hsv[...,1]<130)) | color_ink

    m=(mask.astype(np.uint8)*255)
    # 连接断开的抗锯齿笔画，再去掉极小噪声。
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),iterations=1)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(m,8)
    clean=np.zeros_like(m)
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]
        if area>=2:
            clean[lab==i]=255
    # 稍微扩张，覆盖日文字符边缘，避免残留黑边。
    clean=cv2.dilate(clean,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    return clean


def _safe_ai_inpaint(region, mask):
    """AI 擦字安全封装：强制 mask 与输入图像尺寸一致；AI 异常时仅对当前局部区域降级到 OpenCV，绝不让整页失败。"""
    region=region.convert('RGB')
    m=np.asarray(mask,dtype=np.uint8)
    target=(region.height,region.width)
    if m.ndim!=2 or m.shape!=target:
        m=np.asarray(Image.fromarray(m if m.ndim==2 else np.zeros(target,dtype=np.uint8)).resize((region.width,region.height),Image.Resampling.NEAREST),dtype=np.uint8)
    try:
        out=ai_inpaint(region,m,max_side=1024).convert('RGB')
        if out.size!=region.size:
            out=out.resize(region.size,Image.Resampling.LANCZOS)
        return out, 'ai'
    except Exception as e:
        # 最后一道保险：只在 AI 当前区域失败时用局部 OpenCV，不影响其它区域继续翻译。
        try:
            arr=np.asarray(region)
            mm=(m>32).astype(np.uint8)*255
            bgr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
            fixed=cv2.inpaint(bgr,mm,3,cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(fixed,cv2.COLOR_BGR2RGB)),'opencv_fallback:'+str(e)
        except Exception:
            return region,'failed:'+str(e)


def _erase_text_boxes(img, boxes, mode='bubble'):
    """把多个 OCR 原文框合成一个局部 mask，并交给本地 Manga LaMa AI 擦除。"""
    if not boxes:
        return
    try:
        x1=min(int(b[0]) for b in boxes); y1=min(int(b[1]) for b in boxes)
        x2=max(int(b[2]) for b in boxes); y2=max(int(b[3]) for b in boxes)
    except Exception:
        return
    x1=max(0,x1); y1=max(0,y1); x2=min(img.width,x2); y2=min(img.height,y2)
    if x2<=x1 or y2<=y1:
        return
    mask=np.zeros((y2-y1,x2-x1),dtype=np.uint8)
    for b in boxes:
        bx1=max(x1,int(b[0])); by1=max(y1,int(b[1])); bx2=min(x2,int(b[2])); by2=min(y2,int(b[3]))
        if bx2<=bx1 or by2<=by1:
            continue
        local=_text_ink_mask(img,(bx1,by1,bx2,by2))
        yy1=by1-y1; xx1=bx1-x1
        yy2=yy1+local.shape[0]; xx2=xx1+local.shape[1]
        mask[yy1:yy2,xx1:xx2]=np.maximum(mask[yy1:yy2,xx1:xx2],local)
    if np.count_nonzero(mask)<=2:
        return
    region=img.crop((x1,y1,x2,y2)).convert('RGB')
    # v12.2：统一使用本地 AI 擦字，不再退回 cv2.inpaint。
    fixed,_=_safe_ai_inpaint(region,mask)
    img.paste(fixed,(x1,y1))


def _fit_vertical_font(draw,text,bw,bh,is_bottom=False):
    """根据一个连续竖排栏的真实可用高度自动缩字。"""
    n=max(1,len(text))
    upper=_fs(max(10 if is_bottom else 13, min(30, int(max(10,bw)*.78))))
    lower=_fs(9 if is_bottom else 13)
    by_h=int(max(1,(bh-8)/n-2))
    return max(lower,min(upper,_fs(by_h)))


def _split_vertical_text(text, capacity):
    text=re.sub(r"\s+","",str(text or "")).strip()
    if not text:return []
    return [text[i:i+capacity] for i in range(0,len(text),max(1,capacity))]


def _draw_vertical_region(img,box,text):
    """底部正文专用：一个原文栏对应一个中文栏，绝不跨到相邻栏。"""
    x1,y1,x2,y2=map(int,box)
    bw,bh=x2-x1,y2-y1
    mask=_text_ink_mask(img,box)
    if np.count_nonzero(mask)>2:
        crop=img.crop((x1,y1,x2,y2)).convert("RGB")
        fixed,_=_safe_ai_inpaint(crop,mask)
        img.paste(fixed,(x1,y1))
    clean=re.sub(r"\s+","",str(text or "")).strip()
    if not clean:return
    d=ImageDraw.Draw(img)
    # 先估算能放多少字符；超长时缩小，但仍保持一栏一栏。
    size=_fs(max(9,min(28,int(max(9,bw*.76)))))
    while size>9:
        lh=max(size+1,int(size*1.10))
        cap=max(1,(bh-10)//lh)
        if len(clean)<=cap:break
        size-=1
    f=style_font(img,box,size); lh=max(size+1,int(size*1.10)); cap=max(1,(bh-10)//lh)
    cols=_split_vertical_text(clean,cap)
    max_cols=max(1,(bw-3)//max(1,size+2))
    cols=cols[:max_cols]
    # 右起向左排，模拟日文竖排阅读栏；通常只需要一栏。
    fill,stroke,sw=_text_style_params(img,box,size)
    for ci,col in enumerate(cols):
        cx=x2-(ci+1)*size-2
        cy=y1+5
        for ch in col:
            _draw_styled_text(d,(cx,cy),ch,font=f,fill=fill,stroke=stroke,stroke_width=sw)
            cy+=lh


def _wrap_horizontal_cn(draw,text,font,max_width):
    text=str(text or '').strip()
    if not text:return []
    lines=[]; cur=''
    for ch in text:
        if ch=='\n':
            if cur.strip():lines.append(cur.strip())
            cur=''; continue
        test=cur+ch
        bb=draw.textbbox((0,0),test,font=font)
        if cur and bb[2]-bb[0]>max_width:
            lines.append(cur.strip()); cur=ch
        else:
            cur=test
    if cur.strip(): lines.append(cur.strip())
    return lines


def _text_style_params(img, box, size):
    """高可见度中文：根据局部背景亮度选择黑字/白字，并增加细描边。"""
    x1,y1,x2,y2=map(int,box)
    x1=max(0,x1); y1=max(0,y1); x2=min(img.width,x2); y2=min(img.height,y2)
    if x2<=x1 or y2<=y1:
        mean=220.0
    else:
        g=np.array(img.crop((x1,y1,x2,y2)).convert('L'))
        # 使用高亮区域中位数，避免被黑色原文笔画拉低。
        mean=float(np.percentile(g,65)) if g.size else 220.0
    if mean<125:
        fill=(255,255,255); stroke=(8,8,8)
    else:
        fill=(8,8,8); stroke=(255,255,255)
    stroke_width=5 if size>=22 else (4 if size>=14 else 3)
    return fill,stroke,stroke_width

def _draw_styled_text(draw, xy, text, font, fill, stroke, stroke_width):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke)

def _draw_body_paragraph(img,item,text):
    """正文面板：只擦 member_boxes 的文字，然后在成员框联合区域内自适应横排。"""
    box=item.get('box') or [0,0,img.width,img.height]
    members=item.get('member_boxes') or [box]
    _erase_text_boxes(img,members,mode='panel')
    raw=str(text or '').replace('\r','').strip()
    if not raw:return
    paras=[]
    for p in re.split(r'\n\s*\n+',raw):
        q=re.sub(r'[ \t\f\v]+',' ',p).strip()
        if q:paras.append(q)
    x1,y1,x2,y2=map(int,box); bw=x2-x1; bh=y2-y1
    d=ImageDraw.Draw(img)
    pad=max(12,int(min(bw,bh)*0.035)); usable_w=max(80,bw-pad*2); usable_h=max(60,bh-pad*2)
    chosen=None
    for size in range(_fs(max(12,min(32,int(bw*0.036)))),_fs(9)-1,-1):
        f=style_font(img,box,size); lh=max(size+5,int(size*1.38)); groups=[]; total=0
        for para in paras:
            lines=_wrap_horizontal_cn(d,para,f,usable_w)
            if lines:
                groups.append(lines); total+=len(lines)*lh
        if len(groups)>1: total+=int(lh*0.6)*(len(groups)-1)
        if total<=usable_h:
            chosen=(f,groups,lh,total); break
    if chosen is None:
        size=_fs(10); f=style_font(img,box,size); lh=max(15,int(size*1.38)); groups=[_wrap_horizontal_cn(d,p,f,usable_w) for p in paras]; total=sum(len(g)*lh for g in groups); chosen=(f,groups,lh,total)
    f,groups,lh,total=chosen
    y=y1+(bh-total)//2
    for gi,lines in enumerate(groups):
        for line in lines:
            if y+lh>y2-pad: return
            fill,stroke,sw=_text_style_params(img,(x1+pad,y,x2-pad,min(y+lh,y2-pad)),f.size)
            _draw_styled_text(d,(x1+pad,y),line,f,fill,stroke,sw); y+=lh
        if gi<len(groups)-1: y+=int(lh*0.6)


def draw_chinese(img,box,text,region_type="bubble"):
    if not isinstance(box,(tuple,list)) or len(box)!=4:
        raise ValueError(f"OCR框坐标异常: {box!r}")
    x1,y1,x2,y2=map(int,box)
    x1=max(0,min(img.width-1,x1)); y1=max(0,min(img.height-1,y1))
    x2=max(x1+1,min(img.width,x2)); y2=max(y1+1,min(img.height,y2))
    if region_type=="body_para":
        _draw_body_paragraph(img,{"box":[x1,y1,x2,y2],"member_boxes":None},text); return
    if region_type=="body":
        _draw_vertical_region(img,(x1,y1,x2,y2),text); return
    bw,bh=x2-x1,y2-y1
    is_bottom=(y1>=detect_bottom_panel_top(img)-12)
    mask=_text_ink_mask(img,(x1,y1,x2,y2))
    if np.count_nonzero(mask)>2:
        crop=img.crop((x1,y1,x2,y2)).convert("RGB")
        fixed,_=_safe_ai_inpaint(crop,mask)
        img.paste(fixed,(x1,y1))
    d=ImageDraw.Draw(img);clean_text=re.sub(r"\s+","",str(text or "")).strip()
    if not clean_text:return
    vertical=bh>bw*1.25
    if vertical:
        size=min(_fs(34),max(_fit_vertical_font(d,clean_text,bw,bh,is_bottom=is_bottom), _fs(12 if not is_bottom else 11)));f=style_font(img,(x1,y1,x2,y2),size)
        lh=max(size+1,int(size*1.08));maxn=max(1,(bh-8)//lh);cols=math.ceil(len(clean_text)/maxn);max_cols=max(1,(bw-4)//max(1,size+2))
        if cols>max_cols:
            size=max(_fs(9),min(size,int(((bh-8)*max_cols/max(1,len(clean_text)))-1)));f=style_font(img,(x1,y1,x2,y2),size);lh=max(size+1,int(size*1.08));maxn=max(1,(bh-8)//lh);cols=math.ceil(len(clean_text)/maxn)
        for ci in range(min(cols,max_cols)):
            col_text=clean_text[ci*maxn:min(len(clean_text),(ci+1)*maxn)];cx=x2-(ci+1)*size-2
            if cx<x1:break
            cy=y1+4
            fill,stroke,sw=_text_style_params(img,(x1,y1,x2,y2),size)
            for ch in col_text:
                _draw_styled_text(d,(cx,cy),ch,f,fill,stroke,sw);cy+=lh
    else:
        size=_fs(max(13,min(38,int(min(max(13,bw*.20),max(13,bh*.33))))));f=style_font(img,(x1,y1,x2,y2),size);lines=[];cur=""
        for ch in clean_text:
            if d.textbbox((0,0),cur+ch,font=f)[2]<=bw-10:cur+=ch
            else:
                if cur:lines.append(cur)
                cur=ch
        if cur:lines.append(cur)
        lh=size+4;y=y1+4
        fill,stroke,sw=_text_style_params(img,(x1,y1,x2,y2),size)
        for line in lines[:max(1,(bh-8)//lh)]:
            _draw_styled_text(d,(x1+5,y),line,f,fill,stroke,sw);y+=lh



def _draw_bubble_group(img,item,text):
    members=item.get('member_boxes') or [item.get('box')]
    members=[b for b in members if isinstance(b,(list,tuple)) and len(b)==4]
    if not members:return
    _erase_text_boxes(img,members,mode='bubble')
    x1,y1,x2,y2=map(int,item['box']); bw=x2-x1; bh=y2-y1
    clean=re.sub(r'\s+',' ',str(text or '')).strip()
    if not clean:return
    d=ImageDraw.Draw(img)
    pad=max(10,int(min(bw,bh)*0.10)); uw=max(20,bw-2*pad); uh=max(20,bh-2*pad)
    vertical=bh>=bw*1.18
    if vertical:
        chosen=None
        upper=_fs(max(16,min(48,int(max(16,bw*0.46)))))
        for size in range(upper,_fs(9)-1,-1):
            f=style_font(img,item.get("box") or box,size); lh=max(size+2,int(size*1.08)); cap=max(1,uh//lh)
            cols=math.ceil(len(clean)/cap); total_w=cols*(size+5)
            if total_w<=uw:
                chosen=(f,lh,cap,cols,size); break
        if chosen is None:
            size=_fs(10); f=style_font(img,item.get("box") or box,size); lh=max(12,int(size*1.12)); cap=max(1,uh//lh); cols=math.ceil(len(clean)/cap); chosen=(f,lh,cap,cols,size)
        f,lh,cap,cols,size=chosen
        total_w=min(uw,cols*(size+5));
        start_x=x1+(bw-total_w)//2
        for ci in range(cols):
            part=clean[ci*cap:(ci+1)*cap]
            cx=start_x+total_w-(ci+1)*(size+5)+1
            cy=y1+(bh-min(uh,cap*lh))//2
            fill,stroke,sw=_text_style_params(img,(x1,y1,x2,y2),size)
            for ch in part:
                _draw_styled_text(d,(cx,cy),ch,font=f,fill=fill,stroke=stroke,stroke_width=sw); cy+=lh
    else:
        chosen=None
        upper=_fs(max(16,min(48,int(max(16,bh*0.48)))))
        for size in range(upper,_fs(9)-1,-1):
            f=style_font(img,(x1,y1,x2,y2),size); lines=_wrap_horizontal_cn(d,clean,f,uw); lh=max(size+4,int(size*1.28))
            if len(lines)*lh<=uh:
                chosen=(f,lines,lh,size); break
        if chosen is None:
            f=style_font(img,(x1,y1,x2,y2),_fs(10)); lines=_wrap_horizontal_cn(d,clean,f,uw); lh=max(14,int(_fs(10)*1.28)); chosen=(f,lines,lh,_fs(10))
        f,lines,lh,size=chosen
        total_h=len(lines)*lh; y=y1+(bh-total_h)//2
        for line in lines:
            bb=d.textbbox((0,0),line,font=f); tw=bb[2]-bb[0]
            fill,stroke,sw=_text_style_params(img,(x1,y1,x2,y2),size)
            _draw_styled_text(d,(x1+(bw-tw)//2,y),line,f,fill,stroke,sw); y+=lh


def draw_item_chinese(img,item,text):
    rtype=item.get('type','bubble')
    if rtype=='body_para':
        _draw_body_paragraph(img,item,text); return
    if rtype=='bubble_group':
        _draw_bubble_group(img,item,text); return
    draw_chinese(img,item['box'],text,rtype)


def region_type_for(img,item):
    return item.get("type", "bubble")


def _light_panel_ratio(img, box):
    """估计一个候选长正文区域是否位于浅色纸面/对白框内。"""
    x1,y1,x2,y2=map(int,box)
    x1=max(0,min(img.width-1,x1)); y1=max(0,min(img.height-1,y1))
    x2=max(x1+1,min(img.width,x2)); y2=max(y1+1,min(img.height,y2))
    a=np.array(img.crop((x1,y1,x2,y2)).convert('L'))
    if a.size==0:return 0.0
    return float(np.mean(a>165))


def _merge_gap(a,b):
    return max(0, max(a[0],b[0]) - min(a[2],b[2]))


def _box_dims(b):
    x1,y1,x2,y2=map(int,b)
    return max(1,x2-x1), max(1,y2-y1)


def _local_light_ratio(img, box, pad=3):
    x1,y1,x2,y2=map(int,box)
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(img.width,x2+pad); y2=min(img.height,y2+pad)
    if x2<=x1 or y2<=y1: return 0.0
    g=np.array(img.crop((x1,y1,x2,y2)).convert('L'))
    return float(np.mean(g>=145)) if g.size else 0.0


def _gap_is_light(img, a, b):
    """检查两个 OCR 框之间的连接区域是否像同一个浅色气泡背景。"""
    ax1,ay1,ax2,ay2=map(int,a); bx1,by1,bx2,by2=map(int,b)
    # 垂直相邻：x 有重叠，检查两个框之间的横向带。
    xo=max(0,min(ax2,bx2)-max(ax1,bx1))
    yo=max(0,min(ay2,by2)-max(ay1,by1))
    if xo>0 and (ay2<=by1 or by2<=ay1):
        gx1=max(ax1,bx1); gx2=min(ax2,bx2)
        gy1=min(ay2,by2); gy2=max(ay1,by1)
        if gx2-gx1<4 or gy2-gy1<3: return True
        band=np.array(img.crop((gx1,gy1,gx2,gy2)).convert('L'))
    # 水平相邻：y 有重叠，检查两框之间的纵向带。
    elif yo>0 and (ax2<=bx1 or bx2<=ax1):
        gy1=max(ay1,by1); gy2=min(ay2,by2)
        gx1=min(ax2,bx2); gx2=max(ax1,bx1)
        if gx2-gx1<3 or gy2-gy1<4: return True
        band=np.array(img.crop((gx1,gy1,gx2,gy2)).convert('L'))
    else:
        # 对角相邻：检查两框中心连线附近的小矩形。
        cax=(ax1+ax2)//2; cay=(ay1+ay2)//2; cbx=(bx1+bx2)//2; cby=(by1+by2)//2
        x1=min(cax,cbx); x2=max(cax,cbx); y1=min(cay,cby); y2=max(cay,cby)
        if x2-x1<3 or y2-y1<3: return True
        pad=4
        band=np.array(img.crop((max(0,x1-pad),max(0,y1-pad),min(img.width,x2+pad+1),min(img.height,y2+pad+1))).convert('L'))
    if band.size==0: return True
    # 黑色气泡边框/漫画格线会明显降低亮度；允许灰泡，但不允许大片纯黑连接带。
    dark=float(np.mean(band<105)); light=float(np.mean(band>=145))
    return dark<0.38 and light>=0.30


def _dark_separator_between(img,a,b):
    """判断两 OCR 框之间是否存在明显的黑色边界/格线/气泡轮廓。存在则禁止合并。"""
    ax1,ay1,ax2,ay2=map(int,a); bx1,by1,bx2,by2=map(int,b)
    # 纵向相邻：检查两框之间的水平带。
    if ay2 <= by1 and max(ax1,bx1) < min(ax2,bx2):
        gx1=max(ax1,bx1); gx2=min(ax2,bx2)
        gy1=ay2; gy2=by1
        gap=gy2-gy1
        if 1 <= gap <= 48 and gx2-gx1 >= 8:
            band=np.array(img.crop((gx1,gy1,gx2,gy2)).convert('L'))
            dark=(band<95)
            row_ratio=dark.mean(axis=1) if dark.size else np.array([])
            if row_ratio.size and float(row_ratio.max())>=0.48:
                return True
    # 横向相邻：检查两框之间的竖直带。
    if ax2 <= bx1 and max(ay1,by1) < min(ay2,by2):
        gy1=max(ay1,by1); gy2=min(ay2,by2)
        gx1=ax2; gx2=bx1
        gap=gx2-gx1
        if 1 <= gap <= 48 and gy2-gy1 >= 8:
            band=np.array(img.crop((gx1,gy1,gx2,gy2)).convert('L'))
            dark=(band<95)
            col_ratio=dark.mean(axis=0) if dark.size else np.array([])
            if col_ratio.size and float(col_ratio.max())>=0.48:
                return True
    return False


def _same_region_pair(img,a,b,page_w,page_h):
    """严格保守地判断两个 OCR 框是否属于同一气泡/同一文本块。"""
    ax1,ay1,ax2,ay2=map(int,a); bx1,by1,bx2,by2=map(int,b)
    aw,ah=_box_dims(a); bw,bh=_box_dims(b)
    acx=(ax1+ax2)/2; acy=(ay1+ay2)/2; bcx=(bx1+bx2)/2; bcy=(by1+by2)/2
    union=(min(ax1,bx1),min(ay1,by1),max(ax2,bx2),max(ay2,by2))
    uw,uh=_box_dims(union)
    if uw>page_w*0.26 or uh>page_h*0.30:
        return False
    if _dark_separator_between(img,a,b):
        return False

    va=ah>aw*1.12; vb=bh>bw*1.12
    if va and vb:
        ygap=max(0,max(ay1,by1)-min(ay2,by2))
        xover=max(0,min(ax2,bx2)-max(ax1,bx1))/max(1,min(aw,bw))
        xcenter=abs(acx-bcx)
        same_col = xover>=0.68 and xcenter<=max(14,int(0.65*max(aw,bw))) and ygap<=max(18,int(1.15*min(ah,bh)))
        # 两个并排竖栏只有在横向几乎重叠时才允许合并，避免右侧两句话被吃成一个大框。
        return same_col and _gap_is_light(img,a,b)
    if (not va) and (not vb):
        xgap=max(0,max(ax1,bx1)-min(ax2,bx2))
        yover=max(0,min(ay2,by2)-max(ay1,by1))/max(1,min(ah,bh))
        ycenter=abs(acy-bcy)
        same_line = yover>=0.68 and ycenter<=max(14,int(0.65*max(ah,bh))) and xgap<=max(18,int(1.15*min(aw,bw)))
        return same_line and _gap_is_light(img,a,b)
    # 横竖混合时，除非高度重叠且 IoU 很高，否则绝不合并。
    return box_iou(a,b)>=0.48


def _cluster_text_boxes(img, items):
    """贪心保守聚类：新框必须与组内所有成员兼容，防止 A-B、B-C 链式把两个气泡并成一个大框。"""
    if not items:return []
    page_w,page_h=img.width,img.height
    src=[dict(it) for it in items]
    src.sort(key=lambda q:(q['box'][1],q['box'][0]))
    groups=[]
    for it in src:
        placed=False
        best=None; best_score=1e9
        for gi,g in enumerate(groups):
            if len(g)>=3: continue
            if not all(_same_region_pair(img,it['box'],q['box'],page_w,page_h) for q in g):
                continue
            u=(min([it['box'][0]]+[q['box'][0] for q in g]),
               min([it['box'][1]]+[q['box'][1] for q in g]),
               max([it['box'][2]]+[q['box'][2] for q in g]),
               max([it['box'][3]]+[q['box'][3] for q in g]))
            uw,uh=_box_dims(u)
            if uw>page_w*0.24 or uh>page_h*0.28: continue
            ic=((it['box'][0]+it['box'][2])/2-((u[0]+u[2])/2))**2+((it['box'][1]+it['box'][3])/2-((u[1]+u[3])/2))**2
            if ic<best_score: best_score=ic; best=gi
        if best is None:
            groups.append([it])
        else:
            groups[best].append(it)

    out=[]
    for g in groups:
        g=sorted(g,key=lambda q:(q['box'][1],q['box'][0]))
        srcs=[str(q.get('text','')).strip() for q in g if str(q.get('text','')).strip()]
        if not srcs: continue
        x1=min(q['box'][0] for q in g); y1=min(q['box'][1] for q in g)
        x2=max(q['box'][2] for q in g); y2=max(q['box'][3] for q in g)
        bw,bh=x2-x1,y2-y1
        # 单框只做很小的安全扩张，避免字幕落在框边缘；双框组也不再扩得过大。
        if len(g)==1:
            pad=max(5,min(12,int(min(bw,bh)*0.12)))
        else:
            pad=max(6,min(14,int(min(bw,bh)*0.10)))
        box=[max(0,x1-pad),max(0,y1-pad),min(page_w,x2+pad),min(page_h,y2+pad)]
        out.append({'id':g[0]['id'],'box':box,'text':'\n'.join(srcs),
                    'type':'bubble_group','member_ids':[q['id'] for q in g],
                    'member_boxes':[q['box'] for q in g],'layout':'bubble'})
    return order_manga_items(out)

def _find_body_clusters(img, items):
    """只把真正的“密集正文列”识别成 body_para：至少 4 个高瘦 OCR 栏，纵向高度接近且横向跨度明显。"""
    cand=[]
    H,W=img.height,img.width
    for it in items:
        b=it['box']; bw,bh=_box_dims(b)
        if not (bh>=max(90,int(H*0.16)) and bw<=max(70,int(W*0.09)) and bh>bw*1.8):
            continue
        # 文字栏之间必须处于较亮区域；人物衣服上的竖纹通常无法满足。
        if _local_light_ratio(img,b,3)<0.45: continue
        cand.append(it)
    if len(cand)<4:return []
    cand=sorted(cand,key=lambda q: -((q['box'][0]+q['box'][2])/2))
    used=set(); clusters=[]
    for i,seed in enumerate(cand):
        if id(seed) in used: continue
        g=[seed]
        sx1,sy1,sx2,sy2=seed['box']
        for j in range(i+1,len(cand)):
            q=cand[j]
            if id(q) in used: continue
            bx1,by1,bx2,by2=q['box']
            ov=max(0,min(sy2,by2)-max(sy1,by1))/max(1,min(sy2-sy1,by2-by1))
            center_gap=abs(((sx1+sx2)/2)-((bx1+bx2)/2))
            if ov>=0.50 and center_gap<=max(80,int(W*0.12)):
                g.append(q)
                sx1=min(sx1,bx1); sx2=max(sx2,bx2); sy1=min(sy1,by1); sy2=max(sy2,by2)
        if len(g)>=4:
            span=sx2-sx1; height=sy2-sy1
            if span>=W*0.15 and height>=H*0.20 and span<=W*0.62:
                clusters.append(g)
                used.update(id(q) for q in g)
    return clusters


def _merge_body_clusters(img, clustered):
    """把正文列群合成一个可翻译的段落项，但保留原成员框供 AI 擦字。"""
    if not clustered:return clustered
    body_clusters=_find_body_clusters(img,clustered)
    if not body_clusters:return clustered
    chosen=set(id(q) for g in body_clusters for q in g)
    out=[q for q in clustered if id(q) not in chosen]
    next_id=max([int(x.get('id',0)) for x in clustered if str(x.get('id','')).isdigit()] or [0])+1
    for g in body_clusters:
        g=sorted(g,key=lambda q:-((q['box'][0]+q['box'][2])/2))
        x1=min(q['box'][0] for q in g); y1=min(q['box'][1] for q in g)
        x2=max(q['box'][2] for q in g); y2=max(q['box'][3] for q in g)
        pad_x=max(16,int(img.width*0.02)); pad_y=max(12,int(img.height*0.014))
        # 正文面板不能跨过页面黑色格线太多；限制在成员框附近。
        box=[max(0,x1-pad_x),max(0,y1-pad_y),min(img.width,x2+pad_x),min(img.height,y2+pad_y)]
        text='\n'.join(str(q.get('text','')).strip() for q in g if str(q.get('text','')).strip())
        out.append({'id':next_id,'box':box,'text':text,'type':'body_para',
                    'member_ids':[q['id'] for q in g], 'member_boxes':[q['box'] for q in g],
                    'layout':'horizontal_paragraph'})
        next_id+=1
    return order_manga_items(out)


def build_translation_layout(img, items):
    """v12.4：在 v12.3 基础上进一步收紧气泡区域合并，防止相邻气泡/两句话被一个大框包住。
    1) 绝不再根据“白色连通背景”生成巨型气泡框；
    2) 先用 OCR 框之间的距离/方向/连接带做保守聚类；
    3) 只有真正满足“多条高瘦正文栏”的区域才合成 body_para；
    4) 所有擦字仍只使用 member_boxes，绝不擦掉整块人物/背景。
    手动锁定的框跳过自动拆分/合并。
    """
    if not items:return []
    if all(x.get('locked') for x in items):
        return layout_from_manual(img, items)
    clustered=_cluster_text_boxes(img,items)
    return _merge_body_clusters(img,clustered)


def auto_kind(img, box):
    x1,y1,x2,y2=map(int,box)
    w,h=max(1,x2-x1),max(1,y2-y1)
    pw,ph=img.size
    if w>=pw*0.38 or (w*h>=pw*ph*0.10 and w>=h*0.85):
        return 'long'
    return 'bubble'


def layout_from_manual(img, items):
    """用户确认后的框：只按气泡/长对白排版，绝不再拆分或合并。"""
    out=[]
    for it in items:
        box=list(map(int,it.get('box') or [0,0,0,0]))
        if len(box)!=4: continue
        kind=it.get('kind') or auto_kind(img, box)
        members=it.get('member_boxes') or [box]
        members=[list(map(int,m)) for m in members if isinstance(m,(list,tuple)) and len(m)==4] or [box]
        row=dict(it)
        row['box']=box
        row['member_boxes']=members
        row['locked']=True
        row['kind']=kind
        if kind=='long':
            row['type']='body_para'
            row['layout']='horizontal_paragraph'
        else:
            row['type']='bubble_group'
            row['layout']='bubble'
        out.append(row)
    return out



def _split_body_para_item(item, max_chars=260, max_members=3):
    lines=[x.strip() for x in str(item.get('text','')).splitlines() if x.strip()]
    if not lines:return []
    mids=list(item.get('member_ids') or [])
    chunks=[]; cur=[]; cur_chars=0
    for idx,line in enumerate(lines):
        add=len(line)
        if cur and (len(cur)>=max_members or cur_chars+add>max_chars):
            chunks.append(cur); cur=[]; cur_chars=0
        cur.append((mids[idx] if idx<len(mids) else None,line)); cur_chars+=add
    if cur:chunks.append(cur)
    return chunks


def build_translation_units(items):
    units=[]
    for item in items:
        if item.get('type')=='body_para':
            parts=_split_body_para_item(item)
            if not parts:continue
            for n,part in enumerate(parts,1):
                units.append({'id':f"{item['id']}_{n}",'orig_id':item['id'],'type':'body_para_segment',
                              'text':'\n'.join(t for _,t in part).strip(),
                              'member_ids':[m for m,_ in part if m is not None],'seq':n})
        else:
            units.append({'id':item['id'],'orig_id':item['id'],'type':item.get('type','bubble'),
                          'text':item.get('text',''),'member_ids':[],'seq':1})
    return units


def _translate_unit_batch(endpoint,key,model,units,timeout,maxout):
    if _is_ollama_endpoint(endpoint):
        # Hy-MT2 官方翻译任务格式：只输出译文，不要求 ID/JSON。
        if len(units) != 1:
            raise RuntimeError("Ollama/Hy-MT2 当前要求单条翻译调用")
        u0=units[0]
        base=str(u0.get("text","")).strip()
        if u0.get('type')=='body_para_segment':
            prompt=(
                "下面是同一段日文正文中连续的多个竖排栏。OCR 可能有少量错字、漏字或连字，请结合上下文修正明显识别错误，再把它们理解为一个连续中文正文段落，按原来的阅读顺序准确翻译。不要逐栏生硬直译，不要凭空扩写。只输出简体中文正文，不要解释，不要加标题，不要输出日文。\n\n"
                + base
            )
        else:
            prompt=(
                "将以下 OCR 识别出的日文翻译为简体中文。OCR 可能有少量错字、漏字或连字，请结合日漫语境修正明显的 OCR 误识后再翻译，但不要凭空增加原文没有的信息。只输出翻译后的结果，不要解释。"
                "翻译要自然、简洁，符合日漫对白语气；保留人名、语气、标点和拟声词。\n\n"
                + base
            )
        content,usage=call_chat(endpoint,key,model,prompt,timeout,maxout)
        zh=str(content or "").strip()
        zh=re.sub(r"^```(?:text)?\s*|\s*```$","",zh,flags=re.I).strip()
        if zh and not _looks_like_refusal(zh):
            return {u0["id"]:zh},usage
        return {},usage

    payload=[{"id":u["id"],"type":u["type"],"text":u["text"]} for u in units]
    prompt=(
        "你是专业日漫汉化编辑。下面是同一页漫画中一个小批次的日文文本。请结合这些条目的上下文翻译，保持语气自然、简洁。"
        "body_para_segment 属于同一个长正文，可能只是长段落的一部分；不要擅自扩写，不要解释。"
        "严格返回 JSON 数组：[{{\"id\":\"输入id\",\"type\":\"输入type\",\"zh\":\"中文\"}}]。必须覆盖所有输入 id。"
        "不要把安全提示、拒绝理由、政策说明当成中文翻译返回。\n"
        + json.dumps(payload,ensure_ascii=False)
    )
    content,u=call_chat(endpoint,key,model,prompt,timeout,maxout)
    data=safe_json_array(content); out={}
    for row in data:
        if not isinstance(row,dict): continue
        zid=str(row.get("id","")); zh=str(row.get("zh","")).strip()
        if zid and zh and not _looks_like_refusal(zh): out[zid]=zh
    return out,u

def chunked_translate(endpoint,key,model,items,timeout,maxout,batch_size=4):
    """v11：云端可小批量；Ollama+Hy-MT2 单条调用，按官方译文格式解析。"""
    units=build_translation_units(items)
    if _is_ollama_endpoint(endpoint):
        batch_size=1
    trans_by_orig={}
    usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    failures=[]
    for i in range(0,len(units),max(1,int(batch_size))):
        batch=units[i:i+max(1,int(batch_size))]
        try:
            got,u=_translate_unit_batch(endpoint,key,model,batch,timeout,maxout)
            for k,v in (u or {}).items():
                if k in usage:
                    usage[k]+=int(v or 0)
            missing=[]
            for unit in batch:
                if unit['id'] in got:
                    trans_by_orig.setdefault(unit['orig_id'],[]).append((unit['seq'],got[unit['id']]))
                else:
                    missing.append(unit['id'])
            if missing:
                failures.append({'kind':'missing_translation','ids':missing})
        except AIRejectedError as e:
            failures.append({'kind':'content_rejected','code':e.code,'message':str(e),'ids':[x['id'] for x in batch]})
        except Exception as e:
            failures.append({'kind':'batch_error','message':str(e),'ids':[x['id'] for x in batch]})
    trans={}
    for orig_id,parts in trans_by_orig.items():
        item=next((x for x in items if x['id']==orig_id),None)
        if item and item.get('type')=='body_para':
            need=len(_split_body_para_item(item))
            if len(parts)==need:
                parts=sorted(parts,key=lambda x:x[0])
                trans[orig_id]='\n\n'.join(v for _,v in parts if v).strip()
        else:
            vals=sorted(parts,key=lambda x:x[0])
            if vals:
                trans[orig_id]=vals[0][1]
    return trans,usage,failures


def contextual_translate(endpoint,key,model,items,timeout,maxout):
    """兼容旧调用名：改为 v11 小批次翻译。"""
    return chunked_translate(endpoint,key,model,items,timeout,maxout,batch_size=4)

def list_images(z):return sorted([n for n in z.namelist() if Path(n).suffix.lower() in IMAGE_EXTS and not n.endswith("/")])



@APP.get("/api/inpaint_status")
def inpaint_status():
    return jsonify(ok=True, ready=ai_inpaint_ready(), model="AnimeMangaInpainting-LaMa", message=("本地 AI 擦字模型已就绪" if ai_inpaint_ready() else "请先运行 download_manga_inpaint_model.bat"))

@APP.get("/")
def index():return render_template('index.html')

@APP.get("/api/ocr_status")
def api_ocr_status():
    try:
        eng=get_pp_ocr()
        return jsonify(ok=True,model="PP-OCRv6_manga v0.2",detector="manga_det_v0.2.onnx",recognizer="manga_rec_v0.2.onnx",provider="CPUExecutionProvider")
    except Exception as e:
        return jsonify(ok=False,error=str(e)),500

@APP.get("/api/ollama_status")
def api_ollama_status():
    try:
        r=requests.get("http://127.0.0.1:11434/api/tags",timeout=5)
        if r.status_code>=400:
            return jsonify(ok=False,error=f"Ollama HTTP {r.status_code}"),502
        models=[m.get("name") for m in r.json().get("models",[]) if isinstance(m,dict)]
        has_hymt2=any((m or "").split(":")[0]=="hy-mt2-7b" for m in models)
        return jsonify(ok=True,models=models,has_hymt2=has_hymt2)
    except Exception as e:
        return jsonify(ok=False,error=str(e)),502

@APP.post("/api/preview")
def api_preview():
    started=time.time()
    try:
        img,name=extract_first_image(request.files["file"])
        print(f"[OCR] image={name} size={img.size}")
        items=run_ocr(img)
        print(f"[OCR] PP-OCRv6_manga regions={len(items)}")
        raw_count=len(items)
        print(f"[OCR] PP-OCRv6_manga finished: {raw_count} regions, {time.time()-started:.1f}s")
        valid=[x for x in items if x.get('text') and not str(x.get('text')).startswith("[OCR失败")]
        for x in valid:x["type"]="bubble"
        logical=build_translation_layout(img,valid)
        token=f"{int(time.time()*1000)}_{os.getpid()}";p=WORK/(token+".jpg");make_preview(img,logical).save(p,quality=92)
        txt="\n\n".join(f"[{x['id']}] TYPE={x.get('type')} BOX={x['box']}\n{x.get('text','')}" for x in logical)
        return jsonify(count=len(logical),raw_count=raw_count,paragraph_count=sum(1 for x in logical if x.get('type')=='body_para'),vertical=sum(1 for x in logical if x["box"][3]-x["box"][1]>(x["box"][2]-x["box"][0])*1.25),
                       horizontal=sum(1 for x in logical if x["box"][3]-x["box"][1]<=(x["box"][2]-x["box"][0])*1.25),
                       preview_url="/api/preview_image/"+token,text=txt,elapsed=round(time.time()-started,1))
    except Exception as e:
        traceback.print_exc();return jsonify(error=str(e),elapsed=round(time.time()-started,1)),500

@APP.post("/api/translate_single")
def api_translate_single():
    """v12：保留 v11 OCR/翻译；重做擦字、气泡布局、正文面板布局；失败不擦原文。"""
    try:
        f=request.files.get("file")
        if not f: raise RuntimeError("请选择一张图片")
        endpoint=request.form.get("endpoint","").strip(); key=request.form.get("key","").strip(); model=request.form.get("model","").strip()
        batch=int(request.form.get("batch",4)); timeout=int(request.form.get("timeout",180)); maxout=int(request.form.get("maxout",2400))
        if not _is_ollama_endpoint(endpoint) and not key: raise RuntimeError("请先填写 API Key")
        img=Image.open(io.BytesIO(f.read())).convert("RGB")
        items=run_ocr(img)
        raw_count=len(items)
        items=dedup_ocr_items(order_manga_items(items))
        valid=[x for x in items if x.get("text") and not x["text"].startswith("[OCR失败")]
        for x in valid:x["type"]="bubble"
        valid=build_translation_layout(img,valid)
        if not valid: raise RuntimeError("没有识别到可翻译文字")
        logical_count=len(valid)
        paragraph_count=sum(1 for x in valid if x.get("type")=="body_para")
        trans,u,failures=chunked_translate(endpoint,key,model,valid,timeout,maxout,batch_size=batch)
        # 只有确认拿到有效中文翻译后，才执行擦字+绘制；失败区域完全保留原图。
        for x in valid:
            if x["id"] in trans and trans[x["id"]].strip():
                draw_item_chinese(img,x,trans[x["id"]])
        buf=io.BytesIO();img.save(buf,format="JPEG",quality=94)
        token=f"single_{int(time.time()*1000)}"; (WORK/(token+".jpg")).write_bytes(buf.getvalue())
        (WORK/(token+"_result.json")).write_text(json.dumps({"ocr":valid,"translated":trans,"usage":u,"failures":failures,"mode":"v11_chunked_translation_safe_draw"},ensure_ascii=False,indent=2),encoding="utf-8")
        rejected=sum(1 for f in failures if f.get('kind')=='content_rejected')
        warning=("部分请求失败，相关区域已保留原文。" if failures else "")
        first_error=(failures[0] if failures else None)
        return jsonify(ok=True,count=logical_count,raw_count=raw_count,paragraph_count=paragraph_count,translated=len(trans),failed=len(failures),rejected=rejected,usage=u,image_url="/api/single_image/"+token,result_url="/api/single_result/"+token,warning=warning,first_error=first_error)
    except Exception as e:
        traceback.print_exc(); return jsonify(ok=False,error=str(e),traceback=traceback.format_exc()),500

@APP.get("/api/single_image/<token>")
def single_image(token):
    p=WORK/(token+".jpg")
    return send_file(p,mimetype="image/jpeg") if p.exists() else ("not found",404)

@APP.get("/api/single_result/<token>")
def single_result(token):
    p=WORK/(token+"_result.json")
    return send_file(p,as_attachment=True,download_name=p.name,mimetype="application/json") if p.exists() else ("not found",404)

@APP.get("/api/preview_image/<token>")
def preview_image(token):
    p=WORK/(token+".jpg")
    return send_file(p,mimetype="image/jpeg") if p.exists() else ("not found",404)

def _encode_output_image(img,name):
    ext=Path(name).suffix.lower(); bio=io.BytesIO()
    if ext in {'.jpg','.jpeg'}: img.save(bio,format='JPEG',quality=96,optimize=True,subsampling=0)
    elif ext=='.png': img.save(bio,format='PNG',optimize=True)
    elif ext=='.webp': img.save(bio,format='WEBP',quality=97,method=6)
    else: img.save(bio,format='PNG',optimize=True)
    return bio.getvalue()

def _task_update(task_id,**kw):
    with TASK_LOCK:
        if task_id in TASKS: TASKS[task_id].update(kw)

def _write_task_preview(task_id,img):
    p=WORK/f'task_{task_id}'/'current_preview.jpg'; p.parent.mkdir(parents=True,exist_ok=True)
    im=img.copy(); im.thumbnail((900,1200),Image.Resampling.LANCZOS); im.save(p,format='JPEG',quality=90)

def run_batch_task(task_id,archive_path,endpoint,key,model,batch,maxout,timeout,locked_pages=None):
    task_dir=WORK/f'task_{task_id}'; pages_dir=task_dir/'pages'; started=time.time(); failures=[]
    usage_total={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    src_name=archive_path or ''
    out_ext='.cbz' if str(src_name).lower().endswith('.cbz') else '.zip'; out_path=WORK/f'translated_{task_id}{out_ext}'
    try:
        if locked_pages:
            pages=[{'name':p['name'],'path':p['path'],'boxes':p.get('boxes') or []} for p in locked_pages]
            names=[p['name'] for p in pages]
            total=len(pages)
            _task_update(task_id,status='running',total=total,message=f'已锁定 {total} 页，开始按人工框翻译。')
        else:
            _task_update(task_id,status='extracting',message='正在解压并扫描漫画图片…')
            names=extract_archive_to_dir(archive_path,pages_dir)
            if not names: raise RuntimeError('压缩包里没有图片')
            pages=[{'name':n,'path':str(pages_dir/n),'boxes':None} for n in names]
            total=len(names); _task_update(task_id,status='running',total=total,message=f'已发现 {total} 张图片，开始逐页处理。')
        with zipfile.ZipFile(out_path,'w',zipfile.ZIP_DEFLATED) as zo:
            for page_no,page in enumerate(pages,1):
                name=page['name']; t0=time.time(); raw_bytes=Path(page['path']).read_bytes()
                _task_update(task_id,current=page_no,current_name=name,current_status='OCR' if not page.get('boxes') else '锁定框',completed=page_no-1,message=f'第 {page_no}/{total}：{"使用人工锁定框" if page.get("boxes") is not None else "OCR 识别"}…')
                try:
                    img=Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                    if page.get('boxes') is not None:
                        valid=layout_from_manual(img, page['boxes'])
                        items=valid
                    else:
                        items=dedup_ocr_items(order_manga_items(run_ocr(img)))
                        valid=[x for x in items if x.get('text') and not str(x['text']).startswith('[OCR失败')]
                        for x in valid: x['type']='bubble'
                        valid=build_translation_layout(img,valid)
                    _task_update(task_id,current_status='AI翻译',regions=len(valid),ocr=len(items),message=f'第 {page_no}/{total}：翻译 {len(valid)} 个区域…')
                    trans,u,local_failures=chunked_translate(endpoint,key,model,valid,timeout,maxout,batch_size=batch) if valid else ({},{},[])
                    for x in valid:
                        if x['id'] in trans and trans[x['id']].strip(): draw_item_chinese(img,x,trans[x['id']])
                    _task_update(task_id,current_status='AI擦字+排版',translated=len(trans),message=f'第 {page_no}/{total}：本地 AI 擦字、原文风格字体与对比度增强…')
                    img=enhance_final_image(img)
                    zo.writestr(name,_encode_output_image(img,name))
                    _write_task_preview(task_id,img)
                    for k in usage_total: usage_total[k]+=int(u.get(k,0) or 0)
                    for fe in local_failures: failures.append({'page':page_no,'name':name,**fe})
                    _task_update(task_id,current_status='完成',completed=page_no,failed=sum(1 for f in failures if f.get('kind') in {'batch_error','content_rejected','missing_translation','page_error'}),translated=len(trans),message=f'第 {page_no}/{total} 张完成：{name}，翻译 {len(trans)}/{len(valid)}，耗时 {time.time()-t0:.1f}s')
                except Exception as e:
                    traceback.print_exc(); failures.append({'page':page_no,'name':name,'kind':'page_error','message':str(e)}); zo.writestr(name,raw_bytes)
                    try: _write_task_preview(task_id,Image.open(io.BytesIO(raw_bytes)).convert('RGB'))
                    except Exception: pass
                    _task_update(task_id,current_status='失败（原图保留）',completed=page_no,failed=sum(1 for f in failures if f.get('kind')=='page_error'),message=f'第 {page_no}/{total} 张失败，已保留原图：{name}')
        (task_dir/'result.json').write_text(json.dumps({'task_id':task_id,'total':total,'failures':failures,'usage':usage_total},ensure_ascii=False,indent=2),encoding='utf-8')
        _task_update(task_id,status='done',current_status='完成',message=f'全部完成：{total} 张；失败/部分失败 {len(failures)}；总耗时 {time.time()-started:.1f}s',usage=usage_total,download=f'/api/task_download/{task_id}')
    except Exception as e:
        traceback.print_exc(); _task_update(task_id,status='error',current_status='错误',error=str(e),message=str(e))
    finally:
        if archive_path:
            try: Path(archive_path).unlink(missing_ok=True)
            except Exception: pass

@APP.post('/api/batch_start')
def api_batch_start():
    f=request.files.get('file')
    try:
        if not f: raise RuntimeError('请选择 ZIP / CBZ / 7Z / RAR')
        ext=Path(f.filename or '').suffix.lower()
        if ext not in {'.zip','.cbz','.7z','.rar'}: raise RuntimeError('批量输入支持 ZIP / CBZ / 7Z / RAR')
        endpoint=request.form.get('endpoint','').strip(); key=request.form.get('key','').strip(); model=request.form.get('model','hy-mt2-7b').strip()
        batch=int(request.form.get('batch',4)); maxout=int(request.form.get('maxout',1600)); timeout=int(request.form.get('timeout',180))
        if not _is_ollama_endpoint(endpoint) and not key: raise RuntimeError('请填写 API Key')
        task_id=uuid.uuid4().hex[:12]; td=WORK/f'task_{task_id}'; td.mkdir(parents=True,exist_ok=True)
        archive_path=td/(Path(f.filename or ('input'+ext)).name); f.save(archive_path)
        with TASK_LOCK: TASKS[task_id]={'task_id':task_id,'status':'queued','current':0,'total':0,'current_name':'','current_status':'排队','completed':0,'failed':0,'message':'任务已创建，准备处理…','started_at':time.time()}
        threading.Thread(target=run_batch_task,args=(task_id,str(archive_path),endpoint,key,model,batch,maxout,timeout),daemon=True).start()
        return jsonify(ok=True,task_id=task_id)
    except Exception as e:
        return jsonify(ok=False,error=str(e)),400

@APP.get('/api/batch_status/<task_id>')
def api_batch_status(task_id):
    with TASK_LOCK: st=dict(TASKS.get(task_id) or {})
    if not st: return jsonify(ok=False,error='任务不存在或服务器已重启'),404
    return jsonify(ok=True,**st,preview_url=f'/api/task_preview/{task_id}')

@APP.get('/api/task_preview/<task_id>')
def api_task_preview(task_id):
    p=WORK/f'task_{task_id}'/'current_preview.jpg'
    return send_file(p,mimetype='image/jpeg') if p.exists() else ('not found',404)

@APP.get('/api/task_download/<task_id>')
def api_task_download(task_id):
    p=WORK/f'translated_{task_id}.cbz';
    if not p.exists(): p=WORK/f'translated_{task_id}.zip'
    return send_file(p,as_attachment=True,download_name=p.name,mimetype='application/zip') if p.exists() else ('not found',404)

@APP.post("/api/translate")
def api_translate_compat():
    return api_batch_start()

@APP.get("/api/download/<token>")
def download(token):
    p=WORK/(token+".cbz");return send_file(p,as_attachment=True,download_name=p.name,mimetype="application/zip") if p.exists() else ("not found",404)


def _review_get(sid):
    with TASK_LOCK:
        st=REVIEWS.get(sid)
        return st


def run_review_ocr(sid):
    st=_review_get(sid)
    if not st: return
    try:
        for i,page in enumerate(st['pages']):
            st['current']=i+1; st['current_name']=page['name']; st['current_status']='OCR'
            st['message']=f'第 {i+1}/{st["total"]}：OCR 识别…'
            img=Image.open(page['path']).convert('RGB')
            items=run_ocr(img)
            boxes=[]
            for it in items:
                box=list(map(int,it['box']))
                boxes.append({
                    'id':it['id'],'box':box,'text':str(it.get('text') or ''),
                    'kind':auto_kind(img,box),'locked':False,
                    'member_boxes':[box]
                })
            page['boxes']=boxes; page['width'],page['height']=img.size
            st['completed']=i+1
            st['message']=f'第 {i+1}/{st["total"]} OCR 完成：{page["name"]}，{len(boxes)} 个原始框。请人工确认。'
        st['status']='review'; st['current_status']='人工框选'
        st['message']='OCR 完成。请拖动/缩放/删除/新增/合并框，改文字后锁定，再开始翻译。锁定框不会再被自动拆分或合并。'
    except Exception as e:
        traceback.print_exc(); st['status']='error'; st['error']=str(e); st['message']=str(e)


@APP.post('/api/review_start')
def api_review_start():
    f=request.files.get('file')
    try:
        if not f: raise RuntimeError('请选择图片或 ZIP / CBZ / 7Z / RAR')
        sid=uuid.uuid4().hex[:12]; td=WORK/f'review_{sid}'; pages_dir=td/'pages'; pages_dir.mkdir(parents=True,exist_ok=True)
        name=f.filename or 'input'; ext=Path(name).suffix.lower()
        pages=[]
        if ext in IMAGE_EXTS:
            dest=pages_dir/Path(name).name; f.save(dest)
            pages=[{'name':dest.name,'path':str(dest),'boxes':[],'confirmed':False,'width':0,'height':0}]
        elif ext in {'.zip','.cbz','.7z','.rar'}:
            archive=td/Path(name).name; f.save(archive)
            names=extract_archive_to_dir(archive,pages_dir)
            if not names: raise RuntimeError('压缩包里没有图片')
            pages=[{'name':n,'path':str(pages_dir/n),'boxes':[],'confirmed':False,'width':0,'height':0} for n in names]
        else:
            raise RuntimeError('支持图片、ZIP、CBZ、7Z、RAR')
        st={'id':sid,'status':'ocr','total':len(pages),'completed':0,'current':0,'current_name':'','current_status':'OCR','message':'开始 OCR…','pages':pages,'started_at':time.time()}
        with TASK_LOCK: REVIEWS[sid]=st
        threading.Thread(target=run_review_ocr,args=(sid,),daemon=True).start()
        return jsonify(ok=True,session_id=sid,total=len(pages))
    except Exception as e:
        return jsonify(ok=False,error=str(e)),400


@APP.get('/api/review_status/<sid>')
def api_review_status(sid):
    st=_review_get(sid)
    if not st: return jsonify(ok=False,error='会话不存在或服务器已重启'),404
    confirmed=sum(1 for p in st['pages'] if p.get('confirmed'))
    return jsonify(ok=True,status=st.get('status'),total=st.get('total'),completed=st.get('completed'),current=st.get('current'),current_name=st.get('current_name'),current_status=st.get('current_status'),message=st.get('message'),confirmed=confirmed,error=st.get('error'))


@APP.get('/api/review_page/<sid>/<int:idx>')
def api_review_page(sid,idx):
    st=_review_get(sid)
    if not st: return jsonify(ok=False,error='会话不存在'),404
    if idx<0 or idx>=len(st['pages']): return jsonify(ok=False,error='页码越界'),404
    p=st['pages'][idx]
    return jsonify(ok=True,index=idx,name=p['name'],width=p.get('width') or 0,height=p.get('height') or 0,boxes=p.get('boxes') or [],confirmed=bool(p.get('confirmed')),image_url=f'/api/review_image/{sid}/{idx}')


@APP.get('/api/review_image/<sid>/<int:idx>')
def api_review_image(sid,idx):
    st=_review_get(sid)
    if not st or idx<0 or idx>=len(st['pages']): return ('not found',404)
    p=Path(st['pages'][idx]['path'])
    if not p.exists(): return ('not found',404)
    mime='image/jpeg'
    suf=p.suffix.lower()
    if suf=='.png': mime='image/png'
    elif suf=='.webp': mime='image/webp'
    return send_file(p,mimetype=mime)


@APP.post('/api/review_save/<sid>/<int:idx>')
def api_review_save(sid,idx):
    st=_review_get(sid)
    if not st: return jsonify(ok=False,error='会话不存在'),404
    if idx<0 or idx>=len(st['pages']): return jsonify(ok=False,error='页码越界'),404
    data=request.get_json(silent=True) or {}
    boxes=data.get('boxes') or []
    cleaned=[]
    for i,b in enumerate(boxes,1):
        box=b.get('box')
        if not isinstance(box,(list,tuple)) or len(box)!=4: continue
        members=b.get('member_boxes') or [list(box)]
        cleaned.append({
            'id':int(b.get('id') or i),'box':[int(x) for x in box],
            'text':str(b.get('text') or ''),'kind': 'long' if b.get('kind')=='long' else 'bubble',
            'locked':True,'member_boxes':[[int(x) for x in m] for m in members if isinstance(m,(list,tuple)) and len(m)==4] or [list(map(int,box))]
        })
    st['pages'][idx]['boxes']=cleaned
    st['pages'][idx]['confirmed']=bool(data.get('confirmed',True))
    return jsonify(ok=True,count=len(cleaned),confirmed=st['pages'][idx]['confirmed'])


@APP.post('/api/review_recognize/<sid>/<int:idx>')
def api_review_recognize(sid,idx):
    st=_review_get(sid)
    if not st: return jsonify(ok=False,error='会话不存在'),404
    if idx<0 or idx>=len(st['pages']): return jsonify(ok=False,error='页码越界'),404
    data=request.get_json(silent=True) or {}
    box=data.get('box')
    if not isinstance(box,(list,tuple)) or len(box)!=4: return jsonify(ok=False,error='缺少 box'),400
    img=Image.open(st['pages'][idx]['path']).convert('RGB')
    x1,y1,x2,y2=map(int,box)
    pad=2
    cb=(max(0,x1-pad),max(0,y1-pad),min(img.width,x2+pad),min(img.height,y2+pad))
    try:
        text=get_pp_ocr().recognize(img.crop(cb)).strip()
    except Exception as e:
        text=f'[OCR失败: {e}]'
    return jsonify(ok=True,text=text)


@APP.post('/api/review_translate/<sid>')
def api_review_translate(sid):
    st=_review_get(sid)
    try:
        if not st: raise RuntimeError('请先完成 OCR 框选')
        endpoint=request.form.get('endpoint','').strip(); key=request.form.get('key','').strip(); model=request.form.get('model','hy-mt2-7b').strip()
        batch=int(request.form.get('batch',4)); maxout=int(request.form.get('maxout',1600)); timeout=int(request.form.get('timeout',180))
        if not _is_ollama_endpoint(endpoint) and not key: raise RuntimeError('请填写 API Key')
        locked=[]
        for p in st['pages']:
            boxes=p.get('boxes') or []
            for b in boxes: b['locked']=True
            p['confirmed']=True
            locked.append({'name':p['name'],'path':p['path'],'boxes':boxes})
        task_id=uuid.uuid4().hex[:12]; td=WORK/f'task_{task_id}'; td.mkdir(parents=True,exist_ok=True)
        with TASK_LOCK: TASKS[task_id]={'task_id':task_id,'status':'queued','current':0,'total':len(locked),'current_name':'','current_status':'排队','completed':0,'failed':0,'message':'按人工锁定框开始翻译…','started_at':time.time()}
        threading.Thread(target=run_batch_task,args=(task_id,None,endpoint,key,model,batch,maxout,timeout,locked),daemon=True).start()
        return jsonify(ok=True,task_id=task_id)
    except Exception as e:
        return jsonify(ok=False,error=str(e)),400


if __name__=="__main__":
    print("JP Manga Translator v12.6 - 人工框选 + 批量进度 + 原文风格字体")
    print("打开浏览器: http://127.0.0.1:7860")
    APP.run(host="127.0.0.1",port=7860,debug=False)

