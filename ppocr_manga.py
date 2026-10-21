# -*- coding: utf-8 -*-
"""PP-OCRv6_manga v0.2 ONNX runtime for local manga OCR.
Based on the official Kellenok/PP-OCRv6_manga ONNX/OpenVINO pipeline.
Runs on CPU so Ollama can keep the RTX 3070 for Hy-MT2-7B.
"""
import os, math
from pathlib import Path
import cv2
import numpy as np
import pyclipper
import onnxruntime as ort
from PIL import Image

class PPOMangaOCR:
    def __init__(self, root=None, provider="CPUExecutionProvider"):
        base = Path(root) if root else Path(__file__).resolve().parent / "models" / "PP-OCRv6_manga"
        self.root = base
        det = base / "det" / "manga_det_v0.2.onnx"
        rec = base / "rec" / "manga_rec_v0.2.onnx"
        dic = base / "ppocrv6_dict.txt"
        missing=[str(p) for p in (det,rec,dic) if not p.exists()]
        if missing:
            raise FileNotFoundError("PP-OCRv6_manga 模型文件缺失：\n" + "\n".join(missing) + "\n请先运行 download_ppocr_manga_models.bat")
        available=ort.get_available_providers()
        use=[provider] if provider in available else ["CPUExecutionProvider"]
        self.det_sess=ort.InferenceSession(str(det), providers=use)
        self.rec_sess=ort.InferenceSession(str(rec), providers=use)
        self.det_input=self.det_sess.get_inputs()[0].name
        self.rec_input=self.rec_sess.get_inputs()[0].name
        with open(dic,"r",encoding="utf-8") as f:
            self.vocab=["blank"]+[line.rstrip("\r\n") for line in f]+[" "]

    @staticmethod
    def _box_score_fast(pred, box):
        h,w=pred.shape[:2]
        b=box.copy().astype(np.int32)
        xmin=int(np.clip(np.floor(b[:,0].min()),0,w-1)); xmax=int(np.clip(np.ceil(b[:,0].max()),0,w-1))
        ymin=int(np.clip(np.floor(b[:,1].min()),0,h-1)); ymax=int(np.clip(np.ceil(b[:,1].max()),0,h-1))
        if xmax<xmin or ymax<ymin:
            return 0.0
        score_mask=np.zeros((ymax-ymin+1,xmax-xmin+1),dtype=np.uint8)
        b[:,0]-=xmin; b[:,1]-=ymin
        cv2.fillPoly(score_mask,[b.reshape(-1,2)],1)
        return float(cv2.mean(pred[ymin:ymax+1,xmin:xmax+1],score_mask)[0])

    @staticmethod
    def _unclip(box, ratio=1.4):
        x,y=box[:,0],box[:,1]
        area=0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
        perimeter=np.linalg.norm(np.roll(box,-1,axis=0)-box,axis=1).sum()
        if area<=0 or perimeter<=0:
            return None
        offset=pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(),pyclipper.JT_ROUND,pyclipper.ET_CLOSEDPOLYGON)
        paths=offset.Execute(area*ratio/perimeter)
        return np.asarray(paths[0],dtype=np.float32) if len(paths)==1 else None

    def detect(self, pil_img):
        img=np.array(pil_img.convert("RGB"))
        # v12.3 精确模式：尽量把漫画页面保持在更高检测分辨率。
        # 旧版对长页面使用面积缩放，容易让细小竖排字在检测前就被压小。
        H,W=img.shape[:2]
        target_long=1536.0
        ratio=min(1.0,target_long/max(H,W))
        rh=max(int(round(H*ratio/32)*32),32)
        rw=max(int(round(W*ratio/32)*32),32)
        inp=cv2.resize(img,(rw,rh)).astype(np.float32)/255.0
        inp=(inp-np.array([0.485,0.456,0.406],dtype=np.float32))/np.array([0.229,0.224,0.225],dtype=np.float32)
        pred=self.det_sess.run(None,{self.det_input:inp.transpose((2,0,1))[None,...]})[0]
        pred=np.asarray(pred)
        if pred.ndim==4:
            pred_map=pred[0,0]
        elif pred.ndim==3:
            pred_map=pred[0]
        else:
            pred_map=pred
        mask=(pred_map>0.15)
        contours,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for cnt in contours:
            pts=cnt.squeeze(1)
            if pts.ndim!=2 or pts.shape[0]<4: continue
            if self._box_score_fast(pred_map,pts)<0.25: continue
            pts=self._unclip(pts,1.30)
            if pts is None or len(pts)==0: continue
            rect=cv2.minAreaRect(pts.astype(np.float32))
            _,(bw,bh),_=rect
            if min(bw,bh)<3: continue
            q=cv2.boxPoints(rect).astype(np.float32)
            q[:,0]=np.clip(np.round(q[:,0]/rw*W),0,W)
            q[:,1]=np.clip(np.round(q[:,1]/rh*H),0,H)
            xs,ys=q[:,0],q[:,1]
            x1,y1,x2,y2=int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
            if x2-x1<4 or y2-y1<4: continue
            boxes.append((x1,y1,x2,y2))
        # stable top-to-bottom/left-to-right order before recognition
        boxes.sort(key=lambda b:(b[1],b[0]))
        return boxes

    def _recognize_once(self, crop, variant='orig'):
        c=np.asarray(crop.convert("RGB"))
        if c.size==0:
            return "", 0.0
        if variant == 'gray':
            g=cv2.cvtColor(c,cv2.COLOR_RGB2GRAY)
            c=np.repeat(g[...,None],3,axis=2)
        elif variant == 'clahe':
            g=cv2.cvtColor(c,cv2.COLOR_RGB2GRAY)
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            g=clahe.apply(g)
            c=np.repeat(g[...,None],3,axis=2)
        if c.shape[0]>c.shape[1]:
            c=cv2.rotate(c,cv2.ROTATE_90_COUNTERCLOCKWISE)
        ch,cw=c.shape[:2]
        natural_w=int(round(48.0*cw/max(1,ch)))
        # v12.3 提高识别输入宽上限，小字/长横排不再被过早压扁。
        target_w=max(16,min(1024,natural_w))
        c_inp=cv2.resize(c,(target_w,48),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255.0
        c_inp=((c_inp-0.5)/0.5).transpose((2,0,1))[None,...]
        logits=self.rec_sess.run(None,{self.rec_input:c_inp})[0]
        logits=np.asarray(logits)
        if logits.ndim==3:
            logits=logits[0]
        # CTC greedy + 平均 top-1 概率，供多路识别择优。
        z=logits-logits.max(axis=-1,keepdims=True)
        prob=np.exp(z); prob=prob/prob.sum(axis=-1,keepdims=True)
        idx=np.argmax(logits,axis=-1)
        confs=[]
        out=[]; prev=0
        for pos,i0 in enumerate(idx):
            i=int(i0)
            conf=float(prob[pos,i]) if pos < len(prob) else 0.0
            if i==0 or i==prev:
                prev=i
                continue
            if 0<=i<len(self.vocab):
                out.append(self.vocab[i]); confs.append(conf)
            prev=i
        text="".join(out).strip()
        conf=float(sum(confs)/len(confs)) if confs else 0.0
        return text,conf

    @staticmethod
    def _ocr_suspicious(text, conf, crop_size):
        t=str(text or '').strip()
        if not t:
            return True
        # 极短/低置信度/明显乱码：进入第二、第三路识别。
        if conf < 0.82:
            return True
        if len(t) == 1 and min(crop_size) < 24:
            return True
        bad=sum(1 for ch in t if ch in '¤�□■')
        return bad>0

    def recognize(self, pil_crop):
        c=pil_crop.convert('RGB')
        text,conf=self._recognize_once(c,'orig')
        candidates=[(text,conf,'orig')]
        if self._ocr_suspicious(text,conf,c.size):
            for v in ('gray','clahe'):
                try:
                    tt,cc=self._recognize_once(c,v)
                    candidates.append((tt,cc,v))
                except Exception:
                    pass
        # 置信度优先，同时避免把纯空白结果选为最佳。
        candidates=[q for q in candidates if q[0]] or candidates
        candidates.sort(key=lambda q:(q[1],len(q[0])>0), reverse=True)
        return candidates[0][0] if candidates else ''

    def ocr_page(self,pil_img,progress=None):
        boxes=self.detect(pil_img)
        items=[]
        total=len(boxes)
        for i,b in enumerate(boxes,1):
            x1,y1,x2,y2=b
            # 给识别器留一点抗锯齿/字符边缘上下文。
            pad=max(3,min(7,int(min(max(1,x2-x1),max(1,y2-y1))*0.05)))
            cb=(max(0,x1-pad),max(0,y1-pad),min(pil_img.width,x2+pad),min(pil_img.height,y2+pad))
            try:
                text=self.recognize(pil_img.crop(cb)).strip()
            except Exception as e:
                text=f"[OCR失败: {e}]"
            # Drop completely blank detections, keep explicit OCR errors for diagnostics.
            if text or text.startswith("[OCR失败"):
                items.append({"id":i,"box":[x1,y1,x2,y2],"text":text})
            if progress: progress(i,total)
        return items
