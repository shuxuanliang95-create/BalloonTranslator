import os, time, hashlib, requests

ROOT=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(ROOT,'models','AnimeMangaInpainting')
os.makedirs(OUT,exist_ok=True)
FILE='anime-manga-big-lama.pt'
LOCAL=os.path.join(OUT,FILE)
URLS=[
    'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt',
    'https://huggingface.co/df1412/anime-big-lama/resolve/main/anime-manga-big-lama.pt?download=true',
    'https://hf-mirror.com/df1412/anime-big-lama/resolve/main/anime-manga-big-lama.pt?download=true',
]
MIN_SIZE=100_000_000
if os.path.exists(LOCAL) and os.path.getsize(LOCAL)>MIN_SIZE:
    print('[OK] model exists:',LOCAL, '%.1f MB'%(os.path.getsize(LOCAL)/1048576)); raise SystemExit(0)
for url in URLS:
    tmp=LOCAL+'.part'
    try:
        print('\nSOURCE:',url)
        t=time.time()
        with requests.get(url,stream=True,timeout=(20,180),allow_redirects=True,headers={'User-Agent':'JP-Manga-Translator/12.2'}) as r:
            print('HTTP:',r.status_code,'final:',r.url)
            r.raise_for_status()
            ctype=(r.headers.get('content-type') or '').lower()
            total=int(r.headers.get('content-length') or 0)
            if total and total<MIN_SIZE: raise RuntimeError('response too small: %s bytes'%total)
            got=0; last=0
            with open(tmp,'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if not chunk: continue
                    f.write(chunk); got+=len(chunk)
                    now=time.time()
                    if now-last>1:
                        if total: print('  %.1f/%.1f MB'%(got/1048576,total/1048576),end='\r')
                        else: print('  %.1f MB'% (got/1048576),end='\r')
                        last=now
        print()
        if os.path.getsize(tmp)<MIN_SIZE: raise RuntimeError('downloaded file too small: %.1f MB'%(os.path.getsize(tmp)/1048576))
        os.replace(tmp,LOCAL)
        print('[OK] model:',LOCAL)
        print('SIZE: %.1f MB  TIME: %.1fs'%(os.path.getsize(LOCAL)/1048576,time.time()-t))
        raise SystemExit(0)
    except Exception as e:
        print('\n[FAIL]',type(e).__name__,str(e)[:600])
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except: pass
print('\n[ERROR] all model sources failed.')
print('You can manually download anime-manga-big-lama.pt and place it in:',LOCAL)
raise SystemExit(1)
