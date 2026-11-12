from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, math
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'contact_sheets'; OUT.mkdir(exist_ok=True)
font=ImageFont.load_default()
for model_dir in sorted((ROOT/'results').glob('*')):
    if not model_dir.is_dir(): continue
    for scen_dir in sorted(model_dir.glob('*')):
        imgs=[]
        for seed_dir in sorted(scen_dir.glob('seed_*')):
            p=seed_dir/'image.png'
            if p.exists(): imgs.append((seed_dir.name,p))
        if not imgs: continue
        thumb_w=320; thumb_h=360; cols=3; rows=math.ceil(len(imgs)/cols)
        sheet=Image.new('RGB',(cols*thumb_w,rows*thumb_h),(245,245,245)); d=ImageDraw.Draw(sheet)
        for i,(label,p) in enumerate(imgs):
            im=Image.open(p).convert('RGB'); im.thumbnail((thumb_w-12,thumb_h-42))
            x=(i%cols)*thumb_w+(thumb_w-im.width)//2; y=(i//cols)*thumb_h+5
            sheet.paste(im,(x,y)); d.text(((i%cols)*thumb_w+8,(i//cols)*thumb_h+thumb_h-30),label,fill=(0,0,0),font=font)
        out=OUT/f'{model_dir.name}__{scen_dir.name}.jpg'; sheet.save(out,quality=92)
print('wrote',len(list(OUT.glob('*.jpg'))),'contact sheets to',OUT)
