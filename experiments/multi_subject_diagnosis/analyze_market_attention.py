import json, numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent; d=ROOT/'market_attention/matrices'; out=ROOT/'market_attention'
manifest=json.loads((out/'manifest.json').read_text()); names=list(manifest['groups'])
summary={}
for name in names:
 vals=[]
 for step in range(30):
  f=d/f'step_{step:02d}__{name}.npy'
  if not f.exists(): continue
  m=np.load(f).astype(float); p=np.clip(m.ravel(),0,None); s=p.sum(); p=p/s if s else p
  ent=-(p*np.log(p+1e-12)).sum()/np.log(len(p)) if s else 1.0
  peak=np.unravel_index(np.argmax(m),m.shape)
  vals.append({'step':step,'image_mass':float(s),'concentration':float(1-ent),'peak_xy_patch':[int(peak[1]),int(peak[0])],'peak_value':float(m[peak])})
 summary[name]=vals
(out/'attention_summary.json').write_text(json.dumps({'definition':'image_mass is the sum of full-softmax attention from group text tokens to image patches; concentration is 1-normalized entropy; peak coordinates are 32x32 image-token grid','groups':summary},indent=2))
base=Image.open(out/'market_flux1dev2_seed4242.png').convert('RGB').resize((256,256)); cols=4; cell=300; rows=(len(names)+cols-1)//cols
sheet=Image.new('RGB',(cols*cell,rows*cell),'white'); dr=ImageDraw.Draw(sheet)
for i,name in enumerate(names):
 f=d/f'step_29__{name}.npy'
 if not f.exists(): continue
 m=np.load(f).astype(float); lo,hi=np.percentile(m,[65,99.5]); z=np.clip((m-lo)/(hi-lo+1e-9),0,1); z=Image.fromarray((z*255).astype('uint8')).resize((256,256)); heat=Image.merge('RGB',(z,Image.new('L',(256,256),0),Image.new('L',(256,256),0))); panel=Image.blend(base,heat,.55); x=(i%cols)*cell; y=(i//cols)*cell; sheet.paste(panel,(x,y)); dr.text((x+4,y+260),name,fill='black')
sheet.save(out/'attention_final_step_contact_sheet.jpg',quality=92)
print('wrote',out/'attention_summary.json',out/'attention_final_step_contact_sheet.jpg')
