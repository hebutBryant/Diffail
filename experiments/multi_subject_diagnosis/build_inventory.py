import json, pathlib, os, hashlib
paths={
'Qwen-Image-2.1':'/home/hdd4/lpz/Qwen-Image-2.1',
'FLUX.2-klein_requested':'/home/hdd4/lpz/FLUX.2-klein',
'FLUX.2-klein_fallback':'/home/hdd4/lpz/FLUX.2-klein-9B',
'FLUX.1-Kontext-dev':'/home/hdd4/lpz/FLUX.1-Kontext-dev',
'FLUX.1-dev2':'/home/hdd4/lpz/FLUX.1-dev2'}
out={}
for name,p in paths.items():
 d=pathlib.Path(p); rec={'path':p,'exists':d.exists(),'is_dir':d.is_dir(),'size_gb':round(sum(x.stat().st_size for x in d.rglob('*') if x.is_file())/1e9,2) if d.is_dir() else 0}
 for fn in ['model_index.json','transformer/config.json','scheduler/scheduler_config.json','README.md']:
  f=d/fn
  if f.exists():
   if fn.endswith('.json'): rec[fn]=json.loads(f.read_text())
   else: rec[fn+'_first_lines']=f.read_text(errors='ignore').splitlines()[:80]
 out[name]=rec
pathlib.Path('/home/lipz/Diffail/experiments/multi_subject_diagnosis/model_inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
