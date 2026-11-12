#!/usr/bin/env python
"""Detector-based, auditable proxies for object/count/spatial metrics.

These are deliberately conservative: detector failure is recorded separately from
semantic judgement, and interaction/attribute scores remain proxies requiring review.
"""
import argparse, csv, json, math, os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
ALIASES = {"teddy bear": ["teddy bear", "bear"], "person": ["person"], "bicycle": ["bicycle"], "apple": ["apple"], "banana": ["banana"], "cup": ["cup"], "book": ["book"], "backpack": ["backpack"], "umbrella": ["umbrella"], "green backpack": ["backpack"], "chair": ["chair"], "dog": ["dog"], "cat": ["cat"], "skateboard": ["skateboard"], "bird": ["bird"]}

def relation_ok(a, relation, b, boxes):
    if not boxes.get(a) or not boxes.get(b): return None
    ax = max(boxes[a], key=lambda x:x[4]); bx = max(boxes[b], key=lambda x:x[4])
    ac=((ax[0]+ax[2])/2,(ax[1]+ax[3])/2); bc=((bx[0]+bx[2])/2,(bx[1]+bx[3])/2)
    dx,dy=ac[0]-bc[0],ac[1]-bc[1]
    if relation == 'left_of': return dx < -0.03*512
    if relation == 'right_of': return dx > 0.03*512
    if relation == 'above': return dy < -0.03*512
    if relation == 'below': return dy > 0.03*512
    if relation == 'beside': return abs(dx) > abs(dy)
    if relation == 'behind': return None  # monocular 2D detector cannot establish depth
    return None  # holding/riding/contact require human/VLM review

def hue_color(rgb):
    # rough color proxy from pixels inside a detection box; texture/background makes this noisy
    import colorsys, numpy as np
    arr=np.asarray(rgb).reshape(-1,3).astype(float)/255
    if len(arr)==0:return None
    # saturation-weighted mean hue, reject near-white/gray pixels
    hsv=np.array([colorsys.rgb_to_hsv(*p) for p in arr])
    sat=hsv[:,1]; keep=(sat>0.25)&(hsv[:,2]>0.15)
    if keep.sum()<10:return None
    h=float(np.median(hsv[keep,0]))
    if h<.04 or h>.96:return 'red'
    if h<.12:return 'orange'
    if h<.20:return 'yellow'
    if h<.45:return 'green'
    if h<.58:return 'cyan'
    if h<.75:return 'blue'
    if h<.90:return 'purple'
    return 'red'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,default=3); ap.add_argument('--results',type=Path,default=ROOT/'results'); ap.add_argument('--out',type=Path,default=ROOT/'metrics.csv'); ap.add_argument('--detector',default='/home/hdd4/lpz/yolo26/yolo26l.pt'); args=ap.parse_args()
    from PIL import Image
    from ultralytics import YOLO
    import torch
    model=YOLO(args.detector)
    rows=[]; det_dir=ROOT/'detector_outputs'; det_dir.mkdir(exist_ok=True)
    for meta_path in sorted(args.results.glob('*/*/seed_*/metadata.json')):
        meta=json.loads(meta_path.read_text()); spec=meta['scenario']; img_path=meta_path.parent/'image.png'; im=Image.open(img_path).convert('RGB')
        pred=model.predict(str(img_path), imgsz=512, conf=0.20, device=f'cuda:{args.gpu}', verbose=False)[0]
        names=pred.names; boxes=defaultdict(list); all_det=[]
        if pred.boxes is not None:
            for xyxy,conf,cls in zip(pred.boxes.xyxy.detach().cpu().tolist(), pred.boxes.conf.detach().cpu().tolist(), pred.boxes.cls.detach().cpu().tolist()):
                label=str(names[int(cls)]).lower(); x1,y1,x2,y2=map(float,xyxy); item=[x1,y1,x2,y2,float(conf),label]; all_det.append(item)
                for key,aliases in ALIASES.items():
                    if label in aliases: boxes[key].append(item)
        (det_dir / f"{meta['model_id']}__{spec['id']}__seed{meta['seed']}.json").write_text(json.dumps({'image':str(img_path),'detections':all_det},indent=2))
        exp_counts={}
        for obj in spec['objects']: exp_counts[obj['class']]=exp_counts.get(obj['class'],0)+obj['count']
        actual_counts={k:len(v) for k,v in boxes.items()}
        existence=[]; binding=[]
        for obj in spec['objects']:
            got=boxes.get(obj['class'],[]); existence.append(1.0 if len(got)>=obj['count'] else min(1.0,len(got)/max(1,obj['count'])))
            if obj.get('color') and got:
                colors=[]
                for b in sorted(got,key=lambda x:x[4],reverse=True)[:obj['count']]: colors.append(hue_color(im.crop((int(b[0]),int(b[1]),int(b[2]),int(b[3])))))
                binding.append(sum(c==obj['color'] for c in colors)/len(colors))
        rels=[]; rel_unknown=0
        for r in spec['relations']:
            v=relation_ok(r['a'].split('_')[0],r['relation'],r['b'].split('_')[0],boxes)
            if v is None: rel_unknown+=1
            else: rels.append(float(v))
        total_expected=sum(exp_counts.values()); total_detected=sum(actual_counts.values())
        rows.append({'model_id':meta['model_id'],'scenario_id':spec['id'],'factor':spec['factor'],'level':spec['level'],'seed':meta['seed'],'image_path':str(img_path),
                     'expected_object_instances':total_expected,'detected_object_instances':total_detected,'object_existence_score':round(sum(existence)/len(existence),4) if existence else '',
                     'count_exact_all_classes':int(all(actual_counts.get(k,0)==v for k,v in exp_counts.items())),'count_error_abs':sum(abs(actual_counts.get(k,0)-v) for k,v in exp_counts.items()),
                     'attribute_binding_proxy':round(sum(binding)/len(binding),4) if binding else '',
                     'spatial_relation_observable_accuracy':round(sum(rels)/len(rels),4) if rels else '', 'spatial_relations_unobservable':rel_unknown,
                     'interaction_geometry_manual':'REVIEW','vlm_review':'REVIEW','manual_notes':''})
    fields=list(rows[0]) if rows else []
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')

if __name__=='__main__': main()
