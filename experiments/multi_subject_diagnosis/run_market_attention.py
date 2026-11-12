#!/usr/bin/env python
"""One-shot FLUX market scene + read-only text-to-image attention maps."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'market_attention'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
sys.path.insert(0,'/home/lipz/GraphDiff')
from experiments.uncontrolled_relation_attention_visualization import RelationAttnRecorder, install_recorder, uninstall_recorder, locate_relation_tokens

PROMPT="""In the early morning, at a bustling traditional wet market, sunlight filters through blue-and-green striped awnings, casting patterns onto the damp stone-paved floor. In the center of the scene, a middle-aged female vendor wearing a yellow apron stands behind a vegetable stall, handing a bag of tomatoes to a young woman in a blue jacket who carries a bamboo basket. The woman reaches out with her left hand to take the plastic bag, while holding a red apple in her right hand. To the left of the stall, an elderly man in a gray hat and a dark green jacket selects some purple eggplants; nearby, a young boy with a red backpack stands on tiptoe, pointing toward a bunch of yellow bananas hanging above the display."""

# Groups are semantic probes, not generation controls. Repeated words are retained
# so the saved token-index manifest makes ambiguity explicit.
GROUPS={
 'vendor':['vendor'], 'yellow_apron':['yellow','apron'], 'vegetable_stall':['vegetable','stall'],
 'tomatoes':['tomatoes'], 'bag':['bag','plastic'], 'woman':['woman'], 'blue_jacket':['blue','jacket'],
 'bamboo_basket':['bamboo','basket'], 'left_hand':['left','hand'], 'red_apple':['red','apple'],
 'right_hand':['right','hand'], 'elderly_man':['elderly','man'], 'gray_hat':['gray','hat'],
 'green_jacket':['dark','green','jacket'], 'eggplants':['purple','eggplants'], 'boy':['boy'],
 'red_backpack':['red','backpack'], 'tiptoe':['tiptoe'], 'pointing':['pointing'],
 'yellow_bananas':['yellow','bananas'], 'awnings':['blue','green','striped','awnings'], 'floor':['damp','stone','floor']}

def main():
    gpu=1; device=f'cuda:{gpu}'; seed=4242; steps=30; width=height=512
    from diffusers import FluxPipeline
    print('loading FLUX',flush=True); t0=time.time()
    pipe=FluxPipeline.from_pretrained('/home/hdd4/lpz/FLUX.1-dev2',torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    print('loaded',round(time.time()-t0,2),flush=True)
    tokenizer=pipe.tokenizer_2
    groups_json, tracked=locate_relation_tokens(PROMPT,tokenizer)
    # locate_relation_tokens expects built-in relation keys; convert its token map
    # into our own groups with offset mapping directly through tokenizer.
    ids, offsets, _ = tokenizer(PROMPT, return_offsets_mapping=True, max_length=512, truncation=True, padding='max_length').input_ids, tokenizer(PROMPT, return_offsets_mapping=True, max_length=512, truncation=True, padding='max_length').offset_mapping, None
    low=PROMPT.lower()
    import re
    token_groups={}
    all_ids=[]
    for name,words in GROUPS.items():
        toks=[]
        for w in words:
            for m in re.finditer(r'\b'+re.escape(w.lower())+r'\b',low):
                s,e=m.start(),m.end()
                toks += [i for i,(os_,oe) in enumerate(offsets) if oe>os_ and oe>s and os_<e]
        toks=sorted(set(toks)); token_groups[name]=toks; all_ids += toks
    all_ids=sorted(set(all_ids))
    manifest={'prompt':PROMPT,'groups':GROUPS,'group_token_indices':token_groups,'tracked_token_ids':all_ids,
              'model':'/home/hdd4/lpz/FLUX.1-dev2','seed':seed,'width':width,'height':height,'steps':steps,
              'guidance_scale':3.5,'dtype':'bfloat16','device':device,'cpu_offload':True,
              'attention_semantics':'read-only text-token query to image-token attention; averaged over heads and recorded layers; no attention modification'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rec=RelationAttnRecorder(all_ids,list(range(steps)))
    installed=install_recorder(pipe.transformer,rec)
    def cb(p,step,timestep,vals):
        rec.set_step(min(step+1,steps-1)); return vals
    rec.set_step(0)
    g=torch.Generator(device=device).manual_seed(seed)
    try:
        result=pipe(prompt=PROMPT,height=height,width=width,num_inference_steps=steps,guidance_scale=3.5,generator=g,callback_on_step_end=cb,callback_on_step_end_tensor_inputs=['latents'])
    finally:
        uninstall_recorder(pipe.transformer,installed)
    image=result.images[0]; image.save(OUT/'market_flux1dev2_seed4242.png')
    # save each step's aggregated group x image-token matrix (R,P), plus per-layer
    # raw tracked rows in compressed NPZ.  Matrix is unnormalized image mass from
    # full softmax, so row sums measure how much text-token attention reaches image.
    mat_dir=OUT/'matrices'; mat_dir.mkdir(exist_ok=True)
    for step in range(steps):
        layer_mats=[v.astype(np.float32) for (s,_),v in rec.records.items() if s==step]
        if not layer_mats: continue
        layer_mean=np.mean(np.stack(layer_mats),axis=0) # [tracked_tokens,P]
        group_rows={name: layer_mean[[all_ids.index(i) for i in toks]].mean(axis=0) if toks else np.zeros(layer_mean.shape[1],np.float32) for name,toks in token_groups.items()}
        np.savez_compressed(mat_dir/f'step_{step:02d}.npz', group_names=np.array(list(group_rows),dtype=object), group_matrix=np.stack(list(group_rows.values())).astype(np.float32), tracked_token_ids=np.array(all_ids), layer_mean=layer_mean.astype(np.float32))
        for name,row in group_rows.items(): np.save(mat_dir/f'step_{step:02d}__{name}.npy',row.astype(np.float32).reshape(32,32))
    meta={'elapsed_seconds':round(time.time()-t0,2),'captured_steps':rec.steps_available(),'captured_layers':rec.captured_layers,'text_seq_len':rec.text_seq_len,'num_image_tokens':rec.num_image_tokens,'image_path':str(OUT/'market_flux1dev2_seed4242.png')}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__': main()
