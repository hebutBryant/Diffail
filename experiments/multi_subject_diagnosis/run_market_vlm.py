import json, os, torch
from pathlib import Path
from PIL import Image
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if 'default' not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS['default'] = ROPE_INIT_FUNCTIONS.get('linear', next(iter(ROPE_INIT_FUNCTIONS.values())))
except Exception:
    pass

ROOT=Path(__file__).resolve().parent; img=ROOT/'market_attention/market_flux1dev2_seed4242.png'; out=ROOT/'market_attention/vlm_review.json'
prompt='''Inspect this generated wet-market image against the requested scene. Return strict JSON with keys: present_objects (list), missing_or_wrong_objects (list), wrong_attributes (list), wrong_relations (list), short_reason (string). Check these requested elements: blue-green striped awnings; damp stone floor; middle-aged female vendor wearing yellow apron behind vegetable stall; vendor handing bag of tomatoes; young woman in blue jacket with bamboo basket; woman left hand taking plastic bag and right hand holding red apple; elderly man in gray hat and dark green jacket selecting purple eggplants; young boy with red backpack on tiptoe pointing at yellow bananas. Do not infer hidden objects; only report visibly supported elements.'''
print('loading VLM',flush=True)
processor=AutoProcessor.from_pretrained('/home/hdd4/lpz/LLaVA-OneVision-1.5-8B-Instruct',trust_remote_code=True,local_files_only=True)
processor.tokenizer.padding_side='left'
if processor.tokenizer.pad_token_id is None: processor.tokenizer.pad_token_id=processor.tokenizer.eos_token_id
cfg=AutoConfig.from_pretrained('/home/hdd4/lpz/LLaVA-OneVision-1.5-8B-Instruct',trust_remote_code=True,local_files_only=True)
cfg.pad_token_id=processor.tokenizer.pad_token_id
if hasattr(cfg,'text_config'):
    cfg.text_config.pad_token_id=processor.tokenizer.pad_token_id
    cfg.text_config.rope_type='linear'
    cfg.text_config.rope_parameters={'rope_type':'linear','factor':1.0}
model=AutoModelForCausalLM.from_pretrained('/home/hdd4/lpz/LLaVA-OneVision-1.5-8B-Instruct',config=cfg,torch_dtype=torch.bfloat16,device_map={'':1},trust_remote_code=True,local_files_only=True,attn_implementation='eager',low_cpu_mem_usage=True)
messages=[{'role':'user','content':[{'type':'image','image':str(img)},{'type':'text','text':prompt}]}]
try: text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
except Exception: text=prompt
inputs=processor(text=[text],images=[Image.open(img).convert('RGB')],padding=True,return_tensors='pt')
inputs={k:v.to('cuda:1') if hasattr(v,'to') else v for k,v in inputs.items()}
with torch.no_grad(): ids=model.generate(**inputs,max_new_tokens=700,do_sample=False)
trim=[o[len(i):] for i,o in zip(inputs['input_ids'],ids)]
answer=processor.batch_decode(trim,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
out.write_text(json.dumps({'prompt':prompt,'raw_answer':answer,'image':str(img)},ensure_ascii=False,indent=2))
print(answer)
