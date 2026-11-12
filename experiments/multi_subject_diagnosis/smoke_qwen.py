import os, time, torch
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from diffusers import QwenImage21Pipeline
p='/home/hdd4/lpz/Qwen-Image-2.1'
print('loading',flush=True); t=time.time()
pipe=QwenImage21Pipeline.from_pretrained(p,torch_dtype=torch.bfloat16,local_files_only=True)
pipe.enable_model_cpu_offload(gpu_id=3)
print('loaded',time.time()-t,flush=True)
g=torch.Generator(device='cuda:3').manual_seed(0)
out=pipe(prompt='A studio photo of exactly three red apples arranged in a row on a white table, each apple separate and fully visible.',width=512,height=512,output_resolution=512,num_inference_steps=8,true_cfg_scale=1.0,use_kv_cache=True,generator=g)
out.images[0].save('/home/lipz/Diffail/experiments/multi_subject_diagnosis/smoke_qwen.png')
print('done',time.time()-t,flush=True)
