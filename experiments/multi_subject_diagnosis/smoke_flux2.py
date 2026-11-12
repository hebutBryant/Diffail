import os,time,torch
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from diffusers import Flux2KleinPipeline
p='/home/hdd4/lpz/FLUX.2-klein-9B'
print('loading',flush=True); t=time.time()
pipe=Flux2KleinPipeline.from_pretrained(p,torch_dtype=torch.bfloat16,local_files_only=True)
pipe.enable_model_cpu_offload(gpu_id=3)
print('loaded',time.time()-t,flush=True)
g=torch.Generator(device='cuda:3').manual_seed(0)
out=pipe(prompt='A studio photo of exactly three red apples arranged in a row on a white table, each apple separate and fully visible.',width=512,height=512,num_inference_steps=4,guidance_scale=1.0,generator=g)
out.images[0].save('/home/lipz/Diffail/experiments/multi_subject_diagnosis/smoke_flux2.png')
print('done',time.time()-t,flush=True)
