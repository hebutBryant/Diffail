#!/usr/bin/env python
"""Run controlled one-shot generations sequentially, without editing or attention control."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SCENARIOS = json.loads((ROOT / "scenarios.json").read_text())
SCENARIO_BY_ID = {x["id"]: x for x in SCENARIOS}
SEEDS = [0, 1, 2]
MODEL_INFO = {
    "qwen_image_2_1": {
        "path": "/home/hdd4/lpz/Qwen-Image-2.1",
        "pipeline": "QwenImage21Pipeline",
        "architecture": "7B single-stream DiT, Qwen3-VL text encoder, FlowMatchEulerDiscreteScheduler, VAE",
        "dtype": "bfloat16", "steps": 8, "guidance": 1.0, "cpu_offload": True,
        "resolution": [512, 512], "mode": "text-to-image; also supports editing/RGBA but those are excluded here",
    },
    "flux_2_klein_9b": {
        "path": "/home/hdd4/lpz/FLUX.2-klein-9B",
        "requested_path": "/home/hdd4/lpz/FLUX.2-klein",
        "pipeline": "Flux2KleinPipeline",
        "architecture": "9B distilled rectified-flow transformer, Qwen3 causal text encoder, AutoencoderKLFlux2",
        "dtype": "bfloat16", "steps": 4, "guidance": 1.0, "cpu_offload": True,
        "resolution": [512, 512], "mode": "text-to-image and image-to-image/multi-reference editing",
        "note": "Requested directory was absent; this complete local FLUX.2-klein-9B directory is used as a same-family fallback.",
    },
    "flux_1_dev2": {
        "path": "/home/hdd4/lpz/FLUX.1-dev2",
        "pipeline": "FluxPipeline",
        "architecture": "12B rectified-flow transformer, CLIP + T5 text encoders, AutoencoderKL",
        "dtype": "bfloat16", "steps": 8, "guidance": 3.5, "cpu_offload": True,
        "resolution": [512, 512], "mode": "text-to-image",
    },
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def load_pipe(model_id, gpu):
    info = MODEL_INFO[model_id]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if model_id == "qwen_image_2_1":
        from diffusers import QwenImage21Pipeline
        pipe = QwenImage21Pipeline.from_pretrained(info["path"], torch_dtype=torch.bfloat16, local_files_only=True)
    elif model_id == "flux_2_klein_9b":
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(info["path"], torch_dtype=torch.bfloat16, local_files_only=True)
    else:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(info["path"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    return pipe

def generate_one(pipe, model_id, spec, seed, gpu):
    info = MODEL_INFO[model_id]
    prompt = spec["prompt"]
    gen = torch.Generator(device=f"cuda:{gpu}").manual_seed(seed)
    kwargs = dict(prompt=prompt, width=512, height=512, generator=gen)
    if model_id == "qwen_image_2_1":
        kwargs.update(output_resolution=512, num_inference_steps=info["steps"], true_cfg_scale=info["guidance"], use_kv_cache=True)
    elif model_id == "flux_2_klein_9b":
        kwargs.update(num_inference_steps=info["steps"], guidance_scale=info["guidance"])
    else:
        kwargs.update(num_inference_steps=info["steps"], guidance_scale=info["guidance"], max_sequence_length=256)
    started = time.time()
    result = pipe(**kwargs)
    return result.images[0], round(time.time() - started, 3), kwargs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen_image_2_1", "flux_2_klein_9b", "flux_1_dev2", "both", "all"], default="both")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected = [SCENARIO_BY_ID[x] for x in args.scenarios] if args.scenarios else SCENARIOS
    models = list(MODEL_INFO) if args.model == "all" else (["qwen_image_2_1", "flux_2_klein_9b"] if args.model == "both" else [args.model])
    configs = {"models": MODEL_INFO, "seeds": args.seeds, "resolution": [512,512], "prompt_rewriting": False,
               "external_images": False, "attention_control": False, "sampling": "one complete text-to-image diffusion/flow sampling run"}
    (ROOT / "generation_configs.json").write_text(json.dumps(configs, ensure_ascii=False, indent=2))
    for model_id in models:
        print(json.dumps({"stage":"load","model":model_id,"gpu":args.gpu}), flush=True)
        pipe = load_pipe(model_id, args.gpu)
        info = MODEL_INFO[model_id]
        for spec in selected:
            for seed in args.seeds:
                folder = args.out / model_id / spec["id"] / f"seed_{seed}"
                image_path = folder / "image.png"
                meta_path = folder / "metadata.json"
                if meta_path.exists() and image_path.exists():
                    print(json.dumps({"stage":"skip","folder":str(folder)}), flush=True); continue
                folder.mkdir(parents=True, exist_ok=True)
                image, elapsed, call_kwargs = generate_one(pipe, model_id, spec, seed, args.gpu)
                image.save(image_path)
                meta = {"model_id":model_id,"model":info,"scenario":spec,"seed":seed,
                        "prompt":spec["prompt"],"call_kwargs":{k:(str(v) if k=='generator' else v) for k,v in call_kwargs.items()},
                        "elapsed_seconds":elapsed,"image_sha256":sha256(image_path),"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                print(json.dumps({"stage":"saved","model":model_id,"scenario":spec["id"],"seed":seed,"seconds":elapsed,"path":str(image_path)}), flush=True)
        del pipe
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"stage":"unload","model":model_id}), flush=True)

if __name__ == "__main__": main()
