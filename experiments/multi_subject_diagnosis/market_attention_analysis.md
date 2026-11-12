# Wet-market prompt: generation and attention analysis

## Run

The structured decomposition is in [`market_prompt_decomposition.json`](market_prompt_decomposition.json). FLUX.1-dev2 was run from random noise with the complete prompt, 512×512 output, seed 4242, guidance scale 3.5, and 30 inference steps. No layout, editing, attention bias, or external image was used.

The generated image is [`market_flux1dev2_seed4242.png`](market_attention/market_flux1dev2_seed4242.png). The attention recorder was read-only and restored all 57 attention processors after sampling. For every denoising step 0–29 it saved 32×32 image-token maps for 22 semantic token groups and compressed matrices containing the tracked-token rows.

Attention files:

- [`manifest.json`](market_attention/manifest.json) records prompt, groups, token indices and sampling parameters.
- [`matrices/`](market_attention/matrices/) contains `step_00.npz` through `step_29.npz` and per-group 32×32 maps.
- [`attention_summary.json`](market_attention/attention_summary.json) contains image-mass, entropy concentration and peak patch per group and step.
- [`attention_final_step_contact_sheet.jpg`](market_attention/attention_final_step_contact_sheet.jpg) overlays the final-step group maps for manual inspection.

## Visual result

The scene is visually coherent as a market, and it contains the awnings, wet stone floor, central pair, boy with a red backpack, hanging bananas, fruit/vegetable displays and a basket. However, the requested constraints are not all satisfied:

- The central vendor appears male/elderly rather than clearly a middle-aged woman.
- The vendor and the woman both appear in yellow apron-like clothing, weakening the requested attribute separation.
- A clear tomato bag being transferred is not visible; the hands meet, but the object ownership is ambiguous.
- The woman visibly carries a basket, but the red apple in her right hand is not clearly present. A detector found an apple-like region in the lower-right produce area, not a hand-held apple.
- The elderly-man/gray-hat figure is not clearly selecting purple eggplants; the left figure is more plausibly the boy with a backpack, while the central gray-haired figure is the person in the apron.
- Bananas are clearly visible overhead, and the boy/backpack are present, but the pointing gesture toward the bananas is weak or absent.
- The vegetable stall and general market prior are strong, but person identity, hand-object ownership and role binding are not.

## Attention observations

The final-step overlays show broad attention for most groups rather than clean, isolated object-specific peaks. `vendor`, `woman`, `elderly_man`, `green_jacket`, and `yellow_apron` frequently peak on the central person pair or nearby regions. `red_backpack` and `boy` peak more consistently on the left child. `yellow_bananas` has strong response near the upper-right hanging bananas, while `tomatoes` and `vegetable_stall` often spread over the right-side produce display. `red_apple` does not form a stable, distinct hand-localized response.

These are correlational observations. The maps are full-softmax text-to-image attention fractions averaged over heads and layers; they do not prove that attention caused a particular visual error. The main evidence for missing/wrong objects comes from the generated image and post-hoc detector output, not attention alone.

## VLM status

The local LLaVA-OneVision-1.5-8B checkpoint was attempted as an independent visual reviewer, but the available Transformers 5.17 environment is incompatible with its custom configuration: the model first required a missing `pad_token_id`/RoPE compatibility patch, then reported unexpected and missing language-model weights, and generation rejected `mm_token_type_ids`. The older local environment could not instantiate the processor. Because this would make the VLM result unreliable, no VLM answer is treated as evidence. The raw failed attempts are represented by `run_market_vlm.py`; manual inspection and the YOLO detector output were retained instead.

## Likely failure mechanism

The image demonstrates a **role/attribute/interaction binding failure** more than a global scene-generation failure. The model learned the market prior and many individual objects, but it did not maintain separate slots for vendor, recipient, elderly shopper and child while also assigning hand ownership and transfer relations. This is consistent with the broader controlled experiments: coarse object presence survives, while joint relations, role attributes and contact topology fail.
