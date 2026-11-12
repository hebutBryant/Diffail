# Fundamental factors in one-shot multi-subject image generation

## Scope and execution

This study uses local, frozen checkpoints and one complete sampling call per image. Every main image starts from random noise, receives the complete prompt, uses no external image, no editing pass, no attention modification, and no prompt rewriting. We used 512×512 outputs and seeds 0, 1, and 2.

The full controlled scenario matrix is in [`scenarios.json`](scenarios.json), generation parameters in [`generation_configs.json`](generation_configs.json), raw images in [`results/`](results/), visual comparisons in [`contact_sheets/`](contact_sheets/), detector proxies in [`metrics.csv`](metrics.csv), and per-factor observations in [`failure_analysis.md`](failure_analysis.md).

## Model audit

| Requested model | Local status | Architecture/interface | Main-experiment status |
|---|---|---|---|
| Qwen-Image-2.1 | present, 33.13 GB | 7B visual single-stream DiT, Qwen3-VL processor/encoder, `QwenImage21Pipeline` | complete, 63 images |
| FLUX.2-klein | requested path absent | complete local same-family fallback `/home/hdd4/lpz/FLUX.2-klein-9B`: 9B distilled rectified-flow transformer, Qwen3 text encoder, `Flux2KleinPipeline` | complete, 63 images; clearly labelled fallback |
| FLUX.1-Kontext-dev | directory exists but is empty (4 KB) | no local weights or usable pipeline | excluded from main comparison; no reliable pure-text run possible |
| FLUX.1-dev2 | present, 89 GB | 12B rectified-flow transformer, CLIP + T5, `FluxPipeline` | smoke/partial run only because its 8-step offloaded inference is much slower; its completed images are retained and marked partial |

The exact machine-readable audit is [`model_inventory.json`](model_inventory.json). Qwen and FLUX.2-klein use CPU offload on GPU 3 and were loaded sequentially. No weights were changed or downloaded.

## What most affects one-shot generation?

The dominant factor is **joint constraint density**, which combines object count, attribute assignments, relations, occlusion, and interaction topology. A single class of six apples in a regular arrangement was reliable for both Qwen and FLUX.2-klein. Adding object categories and colours, then adding relations and physical contact, caused failures even when the objects themselves remained recognizable.

The second factor is **spatial/interaction realization**. In high spatial prompts, both models frequently generated the requested objects but arranged them as a plausible still-life or catalogue scene. In interaction prompts, they often generated people, umbrella and book but failed to preserve exact hand ownership and contact. This is stronger evidence than a raw “missing object” count because it separates semantic presence from scene correctness.

The third factor is **scene density and occlusion**. Closely packed objects are merged, nested, or made partially illegible. A larger prompt does not create more image canvas or more object slots; it increases the number of constraints competing for the same latent spatial support.

The fourth factor is **compositional prior mismatch**. Common combinations such as person riding bicycle were stable. Rare configurations such as a teddy bear riding a bicycle while a cat sits on a skateboard and a bird perches on the handlebar were replaced by familiar, visually coherent variants or augmented with extra animals. This is prior substitution, not failure to render the individual nouns.

## Cross-model consistency

Qwen and FLUX.2-klein differ in visual style and robustness, but the failure ordering is consistent:

1. isolated single-class counting is easiest;
2. several isolated objects with attributes are harder;
3. directed spatial relations are less reliable than object existence;
4. contact, support, handing and occlusion are least reliable;
5. rare compositions invite familiar-scene substitution.

FLUX.2-klein was generally stronger at preserving isolated objects and realistic interactions. Qwen produced more soft, catalogue-like arrangements and more severe blending in dense scenes. These are quality differences, not different fundamental failure classes.

## Representation, grounding, prior, and realization

**Semantic representation.** The models usually represent the nouns and broad attributes: apples, cups, books, people, umbrellas and bicycles are visibly present. The count negative control confirms that basic object concepts can be represented.

**Conditional grounding.** Attribute binding and relation binding degrade when many tokens must be assigned to many instances. The output evidence is compatible with attention competition or leakage, but this experiment did not collect read-only attention traces; it therefore does not establish attention leakage as the cause. A text encoder change alone should not be expected to solve the observed failures.

**Compositional prior.** Rare object combinations are replaced by familiar scene templates or augmented with plausible distractors. This is visible even when each requested object is individually well rendered.

**Visual realization.** Dense layouts, occlusions, hand contact, support, depth and front/behind relations fail after coarse semantics are present. These require object-level identity persistence, geometry and collision/contact reasoning through the entire denoising trajectory.

## Can prompt rewriting solve it?

Prompt rewriting can help with lexical ambiguity, explicit object lists, and simple left/right relations. It cannot reliably solve exact counting under overlap, precise ownership of hands, front/behind depth, collision-free support, or rare multi-object interactions. The one-shot failures persisted with explicit wording such as “each object separate and fully visible”; this is evidence that extra prose cannot substitute for object/layout structure.

## Which objects can be separated?

Objects with weak coupling—independent still-life items, separated by space, with distinct colours and no interaction—can often be handled jointly. Objects coupled by geometry must be jointly represented: a person and the object being held, rider and bicycle, child and umbrella, book and hands, or any pair linked by support/occlusion. Splitting these into independent generations would destroy contact topology and depth ordering unless a later composition stage restores it.

## Decomposition strategy implied by each mechanism

| Failure mechanism | Suitable decomposition |
|---|---|
| lexical/attribute ambiguity | structured object records and attribute checks; prompt normalization may be enough |
| object omission/counting | object slots or count-aware generation, then verify count |
| spatial relation failure | explicit layout/bounding boxes or a scene graph before rendering |
| dense occlusion | depth-aware grouping and ordered layer rendering |
| interaction/contact failure | joint interaction groups (person–hand–object–support), pose/geometry constraints |
| rare composition prior substitution | retrieve/compose object-level priors or use adaptive scene planning |

This supports the hypothesis **different failure mechanisms require different decomposition strategies**. A single fixed rule such as “generate every object independently” fails for interactions; a single monolithic prompt fails for dense scenes; a fixed layout decomposition can over-constrain ordinary scenes.

## Fixed versus adaptive decomposition

A fixed strategy cannot resolve all cases because the correct unit of decomposition changes with the relation graph. Independent apples and cups can be separated, while a rider/bicycle pair must stay together. A rare scene may need a layout plan, whereas a common isolated still life does not. The appropriate strategy is therefore adaptive:

1. parse the prompt into objects, attributes, and relations;
2. build connected components over strong geometric relations (holding, riding, supporting, hugging, occluding);
3. keep each component jointly generated;
4. place weakly coupled components with explicit layout/depth constraints;
5. verify object existence, attributes, count, relations and contact;
6. re-decompose only the failed component.

This is a research implication, not an intervention performed in the main experiment. The current results show why the strategy is needed; they do not claim to have demonstrated a complete adaptive generator.

## Limitations

The primary comparison is Qwen versus FLUX.2-klein fallback. FLUX.1-Kontext cannot be compared because its requested local directory contains no weights. FLUX.1-dev2 was run only partially due to substantially slower offloaded inference. Detector metrics are conservative proxies and cannot establish colour binding, depth, contact, or interaction correctness. Manual inspection is included through contact sheets. No VLM score is used as sole ground truth, and no causal attention experiment was performed in this run.
