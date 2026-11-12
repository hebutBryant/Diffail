# Failure analysis by scenario

This file records the first-pass human inspection of the contact sheets. `metrics.csv` contains detector-based proxies; detector results are not treated as ground truth for colour, depth, contact, or interaction. Each statement below is based on the three generated seeds and the linked contact sheet.

## Count

- `count_simple`, `count_medium`, and `count_high`: both Qwen-Image-2.1 and FLUX.2-klein produced the requested 1, 3, and 6 apples in the inspected sheets. The objects were separate and the 6-apple grid was stable. This is a negative control: count alone is easy when all instances share one class, colour, material, and a simple layout.
- This result rules out the claim that the models are globally unable to count. Counting becomes a failure candidate when count is coupled to category diversity, occlusion, relations, or a less stereotyped composition.

## Attribute binding

- `attr_simple`: both models generally produced the apple and banana, but Qwen showed weaker visual distinctness in some seeds.
- `attr_medium`: Qwen frequently merged or visually entangled the apple/banana/backpack arrangement; FLUX.2-klein usually preserved three isolated objects.
- `attr_high`: Qwen generated the five named objects as isolated catalogue-like items, but object placement was arbitrary and some shapes/colours were only approximately correct. FLUX.2-klein preserved the five object identities and colour families much more consistently.
- Evidence supports an interaction between object density and binding. It does not by itself prove attention leakage: the same output can arise from weak text grounding, prior retrieval, or the latent layout process.

## Spatial relations

- `spatial_simple` through `spatial_high`: object existence was often better than relation satisfaction. In the high case, Qwen commonly showed apple, cup, book, bicycle and chair but arranged them in a visually plausible catalogue/table scene rather than enforcing all requested directed relations.
- FLUX.2-klein showed the same pattern more strongly in the high spatial sheet: the five objects were present and photorealistic, but `apple left of cup`, `book above cup`, and `bicycle behind chair` were not jointly maintained.
- The failure is therefore conditional grounding plus visual realization of relations, not simple object recognition. A detector can verify left/right in 2D, but it cannot verify behind without depth; those rows are marked accordingly in `metrics.csv`.

## Geometric interaction

- `interaction_simple`: holding an umbrella was usually visually plausible.
- `interaction_medium`: handing a book between two people was unstable; the book was often held by one person or simply placed near both.
- `interaction_high`: Qwen often produced a woman/child/umbrella/book scene, but hand contact and who held which object were inconsistent. FLUX.2-klein generated realistic mother/child/umbrella/book scenes more often, yet exact hand ownership remained variable.
- This is a visual-realization bottleneck: the nouns are present, but contact topology, pose, and ownership are not preserved. Prompt rewriting alone is unlikely to guarantee it.

## Compositional prior

- `prior_simple` (person riding bicycle) was easy for both models because it is a common image prior.
- `prior_medium` remained mostly plausible.
- `prior_high` was revealing. Qwen often changed the requested unusual animal/vehicle arrangement into a crowded but related scene; FLUX.2-klein tended to preserve the teddy bear on a bicycle, cat/skateboard and bird, but added extra animals or altered the exact support/contact relation.
- Rare combinations therefore fail through prior substitution and scene completion. The model can generate each object independently while still replacing the requested joint configuration with a familiar scene template.

## Scene density

- `density_simple`: separated objects were reliable.
- `density_medium`: close placement caused overlap and occasional shape/identity ambiguity.
- `density_high`: Qwen commonly packed objects into a single pile, with severe occlusion and weakened object boundaries. FLUX.2-klein handled the same dense arrangement more cleanly but still merged or nested objects (e.g. apple/cup/banana/book around the backpack).
- Density is a direct visual-realization constraint: all tokens can be semantically represented while the finite image canvas cannot provide separable, legible instances.

## Semantic complexity and order

- `semantic_simple` and `semantic_medium` were often close to the requested scene.
- `semantic_high` produced more variability, but the dominant failure remained the spatial relation realization rather than total object omission. The nested descriptions were not catastrophic, yet they reduced robustness across seeds.
- The current experiment compares equivalent meanings with different syntax, but does not isolate every possible token permutation. The observed sensitivity is suggestive, not a causal estimate of token-order effects.

## Mechanism diagnosis

The strongest evidence is a dissociation: object existence can remain high while joint relation, contact, or density correctness falls. This points to a pipeline bottleneck after coarse semantic recognition: maintaining multiple instance identities and converting them into a globally consistent layout/geometry. The experiment does not support the stronger claim that cross-attention leakage is the sole or primary cause. Read-only attention traces were not collected for these pure-text runs; therefore attention competition is a plausible mechanism, not a measured causal explanation here.

## FLUX.1-dev2 partial control

The available FLUX.1-dev2 checkpoint was additionally run on the high-complexity cases and selected count controls with one seed. Its 8-step offloaded samples were visibly under-refined/blurred in several controls, so they are retained as an execution check and qualitative reference rather than pooled with the Qwen/FLUX.2-klein comparison. This is why the main cross-model conclusions use the two models with complete 21-scenario × 3-seed coverage.
