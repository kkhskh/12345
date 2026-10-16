# Paper claims after Experiment 2D

The 2D run on Pythia-1B (`clean_pca` r8, 64 examples, 144 sites) forces a split of the original
single-loop story into three claims. Do not collapse them.

Organizing principle:

```text
represent → detect → monitor → intervene → track emergence → generalize to LLMs
```

## The three claims

| Claim | Status | Evidence |
|---|---|---|
| Detect a task-related internal representation | Not established on LLMs. To be tested in 0B, where the latent state is known. | Factual `clean_pca` is a variance subspace, not a proven concept. Toy modular addition has exact cyclic ground truth. |
| Obstruction monitors inconsistency beyond margin / confidence | Promising (2B). Needs the confound audit (`run_04d`) and, later, paraphrase transfer. | 2B `clean_pca` r8: AUC 0.976, partial corr 0.70. 2D: obstruction-only AUC for actual `ΔO` is 0.943; attribution's is 0.518. Predicting that `O` falls is not the same as proving `O` is the right semantic energy. |
| Obstruction-response beats attribution at causal repair-site ranking | **Not established. 2D is negative for this.** | Held-out repair AUC: attribution 0.979 vs best obstruction-using score 0.955. Lift −0.0245, CI entirely negative. Pareto lift +0.0086, CI includes 0. The selected score is `β=2`, i.e. almost attribution. |

`PASS_WEAK` is the predefined 2D gate firing on a tiny Pareto mean. It is **not** a claim of superior steering.

## Roles the evidence supports

```text
Obstruction  = monitoring  (does distributed evidence glue?)
Attribution  = intervention prediction  (which site changes the margin?)
```

Keep 2B and 2C. Do not market 2D as a better patcher.

## What 2D did establish

- A linearized score computed **before** patching predicts observed `Δm` (Spearman ~0.91). That score is `grad m · Δh` (attribution).
- `pred_ΔO` predicts observed obstruction reduction (AUC 0.94). That validates the local Taylor map for `O`, not the meaning of `O`.
- Combining them does not improve repair-site selection.

Statistical unit: 32 held-out **examples**. The 4,608 rows are sites inside those examples. `run_06e` already resamples whole examples.

## 2B audit still owed

`run_04d_factual_obstruction_vs_confound_baselines.py` compares `O` to `|margin|` and fragility with example-level bootstrap, on existing 04b CSVs. It does **not** add paraphrase / prompt-shift. That remains a later data collection.

## Next experiment

Experiment 0B, modular addition. The question is not another 0.9 AUC.

Can obstruction say something true about a learned mathematical representation that attribution, confidence, and ordinary probes cannot — and can we intervene to verify it?

Pipeline already in the repo:

1. Detect: `run_07a`, `run_07b`
2. Monitor vs loss / margin / `A_cyc` / probes: `run_07c`
3. Intervene, including norm-preserving phase rotation: `run_07d`
4. Plot emergence: `run_07e`
