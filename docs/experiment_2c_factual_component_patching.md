# Experiment 2C: Component-Level Factual Obstruction and Patching

## Status

Experiment 2C is the causal response follow-up to Experiment 2B. It should be run only after preserving the Experiment 2B low-rank factual obstruction result.

Experiment 2C tests whether component-level clean-to-conflict patching reduces low-rank obstruction and repairs the factual answer margin.

Current bounded result: PASS_WEAK. Component sites that causally improve factual conflict margins tend to reduce low-rank obstruction, but this is not yet a full-dataset/full-site or bootstrap-validated result.

## Motivation

Experiment 2B showed that clean-PCA low-rank residual germs strongly separate factual conflict from clean prompts on `EleutherAI/pythia-1b`:

- `clean_pca`, rank 4: n = 126, AUC = 0.9458, partial corr = 0.5834
- `clean_pca`, rank 8: n = 126, AUC = 0.9758, partial corr = 0.7044
- `clean_pca`, rank 16: n = 126, AUC = 0.9740, partial corr = 0.7910

Experiment 2C asks whether this representation-level obstruction signal has causal response value.

## Intervention

For each factual example and component site, patch the clean activation into the conflict prompt at the final prompt token.

Implemented site types:

- attention head output: `blocks.L.attn.hook_z`, selected by head
- MLP output: `blocks.L.hook_mlp_out`
- optional residual controls: `blocks.L.hook_resid_pre`

The current implementation measures:

- multi-token conflict margin before patching
- multi-token conflict margin after patching
- low-rank normalized obstruction before patching
- low-rank normalized obstruction after patching
- obstruction reduction
- activation-delta norm baseline
- optional multi-token gradient attribution and saliency baselines
- repair success by margin increase
- repair success by margin flip

## Scripts

Run the bounded patching oracle:

```bash
python run_05a_factual_component_patching.py \
  --input data/factual_conflict/factual_conflict_margin_matched.jsonl \
  --model EleutherAI/pythia-1b \
  --variant clean_pca \
  --rank 8 \
  --max-examples 24 \
  --max-sites 64
```

Summarize component sites:

```bash
python run_05b_factual_component_patching_summary.py
```

For expensive attribution baselines, add:

```bash
--compute-gradient-baselines
```

Default outputs:

- `artifacts/factual_component_patching/component_patch_oracle.csv`
- `artifacts/factual_component_patching/component_patch_site_summary.csv`
- `artifacts/factual_component_patching/component_patch_metric_summary.csv`
- `artifacts/factual_component_patching/component_patch_repair_at_k.csv`

## Go/No-Go Criteria

Experiment 2C should not be treated as successful just because patching changes margins. The stronger claim requires obstruction reduction to predict causal repair.

Initial weak pass:

- obstruction reduction has AUC > 0.65 for top repair sites
- Spearman correlation between obstruction reduction and observed delta margin > 0.30

Stronger pass:

- obstruction-ranked top-k sites outperform random and simple repair-frequency baselines
- patching reduces obstruction and improves margin on the same examples
- effect is not dominated by a tiny number of examples or one relation class

## Results So Far

Bounded `clean_pca` rank 8 run:

- examples: 32
- sites: 96
- Spearman(obstruction reduction, delta margin): 0.4225
- AUC(obstruction reduction detects top repair sites): 0.7807
- random AUC: 0.3818
- decision: PASS_WEAK

Bounded `clean_pca` rank 16 run:

- examples: 32
- sites: 96
- Spearman(obstruction reduction, delta margin): 0.3645
- AUC(obstruction reduction detects top repair sites): 0.7679
- random AUC: 0.3818
- decision: PASS_WEAK

Larger bounded `clean_pca` rank 8 run:

- examples: 64
- sites: 144
- Spearman(obstruction reduction, delta margin): 0.4367
- AUC(obstruction reduction detects top repair sites): 0.7441
- random AUC: 0.3984
- decision: PASS_WEAK

Bounded `clean_pca` rank 8 run with gradient/activation baselines:

- examples: 32
- sites: 96
- Spearman(obstruction reduction, delta margin): 0.4225
- AUC(obstruction reduction detects top repair sites): 0.7807
- activation norm AUC: 0.5636
- gradient saliency AUC: 0.6610
- absolute gradient-attribution AUC: 0.7861
- random AUC: 0.3818
- decision: PASS_WEAK

Outcome-derived repair-success summaries reached AUC 0.9166, but this should not be treated as a predictive baseline because it is computed from patch outcomes.

Interpretation: obstruction reduction is associated with causal patch repair across bounded component sweeps. It beats activation norm, gradient saliency, and random baselines, and is competitive with gradient attribution. It does not yet cleanly dominate attribution on every metric.

## Caveats

- This is an oracle patching pass: it measures actual patch effects per site.
- It is not yet a fully predictive site-selection algorithm.
- The default run is intentionally bounded with `--max-examples` and `--max-sites`.
- Full sweeps can be expensive and should be run only after smoke tests.
- Baseline comparisons should include activation norm and optional gradient attribution before claiming specificity over standard causal/attribution measures.

## Interpretation

If Experiment 2C passes, the project has evidence for the full obstruction-response loop on factual conflict:

```text
low-rank local evidence
-> obstruction
-> component intervention
-> measured obstruction reduction
-> measured margin repair
```

If it fails, Experiment 2B remains a strong representation-level obstruction result, but causal response at component granularity is not yet established.
