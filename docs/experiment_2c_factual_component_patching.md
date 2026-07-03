# Experiment 2C: Component-Level Factual Obstruction and Patching

## Status

Experiment 2C is the causal response follow-up to Experiment 2B. It should be run only after preserving the Experiment 2B low-rank factual obstruction result.

Experiment 2C tests whether component-level clean-to-conflict patching reduces low-rank obstruction and repairs the factual answer margin.

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

## Caveats

- This is an oracle patching pass: it measures actual patch effects per site.
- It is not yet a fully predictive site-selection algorithm.
- The default run is intentionally bounded with `--max-examples` and `--max-sites`.
- Full sweeps can be expensive and should be run only after smoke tests.

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
