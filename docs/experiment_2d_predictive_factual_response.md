# Experiment 2D: Predictive Factual Obstruction-Response Ranking

## 1. Motivation

Experiment 2D tests the stronger causal-response claim for margin-matched factual conflict: can a linearized obstruction-response score predict repair-effective component sites before patching?

Experiment 2B established that low-rank residual germs reveal margin-controlled factual-conflict obstruction.

Experiment 2C showed that actual obstruction reduction is associated with causal margin repair under component patching.

Experiment 2D tests the stronger causal-response claim: whether a linearized obstruction-response score predicts repair-effective component sites before patching.

## 2. Relationship to 2B and 2C

Experiment 2B is the representation result. It showed that `clean_pca` low-rank residual germs strongly separate clean and conflict examples on `EleutherAI/pythia-1b`.

Experiment 2C is the oracle patching result. It measured actual component patch effects and found that actual obstruction reduction is associated with actual factual margin repair.

Experiment 2D separates prediction from evaluation:

```text
predicted response score
-> ranked site
-> patch selected/ranked sites
-> measured Delta O
-> measured Delta m
-> repair/stabilization outcome
```

## 3. Predictive Response Score

For each component site `u`, define clean-to-conflict patch direction:

```text
delta_u = h_u_clean - h_u_conflict
```

Linearized predicted margin response:

```text
pred_delta_m_u = grad_{h_u} m_c(x_conflict)^T delta_u
```

Linearized predicted obstruction response:

```text
pred_delta_O_u = grad_{h_u} O_tilde_c(x_conflict)^T delta_u
```

Cost:

```text
cost_u = ||delta_u||^2
```

Response score:

```text
S_u = - pred_delta_O_u + beta * pred_delta_m_u - gamma * cost_u
```

Default grids:

- `beta in {0.0, 0.25, 0.5, 1.0, 2.0}`
- `gamma in {0.0, 1e-4, 1e-3, 1e-2}`

## 4. Site Set

Supported component sites:

- attention head outputs: `blocks.{layer}.attn.hook_z`
- MLP outputs: `blocks.{layer}.hook_mlp_out`
- optional residual controls: `blocks.{layer}.hook_resid_pre`

Default model: `EleutherAI/pythia-1b`.

Default representation: `clean_pca`, rank 8.

## 5. Prediction Phase

`run_06a_factual_predictive_response_scores.py` computes scores before patching. It does not read patch outcome files.

Outputs:

- predicted obstruction reduction
- predicted margin response
- activation norm
- gradient saliency
- attribution score
- response score for beta/gamma grid

## 6. Patch-Evaluation Phase

`run_06b_factual_predictive_patch_eval.py` evaluates the fixed example-site pairs from the prediction file.

Patch outcomes include:

- actual margin before and after patching
- actual obstruction before and after patching
- actual obstruction reduction
- margin increase success
- margin flip success
- optional next-token KL

Patch outcomes are used only for evaluation, not for computing `S_u`.

## 7. Baselines

Baselines include:

- `obstruction_only_score = -pred_delta_O`
- `attribution_score = pred_delta_m`
- `abs_attribution_score = abs(pred_delta_m)`
- `grad_saliency_score`
- `activation_norm_score`
- `random_score`
- oracle outcome columns for upper-bound context only

## 8. Metrics

`run_06c_factual_response_baselines.py` computes:

- Spearman with actual delta margin
- Spearman with actual obstruction reduction
- Spearman with observed Pareto score
- AUC for top repair sites
- AUC for top obstruction-reduction sites
- repair@k by example
- relative lift over attribution and other baselines

Pareto score:

```text
pareto = actual_delta_m + lambda * actual_obstruction_reduction - kl_gamma * off_target_kl
```

## 9. Go/No-Go Criteria

PASS_STRONG if:

- Spearman(response_score, observed Pareto) > 0.35
- AUC(response_score detects top repair sites) > 0.75
- repair@k or Pareto@k beats attribution by at least 5 percent relative
- response_score predicts actual obstruction reduction better than attribution
- bootstrap confidence interval does not collapse

PASS_WEAK if:

- response_score beats norm, saliency, and random
- response_score is competitive with attribution on delta margin
- response_score beats attribution on actual obstruction reduction or Pareto score

NO_GO if:

- response_score collapses to attribution
- attribution explains all predictive value
- pred_delta_O does not correlate with actual obstruction reduction
- top-k repair does not beat simple baselines

## 10. Interpretation Rules

Do not claim final paper success from Experiment 2D alone. A positive result supports the predictive response loop for this task/model/site set.

If response scores do not beat attribution, interpret 2D as attribution-competitive or attribution-limited, not as a failure of the 2B representation result.

If obstruction-only predicts actual obstruction reduction but not margin repair, interpret the result as diagnostic rather than causal repair.

## 11. What Happens Next

If 2D passes, the project can present a coherent factual-conflict obstruction-response ladder:

```text
2A scalar no-go
-> 2B low-rank representation PASS_STRONG
-> 2C bounded causal patching PASS_WEAK
-> 2D predictive response ranking
```

If 2D fails, the project still has a strong representation-level obstruction result and bounded causal association, but not yet a predictive site-selection result.
