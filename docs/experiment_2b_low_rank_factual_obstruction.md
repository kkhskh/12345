# Experiment 2B: Low-Rank Factual Obstruction

## 1. Motivation

Experiment 2B tests whether vector-valued local germs reveal factual conflict obstruction that scalar residual-chain germs missed. It stays on the existing margin-matched factual conflict task and does not implement component patching.

## 2. Previous Scalar/Sequence No-Go

Experiment 2A used scalar and sequence residual-chain obstruction with multi-token factual answer scoring. Multi-token scoring fixed the data-size bottleneck for `EleutherAI/pythia-1b`, yielding 126 matched facts.

Latest `pythia-1b` results:

- first-token scalar with multi-token margins: AUC 0.6369, partial corr 0.1665
- `sequence_mean` germ: AUC 0.6536, partial corr 0.1528
- `sequence_sum` germ: AUC 0.5437, partial corr 0.0163

Decision: scalar/sequence residual obstruction is a no-go for component patching. The next scientific move is a richer vector/low-rank representation test.

## 3. Hypothesis

For a candidate factual contrast, internal incompatibility may be distributed across a low-rank residual subspace rather than a single answer-token direction. A vector-valued germ may preserve incompatibility that scalar projection erases.

## 4. Low-Rank Germ Definitions

For model `M`, prompt `x`, and contrast `c = (true answer, false answer)`, the multi-token margin is:

```text
m_c(x) = log p(true_answer | x) - log p(false_answer | x)
```

At each residual-chain layer `l`, collect the final-prompt-token residual activation:

```text
h_l(x)
```

The low-rank germ is:

```text
s_l(x) = P_l^T (h_l(x) - mu_l) in R^r
```

Implemented variants:

- `clean_pca`: non-leaky clean-train PCA basis per layer.
- `answer_augmented_clean_pca`: candidate-specific direction from mean unembedding contrast, plus clean-PCA directions orthogonalized against it.
- `delta_pca_diagnostic`: diagnostic basis from clean-conflict deltas; not main evidence because conflict deltas choose the representation.

## 5. Transport Fitting

Residual-chain vertices are:

```text
v_l = residual stream at final prompt token, l = 0,...,L
```

Edges are:

```text
e_l: v_l -> v_{l+1}
```

For clean train examples, fit:

```text
R_l = argmin_R sum_i ||R s_l(x_i_clean) - s_{l+1}(x_i_clean)||^2 + lambda ||R||_F^2
```

Closed form:

```text
R = Y X^T (X X^T + lambda I)^(-1)
```

Weights are:

```text
w_l = max(heldout_R2_l, 0)
```

## 6. Obstruction Metrics

Raw obstruction:

```text
O_c(x) = sum_l w_l ||R_l s_l(x) - s_{l+1}(x)||^2
```

Normalized obstruction:

```text
O_tilde_c(x) = O_c(x) / (sum_l ||s_l(x)||^2 + eps)
```

Fragility:

```text
Frag_c(x) = O_tilde_c(x) / (|m_c(x)| + eps)
```

## 7. Controls

Experiment 2B evaluates learned transports against:

- identity transports
- random Gaussian transports
- random orthogonal transports if requested
- shuffled learned transports
- uniform-weight learned transports if requested

Learned transports should beat identity/random/shuffled controls before component patching is considered.

## 8. Robustness Checks

Robustness checks include:

- bootstrap confidence intervals for AUC, partial correlation, and mean delta obstruction
- stricter margin subsets at abs margin gap <= 4.0, <= 2.0, and <= 1.0 when enough examples remain
- answer-length controls
- relation breakdowns
- learned-vs-control transport comparisons
- edge contribution summaries when edge terms are available

## 9. Go/No-Go Criteria

Before any future component patching:

- matched facts >= 100
- AUC(`O_conflict > O_clean`) > 0.65
- partial corr(`O`, conflict_label | margin) > 0.20
- preferably partial corr > 0.25 or clear lift over scalar baseline
- learned transports beat identity/random/shuffled controls
- held-out transport R2 is nontrivial

`PASS_STRONG` requires data size, AUC, partial corr > 0.25, control lift, and a non-weak bootstrap lower bound.

`PASS_WEAK` requires data size, AUC > 0.65, and partial corr > 0.20.

`NO_GO` otherwise.

## 10. What Happens Next

If Experiment 2B is `PASS_STRONG`, or a deliberately accepted `PASS_WEAK`, the project may proceed to component-level factual obstruction and patching.

If Experiment 2B is `NO_GO`, do not run expensive component patching. The next move should be a different representation family, such as SAE feature-space germs or more structured candidate-specific vector germs.
