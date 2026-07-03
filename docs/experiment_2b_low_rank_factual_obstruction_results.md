# Experiment 2B Results: Low-Rank Factual Obstruction

## Status

Experiment 2B is the first strong positive obstruction result in the project timeline.

This is a representation-level obstruction result, not component patching evidence. No component patching was implemented in Experiment 2B.

## Input Task

Task: margin-matched factual conflict.

Model: `EleutherAI/pythia-1b`.

Matched facts: 126.

Candidate contrast:

```text
true answer - false answer
```

Multi-token candidate margin:

```text
m_c(x) = log p(true_answer | x) - log p(false_answer | x)
```

## Previous Experiment 2A No-Go

After multi-token factual answer scoring fixed the data-size bottleneck, scalar / sequence residual-chain obstruction still failed the margin-controlled gate:

- first-token scalar with multi-token margins: AUC 0.6369, partial corr 0.1665
- `sequence_mean` germ: AUC 0.6536, partial corr 0.1528
- `sequence_sum` germ: AUC 0.5437, partial corr 0.0163

Decision: no component patching from scalar / sequence residual obstruction.

## Experiment 2B Positive Result

Low-rank vector germs were fit over final-prompt-token residual activations:

```text
s_l(x) = P_l^T (h_l(x) - mu_l)
```

Clean-PCA is the main non-leaky result because the representation is fit on clean train activations only.

### Clean-PCA Results

All-split Pythia-1b:

- `clean_pca`, rank 4: n = 126, AUC = 0.9458, partial corr = 0.5834, decision = PASS_STRONG.
- `clean_pca`, rank 8: n = 126, AUC = 0.9758, partial corr = 0.7044, decision = PASS_STRONG.
- `clean_pca`, rank 16: n = 126, AUC = 0.9740, partial corr = 0.7910, decision = PASS_STRONG.

### Answer-Augmented Clean-PCA Results

All-split Pythia-1b:

- `answer_augmented_clean_pca`, rank 4: n = 126, AUC = 0.9808, partial corr = 0.5461, decision = PASS_STRONG.
- `answer_augmented_clean_pca`, rank 8: n = 126, AUC = 0.9760, partial corr = 0.7010, decision = PASS_STRONG.
- `answer_augmented_clean_pca`, rank 16: n = 126, AUC = 0.9728, partial corr = 0.7868, decision = PASS_STRONG.

### Delta-PCA Diagnostic

Delta-PCA was strong, but it should be treated as diagnostic only because the representation is selected from clean-conflict deltas:

- `delta_pca_diagnostic`, rank 4: n = 126, AUC = 0.9949, partial corr = 0.7095, decision = PASS_WEAK.
- `delta_pca_diagnostic`, rank 8: n = 126, AUC = 0.9987, partial corr = 0.7397, decision = PASS_STRONG.
- `delta_pca_diagnostic`, rank 16: n = 126, AUC = 1.0000, partial corr = 0.7719, decision = PASS_STRONG.

## Interpretation

Scalar / sequence scalar germs produced only a weak factual-conflict signal. Low-rank vector-valued residual germs produced a strong obstruction signal after margin control.

Reviewer-safe interpretation:

> In this factual conflict setting, obstruction appears to be vector-valued and distributed in residual space. A one-dimensional answer-direction germ misses much of the incompatibility, while clean-trained low-rank residual germs recover it.

This supports the local-to-global obstruction-response framework for this task/model pair, but only at the representation and transport level.

## Caveats

- These results are for `EleutherAI/pythia-1b` on the current factual conflict dataset.
- The main evidence should emphasize `clean_pca`, not `delta_pca_diagnostic`.
- All-split metrics evaluate the full 126 matched facts; train/val/test split rows are smaller and should be read as diagnostics.
- Component patching has not yet tested causal repair.
- Future claims require preserving learned-vs-control comparisons and bootstrap summaries from `run_04c`.

## Decision

Experiment 2B passes the representation-level gate.

Next step:

```text
Experiment 2C: component-level factual obstruction and patching
```

Do not merge patching into Experiment 2B. Treat 2C as a separate causal response experiment.
