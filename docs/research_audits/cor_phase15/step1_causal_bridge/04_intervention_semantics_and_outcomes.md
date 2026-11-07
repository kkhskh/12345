# 04 — Intervention semantics and outcome variables

The distinctions below are audit conventions forced by the difference between a recomputed forward pass and an assigned cochain. They are not theorems of the COR PDFs. The PDFs write `h ↦ h + B_u δ_u` and a Jacobian `∂m_c/∂h_u`, which is the first-order effect of one activation change. They do not define stitched cochains as interventions.

## 1. Frozen computation and one intervention

`F_θ(x)` is the frozen network's output on an ordinary forward pass.

`F_θ(x; do(h_v ← h_v+δ))` means: run the network on `x`, replace the activation at site `v` by that value plus `δ`, and recompute every descendant with the network's own maps. Ancestors are left as in the original pass. Sites that are not descendants and not `v` are left as in the original pass.

The candidate output `L_c(x)` used in this step is the manuscript's margin `m_c = z_a - z_b`, unless a sentence names a different scalar. The intervention response is

```text
R_c(x, v, δ) = L_c(x; do(h_v ← h_v+δ)) - L_c(x).
```

## 2. Outcomes that are not interchangeable

| Symbol | Meaning | Determined by `D` alone? |
|---|---|---|
| Candidate logit | one coordinate of `F_θ` | No. Theorem NFL |
| Margin `m_c` | difference of two logits | No |
| Probability | softmax of logits | No. Monotone in the logit only given the others |
| Prediction reversal | sign change of the margin | No |
| Accuracy | agreement with a label | No. Zero `D` can be wrong (CE-04) |
| Target-state response | a task coordinate such as legal-move mass | A different functional. Not `m_c` |
| Repair success | decrease of a chosen objective | Depends on what the objective contains |
| Output JS | change of the full distribution | Not a sign of `m_c` |
| Intervention cost | a penalty on `δ` | Chosen by the analyst |
| `ΔO`, `ΔD` | change of the consistency scalar | Not `R_c` |

The repair objective in Update Proposition 1 is `O - β m + cost`. Its minimizer moves `m` because `-β m` is in the objective. That is not `R_c` being predicted by `D`.

## 3. What the intervention is allowed to change

A single-site `do(h_v ← h_v+δ)` changes one activation bundle at one site and one token position, then recomputes descendants. It does not assign new values to those descendants. It does not replace a map `f` by a different map. It does not edit a weight.

An edit of several sites at once, with descendants not recomputed from one parent, is a different operation. It produces a Domain C or Domain D state in file 03. Consistency of that tuple under a chosen `ρ` is not the causal effect of a single-site intervention.

## 4. Worked contrast

Exact copy scaled by 2, logit read at the parent. Intervening `h ← h+1` and recomputing the child leaves the residual at 0 and changes the logit by `+1` (`test_ce19_recomputation_preserves_consistency_and_changes_the_logit`).

Setting the child to 100 while leaving the parent at 0 produces residual 100 and, when the logit does not read the stitched pair, output change 0 (`test_ce20_stitch_can_have_large_discrepancy_and_no_output_effect`).

The first row is a causal effect. The second row is an inconsistent assignment. A theory that scores both with `||ρ s_u - s_v||` is using one number for two operations.

## 5. Feedforward recomputation

Across layers, recomputation means applying the later blocks. It does not mean iterating a state until `d s = 0`. Nothing in the Transformer block, and nothing in the COR definitions, supplies that iteration. Fixed-point language is not used for these maps in this step.
