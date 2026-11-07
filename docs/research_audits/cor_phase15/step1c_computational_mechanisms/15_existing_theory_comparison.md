# 15 — Comparison with existing theory

The comparisons use the mathematical content of the cited work, not a restatement of abstracts. No new bibliographic theorem number is introduced. Where this step proves a statement, the proof is in `proofs/` and is self-contained. Agreement with a known fact is recorded so that the statement is not claimed as a COR theorem.

## Causal abstraction and interchange interventions

Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, Icard, JMLR 2025, give a criterion under which a high-level variable is an abstraction of a low-level variable: interventions on the high-level variable correspond to interventions on the low-level variable for the experimental set. The fiber theorem in `proofs/fiber_and_trace.md` is the special case in which the high-level variable is a function `P` of the low-level state and the correspondence demanded is equality of a named response. It is not a new abstraction theorem. It is the definition of "this `P` determines this response," written so it can be checked on a two-token attention map.

Distributed alignment search (Geiger, Wu, Potts, Icard, Goodman, CLeaR 2024) learns a subspace intended to support interchange. This step does not learn a subspace and does not test interchange on a model. A projection that drops a key coordinate fails exact sufficiency for residual edits. That is a counterexample to sufficiency, not a result about a learned alignment.

Sutter, Minder, Hofmann, Pimentel, NeurIPS 2025, show that alignment maps which are allowed to be arbitrary can make interchange scores vacuous. The rank-one `\rho` in this step is a one-line instance of the same phenomenon: the map is solved from the pair, and the residual is zero because it was solved that way. The instance is not a new proof of their result. It is a warning, already used in Step 1A, against unrestricted `\rho`.

## Mediation and path patching

An effect under `do(h_v\leftarrow h_v+\delta)` with descendants recomputed is a total effect on the margin. An effect with `A` clamped and `V` edited is a path-specific effect in the sense used by path patching: one edge is moved, another is held. The linearity proof is the reason the path-specific value effect has a closed form. Path patching already computes that effect by running the patched model. The closed form adds the identity `I=0` and the exact Jacobian. It does not add a relational residual, and it does not replace the patched run when `A` is not known.

This step does not recompute the IOI numbers or the historical obstruction comparisons.

## Mechanistic decomposition of attention

The split between a pattern and a value map is the standard QK/OV decomposition of a head: once the pattern is fixed, the output is linear in the residual through `W_V` and `W_O`. Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021), use that decomposition. The proof in `proofs/attention_value_linearity.md` does not depend on their notation. The result is the same fact. It is not a consistency theory.

Vaswani et al., NeurIPS 2017, are the source of the attention formula used in `04`. Ba, Kiros, and Hinton, 2016, are the source of LayerNorm. Zhang and Sennrich, 2019, are the source of RMSNorm. The sign counterexample for a fixed readout under LayerNorm, and the positive-scale lemma for plain RMSNorm, are derived here from those formulas. They are calculus, not a new normalization theory.

## Erasure, redundancy, and robustness

Amnesic probing and related erasure arguments ask what a linear removal of a subspace does to a behavior. The fiber theorem is the complementary question: what the kept subspace still determines. Direction 1 reported probe-invisible causal information in Othello-GPT. That empirical letter is `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED` in the Direction 1 record. It is consistent with the logical possibility shown by the two-token counterexample, and it is not a derivation of that counterexample. This step does not rerun the probe.

Redundancy as equal inner products, and robustness as survival of a sign after deletion, are statements about a linear form. They are what a path-ablation experiment measures. They are not implied by vector cosine.

## Sensitivity

The decomposition of `m(h+\delta)-m(h)` into a gradient term, a Hessian term, and a remainder is Taylor's theorem under its usual hypotheses. The observation that the Hessian term at the baseline does not determine a finite four-term interaction is the distinction between a derivative and a finite difference, which Phase 1 already used against Update Proposition 2. Error 1 applies that distinction to interaction.

## What would have been new

A new constraint would be a relation, fixed without `R`, that implies something about `R` which the Jacobian, the Hessian, the path effect, and the abstraction criterion do not already imply under the same hypotheses.

No candidate in `11` is of that kind. The positive theorems are separability, value-linearity, fiber constancy, the kernel of the actual measurement, the independence of `O` from `M`, and the LayerNorm sign counterexample. Each is one of the frameworks above, or a correction of a misstatement about those frameworks.

## Historical empirical record, as a constraint only

The following are not tests of the Step 1C formulas. They constrain interpretation.

| record | what it already says | what it does not say |
|---|---|---|
| Probe-invisible causal information, Direction 1, `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED` | a frozen board probe's subspace does not determine the intervention response that was measured | that every projection fails, or that a coboundary would have succeeded |
| Fixed-JS correct-target specificity | a specificity result at matched divergence | a test of `Q,K,V` interaction |
| Donor-free target-conditioned transfer | a transfer result under that protocol | identification of `\rho` |
| Stage-only contributions | a share of a contrast between full and editor states | a path decomposition of attention |
| Negative symmetric target switching | that manipulation was not supported | a theorem about softmax |
| Negative visible/nullspace interaction | that factorial interaction was not supported | `I=0` or `I\neq 0` for value edits in the block |
| Negative original 08R2 repair | the repair result was negative | a new theoretical identification |
| Unfavorable obstruction comparisons in factual repair and IOI | obstruction did not beat the attribution baseline in those comparisons | a reason to refit `W` |

Attribution dominated obstruction in the comparison already recorded in Phase 1 (`08_existing_evidence_adjudication.md`, citing the 2D summary): attribution repair AUC `0.979` against the best obstruction-using score `0.955`; obstruction-only Spearman with `\Delta m` about `0.095` against attribution about `0.905`. Those numbers are not re-estimated here. They are the empirical reason a relational residual has to beat a gradient method on information, not only on vocabulary. It does not, in the theorems above.

The pair `0.936` versus `0.141` is not used. It was not found in the record.
