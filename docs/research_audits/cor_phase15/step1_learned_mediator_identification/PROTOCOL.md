# Frozen mediator-identification and causal-test protocol

This protocol was frozen **before any evaluation run**. The commit hash is in `RESEARCH_LOG.md`. The protocol is implemented exactly by `src/run_protocol.py` together with `src/mediator.py`.

**Development.** The protocol was developed on historical dev seeds 0 and 1 only, using the scripts `/tmp/dev_probe.py`, `/tmp/dev2.py`, `/tmp/dev3.py` and the smoke runs. Those runs are exploratory and are listed in `RESEARCH_LOG.md`.

**Evaluation sets.** All are run with **no code change**:

- **E1.** Historical seeds 2, 3 and 4. These checkpoints have never been used for mediator analysis.
- **E2.** New seeds 7, 8 and 9 for REL-XNOR, REL-GATE, REL-MIS and K3-GATE. They were trained by `scripts/train_new.py` and have never been analysed.
- **E3.** The new AGG-NL control, seeds 0–4.
- **E4.** A new input distribution, `gauss`, applied to E1 and E2. It was never used for selection.

## 1. Relation sought

**D1: semantic candidate agreement.**

- Binary tasks: `s = 1[ab > 0]`.
- K = 3 tasks: `s = 1[argmax z_A = argmax z_B]`.

**Aggregate:**

- Binary tasks: `z = a + b`, with p where present.
- K = 3 tasks: `z = z_A + z_B`, with p.

The relation is defined by the generator and was never selected from model margins.

**Identifiability limit (stated in advance).** For binary inputs at fixed z, `s = 1[|a−b| < |z|]`. "Agreement" and "spread thresholded at |z|" are therefore the same function on this input family.

## 2. Sites (frozen per architecture; complete cuts are checked, not assumed)

| arch | primary site | why |
|---|---|---|
| TF1 | `resid_mid0` at R | Last-token residual after attention and before the MLP. The R MLP and the readout read only this state. |
| TF2 | `resid_mid1` at R | Before the final MLP. |
| TF2 | `resid_post0` at all positions | Locates layer-0 versus layer-1 computation. |
| TF2-AO | `resid_post0` at all positions | The whole layer-0 output, which is all that layer 1 reads. |

Final residual sites are excluded. A readout-aligned direction there transfers decisions trivially, as the previous review established.

## 3. Candidate mediators

| candidate | role |
|---|---|
| S_rel (primary) | **Output-free.** Top-k right singular vectors of the differences `h(agree) − h(conflict)` over 800 matched pairs with **identical** aggregate (z, p). k is the smallest value reaching 90% of the singular energy, capped at 8. |
| S_das | Target-using. Distributed alignment search trained on the same pairs, with the same k. |
| S_grad | Output-derived. Directions of largest margin sensitivity. |
| S_pca | Generic baseline: top principal components. |
| S_rand | 20 random subspaces. |
| S_alt | k directions **orthogonal** to S_rel, chosen to decode s. These are equally decodable but not the identified subspace. |

## 4. Tests

All tests run on fresh evaluation pairs (seed 6002, or 6102 for `gauss`) and are **model-relative**. They are restricted to matched pairs on which the model's own decisions differ.

1. **Cut.** IIA when the whole site state is interchanged. Pass if ≥ 0.99.
2. **Interchange.** IIA for each candidate subspace, in both directions.
3. **Necessity.** Set the S-component of both pair members to the pair mean. The fraction of pairs whose decisions still differ must be ≤ 0.2.
4. **Sufficiency.** Replace everything except S with the partner's state. The base decision must be kept on ≥ 0.95 of pairs.
5. **Bypass.** At TF1 `resid_mid0`, split the effect into the MLP path and the direct residual path. In TF2, compare the R-only cut with the all-position cut.
6. **Manifold.** Distance from the patched state to the nearest natural state, relative to typical nearest-neighbour distances between natural states. Reported only; it is not a gate.
7. **Content: is the identified mediator a cross-source interaction?** Replace the S-coordinates with their best:
   - **source-additive fit** `f(a, p) + g(b, p)`;
   - affine fit in (a, b);
   - function of z (the aggregate);
   - **exact** fit, the reference. For K = 3 the reference is a regression in the same family with interaction features added.

   **Interactive** means the additive surrogate loses at least 0.10 of matched-pair decision differences, or at least 0.05 accuracy, relative to the reference.
8. **Input level.** Flip A while holding B, and flip B while holding A. Report the decision-change rate against the target-change rate.

## 5. Verdict per model and site (frozen)

| verdict | conditions |
|---|---|
| **RELATION_COMPUTING_MEDIATOR** | Cut ≥ 0.99; S_rel IIA ≥ 0.95; mean S_rand IIA ≤ 0.2; necessity ≤ 0.2; sufficiency ≥ 0.95; and content interactive |
| **SOURCE_ADDITIVE_MEDIATOR** | All causal gates pass, but the content is additive |
| **NO_CUT** | Cut < 0.99 |
| **MEDIATOR_NOT_IDENTIFIED** | Otherwise |
| (aggregate-only behaviour) | The model's decisions differ on fewer than 20 matched pairs; contrast energy is reported instead |

## 6. Additional analyses (descriptive, run after the verdicts)

1. **Pattern freeze**, the locus test: replace layer-0 attention patterns with a constant reference pattern, then measure the interaction share of the S-coordinates.
2. **Interaction R² by `sign(ab)`.**
3. **Cosines** between S_rel and S_grad, and between S_rel and S_das.

These do not affect any verdict.
