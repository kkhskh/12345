# Multi-layer mechanisms: C (MLP rereading) and C′ (attention rereading)

The theorems are in `THEOREMS_AND_COUNTEREXAMPLES.md` §4. Constructions: `src/cbmath.py::reconciliation` (C) and `::rereading` (C′).

Review R1 (S7) observed that C has no attention re-reading the source positions: it is effectively a two-layer MLP at the read position. Without more, deep rereading would have been left unexamined. C′ was therefore built after the review. Its predictions were committed in `PREREGISTRATION_ADDENDUM.md` before its code was written.

## C: two blocks, MLP rereading

**Construction.**
- **Block 1.** A uniform head copies `x e_1 + y e_2` to the read token. A GELU MLP using the Mechanism-A pairing writes the agreement signal `q e_3`.
- **Block 2.** No attention. A ReLU MLP reads net evidence `Ŝ` and agreement `q̂`. It has gated units `ReLU(±βŜ + γq̂ − θ)` writing `±λ_2 f_U`, plus linear units writing `εŜ`.
- An optional prior `κ_0` toward `c'` is added.

**Results.** Pairs are matched exactly at the block-2 reader. This correction comes from R1 S6 and R2 S2: the original analysis matched at block 1, and those numbers are still in the log.

| | agreeing pair | disagreeing pair | gap |
|---|---|---|---|
| designed, `Ŝ = 1.0795` | `m = 4.588` | `m = 0.270` | 4.318 |
| block-1 MLP ablated (re-matched) | | | **exactly 0.0000** |

- Agreement amplifies the net-evidence candidate. `sign m = sign Ŝ` in all 156 cells.
- With a prior between the two margins, agreement decides `c` versus `c'` (C1.3). The prior was chosen to *demonstrate* this, not predicted.
- Interchanging block 1's MLP output from the agreeing run into the disagreeing run moves `m` from 0.243 to 3.928.
- With a bottleneck that loses source identity, matched-sum pairs give identical margins (C2).

**Limits.**
- C1.2's monotonicity is flat where both gate units are active (corrected).
- The sign law is fragile under weight noise: it holds on the whole grid in 0 of 20 seeds at noise 0.03. Amplification is robust: 20 of 20.
- A final LN with bias gives 34 sign errors.
- The mirrored gate amplifies *dis*agreement just as easily (3.185 against 0.246).
- With no prior, the *decision* is `sign Ŝ`, which a single sum-only unit reproduces. C's agreement effect is a stability property unless a prior is present.
- A one-layer ReLU network cannot represent C exactly (it is piecewise affine; C is not). A wide one approximates C's margin, with relative RMSE 0.032 and sign agreement 0.923.

## C′: two blocks, conditional rereading through attention

**Construction** (`d = 10`; positions: prior token P, A, B, read).
- **Block 1.** A uniform head copies the evidence. A GELU MLP writes `q` onto `e_3`.
- **Block 2.** A head whose **query reads `e_3`**, whose keys read `(marker_src − marker_P)`, and whose values carry the sources' votes and P's prior vote for `c'`.

This is the structure the mandate lists as "a later attention head whose query depends on earlier combined evidence" and "conditional suppression".

**Results.**

| Cells | margins | block-2 attention |
|---|---|---|
| agreeing | +1.12 to +2.33 (and the mirror image for negative sources) | ≈ 1 on the sources |
| conflicting | −2.631, in every cell | ≈ 1 on P |

- In conflict the prior wins whichever source is stronger. The margin stays −2.631 as the stronger source grows from 0.6 to 4.0 (PR1, PR5).
- **Depth is necessary (C′2, proved).** In one pre-norm attention layer, `log(α_A/α_P)` cannot depend on B. Across 30 random heads the spread is at most `3.6e−15`. Routing that depends on a relation needs an earlier layer to write that relation into the query.
- **Mediation (C′3).**
  - Interchanging block 1's MLP output from an agreeing run into a disagreeing run flips the decision, from −2.631 to +1.879.
  - The same interchange moves source attention mass from 0.000 to 1.000.
  - Ablating block 1's MLP makes block 2's routing uniform (0.25 each), whether the sources agree or not.

**Failed prediction (kept).** PR2 predicted that source mass is monotone in `q̂` across cells. It is not. The score also depends on each source's LN-scaled marker coordinate, so `q̂` alone does not order the routing.

## Necessary, sufficient, merely correlated

| Variable | C | C′ |
|---|---|---|
| block-1 agreement signal `q̂` | necessary (gap → 0 when ablated); sufficient by interchange (0.243 → 3.928) | necessary (routing uniform when ablated); sufficient by interchange (decision flips) |
| net evidence `Ŝ` | sets the decision sign | sets the sign in agreeing cells only |
| an unread residual direction | causally irrelevant except through the LN scale (no flips in 169 cells) | not tested separately |
| raw source magnitudes | act only through `Ŝ`, `q̂` and the LN scale | do not affect the conflict decision (PR5) |

## Comparison with single-block and direct-evidence baselines

- **C.** The decision is reproducible by a single sum-only unit when there is no prior. The confidence profile needs depth for an exact implementation, but it can be approximated in one wide layer.
- **C′.** No single attention layer can do this (C′2). The extra block contributes real causal structure: a relation-dependent routing switch, mediated by `q̂`.

In standard terms, this is **Q-composition gated by an MLP feature** (Elhage et al. 2021). It is the most transformer-specific of the constructions.

## What this does not establish

- That trained models reconcile or defer this way.
- That the switch direction (prior under conflict) is preferred by the architecture. Swapping the key signs gives the opposite rule.
- Novelty: these are known composition and gating mechanisms (see `BASELINES_AND_NOVELTY.md`).
- Any role for `O`/`D`.
