# Identifiability and reparameterization

The formal statements are `THEOREMS_AND_COUNTEREXAMPLES.md` §1 (ID1, ID2), and the checks are PI1–PI5 in `logs/experiments.txt`.

## 1. Correction of the previous package (correction B)

The foundational-bridge package (`step1_foundational_bridge_adjudication/CAUSAL_BRIDGE_MAIN_ARGUMENT.md` §3) said that per-head `w_j`, `v_j`, `M` and centered writes are invariant under the architecture's function-preserving symmetries. **That is false** and is withdrawn.

Take two heads with identical `W_Q`, `W_K` and `W_V`, and apply `W_O^1 + E`, `W_O^2 − E`. This preserves the function on every input and changes each head's write, direct effect, consensus and patching effect:

- logits change by at most 4.4e−16;
- per-head direct effects go from `[2.111, 2.111]` to `[1.703, 2.520]`;
- per-head patch effects go from `[−2.106, −2.106]` to `[−1.699, −2.514]`, while their sum stays fixed.

The audit's own numbers reproduce exactly: a consensus change of 14.963 with logits fixed to 1.8e−15.

## 2. Three levels, applied

| Quantity | Fixed-parameterization observable | Causal intervention attribution (given θ) | Identifiable modulo G | Function-level |
|---|---|---|---|---|
| input-level effects `m(a',b) − m(a,b)` | yes | yes | yes | **yes** |
| matched-reader-input contrasts (Theorem A4) | yes | yes | yes, given an identified reader | given the reader |
| logits, margins | yes | — | yes | yes |
| attention scores and patterns | yes | yes | yes under the QK gauge | no |
| Q, K, V vectors | yes | yes | **no** (QK and VO gauges) | no |
| per-head writes, DLA, patching effects | yes | yes | **no** when heads share Q/K/V (ID1) | no |
| summed same-layer writes | yes | yes | yes under ID1 and ID2 | no |
| MLP preactivations | yes | yes | up to unit permutation, and positive rescaling for ReLU | no |
| per-unit attributions | yes | yes | **no** (splitting, ReLU rescaling) | no |
| raw residual coordinates | yes | — | **no** (orthogonal `R` with `R1 = 1`) | no |
| reader coordinates `x, y` as compositions | yes | yes | yes under ID2 | no |

## 3. The strongest mechanisms: equivalent implementations, different "consistency" values

The mandate's question: would two mathematically equivalent implementations give different values of the proposed consistency quantity while producing identical decisions? **Yes, in three distinct ways.**

1. **Same behavior on the binary family, different relation (Theorem A2).**
   - The designed circuit reads `x − y`.
   - The sum-only reader with an offset does not.
   - Their margins agree on all four binary cells to 3.1e−15.

   The "relation the computation uses" is not identified by the binary intervention family at all. It is identified only after extending the family to matched reader-input contrasts.
2. **Same function to approximation, different decomposition (C3).** The two-block mechanism, with `q̂` as an explicit mediator, and a one-layer wide ReLU network fitted to the same function (relative RMSE 0.032) give the same behavior on the tested family. Only one of them has a `q̂` variable.
3. **Exactly equal functions, different per-component values (ID1, ID2).** Head redistribution changes per-head gate contributions in Mechanism B with two identical gate heads. Unit splitting and ReLU rescaling change per-unit agreement attributions in A and C.

## 4. What is identifiable, and the convention that makes the claims meaningful

**Identifiable without any convention.** Input-level interventional effects on the frozen function. This includes the sign law of A1 as a function of the *input* evidence: `sign m(a,b) = sign(ab)` holds for the positive control and the continuous family as a property of the function. Also identifiable: candidate-relativity or its absence (`m(a,b) = m(−a,−b)`), and B's preference-reversal band relative to the qk-off model. The last needs a second model, so it is a comparison between implementations.

**Identifiable modulo G, once the reader is fixed.**
- The reader coordinates as compositions "source token → LN → write → LN → reader direction". These are invariant under QK and VO gauges, unit permutation, positive ReLU rescaling, and orthogonal residual maps fixing `1`.
- The matched-`Ŝ` contrast that separates relation computation from a function of net evidence.

**Restricted to a selected parameterization.** Which unit or head "is" the gate; per-head and per-unit magnitudes; raw residual coordinates. These are reported only as fixed-parameterization statements.

**Source identification.** Here the evidence sources are input positions with semantically fixed tokens. That is independent of the outcome. In a trained model, finding *which* internal directions carry "evidence A" is not solved by anything in this package. A probe trained on the evidence labels would reintroduce the Phase 1 problem with `P`. A direction chosen because it predicts the margin would be circular. The protocol in `FINAL_SCIENTIFIC_DECISION.md` §5 fixes sources at the input level and finds readers by intervention, not by fitting to the response.

## 5. Not covered

- Function-preserving maps that move computation between attention and MLP. These exist only approximately in general.
- Non-uniform LN gains. The orthogonal residual gauge then shrinks to maps commuting with `diag(γ)`.
- Symmetries of trained-weight degeneracies.
