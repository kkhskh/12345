# Dynamic attention and deep mechanisms (Programs B and C)

| program | code | log |
|---|---|---|
| B | `src/program_b.py` | `logs/program_b.txt` |
| C | `src/program_c.py` | `logs/program_c.txt` |

---

# Program B: moving Q/K gates with three or more candidates

## B-construction `qk3` (K = 3, d = 16)

Positions are A, B and R, with causal attention.

**Layer 1.** R attends to A through a marker, with score gain 12. It copies A's evidence `z_A` onto two places:

- the query features `fQ_k`, with weight 2;
- the direct-vote features `fU_k`, with weight `wU = 0.3`.

**Layer 2, gate head G.**

- Query `= λ Σ_k fQ_k`-coordinates, with `λ = 3`.
- Key `= fB_k`-coordinates.
- Value `= fB_k`-coordinates, written to `fU_k` with weight 1.5.

**Score and value equations.** Let `z̃` denote the LN-scaled evidence coordinates, so `z̃_A = (wQ/3)·LN-scale·z_A`, and likewise for `z̃_B`. Then:

- `s_RB = λ⟨z̃_A, z̃_B⟩/√3`
- `s_RA = s_RR = 0`, because those positions carry no `fB` content.
- `α_B = e^{s_RB}/(e^{s_RB} + 2)`
- `logit_k = wU·z̃'_A,k + 1.5·α_B·z̃_B,k`

The queries and keys are nonzero, and the routing pattern varies with the input.

## B1. Bilinear impossibility for argmax agreement (K ≥ 3)

**Theorem B-IMP.**

- **Setting.** Let K ≥ 3, and let `f(z_A, z_B) = z_Aᵀ M z_B + αᵀz_A + βᵀz_B + γ` for any `M, α, β, γ`.
- **Claim.** `sign f` does not equal `2·1[argmax z_A = argmax z_B] − 1` on any set of full measure.

*Proof.*

1. Fix `z_B = e_0`. On the unique-argmax set, the agreement region in `z_A` is the open polyhedral cone `C_0 = {z : z_0 > z_j for all j ≠ 0}`.
2. Because `f(·, e_0)` is affine, `{z_A : f > 0}` is an open half-space H, or else empty or everything.
3. The complement of a half-space is convex. The complement of `C_0` is not convex, even up to null sets:
   - `p = (0, 1, −2)` and `q = (0, −2, 1)` both lie outside `C_0`;
   - their midpoint `(0, −½, −½)` lies in `C_0`;
   - so do small balls around these points.
4. Hence `H` and `C_0` differ on a set of positive measure. ∎

**Corollary for attention.** A pre-norm head with zero LN bias has score `⟨W_Q LN(x_r), W_K LN(x_j)⟩`. Since `LN(x) = Px/σ(x)` and `σ > 0`, the sign of the score is the sign of a bilinear form in the linearly encoded evidence. So **no single attention score can switch sign exactly on candidate-relative agreement when K ≥ 3.** For K = 2 with `z = (t, −t)`, agreement is `sign(t_A t_B)`, which is bilinear, and the obstruction disappears.

**Quantitative check.** Trained on 20,000 draws (seed 42) and tested on 20,000 (seed 43):

| classifier | test accuracy |
|---|---|
| best bilinear + linear | 0.821 |
| MLP (64, 64) | 0.987 |
| base rate | 0.587 |

## B2. Routing tracks the inner product, not semantic agreement

Over 3,000 T5b draws (seed 41):

| quantity | value |
|---|---|
| sign(s_RB) = sign⟨z_A, z_B⟩ | always (1.000) |
| routing toward B suppressed below uniform, given argmax agreement | 8.4% |
| routing enhanced, given disagreement | 43.5% |

Example pair: `z_A = (1, −3, 0.5)` and `z_B = (1, 0.5, −3)`. They agree on candidate 0, yet `⟨z_A, z_B⟩ = −1.75`. Then:

- `α_B = 0.017`, so B's supporting evidence is suppressed;
- the 0-vs-2 margin is 0.221 under full routing, against 1.905 under uniform routing.

## B3. Total-margin theorem

**Theorem B-TM.**

- **Setting.**
  - Let the logit difference for candidates `(k, j)` at R be `Δ = c + Σ_i α_i w_i`.
  - Here `w_i` is the gate head's value-output margin at position i, and c is everything else (bypass and direct votes).
  - Assume `w_i` and c do not depend on the gate's scores: no competing writer reads the gate's output, and there is no rereader.
  - Assume no nonlinearity sits between the residual and the readout, except a final LN with zero bias and a mean-zero readout difference.
- **Claim.** Suppose only `s_B` changes, from `s` to `s′`. Then `ΔΔ = (α′_B − α_B)·(w_B − w̄_{−B})`, where `w̄_{−B} = Σ_{i≠B} α_i w_i/(1 − α_B)`. The sign of the total margin consequence is therefore `sign(Δs_B)·sign(w_B − w̄_{−B})`.

*Proof.*

1. Under softmax, changing only `s_B` rescales the other weights proportionally: `α′_i = α_i(1 − α′_B)/(1 − α_B)` for `i ≠ B`.
2. Substitute into `Σ_i α_i w_i` and subtract.
3. The final LN with zero bias multiplies the logit difference by `1/σ > 0`, so the sign is unchanged. ∎

**Check.** Against uniform-routing ablation on 3,000 draws, the predicted sign is correct on **3,000/3,000**.

## B4. Causal tests

All tests use `z_A = (2, 0, 0)` and report the margin for candidate 0 over 1.

| test | margin |
|---|---|
| recomputed source change: disagreeing `z_B = (0, 2, 0)` → agreeing `(2, 0, 0)` | −0.617 → **+4.311** |
| Q-only patch from the agreeing run | −0.617 (the query encodes `z_A` only, so it is unchanged: the query route carries **none** of this source change) |
| K-only patch (agreeing keys, disagreeing values) | **−3.081** (routing amplifies the *disagreeing* value) |
| V-only patch (disagreeing scores, which are uniform, with agreeing values) | +1.847 |
| routing ablation (uniform) on the agreeing run | +1.847, so the routing effect is +2.464 |
| support-preserving change `z_B × 0.25` | +1.734 (still agreeing, weaker routing) |
| support-changing change at equal dot product: `z_B = (0, −2, 0)` vs `(0, 2, 0)`, both with `⟨z_A, z_B⟩ = 0`. The first vector votes *against* candidate 1; its argmax is tied between 0 and 2. The second supports candidate 1. | +1.847 vs −0.617 (decision differs; routing is identical and uniform) |

**Routing and value effects are not additive.** The K-only and V-only effects do not sum to the total effect.

**Routing is blind to this support change.** In the equal-dot-product pair, the decision flips through the **value** path. The gate is not involved at all: routing cannot see the change because the dot product did not move.

## B5. Attacks on the total-margin conclusion

| variant | m(agree) | m(disagree) | routing effect on agree |
|---|---|---|---|
| baseline | +4.311 | −0.617 | +2.464 |
| competing head (uniform; reads B, writes −kC), kC = 3 | +1.847 | +1.847 | +2.464 |
| rereading MLP (3.0) on fU | **−2.555** | +0.791 | **−1.201** |
| final LN, zero bias | +2.589 | −0.469 | +1.222 |
| final LN, bias +2 on fU₀ | +4.589 | **+1.531** | +1.222 |

**Competing-head sweep.** As kC runs over 0, 1, 2, 3, 4, 6, `m(agree) − m(disagree)` takes the values +4.93, +3.29, +1.64, 0.00, −1.64, −4.93. A competing writer cancels the agreement contrast at kC = 3 and reverses it beyond. The gate's own routing effect is unchanged throughout, so a head-level conclusion does not fix the total response.

**What each attack breaks.**

- The rereader breaks B-TM's no-rereader assumption, and the sign reverses.
- The final-LN bias preserves the sign of the routing effect, but changes the decision.

The **total-margin consequence holds exactly within B-TM's class** and fails in each direction outside it.

---

# Program C: multi-layer reconciliation

## C1. Corrected C′ operating region

See `CPRIME_CORRECTED_OPERATING_REGIONS.md`:

- Theorem C′1*;
- certified regions;
- the continuity obstruction C′-CO;
- the trade-off between λ and π.

## C2. Mediation graph

The graph is: sources `(a, b)` → block-1 LN coordinates `(x, y)` at A, B and R → MLP1 relation feature (`q_R` at R and `q_B` at B) → block-2 query gain `G = λq̂/√2` → scores `s_j` → routing `α` → values `v_j` (which depend on the sources and on `q_B` through `σ_B2`) → margin.

**Interventions.** The source change is `(a, b) → (a, −b)` on 200 random points (seed 51). The median total effect is −4.92.

| site interchanged (from the conflict run into the agree run) | median share of TE | decision → counterfactual |
|---|---|---|
| `q_R` (MLP1 output at R) | **1.000** | 1.000 |
| `q_B` (MLP1 output at B) | 0.000 | 0.000 |
| `q_R + q_B` | 1.000 | 1.000 |
| block-2 scores | 1.000 | 1.000 |
| block-2 values | 0.018 | 0.000 |

**On or off the manifold.**

*Proposition C2-OFF.* After block 1, R's e1 and e2 coordinates are exactly `(x, y)`, and its e3 coordinate is `λ1·q(x, y)`. The `q_R`-patched state has e3 coordinate `λ1·q(x′, y′)`. That is reachable only if `q(x′, y′) = q(x, y)`, which fails whenever the patch moves the state. So **the `q_R` interchange is off-manifold.*

A grid search agrees:

| base point | distance from patched state to reachable states | resolution control |
|---|---|---|
| (1, 1) | 5.49 | 0.35 |
| (2, 0.5) | 3.74 | 0.29 |

The **on-manifold** counterpart is the matched relation-blind pair (`MATCHED_RELATION_INTERVENTIONS.md` §4). There, two real inputs share an identical relation-blind margin and still reach opposite decisions.

**Path-specific conclusion.** The relation effect travels entirely through the query route, by way of `q_R` and the scores. The value route through `q_B` carries nothing measurable.

## C3. Composition

These results are on a 30 × 30 grid with `|a|, |b| ≥ 0.1`. The target rule is the corrected C′ rule outside its dead bands.

| variant | rule holds (agree) | rule holds (conflict) | reading |
|---|---|---|---|
| base C′ | 1.000 | 1.000 | |
| bypass β = 0.5 / 2 | 1.000 | 1.000 | bypass dominated by gated values |
| bypass β = 6 | 1.000 | 0.978 | bypass starts to dominate conflict cells |
| extra head reading the relation, reversed values, w = 0.5 | 0.998 | 1.000 | masked |
| same, w = 1.5 | **0.002** | 1.000 | **reversed** in agreement |
| downstream MLP γ = 1 (compressive) | 1.000 | 1.000 | monotone, preserves |
| downstream MLP γ = 3 (non-monotone) | 0.680 | 0.440 | **reverses** large margins |
| later rereading head w = 1 | 1.000 | 1.000 | |
| later rereading head w = 4 | 1.000 | 0.844 | net-evidence vote overrides the prior |
| redundant relation feature | 1.000 | 1.000 | function unchanged |
| competing reversed relation feature 0.5 | 0.998 | 1.000 | weakened |
| competing reversed relation feature 1.5 | 0.500 | 0.502 | **gate inverted**: agreement → prior |
| permuted source positions A ↔ B | identical decisions on 100% of cells | | C′'s decision is symmetric in the sources |

**Redundancy breaks unique mediation.** With the relation stored in two copies, ablating either copy preserves 200/200 decisions. Ablating both preserves 167/200. The 33 changed decisions are those where uniform routing gives the other answer. A single-site necessity test would wrongly report "not used".

## C4. Conditional theorem

See `CONDITIONAL_CAUSAL_THEOREMS.md`, Theorem C4.
