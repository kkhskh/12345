# MLP agreement law under shared LN2: corrected theorems

This document covers correction 2.2. Code is in `src/mlp_norm.py` (experiments M1–M7) and the log is `logs/mlp_norm.txt`. It supersedes the historical hypothesis (A0), which said that the read-token coordinate of source A is a function of `a` alone. The historical file is unchanged.

## 0. What was wrong

The read token's pre-LN2 residual is `r = x e1 + y e2 + w`, where `x = a/σ_A`, `y = b/σ_B`, and `w ⟂ e1, e2`. In this family `w = ρ e5`. LN2 divides `r` by **one shared scale**:

```
σ(r) = sqrt((x² + y² + ‖w‖²)/d + ε)
```

So the coordinate `u = x/σ(r)` depends on `b`. At `a = 0.8`, `u` = 0.708674, 0.698383 and 0.643596 for `b` = 0.2, 0.5 and 1.5, as the audit reported (M1). (A0) is **false**.

The conclusion that rested on (A0), `sign q = sign(ab)`, survives. It does not need (A0). It needs only the weaker fact below.

## 1. Five properties, kept apart

Write `D = diag(g)` for the LN gain and `β` for the LN bias.

| property | statement | holds under shared LN2? |
|---|---|---|
| P1 raw separable encoding | the pre-LN residual carries `x(a)` on e1 and `y(b)` on e2 | yes, by construction (block-1 LN is per position) |
| P2 jointly normalized coordinates | `(u, v) = (x, y)/σ(x, y, w)` | yes; this is what the MLP reads |
| P3 sign preservation | `sign u = sign x` and `sign v = sign y` | **yes** when `D = γI` with `γ > 0` and `β ⟂ e1, e2` |
| P4 magnitude preservation | `u` is a function of `x` alone | **no**, generically: `∂u/∂y = −xy/(dσ³)`, which is nonzero whenever `xy ≠ 0` |
| P5 ratio preservation | `u/v = x/y` exactly | **yes**: a common positive scale cancels (M1, to 1e−12) |

**Normalization-created interaction.** Any reader of `(u, v)` that is not invariant to a common positive rescaling picks up dependence on `x² + y² + ‖w‖²`, and so on the relation between the sources. This happens even if the reader is a pure sum unit (§3).

## 2. Theorem A1* (corrected agreement law)

**Lemma L1.** Let `h(t) = φ(t) + φ(−t)`.

- For ReLU, `h(t) = |t|`.
- For tanh-GELU, `h(t) = t·tanh(K(t + 0.044715 t³))`.

Both are even. Both are strictly increasing in `|t|`: for `t > 0`, the GELU form is a product of two positive increasing factors.

**Theorem A1*.** Suppose the read-token pre-LN2 residual is `r = x e1 + y e2 + w`, with `w ⟂ e1, e2`, and the following conditions hold:

- (N1) The LN2 gain is a scalar, `D = γI` with `γ > 0`.
- (N2) The LN2 bias is orthogonal to the read directions: `β·e1 = β·e2 = 0`.
- (N3) The four units read exactly `±c(e1+e2)` and `±c(e1−e2)`, with `c > 0`.
- (N4) The output weights are `(w, w, −w, −w)` onto one direction, with `w > 0`.

Then `q = w[h(c(u+v)) − h(c(u−v))]` satisfies `sign q = sign(uv) = sign(xy) = sign(ab)`. This holds for every `w` and every anchor or distractor content orthogonal to e1, e2.

**Proof.**

1. By (N1) and (N2), `u = γx/σ` and `v = γy/σ`, with `σ > 0`. This is P3.
2. By L1, `h(c|u+v|) > h(c|u−v|)` if and only if `|u+v| > |u−v|`, which holds if and only if `uv > 0`. ∎

Because the proof uses only sign preservation (P3) and never separability (P4), it survives the shared normalization.

**Check (M2).** Over 20,000 fresh draws (seed 11), with distractor content `w` of random norm in [0, 5], GELU and ReLU, and `c ∈ {0.3, 1.5, 6}`, there are 0 violations in 120,000 evaluations.

## 3. Normalization-created interaction (Proposition N1)

Consider a pure sum reader `s = c⟨LN(r), e1+e2⟩ = cγ(x+y)/σ`. This is relation-blind in raw coordinates before LN.

- **Its sign is still relation-blind:** `sign s = sign(x+y)`.
- **Its magnitude is not.** At fixed `S = x+y`, `σ² = (S² − 2xy + ‖w‖²)/d + ε`. So `|s|` is strictly increasing in `xy` whenever `S ≠ 0`.

In words: at a fixed net evidence, agreement inflates a sum unit's output, and conflict shrinks it, with no relation computation anywhere in the circuit.

**Check (M3).** At `S = 1`, `⟨LN(r), e1+e2⟩` runs from 0.7947 at `xy = +0.25` down to 0.5855 at `xy = −3.75`.

**Consequence.** Any test that labels a circuit "relation-sensitive" because its magnitude, confidence or margin covaries with agreement at a fixed raw sum is confounded by LN. This is the theoretical root of the historical PA9 failure. The matched contrasts in `MATCHED_RELATION_INTERVENTIONS.md` therefore hold the post-LN sum `Ŝ = u+v` fixed, not the raw sum.

## 4. Removing the hypotheses: derived operating cones

### (N3) violated: cross-talk in the read directions

Suppose the read directions are `e1' = e1 + t12 e2` and `e2' = e2 + t21 e1`.

- Then `q`'s sign law becomes `sign q = sign(u'v')`, with `u' = u + t12 v` and `v' = v + t21 u`. The proof is identical, applied to `(u', v')`.
- With `ρ = v/u`, `sign(u'v') = sign(uv)·sign((1 + t12ρ)(1 + t21/ρ))`. So the original law holds exactly where `(1 + t12ρ)(1 + t21/ρ) > 0`.
- For `0 < t12, t21` with `t12·t21 < 1`, agreement cells (`ρ > 0`) are always correct. A conflict cell is correct if and only if `|ρ| ∈ (t21, 1/t12)`. This is the **operating cone**: the law fails when the second source is too weak relative to the first (`|ρ| ≤ t21`) or too strong (`|ρ| ≥ 1/t12`).
- For `t12, t21 < 0`, the roles of agreement and conflict are swapped.
- For mixed signs, there are **two** failure cones, one for each sign of ρ (post-review, R1 F10).

**Check (M4).** Over 200 random `(t12, t21)` in [−0.6, 0.6]², with 100 fresh points each (seed 12):

- the derived law has **0 violations in 20,000**;
- the original law fails on 3,504 of 20,000 points.

The thresholds come from the perturbation, not from any outcome.

### (N1) or (N2) violated: general LN gain and bias

`u = (x e1ᵀDe1 + y e1ᵀDe2 + e1ᵀDw)/σ + e1ᵀβ`, and likewise for `v`.

- A non-scalar gain creates cross-talk `e1ᵀDe2` and an anchor leak `e1ᵀDw`.
- A bias creates a constant offset.
- The offsets move the cone's apex away from the origin. Near-zero evidence then fails, which is the same structural reason C′ fails near the origin.

**Check (M6).** Over 200 random gain/bias draws × 50 points (seed 13):

- the derived law `sign q = sign(u_pred·v_pred)` has **0 violations in 10,000**;
- the original law fails on 866 of 10,000 points.

### (N4) violated: unequal output weights, ratio `r = w2/w1`

`q > 0` if and only if `h(c|u+v|) > r·h(c|u−v|)`.

- **ReLU:** `h` is homogeneous of degree 1, so the boundary is the scale-free cone `|u+v| > r|u−v|`. For agreement with `t = v/u > 0`, this is `(r−1)/(r+1) < t < (r+1)/(r−1)`.
  - The upper bound was missing in the reviewed version and is added after review (R1 F4). At `r = 2`, `t = 3.1` gives `q < 0`.
  - Because the boundary is homogeneous, **the LN scale σ is irrelevant**.
- **GELU:** `h` is not homogeneous. `h ≈ Kt²` near 0 and `h ≈ |t|` far out, so the boundary moves from `(√r−1)/(√r+1)` at small scale to `(r−1)/(r+1)` at large scale. **The shared LN scale σ, and hence the distractor content `‖w‖` and the other source's magnitude, move the decision boundary.**

**Check (M5)**, at `r = 2`. The scan covers `t ≤ 1` only, so it checks the lower bound alone:

| activation | t* at input scale 0.2 | t* at 1 | t* at 5 | analytic limits |
|---|---|---|---|---|
| ReLU | 0.3335 | 0.3335 | 0.3335 | 1/3 |
| GELU | 0.1741 | 0.2321 | 0.3335 | 0.1716 (small) and 1/3 (large) |

## 5. Architectural and semantic stability

- **Architectural stability:** the law holds across the parameter class. A1* is exactly invariant to:
  - the scalar LN gain;
  - the unit scale `c`;
  - the output scale `w`;
  - activation choice within {GELU, ReLU};
  - every gauge that preserves (N1)–(N4).

  It is **not** stable to generic perturbation of (N3) or (N4). Under those perturbations it degrades to the derived cones above, so it holds on an open set of inputs rather than everywhere.
- **Semantic stability:** the law holds across source semantics, meaning anchor norms κ, distractor content `w`, source magnitudes, and distributions of `(a, b)`.
  - Under (N1)–(N4) the sign law is semantically stable for every input. M2 varies `w` freely with 0 violations.
  - Under (N4) violations with GELU, the boundary depends on input scale. A change in source *semantics* alone, such as larger anchors or a stronger third source, moves the decision.

  These are different failure modes, and the cones say which one applies.

## 5b. Proposition N2: shared LN is itself a cross-source operation (added after review, R1 F5)

Suppose each source's magnitude carries its sign, for example `P(a) = 1.5 − 0.5·tanh(5a)` and `Q(b) = 2 − tanh(5b)`.

- After shared LN, a **single** coordinate `X = P(a)/σ(P, Q)` takes four distinct values, one per sign pattern.
- A ReLU bump reading only `X` then computes XNOR exactly: 100% on a 120 × 120 grid (`reviews/r1_scripts/xnor_ln.py`).

So in a pre-norm Transformer, **"no unit reads two sources" does not imply "relation-free"**, because the normalization couples the sources. A relation-free claim must instead be stated at the level of the computed function, as in T-RF and T-RF′ (`CONDITIONAL_CAUSAL_THEOREMS.md`).

## 5c. Parameterization caveat (R4 F11)

The architectural-vs-semantic split in §5 depends on parameterization. For example, folding a non-scalar LN gain into `W1` turns a gain violation (N1) into a cross-talk violation (N3). The *derived law* is invariant to this choice; the *labels* are not.

## 6. Verification

- M7: the read-token formula with `c = 1` matches the `cbmath.agreement_family` forward margin to `8.9e−16` on 300 fresh points.

## 7. Status

| claim | status |
|---|---|
| (A0) separable coordinate | **FALSE** (M1) |
| A1* sign law under shared LN2, assuming (N1)–(N4) | **PROVED**; 0/120,000 (M2) |
| P5 ratio invariance | **PROVED**; exact to 1e−12 (M1) |
| N1, normalization-created interaction in sum readers | **PROVED**; demonstrated (M3) |
| cross-talk, gain and bias, and unequal-weight cones | **DERIVED**; 0 violations of the derived laws on fresh points (M4–M6). The ReLU cone's upper bound and the second mixed-sign cone were added after review |
| N2, shared LN couples the sources | **SHOWN** by a reviewer construction |
| GELU boundary depends on LN scale; ReLU boundary does not | **PROVED**; matches analytic limits (M5) |
