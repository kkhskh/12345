# Protocol addendum: agreement as an intermediate variable (REL-XOR3)

**Status.** Frozen before any REL-XOR3 model was trained.

**Why this addendum exists.** The causal reviewer (finding R_causal F1) showed that in REL-XNOR and REL-MIS the relation and the target are confounded on matched pairs. On those pairs, S_rel cannot be told apart from a target or decision direction. This addendum adds the deciding experiment that the reviewer asked for.

## Task

**REL-XOR3** (internal code T7): `y = sign(ab) · p`.

- Tokens are `[A, B, P, R]`.
- The training distribution is the historical mixture.
- Agreement `s_AB = 1[ab > 0]` is an intermediate variable that **differs from the target**: when p is flipped together with agreement, the target stays the same.
- The rule is also equal to `sign(ap)·sign(b)` and to `sign(bp)·sign(a)`. The test below therefore asks which intermediate the model actually uses, if any.

The aggregate-only `(z, p)` ceiling is 0.754.

## Models

Training is identical to the earlier models (`scripts/train_xor3.py`).

- Architectures: TF1, TF2 and TF2-AO.
- **Dev seeds:** 0 and 1, used only to check that the code runs.
- **Evaluation seeds:** 2–6.

## Sites

The sites are the primary sites from `PROTOCOL.md` §2.

## Candidate subspaces at each site

| subspace | how it is obtained | uses outputs or targets? |
|---|---|---|
| **S_rel** | Same rule as `PROTOCOL.md`: matched pairs with the same z and the same p, and opposite `s_AB`. | output-free |
| **S_das_AB** | DAS trained so that interchange reproduces hypothesis H_AB | uses H_AB's predicted labels |
| **S_das_y** | DAS trained so that interchange reproduces hypothesis H_y | uses H_y's predicted labels |

The k for each DAS subspace equals S_rel's k. Both DAS subspaces are trained on the fit pairs (seed 7001).

## Decisive interchange set D

Pairs `(x, x′)` with the same z, opposite `s_AB`, and **opposite p** (seed 7002, 600 pairs). A control set C uses the same p.

The U-component is patched from the source x′ into the base x. The rest of the base state keeps the base's p and sources.

**Predictions for the patched decision under each hypothesis:**

| hypothesis | U carries | predicted decision |
|---|---|---|
| H_AB | agreement | `sign(a′b′) · p_base` |
| H_y | the target | `y(x′) = sign(a′b′) · p′` |
| H_AP | `sign(ap)` | `sign(a′p′) · sign(b_base)` |
| H_BP | `sign(bp)` | `sign(b′p′) · sign(a_base)` |

On D, H_AB and H_y always predict opposite decisions.

## Metrics

For each subspace, report the fraction of D pairs whose patched decision equals each hypothesis's prediction. Report the same on C.

## Verdict for each site

| verdict | condition |
|---|---|
| **AGREEMENT_INTERMEDIATE** | S_rel agrees with H_AB on at least 0.9 of D. **Or**: S_das_AB reaches at least 0.9 on held-out D (seed 7003) and beats S_das_y by at least 0.2. |
| **TARGET_ONLY** | S_rel agrees with H_y on at least 0.9 of D, and S_das_AB is below 0.9. |
| **OTHER_INTERMEDIATE** | S_rel agrees with H_AP or H_BP on at least 0.9 of D. |
| **UNRESOLVED** | none of the above |

These verdicts are frozen together with `src/intermediate.py`.
