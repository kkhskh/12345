# 23 — Verdict

`STEP1C_R_ONLY_GENERIC_COMPUTATIONAL_RESULTS_SURVIVE`

The block in `04` has a proved derivative, a proved constant-shift response, and a set of counterexamples. The constant-shift response is per-token mean subtraction plus the residual add. The derivative is the chain rule. Neither is a consistency score, and neither restores `D_c ↓ ⇒ L_c ↑`. Phase 2 stays `NO GO`. `OBSTRUCTION_READY` stays `NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` stays.

The clean-room log in this archive records `pytest` and the manifest check on an extraction of the archive before those log files were inserted. After the logs were inserted, the archive was rebuilt. The release claim is that a further extraction of the rebuilt archive runs `python -m pytest tests -q` and `python scripts/verify_manifest.py` with exit code 0. That further run is not copied back into the archive, because copying it would change the archive. A rerun that disagrees with exit code 0 supersedes this paragraph.

## Q1

The `ε=0` claims are no longer applied to `ε>0`. Shift invariance holds for every `ε≥0`. Scale invariance of the normalized vector, for `d≥2` and `α>0` with `α≠1`, holds for every admissible `h` if and only if `ε=0`. The full map has the same biconditional when some gain is nonzero. If every gain is zero the map is the bias at every `ε`. The identity scale `α=1` holds at every `ε`. Dimension `1` has a zero centered vector.

## Q2

For `ε>0` the bracket has kernel `span{1}` and image `1^⊥`. If every gain is nonzero, the Jacobian has rank `d−1`. Exactly one zero gain, with the others nonzero, leaves that rank. Exactly two zero gains, with the others nonzero, drop it to `d−2`. A tolerance of `10^{-3}` on the checked vector reports rank `2` because the radial singular value is about `6.7×10^{-5}`. The algebraic rank at `ε=10^{-5}` is still `3`.

## Q3

A constant shift is invisible to both normalizations and is added by the skip. If the edited token is the readout, `R=a 1^T w`. Otherwise `R=0`. No attention or MLP weight is used. A centered direction at `ε>0` is not in the kernel when the gains on its support are nonzero. The block derivative along it is the skip plus a branch term. Those terms can cancel. Invisibility to the normalization is not invisibility to the block, and visibility of the skip term is not visibility of the sum.

## Q4

The pair `c_1=1`, `c_2=−0.5` does not make both deletions necessary. Deleting `c_2` leaves `1`. If `c_1+c_2>0` and both deletions fail to stay positive, then both coefficients are nonpositive, which contradicts the sum. The replacement `b=−1.5`, `c_1=c_2=1` makes both deletions necessary and has four-term interaction `0`.

## Q5

Necessity, redundancy, and sufficiency are properties of a deletion and a threshold. Interaction is the four-term contrast of two increments from a stated baseline. Synergy, in this note, means that contrast is nonzero. Compensation means one parent lies in the equation of another, so recomputation can change the second term. It need not restore the margin. The biased sum has both deletions necessary, interaction zero, and no compensation. All three of necessity, redundancy, and sufficiency move when the baseline or the threshold moves. Interaction of an affine map does not.

## Q6

`H` at token `t` is replaced by `H_t+δ`, and the block is recomputed. Queries, keys, and values are not held fixed. The first normalization of token `t` changes unless `δ` is constant. Later tokens change only through attention. Earlier tokens do not.

## Q7

`Q`, `K`, and `V` change in row `t` only, at the first attention. Scores in row `t` change through the query, for legal keys. Scores in column `t` change through the key, for queries `i≥t`. The value row `t` changes outputs at those same queries even if the pattern is clamped. Patterns change wherever those scores change. Value rows `i≠t` do not change, because their normalized inputs do not.

## Q8

The mask sets scores with `j>i` to `−∞` before the softmax, and those scores do not depend on `X`. Attention outputs at `i<t` do not read token `t`. For an edit at token `2`, tokens `0` and `1` are unchanged at `H_2`. The readout is token `2`, so an edit at token `0` or `1` can still change `m`.

## Q9

`dH_1=dH+dA`. The first term is the skip. For a constant `δ` the second term is zero at every token, and `H_2` changes by that constant at the edited token only.

## Q10

The second normalization is per token and kills constants again. A non-constant increment is not killed when `ε>0` and every gain is nonzero. The MLP derivative multiplies by `1−tanh(pre)^2`, which is smooth. Then `dH_2=dH_1+dM`, and `Dm=w·dH_2[2]`.

## Q11

`Dm` is that composition. On seed `0`, token `1`, `δ=(0.2,−0.1,0.05,0.3)`, it equals `0.9989135623527045`. The central difference at step `10^{-6}` differs by relative error `3.22×10^{-6}`. The attention map alone matches its own central difference in `test_attention_directional_derivative_matches_finite_differences`. This is a check of the implementation.

## Q12

The linear sign is the wrong guide when the direction is orthogonal to the token gradient and the remainder dominates, and when the step is not small. On seed `0` the unit vector `u∝(g_1,−g_0,0,0)` has directional derivative about `−10^{-16}` and finite response `−0.629265931` at step `0.2`. The ratio `R/step²` at step `10^{-3}` is `−14.0474795`. That ratio is not the pure second-order coefficient. It is enough that the first derivative does not account for the response. A value-only derivative can also have the opposite sign from `Dm` while both are first-order. That is a missing path, not a remainder. Seed `16` has total derivative `+0.0201953835` and value-only derivative `−0.0085806918`.

## Q13

The sharp finite examples are that seed-`0` direction, the seed-`16` value-only sign, and the seed-`16` pair that shares a projection and not a response. The constant shift is the opposite kind of example: an arbitrarily large change of one token with response exactly `0` when that token is not read.

## Q14

Not for every intervention. Deleting feature `3` leaves seed-`16` `H` and the state with `H_{0,3}` increased by `2` indistinguishable. The intervention `δ=(0.4,−1.2,0.3,0.5)` at token `1` has responses `−0.121187908` and `+0.108445402`. On the constant-shift family every projection is sufficient, because `R` does not depend on `H`.

## Q15

The missing object in that pair is the coordinate `H_{0,3}`. It can enter token `0`'s query, the keys and values seen by later tokens, and everything downstream of those. The example does not isolate which route carries the sign. It shows that deleting the coordinate is enough.

## Q16

Putting `H_{0,3}` back next to the projection separates this pair, because the states already agreed on feature `3` at the other tokens. That joint measurement is not a scalar discrepancy. A function of the pair can collapse the pair again. Several partial maps restore information only to the extent that their coordinates separate the states.

## Q17

Yes, on the recomputed family. `H_1` and `H_2` are assigned by the structural equations, so the residual of those equations is zero. Seed `0` has margin about `+0.464` and seed `2` has margin about `−1.356`, with that residual still zero. The residual does not order the margins.

## Q18

When the relation is not the structural equation. The rank-one map built from token `0` of seed `1` has residual `1.25789191` on the base state and `1.79690659` after a constant shift at token `1`. The two margins agree. The residual is the error of the map. A nonzero residual obtained by clamping `H` and `H_1` to inconsistent values is a different intervention from the one that defines `R`.

## Q19

Not on a general `δ`. The sign of `Dm·δ` is the linear sign. Real values of that form on a basis are the gradient. Signs on a basis are only an orthant. A statistic unchanged under `δ↦−δ` cannot return both signs when `Dm·δ≠0`. The constant-shift evaluation predicts an exact finite `R` without `W_Q`, `W_K`, `W_V`, `W_O`, or the MLP. That prediction is the architecture, and it stops at constant vectors.

## Q20

The constant-shift identity constrains more than a pure constant. For every finite edit `E`, adding `a1` at token `t` changes `R` by exactly `a 1^T w` when `t` is read and by `0` otherwise. The interaction with `E` is zero. This is `25`. It does not determine the centered part of a general edit. That part still requires an evaluation of `m` or a remainder bound. `Dm=0` does not imply `R=0`. A centered direction is not weight-free: on seed 7 the live block and the zero-branch block disagree.

## Q21

On constant shifts the identity is an equality, which is tighter than a Lipschitz bound and does not need the Jacobian. No inequality of the form `|R|≤f(P,ρ,H)`, free of the branch weights and valid for all `δ`, was proved. The feature-`3` pair has responses of both signs, so no function of that projection determines the sign.

## Q22

Recomputation uses the weights, the state, and `δ`. The derivative uses the same objects through the chain rule. The constant-shift evaluation uses the architecture, the read-token index, and `1^T w` when the edited token is read. A projection uses the coordinates it keeps. The structural residual uses the activations and returns zero.

## Q23

Recomputation restates `R` because it is `R`. The derivative restates the linear term. The constant-shift identity does not restate a general response. It evaluates one symmetry. Assuming "the margin moves with the proposed score" would put the sign in the hypothesis. None of the proved rows uses that assumption.

## Q24

An invertible change of basis of `H`, with the weights and `w` and `δ` transformed with it, preserves the computation. The numerical coordinates of `δ` change. LayerNorm is not that conjugation. It is a function inside the block. A fixed readout of `H` and of `LN(H)` are different scalars. The constant-shift identity is tied to mean subtraction. A normalization that keeps the mean does not inherit it. The identity is meaningful inside this architecture. It is not a coordinate-free obstruction.

## Q25

There is a mathematically valid identity: constant residual shifts pass only through the skip. It is scientifically empty as a new mechanism. It does not rank natural decisions, it does not see the seed-`16` sign, and it does not see the remainder along `u`. Calling it a COR object renames the skip.

## Q26

No. The derivative is calculus on Vaswani attention, Ba LayerNorm, and a pre-norm residual in the sense of Xiong et al. The frozen-pattern term is the OV path of Elhage et al., and seed `16` is what happens when the pattern is not frozen. The projection failure is the fiber condition in Geiger et al., JMLR 2025, for one coordinate deletion. The one-pair transport is the kind of fitted map Sutter, Minder, Hofmann, and Pimentel show can be vacuous when it is unrestricted. This step does not re-prove their theorems.

## Q27

R6, the constant-shift evaluation, is the strongest exact finite statement, and it is generic. R2, the positive-`ε` rank, is the strongest correction of the previous package. R12 and R14 are the strongest negative existence results: a value-only linearization can have the wrong sign, and a coordinate deletion can hide opposite responses. R19, "the pieces sum to zero," is a tautology and is not a result.

## Q28

The seed-`16` projection pair is the most direct threat to a relational bridge, because the proposed measurement agrees and the recomputed signs do not. The seed-`0` direction is the most direct threat to reading a causal conclusion off `Dm`. The value-only sign is the most direct threat to treating the OV path as the intervention.

## Q29

Stacks, post-norm, parallel residuals, RMSNorm, rotary positions, and trained operating points were not differentiated. Q/K/V biases are absent from the numerical Jacobian. The shift identity survives an additive bias after `XW`, because `X` does not change, and that extension was checked with nonzero biases by the counterexample pass. Rates were not estimated. R21 does not exclude an unexamined score. It records that none was exhibited.

## Q30

No. The identities that survive are the derivative and one architectural symmetry. They do not identify a reconstruction of the consistency theory, and they do not justify a later experimental stage. Phase 2 remains closed. A future argument inside this same step would need a non-constant family, a quantity computed without `R` and without `Dm`, and a bound that is tighter than the symmetry and the Cauchy estimate. That argument is not a Step 2 protocol.

## Gates

| gate | result |
|---|---|
| A, previous errors | The `ε` split and the both-necessary proof are in `02` and `03`. The false quantifiers found in review were rewritten and checked again. |
| B, reproducibility | Judged from `logs/clean_room.txt` and `logs/manifest_check.txt`, not from the development tree. |
| C, the block | Normalization, Q/K/V, mask, residual, MLP, readout. |
| D, the intervention | `H↦H+E_{t,δ}` is recomputed. Value-only is labeled as a different derivative. |
| E, independent checks | Analytic Jacobian, SymPy, central differences, and the recomputed finite `R`. |
| F, sufficiency | Feature 3 fails. The constant-shift family makes every projection sufficient for a vacuous reason. |
| G, consistency | The structural residual is zero on recomputed traces. A surrogate residual is approximation error. |
| H, where the sign enters | `13` and `15`. |
| I, simpler alternatives | `18` and `19`. |
| J, counterexamples | Proved rows that survived are restricted to the quantifiers in `16`. Existence rows are marked as existence. |
| K, novelty | R6 and the derivative are classified as known structure. |
| L, scope | One block. No stack. No trained model. |

Preserved status, not reopened: `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`, `PHASE_2 = NO GO`, `OBSTRUCTION_READY = NO`.
