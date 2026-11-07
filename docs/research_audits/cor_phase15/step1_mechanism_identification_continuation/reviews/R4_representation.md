# R4: Representation review (source semantics, gauges, feature identification)

Reviewer role: representation. Package at commit `d1aa97d`. The checks are reproducible with the scripts in `reviews/R4_checks/`, run from that folder with `python3 cN_*.py`. No file outside `reviews/` was modified.

Severity scale: **CRITICAL** (a headline claim does not follow), **MAJOR** (a claim is overstated or a method is mis-specified), **MINOR** (wording or bookkeeping).

---

## Findings

### F1. CRITICAL: the "relation unit" criterion mostly flags units that the package's own definition calls non-relational

**Location.**
- `src/learned_analysis.py` lines 174–180 (`base_r2`, `full_r2`).
- PREREG §7.4.
- LEARNED §3, and the T1 "relation units" in §1 and §6.

**Evidence.**
- PREREG §7.4 defines `I_agree` as the variance explained by `ab` "beyond the best functions of `a+b` and of `a` and `b` separately". But `ab = ((a+b)² − a² − b²)/2` lies exactly in that class. LEARNED §1 claim 4 uses this same identity. So under the preregistered definition, the `ab` term can never add anything.
- The code avoids this only because its base is two fixed regressors (ANOVA additive fit and sum bin-mean, one coefficient each), not the joint best `g(a+b) + f1(a) + f2(b)`. It also adds `sign(ab)`, which §7.4 does not mention (see F8).
- `c1_iagree_synthetic.py`, synthetic units:
  - `ab` and `(a+b)² − a² − b²` are both **flagged** (I_agree = 0.266). Their residual outside the non-relational class, by backfitting, is **0.000**.
- `c2_units_real.py`, all 60 binary MLP checkpoints. Among code-flagged units, the number whose share outside `g(a+b) + f1(a) + f2(b)` is ≥ 0.2:

  | task / arch | code-flagged vs ≥ 0.2 outside the class, per seed |
  |---|---|
  | T1 / TF1 | 22→3, 26→6, 23→8, 16→6, 19→6 |
  | T1 / TF2 | 39→13, 53→31, 49→29, 47→20, 50→31 |
  | T3 / TF1 | 20→3, 20→4, 23→2, 17→3, 18→3 |

  So 60–90% of the TF1 "relation units" are, to ≥ 80% of their variance, a nonlinear sum function plus per-source functions.

**Correction.**
- State the reference class explicitly. "Relation-sensitive" here means "not linearly explained by two fixed regressors", not "outside `g(a+b) + f1(a) + f2(b)`".
- Or adopt the backfit residual as `I_agree`, and re-report the counts as a deviation.
- Drop "relation computation" language for units whose activity is an additive-plus-sum function. As F2 shows, such functions alone already solve XNOR.

### F2. CRITICAL: "relation computation" vs "non-relational features" is not drawn consistently

**Location.**
- LEARNED §1 claims 1 and 4.
- SOURCE §2 (D5: "a declared simpler aggregate, e.g. `a + b`").

**Evidence.**
- Claim 4 explains `ab` decodability in T2 as "a linear function of non-relational features". `c3_probe_readout.py` supports this: decoding `ab` from the three decoded features `a²`, `b²`, `(a+b)²` matches the direct `ab` probe (TF1 s0 0.978 vs 0.879; TF2 s0 0.990 vs 0.990).
- But the same identity makes the T1 target `sign(ab) = sign((a+b)² − a² − b²)`. That is a threshold of a sum function plus per-source functions.
- It also makes the matched-sum contrast Δagree degenerate. At fixed `S`, `ab = (S² − (a² + b²))/2`. So on the matched pairs, agree vs conflict is exactly small vs large `a² + b²`, an additive feature. Any model `m = a + b + f(a) + f(b)` with even nonlinear `f` has Δagree ≠ 0.
- Under one consistent definition, therefore, either T1 is also solvable by "non-relational" features (contradicting claim 1's framing), or `ab` is relational (contradicting claim 4's framing).
- B-add (no term reads two sources) failing T1 is consistent with this: `g(a+b)` reads both sources.

**Correction.**
- Define relation relative to a named feature class. Two options:
  - (i) D5 relative to `a + b` only: then T1, T3 and T4 all "use relation" whenever `D_sum > 0`;
  - (ii) relative to `{g(a+b), f1(a), f2(b)}`: then `ab` is non-relational, and only units with residual outside that span count.
- Rewrite claims 1 and 4 under one choice, and note that Δagree, M-flip and the step-6 `cond_diff` cannot distinguish `ab` from `a² + b²` at matched sum.

### F3. CRITICAL: the set-level ablation effect of the T1 "relation units" is barely specific

**Location.**
- LEARNED §1.1 ("ablating the identified units roughly halves Δagree, and drops accuracy to about 0.75 in 7 of the 10…").
- REPARAM §1 ("population … carries a set-level ablation effect E").

**Evidence** (`c8_threshold_and_random.py`):

| model | flagged set: Δagree, acc | 10 random unit sets of equal size | complement set |
|---|---|---|---|
| T1/TF1 s0 (22 of 32 units) | 10.2, 0.746 | Δagree 10.3–16.8; acc ≈ 0.75 in 5 of 10 | 20.4, 1.000 |
| T1/TF1 s3 (16 of 32) | 10.5, 0.749 | Δagree 11.0–15.8; acc ≈ 0.75 in 5 of 10 | 16.5, 0.999 |
| T1/TF2 s0 (39 of 64) | 17.4, 0.9996 | Δagree 12.7–17.5; acc ≤ 0.79 in 5 of 10 | **12.8, 0.749** |

- In TF2 the *unflagged* units are more necessary than the flagged ones.
- The effect is also threshold-fragile. For T1/TF1 s0, the threshold τ gives:

  | τ | units | Δagree |
  |---|---|---|
  | 0.20 | 22 | 10.2 |
  | 0.25 | 7 | 20.7 (no effect) |

- Most flagged units sit just above threshold: 9–30 per model have `min(I_agree, D_sum)` < 0.25 (dev and held-out JSON).
- Mean-ablating 50–70% of a 32-wide MLP collapses one quadrant whichever units are chosen.

**Correction.**
- Add an equal-size random-set ablation control and a threshold sweep to the preregistered procedure, as a declared deviation for the next round.
- Until then, report "ablating ~2/3 of the MLP halves Δagree", not "identified relation units are causally used".
- The gauge-invariant population statement is true but uninformative without this control.

### F4. MAJOR: interchange at the final residual tests readout alignment, not use; T1's "decodable and used" is close to tautological

**Location.**
- LEARNED §1.1 ("1.0 against 0.0"), §4 table, and §6 ("the agreement variable in T1, both decodable and causally used").

**Evidence.**
- After the last residual, the only remaining computation is the final LN and the unembedding. The margin is `(x − μ)·v/σ`, so `sign m = sign(x·(v − v̄))`.
- An interchange along `u` at that site therefore moves the decision only through `cos(u, centred readout)`.
- `c3`: `|cos(agreement probe, centred readout)|`:

  | task | value |
  |---|---|
  | T1 | 0.90–0.99 |
  | T2 | 0.01–0.13 |
  | T3 | 0.01–0.04 |
  | T4 | 0.00–0.07 |

- In T1 the generator label `1[ab > 0]` *is* the target. A probe for it at a site that is linear-up-to-LN in the decision must recover the readout. So the margin is not used to select the probe, but in T1 the probe coincides with the margin direction by construction.
- At the one informative site (`resid_mid0`, before the MLP), the T1/TF1 agreement interchange transfers **0.503** in dev seeds 0 and 2 (1.0 in seed 1), although `sign(ab)` is decodable there at 0.998. The doc does not report this.

**Correction.**
- Report the interchange at pre-final sites as the "use" test, including the 0.50 values.
- State that final-site interchange equals readout alignment.
- Qualify "selection never uses the margin": in T1, label = target, so label-supervised probes inherit the decision direction.

### F5. MAJOR: the unit criterion misses odd and support-conditional interactions; in T5 the `⟨z_A, z_B⟩` regressor is a no-op

**Location.**
- `learned_analysis.py` line 178 (binary) and line 303 (T5).
- PREREG §9 D2.

**Evidence, binary.** The added regressors `ab` and `sign(ab)` are even under `(a, b) → (−a, −b)`, so any odd interaction scores **0**:

| interaction | backfit residual share | code `I_agree` |
|---|---|---|
| `ab(a−b)` | 0.44 | 0.000 |
| `sign(ab)(a−b)` | 0.25 | 0.000 |

- `sign(ab)·sign(a+b)` also scores 0, but it is actually ≈ non-relational (residual share 0.004). So that particular miss is harmless.
- `1[ab > 0]·sign a = (sgn a + sgn b)/2` is additive and is correctly not flagged.
- On real checkpoints, the backfit flags 1–8 units per TF2 model that the code misses; up to 4 per model are > 50% odd (`c2`, `backfit_only` / `oddmiss`).

**Evidence, T5.**
- `⟨z_A, z_B⟩ = Σ_i (s_i² − z_{A,i}² − z_{B,i}²)/2` lies exactly in the base span (poly2 of `s` plus poly2 of each source). The residual std is 4e−15 (`c5`), and the R² gain from the dot is ≤ 3e−16 for every T5 unit.
- So `I_agree^K` reduces to the gain from a single indicator, `1[argmax agree]`.
- The candidate-conditional features `1[agree]·onehot(k*)` and `1[disagree]·onehot(p)`, which T5b's target is built from, raise the maximum unit gain from 0.034 to 0.155. In TF2 L1, 6–11 units exceed 0.1 (`c6`). The "0 relation units in T5b" result still holds at 0.2, but the feature set is mis-specified.
- In T5b, `D_sum^K(unit)` is ≥ 0.2 for **32/32** units in every layer, because the prior token's variance is not a function of `s`. So that filter is vacuous in T5b.
- `D_sum^K` uses poly2 in `s`, where the binary task uses nonparametric bin means: 3 vs 0 units ≥ 0.2 in T5a/TF1 (`c5`). The two criteria are inconsistent.

**Correction.**
- Drop the dot from `I_agree^K`, or state that it is redundant.
- Add odd and candidate-conditional terms (`ab·(a−b)`, `sign(ab)·a`; `1[agree]·onehot(k*)`, `1[¬agree]·onehot(p)`).
- Condition `D_sum^K` on `p`, and use a flexible sum model.

### F6. MAJOR: the reason given for not applying residual rotation is wrong; the gauge exists and was untested

**Location.** REPARAM §1, "Gauges not applied"; the `src/reparam.py` docstring.

**Evidence.**
- With R orthogonal and `R1 = 1`, LN commutes with R. The gains fold into the next matrices.
- The LN1 bias is re-solved to match the query biases: 16 equations in 16 unknowns. Key-bias changes are constant over keys and cancel in softmax. Value-bias changes are absorbed into `bO`. The final LN bias goes into `bU`. No new parameters are needed.
- `c4_resid_rotation.py`:
  - max logit change: 2.7e−6 (T1/TF1) and 8.2e−6 (T3/TF2);
  - probe R² change ≤ 0.0055;
  - unit counts, sign hits and every interchange value unchanged.
- So the residual-stream claims *do* survive. But the doc's reason for calling them intrinsic ("because the residual stream was not transformed") is circular, and the table's "invariant" entries for residual quantities were never exposed to a residual gauge.
- The row "Q/K-space or V/O-space directions: changed" is not measured by `reparam.py`. It is analytic.

**Correction.**
- Add the rotation gauge; the code is in `c4`.
- Replace the stated obstruction.
- Mark the Q/K-direction row "analytic, not measured".
- Note that the random-direction control (seed 999) is a fixed coordinate vector, not a gauge-covariant object. Its value happened to be unchanged under this rotation.

### F7. MAJOR: the per-unit sign-prediction hit rates are not gauge-invariant; step 6 omits final-LN centring and indirect paths

**Location.**
- LEARNED §3 ("86% for TF1 dev and 94% for TF1 held-out").
- `learned_analysis.py` line 165; PREREG §7.6.

**Evidence.**
- Splitting is function-preserving for any output split `(t, 1−t)`.
- Splitting one missed unit into k copies changes 21/22 to 21/(21+k), because both copies always share the same activation statistics. So any hit *rate* is unit-count weighted and not invariant. REPARAM §1's table omits this row.
- The frozen-weight direction `gf ⊙ ΔW_U` ignores the final-LN mean subtraction. The correct direct-path direction is the centred vector `v − v̄`. Using it flips the predicted sign of 1–2 units in 7 of 20 T1/T3 models (`c2`, `signflip_centre`).
- For TF2 layer-0 units, the "direct" weight ignores layer 1. The L0 hit rates are 4/17, 5/16 and 9/20 for T3 dev (at or below chance), against 10/21 to 16/20 for L1.

**Correction.**
- Report hit rates weighted by `|effect|` or per equivalence class of duplicated units.
- Use the centred readout.
- Restrict the "direct-effect" prediction to last-layer units, or compute the full Jacobian path.

### F8. MAJOR: undeclared deviations from PREREG §7.4; the D1–D4 timing is only partly verifiable

**Location.** PREREG §7.4 and §9; git history.

**Evidence.**
- §7.4 specifies `ab` only, with "backfitting" against sum and additive fits. The code adds `sign(ab)` and uses a two-regressor linear base, not backfitting.
- Without `sign(ab)`, the criterion would flag only quadratic-`ab` units. With proper backfitting, it would flag none (F1).
- Neither difference is listed in §9.
- Timeline (git and file mtimes):

  | event | time |
  |---|---|
  | `learned.py` last modified | 20:22:24 |
  | dev training ends | 20:40:40 |
  | dev training starts (end minus 1,037 s wall) | ≈ 20:23:24 |
  | D1–D4 commit `fad7f89` | 20:27:19, about 4 min **after** training started |
  | `learned_analysis.py` last modified | 20:36:22, during training and with dev checkpoints on disk |
  | commit `c4bc553` | 20:42:35 |

- This matches §9's own wording ("after the dev-seed training began"). But:
  - D4 says the initialization was corrected "before any run", and no committed `learned.py` precedes training. `learned.py` is unchanged from `fad7f89` to HEAD.
  - "No mechanism metric had been computed" cannot be verified from git.
  - The final form of `I_agree` was fixed while dev checkpoints existed. That is allowed by §8 (dev development), but it means `I_agree` is a dev-tuned definition, not a preregistered one.

**Correction.**
- List the `sign(ab)` addition and the non-backfit base as deviations D5 and D6.
- Say that the init correction precedes the D-commit but is evidenced only by the mtime of `learned.py`.

### F9. MINOR: unit-splitting invariance holds, but only trivially

**Location.** REPARAM §1: "invariant under permutation and splitting".

**Evidence.**
- For GELU, an exact split must duplicate the pre-activation, so the copies have identical `I_agree` and `D_sum`. "Only one copy passes" therefore cannot happen under an exact split. Set-level invariance is automatic.
- `reparam.py` tests only `u = 0` with `t = 1/2`. For T4/TF1, unit 0 is not a relation unit, so the test there is vacuous.
- Under ε-near-splits (perturbed duplicates), the near-threshold mass in F3 (up to 30 units within 0.05 of τ) can make copies disagree.

**Correction.** Say "invariant under exact duplication splits", and report sensitivity to near-gauges and to τ.

### F10. MINOR: the K = 3 semantics are observation-level, and some recorded variables are not recorded

**Location.** SOURCE §1 and §2; `semantics.sample_k`.

**Evidence.**
- D1 for K = 3 is `argmax z_A = argmax z_B`, but the generator's source claims are `Y_A`, `Y_B`.
- `c7`: `P(argmax z_s = Y_s)` = 0.725, and observed agreement differs from latent agreement `1[Y_A = Y_B]` on **34%** of draws.
- `μ_s` is listed as a recorded generator variable, but `sample_k` discards `μ` and `Y_s`.
- The "candidate-relative margin agreement, accuracy 1.000" is D1 rewritten (positive iff `argmax z_B = argmax z_A`). It is a tautology, not evidence for an alternative definition.

**Correction.**
- Name D1 for K = 3 as observation-level.
- Return `Y_s` and `μ_s` from `sample_k`.
- Label the 1.000 row "by definition".

### F11. MINOR: MLP §5 "architectural stability" partitions failure modes in a gauge-dependent way

**Location.** MLP_NORMALIZATION_THEOREMS §5, and §4 (N1 vs N3).

**Evidence.**
- Folding a non-scalar LN2 gain into `W1` (`g2 → 1`, `W1 → diag(g2) W1`) is function-preserving. It turns an (N1) violation into an (N3) cross-talk violation.
- So "gain failure" vs "cross-talk failure" is not an intrinsic distinction. Only the composite read directions in centred, normalized space are.
- "Exactly invariant to every gauge that preserves (N1)–(N4)" is tautological.
- No learned checkpoint was tested against (N1)–(N4). LEARNED §6 concedes this for (N3) pairing, so §5's stability does not transfer to the learned models.

**Correction.** State the conditions on gain-folded, centred read directions, and add one sentence saying that the learned models were not checked.

---

## Verified claims (reproduced or confirmed)

**V1. Source semantics are model-free.**
- Every label in `semantics.py` and in the probe and unit code is a function of generator variables.
- Unit selection (`I_agree`, `D_sum`) reads activations on the grid only; no margin enters selection.
- The readout enters only step 6, as disclosed. (The T1 label = target caveat is in F4.)

**V2. The K = 3 D2 vs D1 table** reproduces exactly (`c7`):
- `P(dot < 0 | agree)` = 0.080
- `P(dot > 0 | disagree)` = 0.436
- accuracy of `sign(dot)` = 0.713; of the centred dot = 0.819

**V3. LEARNED claim 4** (`ab` decodable in T2 because it is linear in `a²`, `b²`, `(a+b)²`) is supported:
- `(a+b)²` R² is 0.999 at the final site;
- `ab` reconstructed from the three decoded features matches or beats the direct `ab` probe (`c3`).
- TF2-AO s0 is weaker (0.74 vs 0.70).

**V4. Gauges.**
- The QK, VO, head-permutation, unit-permutation and split gauges in `reparam.py` are function-preserving, with logit changes ≤ 8.5e−5, matching `logs/reparam.txt`.
- The new residual rotation (`c4`) is also exact, and leaves every residual-site output unchanged.

**V5. Positive rescaling of GELU units** is correctly excluded, as it is not function-preserving. LN-gain folding (which does preserve the function) leaves unit activations unchanged, so it does not affect unit statistics.

**V6. The binary `D_sum` bin-means are exact conditional means on the grid.** The 121 bins of width 0.0992 hold at most one grid sum each, at spacing 0.1. So `D_sum` is a correct sum-sufficiency deficit; the T2 values (≈ 1e−6) and the T1 value (0.437) are confirmed.

**V7. The T5a "no relation" result is robust** to the F5 fixes:
- maximum unit gain ≤ 0.003 under every feature set;
- `D_sum^K` ≈ 2e−4.

**V8. The T5b "0 relation units at τ = 0.2" result survives** the added conditional features (maximum gain 0.155), though not with much margin.
