# Review B: mathematics and counterexamples

Reviewer B (impossibility and counterexamples). Files read: `CAUSAL_BRIDGE_MAIN_ARGUMENT.md`, `proofs/PROOFS.md`, `proofs/bridge_math.py`, and `tests/test_bridge.py`. The package suite passes (`22 passed`). My scripts are `/tmp/revB/cx.py` and `/tmp/revB/cx2.py`. They import `bridge_math` unchanged, and every number below comes from them.

No FATAL findings. Four SERIOUS findings concern the step from the theorems to the main argument's claims, mainly the one forward-looking claim (§8, the "consensus locus"). One SERIOUS finding is a false clause inside a theorem (NC.3).

---

## SERIOUS

### S1. The claim of independence from the decision metric is false. RB.2 covers transforms of one scalar, not a change of which scalar is used.
**Location.** Main §3 ("invariant to every strictly monotone transformation of the decision variable (logit, log-probability, probability, margin)"); §5 R4 ("invariant to the choice of decision metric"); §5 R6 ("independently of metric and link").
**Argument.** RB.2 is correct for `φ∘m`. But `L_c`, `log p_c`, and `p_c` are not monotone functions of `m = L_c − L_{c'}` once `|V| > 2`, because they depend on the other logits as well.
**Counterexample** (three candidates, 2×2). `L0 = [[0,1],[1,2]]`, `L1 = 0`, `L2 = [[−10,5],[−10,−10]]`.
- The margin `L0−L1` is additive: `is_relation_blind = True`.
- `p0 = [[0.5, 0.0179],[0.731, 0.8808]]` has a crossover: `is_relation_blind = False`.
- `log p0` is also not blind.

The verdict therefore depends on which scalar is chosen, and the choice of competitor `c'` matters too.
**Fix.** Say "invariant to strictly monotone transforms of a fixed decision scalar (equivalently, to the link)". Freeze the scalar (for example the margin against a named `c'`) as part of the preregistration. Drop "logit, log-probability, probability" from the list.

### S2. "sign(v'_j − M)" is the rule for adding a source (CI.3). A token substitution never adds a softmax source, and for substitutions the rule is false.
**Location.** Main §5 R2 ("For a single added source, its sign is sign(v'_j − M)"); §8 (the "distinctive prediction: … the sign of a supporting source's effect is sign(v'_j − M), not sign(v'_j)"); PROOFS Construction D proof ("M ≈ 1.935 > e_W = 0.5, so the sign is negative"); FIB.3.
**Argument.** CI.3 needs `w_j = 0 < w'_j`. In a softmax head every unmasked position has `w_j = exp(s_rj) > 0`. Substituting the token in a slot therefore changes `(w_j, v_j)` from one positive-weight pair to another. Only CI.2 applies, and its sign is that of `w'_j(v'_j − M) − w_j(v_j − M)`. The removal term can dominate.
**Counterexample.** Use `dilution_model` with one change: salience N = 1 and W = 0 (instead of 0 and 1). Everything else is identical, including B = S.
- `M = 3.212`.
- `v'_W = 1.0 < M`, so `sign(v'_j − M) = −1`.
- The head's direct change on `ũ` is +0.670 and the margin change is +0.734.

Supporting evidence that is below the consensus *raises* the margin, contradicting §8's stated sign.

In D itself the stated reasoning is incomplete. The correct criterion is `w'_W(M − e_W) > w_N·M`. Numerically that is `20.09·1.435 = 28.8 > 1.94`, which holds, so D's numbers stand. The test `test_D_crossover_locus_is_the_consensus` checks the CI.2 numerator, not `sign(v'_j − M)`.
**Also.** CI.1 holds for *any* `M'` (`Σw'_j(v'_j−M)/Z' ≡ M'−M`). It is a rewriting, not a law. It has "relational" content only when few `(w_j, v_j)` change. In layer ℓ ≥ 1, a substitution at position `p` changes every `(w_j, v_j)` with `j ≥ p` and the query, so the "relation to consensus" reading becomes empty.
**Fix.** State the prediction with the CI.2 numerator: `sign[w'_j(v'_j − M) − w_j(v_j − M)]`. Say explicitly that it reduces to `sign(v'_j − M)` only when `w_j ≪ w'_j` (a low-salience neutral). Restrict it to first-layer heads, or to designs where only index `j` moves.

### S3. CI signs the numerator `a`, not the margin. The norm term is a second channel that A4 does not close.
**Location.** Main §4 A4, §8 ("in a model where A4 holds … the sign … is sign(v'_j − M)"), §10 arrow 2 ("Established for the direct and final-layer terms"); PROOFS FIB.3.
**Argument.** By CI-5, `Δm = Δa/σ' + a(1/σ' − 1/σ)`. A head output that changes the final residual changes `σ` as well as `a`. In the last layer the head *is* the only path, so A4 ("sign of total = sign of direct effect through the head") holds trivially. Even so, CI's sign need not be the decision's sign.
**Counterexample.** Random 1-layer, 1-head models with the MLP off (`random_model(seed, H=1, V=3)`, `mlp_on=False`), with a token substituted at a non-read position. In 591 of 4000 substitutions (200 seeds × 20), `sign(Δ ũ^T c_h[r]) ≠ sign(Δm)`. Example: seed 0, `Δa = −0.0665`, `Δm = +0.0690`.
**Why D does not show this.** In `dilution_model`, `W_O` writes only along `e_out ∝ ũ`. Then `m(M) = 2(y+M)/√(‖x_⊥‖² + (y+M)² + dε)`, which is strictly increasing in `M` (by Cauchy–Schwarz). The locus test passes by construction, not as a generic property.
**Fix.** Add an assumption A4′: the norm term does not reverse the numerator term, `|Δa/σ'| > |a(1/σ'−1/σ)|` or the two have the same sign. Alternatively, state the prediction for `a` rather than `m`, or for heads whose OV image lies in `span(ũ)` plus `N_C`-neutral directions. Without that change, §10 arrow 2 is overstated for the decision.

### S4. "Writer alignment" in NC is screened by SS's own criterion. The relation it names is the norm of the sum.
**Location.** Main §2 (C) ("relations the architecture actually operates on"); §5 R1 ("A relation can matter only where a nonlinear operation reads two variables separately"); §5 R3; PROOFS NC "Relational reading".
**Argument.** The final residual is one additive cut. By SS(iii), any non-sum statistic of the writers has no effect once `Σc_k` is fixed. The cross terms `Σ_{k≠l} c̃_k^T c̃_l` act on `m` only through `‖P Σ c_k‖`. On the NC fiber, `σ` is the entire channel. So (C) is the same kind of object R1 calls inert: two configurations with different alignments and the same sum give the same decision.

The reverse also holds. LN reads the *single* variable `S`, yet the main argument presents this channel as relational. So R1's dichotomy and R3's reading cannot both stand.
**Fix.** Call NC a norm channel (as §8 already concedes it is known: Stolfo et al.). Drop (C) as a "relation". Alternatively, say explicitly that the alignment reading is a decomposition of `‖S̃‖²` with no causal content beyond the norm.

### S5. NC.3's "changes at most once along any path" is false as stated.
**Location.** PROOFS NC claim 3; main §5 R3 ("Each pair flips at most once").
**Counterexample.** Model seed 5, three candidates, `h` the final residual, and the path `v(t) = t·n_1` for `t: 0→50→0` (`n_1` a basis vector of `N_C`). Pairs (0,1), (0,2), and (1,2) all have `aκ < 0`, and each changes sign **twice** along this continuous path in `N_C`.
**What is true.** The order of each pair is a threshold function of `σ(h+v)`, with the single threshold `σ* = −a/κ`. The number of flips along a path equals the number of times `σ` crosses `σ*`. There is at most one flip along a path on which `σ` is monotone. The proof's last line ("a nonconstant affine function of τ has at most one zero") proves exactly this and nothing more.
**Fix.** Restate the claim as "single-crossing in `τ`" or as "at most once along any σ-monotone path".

---

## MINOR

**M1. RB.3/RB.4 with ties: the definition of crossover differs between the proof and the code.**
- *Location:* `has_crossover` in `bridge_math.py`; `test_RB_two_by_two_without_crossover_is_always_blind`.
- *Problem:* The RB.4 proof uses the weak notion of sign-constancy, with signs in {−1,0,1}. The code flags only strict reversals (`>` in one column and `<` in another). Consider `[[0,0],[0,1]]`: `has_crossover = False` and `is_relation_blind = False`. The same holds for `[[0,1],[1,1]]`, which is `max(a,b)`. So RB.4 fails as implemented. The test uses tie-free Gaussian tables and cannot detect this.
- *Consequence:* Hard saturation (max, clipping) is "relation-sensitive". §8's "monotone-saturation accounts predict RB" holds only for a *strictly* increasing `ψ`.
- *Fix:* Treat "0 in one column, ≠0 in another" as a crossover in the code. State strictness in §8.

**M2. `rb_forces` box [−100, 100].**
- *Correct for ≤3×3:* By the Hadamard bound (4 free coordinates, entries in {−1,0,1}), vertex coordinates are ≤ 16. PROOFS only says "well inside" and should give this bound.
- *Wrong for larger designs:* The function is generic and unguarded, and it gives wrong answers beyond 3×3.
  - A 101×3 design with an A-chain (forcing `f_k = k`) and `g0<g1<g2`: `rb_forces(…, (100,1), (0,2)) = True`. The true answer is "not forced": `g2 = 1000` is admissible.
  - A 102-level additive chain raises "training observations are themselves not additively representable", which is false.
- *Fix:* Bound by the Hadamard bound for `nA+nB−2`, use the homogeneous form, or assert `nA, nB ≤ 3`.

**M3. Invariance claim (main §3, "every one is invariant…").**
- *Problem:* `c_k[t]` is not invariant. Every reader is a mean-subtracting LN, so `W_O → W_O + x1^T` is function-preserving. Tested on a 2-layer model: the maximum logit change is 2.7e−15, while `c_k` changes by up to 22.4 and the cosine between heads moves from −0.55 to 0.92. With `γ` folded into the next weights, rotations of `1^⊥` are also symmetries.
- *Fix:* Only `P c_k` (up to rotation) and the scalars `w, v, M, a, σ, m` are invariant. Say "the scalars are invariant; the writes are equivariant up to `1`-shifts". This incidentally strengthens SS's point about raw cosines.

**M4. SS(iii) cosine range.**
- *Problem:* The range is `[−1, 1]`, not `(−1, 1]`. For example, `c_1 = 2s` and `c_2 = −s` give cos = −1.
- *Fix:* State the closed range, and require both writers to be nonzero.

**M5. Construction D arithmetic.**
- *Problem:* `M = 180/93 = 1.93548` is an approximation. The exact value is `2e^{4.5}/(3+e^{4.5}) = 1.93550`. The table reproduces exactly (−2.4681, 3.2162, −1.2184, 2.7083).
- *Fix:* Write `≈` or use the exact expression.

**M6. Construction LN is attributed to "NC.2".**
- *Problem:* The write `0.5e + 3n` is not in `N_C`, so both `a` and `σ` change. The mechanism is NC.1 (or CI-5), not the null-edit law NC.2. The numbers check: `m = 0, 0.921, 5.601, 4.277`.
- *Fix:* Cite NC.1.

**M7. §9 claims "CI verified with … arbitrary multi-token edits".**
- *Problem:* `test_CI_general_identity…` edits one residual row per trial.
- *Fix:* Either add multi-row edits or change the wording.

**M8. FIB.1 has no construction or test.**
- *Problem:* The intervention whose sign is meant is not specified.
- *Fix:* Give an explicit pair of conflict-fiber cells with opposite signs (D's OPP level would serve).

**M9. CORR quantifier.**
- *Problem:* "π moves the predicted candidate" is the wrong condition. Correctness flips iff `π(y*) ≠ ŷ` (when `ŷ = y*`) or `π(y*) = ŷ` (when wrong). For `|C| ≥ 3`, one global `π` cannot flip correctness on every input, so per-input relabeling is needed.
- *Effect:* The conclusion is unaffected.

**M10. CI.4 unanimity in the Transformer instance.**
- *Problem:* The "sources" are all `j ≤ r`, including BOS and the read token. In D these have `v = 0`, so the strict-positivity hypothesis of CI.4 never holds over evidence slots alone.
- *Fix:* Say "unanimous over all attended positions".

---

## OK (checked, no defect found)

- **SS (i)(ii).** Correct. The test's zero-sum edits are genuine (all positions, zero sum over heads).
- **NC.1, NC.2, NC.4, NC.6.** Correct. The extra inert direction `1` excluded from `N_C` does not change the reachable set.
- **NC.5, including boundaries.**
  - `a = 0` or `κ = 0`: no flip, correct.
  - `σ* = σ_min`: only the tie `m = 0` is reachable, so strict reversal is correctly excluded.
  - If the current sign equals `sign(a)`, then `σ(h) < σ*`, which implies `σ* > σ_min`, so it is always flippable. This is consistent with the criterion.
  - `dim N_C = 0`: the reachable set is `{σ(h)}`.
  - `σ_min ≥ √ε > 0`.
- **NC tests.** The σ-sweep family really spans `[σ_min, ~∞)`. Seed 12 does exercise the "disagree but unreachable" case.
- **CI.1, CI.2, CI.3 (as algebra), CI-5.** Correct.
- **RB.1 (LP).**
  - Correct. Ties are equalities, the gauge is `f_0 = g_0 = 0`, the box is harmless because `t` absorbs scale, and the 1e−9 threshold is fine at these sizes.
  - Equalities within `1e-12` are treated as ties. The 3×4 D table has a true symmetric tie: `(S,OPP) − (N,N) = 2.4e−14`.
- **RB.2, RB.3 (weak-sign form), RB.4 (proof, weak-sign form), RB.5 (table and double-cancellation instance), RB.6 (math).** Correct.
- **D.** The design is exactly realisable: six LN'd vectors in the 7-dimensional space `1^⊥`, with the least-squares solve asserted exact. The targets are fixed design constants, not responses. Salience, evidence, and λ are hand-chosen, which is legitimate for an existence claim. The crossover holds with the MLP off and with a random MLP.
- **D′.** `de < 0 < tot` reproduces.
- **MIX.** Correct and trivial.
- **"Global" and "ranking-level" (R3).** Accurate within `h + N_C` and the set `C`, given A5.

---

## Overall assessment

**Survive as stated:**
- SS (except the cosine endpoint, M4).
- NC.1, NC.2, NC.4, NC.5, NC.6.
- CI.1–CI.4 as weighted-mean algebra, and CI-5.
- RB.1–RB.6 as mathematics, with RB.3 and RB.4 read in the weak-sign sense the proof uses.
- Constructions D, D′, and LN as existence results.
- MIX; CORR (with the quantifier fixed).

**Fails as stated:** NC.3's "at most once along any path" (S5). The correct version is single-crossing in `σ`.

**Main-argument claims that the proofs do not support:**
- Independence from the decision metric (S1).
- The `sign(v'_j − M)` prediction for token substitutions (S2).
- That A4 alone lets CI sign the decision (S3).
- That writer alignment is a relation the architecture operates on (S4).
- That `c_k[t]` is invariant under symmetries (M3).

The "consensus locus" prediction in §8, which the package presents as its distinctive falsifiable claim, needs to be restated in two ways: with the CI.2 numerator in place of `sign(v'_j − M)`, and with an explicit assumption that the norm term does not reverse the result. Until then it is false in simple 1-layer models. RB.6 does not rely on CI and stands on its own, provided the decision scalar is frozen.
