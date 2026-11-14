# Causal abstraction and mathematics (Investigation XII, Workstream C)

Scripts:
- `scripts/math_xii.py` → `logs/math_xii.json` (deterministic; M1–M5);
- `src/xii_bayes.py` (exact shared-copy posterior);
- neural tests: `scripts/vs_linear.py`, `scripts/disc_ident.py`.

**Revision note.** This document was revised after an independent review by a separately initialized agent
(`reviews/INDEPENDENT_MATH_REVIEW.md`). The review found an error in M4/M5: the A1 decoy was evaluated on the wrong
pair distribution. It also found an incorrect quantifier in §4 and an imprecise §1.5. All are corrected below, and the
corrections are listed in §5.

Status labels:
- **[proved]**: a mathematical argument, with numerical confirmation to ≤ 1e-14.
- **[dev]**: observed on development models.
- **[conf]**: confirmed on the fresh cohort; see `FRESH_CONFIRMATION_AND_STATISTICS.md`.

## 1. Re-audit of the record

### 1.1 The XI check count: 67/72 is wrong; the correct count is 73/78

Re-derived from XI's own scorer output (`step1_native_causal_construction/logs/score_conf.json`):
- The arms are 19 parent-level arms × 4 parents, plus the 2-parent C1→full arm.
- That gives 76 + 2 = **78 checks**.
- **73 passed and 5 failed.** The five failures are the ones XI listed.
- The "67/72" in XI's documents was an arithmetic error. The six verdicts are unaffected.

### 1.2 XI's "held-out R²" of the cross-model maps was in-sample

`cross_interchange.py` in XI fitted the ridge map on all 8000 natural states, then reported R² on states 6000–7999.
Those states were part of the fit. The XII version:
- fits on states 0–5999;
- reports in-sample R², R² on the held-out tail, and R² on the separate evaluation stream.

The transfer gains in XI were evaluated on a separate stream (63_500_000) and are not affected. Only the R² label was
wrong.

### 1.3 XI's "true C = 2 Bayes" was the posterior of a different generator

The genlib sampler gives **one shared copied value per question to all colluders**. genlib's `posterior_brute`
multiplies independent per-group colluder factors, so it is the posterior of an *independent-copying* model. The two
coincide for C = 1 and differ for C ≥ 2:

| (G, C) | KL(exact ‖ genlib posterior) | exact Bayes accuracy | genlib-posterior accuracy |
|---|---|---|---|
| (3, 2) | 0.049 | 0.794 | 0.747 |
| (4, 2) | 0.050 | 0.778 | 0.729 |
| (5, 3) | 0.076 | 0.768 | 0.707 |

XI's P5 conclusion (the model applies the C = 1 rule and falls to chance) is unchanged. The model is now *further*
from Bayes (0.33 vs 0.80, not 0.75). The phrase "true C = 2 Bayes" in XI must be read as "the independent-copying
posterior". [proved; `math_xii.py` M2, `src/xii_bayes.py`]

### 1.4 XI §4 ("VS is not a legitimate intervention") is partially retracted

XI argued that VS is illegitimate because it produces S ≠ Σe. Two problems:

1. **In structural-causal-model semantics, do(S := s) is legitimate by definition.** It replaces S's structural
   equation, and the resulting S ≠ Σe is the *point* of the intervention, not a defect.
2. **The same objection applies to VE, which XI accepted.** On the T4 support, and for α ≠ 1/K, e_g is a
   deterministic function of (W_g, L_g); U_g = W_g L_g alone also determines it. W_g determines n_g, because the per-question agreement factors differ; L_g determines agreement at q.
   Check: 36 distinct (W_g, L_g) values, **0 collisions** (M3). So every *informative* E-interchange (e_src ≠ e_base,
   with W and L from the base) produces a high-level state that **no input realizes**. This is measured: 0.000 of
   informative pairs and 1.000 of non-informative pairs are realizable. That is 71.7% of random pairs,
   and 100% of the pairs DAS is evaluated on.

**Corrected statement.**
- VS and VE are both valid do-interventions in high-level models that contain S, respectively e, as nodes.
- **Neither is an input-realizable counterfactual.**
- What XI correctly established stands: in an I2-type implementation (no S node), VS needs a non-projective map, so
  failure of a projection search for S cannot falsify a pooled-S algorithm.

The new taxonomy is in §4.

### 1.5 Proposition A2 (XI): product support

The (⇒) direction of A2 needs the pair support to contain enough (V_s, R_b) combinations. On the T4 support, V = e is
a function of R = (W, L) (§1.4), so the support of (V, R) is **not** a product set. A2's "iff" does not apply to the T4
E-interchange in its literal form.

Corrected statement (after review):
- **(⇐) is unconditional.**
- **On a functional support**, projection-realizability of the freely extended intervention is equivalent to
  Pφ(x) depending on x only through V on *natural* states. So the interchange **does** constrain the natural
  representation. The first XII draft claimed the opposite; that claim is withdrawn.
- No XI result depended on A2's (⇒) direction on T4.
- A precise proof of the functional-support version is not written here, so it is labelled **[stated, not proved]**.

## 2. New results

### Theorem XII-1 (general-C eligibility, independent copying) [proved; elementary]

Setting: G groups, C colluders, β = 1, independent copying, uniform prior over roles, x in the support. Let e_g = 1[n_g = Q − 1 and the pair agrees at q], and let
W_g and L_g be as in Investigation IX. Then

p(y | x) ∝ Σ_g W_g L_g(y) · e_C(e_{−g}),

where e_C is the elementary symmetric polynomial of degree C in the other groups' indicators.

*Proof.*
- A colluder group's pair has probability e_q'/K at each question, so the product over questions is K^{−Q} times the
  indicator that the group agrees at all Q questions. That indicator is exactly e_h.
- Noise groups contribute a role-independent constant (G − 1 − C of them in every role).
- The reliable group contributes W_g L_g, up to the same normalisation as in Theorem 1.
- Summing over colluder sets H gives Σ_{|H| = C} Π_{h∈H} e_h = e_C(e_{−g}). ∎

Numerical check: max |Δp| ≤ 4e-15 for (G, C) ∈ {(3,1), (3,2), (4,1), (4,2), (5,2), (5,3)} (M1).

Consequences:
- The trained C = 1 readout computes the degree-1 aggregation Σ_{h≠g} e_h = S − e_g.
- The correct independent-copying C = 2 rule needs the degree-2 aggregation e_2(e_{−g}). For G = 3 this equals
  Π_{h≠g} e_h, an AND across groups rather than a sum; equality holds only for G = C + 1.
- For the *independent-copying* model, the XI C = 2 failure is an aggregation-degree error. For the real
  (shared-copy) generator, the correct rule uses J_H (XII-2), not e_C.
- That the network's variable is "correct" under C = 2 is an empirical claim: the e read-out is 0.94–1.00. The
  theorem does not establish it.

### Theorem XII-2 (shared copying) [proved]

For the actual sampler, where all colluders copy one shared value, replace Π_{h∈H} e_h by the **joint** indicator J_H.
J_H = 1 iff all 2|H| sources of H agree with each other at every question. Max |Δ| ≤ 2e-15 (M2).

J_H is a relation **between groups**, which the per-group statistic e cannot express. The e-based rule e_C loses
KL 0.047–0.076 and 4.4–6.1 accuracy points.

Consequence for the network (P8): a readout that is correct for the real generator must use cross-group
information beyond e. It *may* therefore be able to succeed without the E-subspace. This is a possibility, not a
consequence; P8 tests it empirically.

For β < 1 and C ≥ 2, genlib's `posterior_brute` is grossly wrong: at G = 3, C = 2, β = 0.6 it gives accuracy 0.50
against 0.79 exact (review item b). The sampler shares both the copied value and the copy *event* across colluder
groups.

### Proposition XII-3 (false-positive bound) [proved; elementary]

Take two high-level variables V, V′ whose counterfactual decisions agree on a fraction 1 − δ of a pair set, and any
intervention subspace R, with both IIAs computed on **that same pair set**. Then

|IIA_V(R) − IIA_V′(R)| ≤ δ.

The pair distributions are the ones XI's `das_decoy.py` actually used: for A1 the base is stratified by the A1 count,
and IIA is evaluated on that rule's own "value differs" set.

| decoy V′ | δ on the rule's IIA set | δ on all pairs |
|---|---|---|
| A1 (perfect-only) | **0.040** | 0.047 |
| PERM | 0.28 | 0.31 |
| RANDE | 0.23 | 0.29 |

*Correction after review.* The first XII draft reported δ(E, A1) = 0.0035. It reused the E-stratified selection for
A1, which is wrong.

Interpretation. On XI's distribution, interchange accuracy can separate E from A1 by at most about 4 points. The
high IIA of XI's A1-trained subspace (0.83–0.93) is therefore not informative about which variable the network
uses. This is a property of the evaluation distribution.

### Proposition XII-4 (preservation power) [proved; elementary]

A preservation test for decoy V′ uses pairs on which V′ is unchanged. It can expose V′ only where the true variable
changes. On XI's A1 preservation pairs (n = 957):
- **E changes on 32%** of pairs;
- the Bayes decision changes on 19%;
- an *ideal* E mediator would show preservation KL ≈ **0.22**. The observed A1-trained subspaces showed 0.16–0.21.

**Correction after review.** The first XII draft said E changes on only 3.9% of pairs, so XI's preservation test was
non-discriminating. That conclusion is **withdrawn**. On the correct pairs, XI's preservation test *could* expose
A1, and the A1-trained subspace's preservation failure is of the size expected if it carries E-like information.

What survives:
- Interchange accuracy alone is weak (XII-3).
- Preservation discriminates **only to the extent** that the preservation set contains pairs on which the true
  variable changes. This has to be checked for each decoy. It was 100% for PERM and 88% for RANDE.

### Discriminating-pair design

This is the task-specific identification tool. Restrict the pairs to those on which cf_E and cf_A1 imply different
decisions (`disc_ident.py`; frozen prediction P6).

[dev] On those pairs, repaired models' E-subspace interchange gives:
- IIA_E 0.79–0.85 on informative pairs (base 0.37–0.50);
- A1 agreement only +0.02–0.04 over base.

C1 models' "E" subspace moves the decision **toward A1** (+0.26–0.39 over base). The C1 subspace mediates
(perfect-only) eligibility, so calling it "E", as Investigations X and XI did, overstated what it carries.

### Proposition XII-5 (linear vs projection, neural test of X1′) [dev; qualified after review]

- [proved, XI] In an I2-type representation, VS is realized by M = ½11ᵀ (eigenvalue 3/2) and by no projection.
- [dev] Inside the network's 6-dim E-subspace (xP81), the best projection (k = 1, 2, 3, or the full E swap) reaches
  VS KL ≥ 0.83. An unconstrained learned linear map reaches **KL 0.40**, with eigenvalues {0.005, 0.005, 0.55, 0.55,
  1.54, 3.83}. These include values > 1, so the map is non-projective.
- The network realizes *part* of VS through a non-projective linear map. The learned spectrum does **not** match
  ½11ᵀ, so "exactly the theorem's class" would overstate it.
- Control: for VE, where a projection (the full E swap) already works (KL 0.015), the free linear map gains nothing
  (0.013). So the linear class does not simply fit any target better.
- The linear-map control on a *random* 6-dim subspace **was** run (development, commit 5425317; fresh, exploratory).
  It **also** beats every projection (KL 0.69–0.85 vs ~1.25–1.30). The "linear beats projection" criterion is
  therefore **not diagnostic**. Only the E-vs-random gap (0.24–0.47 vs 0.69–0.85) is E-specific.
- DAS-patched states are themselves class III (synthetic).
- It is not complete: IIA 0.62.
- Frozen as P6c.

## 3. When do the diagnostics identify a mediator?

A candidate subspace R for variable V is supported over an alternative V′ only if the following hold. These are
necessary for a **sole-mediator** claim within behavioural intervention evidence; they are not sufficient.

1. **Discriminating support.** The evaluation pairs include a non-negligible fraction δ on which V and V′ imply
   different decisions (XII-3).
2. **Preservation on the discriminating complement.** There are pairs with V′ equal and V different (XII-4).
3. **Natural-activation necessity.** Removing the upstream computation by input-realizable or natural-activation
   interventions (P2's designs) changes the decision as predicted.
4. **Restoration** through R alone, with a random-subspace control.
5. **Transfer relative to a source-free mean patch.** Plain IIA_disc reaches 0.56–0.95 with no source information.

Conditions 3 and 4 are **not** necessary when the variable is encoded redundantly. With two independent copies of V,
removing either copy leaves the decision unchanged (3 fails), and restoring one copy after removing both may suffice
or not depending on the readout (4 is uninformative).

Within layer 1 we found exactly such redundancy (NATIVE §2.3). At the level of the E-subspace, conditions 3–4 held in
development.

Even all five conditions do not establish *uniqueness*: any bijective re-encoding of V passes identically.
## 4. Admissible interventions: a taxonomy

| class | example in this program | realizable by inputs? | what failure can falsify |
|---|---|---|---|
| I. input edits on the support | factorial cells x00/xA/xP/xAP; D1/D2/F45 edits | yes | the input–output function (behavioural rule) |
| II. natural-activation path interventions | path_inA / path_inP, resampling from a natural run, scrubbing with matched natural donors | each component state is natural; the *combination* is not | that a specific component carries the effect, under the stated path |
| III. synthetic internal states | additive completion, mean ablation, interpolation | no | only relative to the assumption that the network behaves smoothly off its natural manifold |
| IV. high-level do() on deterministic intermediates | VE (e := e_s), VS (S := s) | no, for informative pairs (§1.4) | an abstraction *with that node and that alignment class* (projection vs linear) |
| V. artificial targets | decoy rules A1, PERM, RANDE | n/a | nothing about the model; they calibrate the diagnostics |

**Rule (corrected after review).** Failure under a class-IV intervention falsifies a hypothesised *class* of
implementations only if:
- (a) the alignment class realizes that intervention in **every** implementation in the hypothesised class. For
  example, projections realize VS in I1 = (S, A, B) but not in I2 = (m, U) (XI Theorem X1′). So projection-VS failure
  falsifies "I1-like" but not "pooled-S algorithms in general"; and
- (b) the network's downstream computation is expected to extrapolate to the off-image states.

(b) is an empirical assumption that the XII results bear on directly. VS via a linear map is realized only partially
(IIA 0.62), so the network's readout extrapolates imperfectly.

## 5. Corrections made after the independent review

The review is at `reviews/INDEPENDENT_MATH_REVIEW.md`. It was written by a separately initialized agent with no shared
working context, so it is independent of this author's reasoning. It is not external human review.

| # | review item | action |
|---|---|---|
| 1 | M4/M5 computed on the wrong pair distribution for A1 | recomputed. δ = 0.040 (not 0.0035); E changes on 32% (not 3.9%); the non-discrimination conclusion is withdrawn |
| 2 | §4 quantifier "at least one implementation" | changed to "every implementation in the hypothesised class"; the incorrect parenthetical removed |
| 3 | §1.5 restricted statement | replaced; the interchange does constrain natural states; labelled [stated, not proved] |
| 4 | §3 "each necessary", "not separable at all" | qualified to sole-mediator claims, with the redundancy counterexample; the inseparability claim removed |
| 5 | XII-1 consequences, statement conditions | e_C used in general; uniform prior and support conditions added; "precisely … correct variable" removed |
| 6 | β < 1 posterior_brute error; P8 "therefore" | reported; reworded as a possibility |
| 7 | α ≠ 1/K; hard-coded M3 entry | condition added; M3 now measured (0.000 informative, 1.000 non-informative realizable) |
| 8 | XII-5 overclaim, controls | qualified; VE linear-map control reported; random-subspace control not run |
| 9 | novelty | XII-1 to XII-4 are labelled elementary or task-specific. The general phenomena are standard in the causal-abstraction literature: non-uniqueness of abstractions, dependence on the intervention set, off-manifold interchange states, alignment-class richness, and redundancy. The task-specific contributions are the shared-copy correction, the measured δ and preservation power, and the discriminating-pair design |

**The P6 rationale in the frozen predictions quoted δ = 0.35%.** That background sentence was wrong for the reason in
item 1. The frozen arms and thresholds of P6 do not depend on it and are not changed. The error is recorded in
`FRESH_CONFIRMATION_AND_STATISTICS.md` Part II.
