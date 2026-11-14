# Causal abstraction and mathematics (Investigation XII, Workstream C)

Scripts:
- `scripts/math_xii.py` → `logs/math_xii.json` (deterministic; M1–M5);
- `src/xii_bayes.py` (exact shared-copy posterior);
- neural tests: `scripts/vs_linear.py`, `scripts/disc_ident.py`.

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
2. **The same objection applies to VE, which XI accepted.** On the T4 support, e_g is a deterministic function of
   (W_g, L_g). W_g determines n_g, because the per-question agreement factors differ; L_g determines agreement at q.
   Check: 36 distinct (W_g, L_g) values, **0 collisions** (M3). So every *informative* E-interchange (e_src ≠ e_base,
   with W and L from the base) produces a high-level state that **no input realizes**. That is 71.7% of random pairs,
   and 100% of the pairs DAS is evaluated on.

**Corrected statement.**
- VS and VE are both valid do-interventions in high-level models that contain S, respectively e, as nodes.
- **Neither is an input-realizable counterfactual.**
- What XI correctly established stands: in an I2-type implementation (no S node), VS needs a non-projective map, so
  failure of a projection search for S cannot falsify a pooled-S algorithm.

The new taxonomy is in §4.

### 1.5 Proposition A2 (XI): product support

The (⇒) direction of A2 needs the pair support to contain enough (V_s, R_b) combinations. On the T4 support, V = e
is a function of R = (W, L) (§1.4), so the support of (V, R) is **not** a product set. A2's "iff" therefore does not
apply to the T4 E-interchange in its literal form. What survives:

- **(⇐) is unconditional.** A direct-sum decomposition a(V) + b(R) aligned with P always realizes the intervention,
  including on off-image pairs.
- **On a non-product support**, realizability by projection is equivalent to the decomposition *restricted to the
  pairs actually used*. The interchange then constrains the representation only on states never produced naturally.
  This is exactly why interchange results are weak evidence about natural computation (§3).

[proved; the restriction is new in XII]

## 2. New results

### Theorem XII-1 (general-C eligibility, independent copying) [proved]

Setting: G groups, C colluders, β = 1, independent copying. Let e_g = 1[n_g = Q − 1 and the pair agrees at q], and let
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
- The correct C = 2 rule needs the degree-2 aggregation Π_{h≠g} e_h: an AND across groups rather than a sum.
- The XI C = 2 failure is therefore precisely an **aggregation-degree** error with a correct variable.

### Theorem XII-2 (shared copying) [proved]

For the actual sampler, where all colluders copy one shared value, replace Π_{h∈H} e_h by the **joint** indicator J_H.
J_H = 1 iff all 2|H| sources of H agree with each other at every question. Max |Δ| ≤ 2e-15 (M2).

J_H is a relation **between groups**, which the per-group statistic e cannot express. The e-based rule e_C loses
KL 0.047–0.076 and 4.4–6.1 accuracy points.

Prediction for the network (P8): a readout that is correct for the real generator must use cross-group information
beyond e. It can therefore succeed **without** the E-subspace.

### Proposition XII-3 (false-positive bound) [proved; elementary]

For any intervention subspace R and two high-level variables V, V′ whose counterfactual decisions agree on a fraction
1 − δ of the evaluation pairs:

|IIA_V(R) − IIA_V′(R)| ≤ δ.

On XI's decoy-DAS pair distribution:

| decoy V′ | δ(E, V′) |
|---|---|
| A1 (perfect-only) | **0.0035** |
| PERM | 0.31 |
| RANDE | 0.29 |

So **no interchange-accuracy comparison on that distribution can separate E from A1 by more than 0.35 points.** XI's
"A1 false positive" (A1-trained subspace IIA 0.83–0.93) was a property of the *evaluation distribution* before it was a
property of the network.

### Proposition XII-4 (preservation power) [proved; elementary]

A preservation test for decoy V′ uses pairs on which V′ is unchanged. It can expose V′ only on the subset where the
true variable changes. On XI's distribution:
- **for A1, E changes on only 3.9%** of A1-preservation pairs, and the Bayes decision changes on only 2.5%;
- for PERM, E changes on 100% of pairs; for RANDE, on 88%.

Hence a *clean* E mediator, relabelled as an A1 mediator, would pass interchange (to within 0.35 points) and pass
preservation on 96% of pairs.

**Counterexample to "interchange + preservation identifies the variable".** On XI's distribution, a genuine E
mediator scored against A1 passes interchange (within 0.35 points) and preservation (on 96% of pairs), although A1 is
not the variable it carries.

XI's A1 decoy failed preservation (KL 0.16–0.21) because the *A1-trained subspace* carried other information, not
because the test discriminated A1 from E.

**What does discriminate them:** a pair distribution concentrated where the cf_E and cf_A1 decisions differ
(`disc_ident.py`; frozen prediction P6).

[dev]: on discriminating pairs, repaired models' E-subspace interchange gives:
- IIA_E 0.79–0.85 on informative pairs (base 0.37–0.50);
- A1 agreement only +0.02–0.04 over base.

C1 models' "E" subspace moves the decision **toward A1** (+0.26–0.39 over base). The C1 subspace mediates perfect-only
eligibility. Calling it "E" (as Investigations X/XI did) is a false positive that XI's diagnostics could not detect.

### Proposition XII-5 (linear vs projection, neural test of X1′)

- [proved, XI] In an I2-type representation, VS is realized by M = ½11ᵀ (eigenvalue 3/2) and by no projection.
- [dev] Inside the network's 6-dim E-subspace (xP81), the best projection (k = 1, 2, 3, or the full E swap) reaches
  VS KL ≥ 0.83. An unconstrained learned linear map reaches **KL 0.40**, with eigenvalues {0.005, 0.005, 0.55, 0.55,
  1.54, 3.83}. These include values > 1, so the map is non-projective.
- The network therefore realizes part of the "inadmissible" VS intervention through exactly the non-projective
  linear class the theorem identifies. This is a **false negative of projection-DAS** in a real network.
- It is not complete: IIA 0.62.
- Frozen as P6c.

## 3. When do the diagnostics identify a mediator?

Combining §1.5 with XII-3 and XII-4, a candidate subspace R for variable V is supported over an alternative V′ only if:

1. **Discriminating support.** The evaluation pairs include a non-negligible fraction δ of pairs on which V and V′
   imply different decisions. Otherwise interchange cannot separate them (XII-3).
2. **Preservation on the discriminating complement.** Pairs with V′ equal and V different must be present (XII-4).
3. **Natural-activation necessity.** Removing the upstream computation with *input-realizable* or
   natural-activation interventions changes the decision as predicted. These are the path-patching, resampling and
   scrubbing designs of P2; additive completion alone is not enough.
4. **Restoration** through R alone, with a random-subspace control.
5. **Transfer relative to a source-free mean patch.** Plain IIA can be inflated by moving every base toward a typical
   state; IIA_disc reaches 0.56–0.95 with no source information.

None of these is sufficient alone. Even all five do not establish *uniqueness*:
- any bijective re-encoding of V passes identically;
- variables that coincide on the support (A1 vs E on XI's distribution) are not separable at all.

## 4. Admissible interventions: a taxonomy

| class | example in this program | realizable by inputs? | what failure can falsify |
|---|---|---|---|
| I. input edits on the support | factorial cells x00/xA/xP/xAP; D1/D2/F45 edits | yes | the input–output function (behavioural rule) |
| II. natural-activation path interventions | path_inA / path_inP, resampling from a natural run, scrubbing with matched natural donors | each component state is natural; the *combination* is not | that a specific component carries the effect, under the stated path |
| III. synthetic internal states | additive completion, mean ablation, interpolation | no | only relative to the assumption that the network behaves smoothly off its natural manifold |
| IV. high-level do() on deterministic intermediates | VE (e := e_s), VS (S := s) | no, for informative pairs (§1.4) | an abstraction *with that node and that alignment class* (projection vs linear) |
| V. artificial targets | decoy rules A1, PERM, RANDE | n/a | nothing about the model; they calibrate the diagnostics |

**Rule.** Failure under a class-IV intervention falsifies a proposed abstraction only if:
- (a) the alignment class is rich enough to realize that intervention in at least one exact implementation of the
  function (XII-5: projections are not, for VS); and
- (b) the network's downstream computation is expected to extrapolate to the off-image states.

(b) is an empirical assumption that the XII results bear on directly. VS via a linear map is realized only partially
(IIA 0.62), so the network's readout extrapolates imperfectly.
