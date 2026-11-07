# 01 — Adjudication of Step 1A

Step 1A verdict `PHASE15_STEP1_ONLY_GENERIC_BOUNDS_SURVIVE` stands inside its stated scope: the eight families built around `O`, `D`, linear `ρ`, and a scalar residual. This file records what that scope does and does not cover, and three corrections required by the Step 1B brief.

## 1. Counterexamples that disprove universal claims

CE-01 through CE-04, CE-07, CE-08, CE-16, and Theorem Sign disprove unrestricted implications from `O` or `D` to decisions, rankings, or repair. Those claims stay `DISPROVED`. They are not limited to two dimensions: Theorem Blind in this step extends the same information obstruction to any statistic that is invariant along a direction that `L` uses.

## 2. Counterexamples that only defeat a restricted assumption

| Example | What it actually kills | What it does not kill |
|---|---|---|
| CE-05, Theorem Z | Ranking decisions with an exact intertwiner residual | Projected residuals, joint constraints, stitched states |
| CE-06 | Reading a least-squares leftover as a candidate conflict | A later claim that uses a different object than `ρ s_u - s_v` |
| CE-13 | Commutation as use | Causal sufficiency of a different representation |
| CE-19 vs CE-20 | Equating a stitch with a recomputed `do` | Well-defined simultaneous clamps under truncation |
| Family F printed proposition | Pointwise Jacobian as a Lipschitz bound | Ordinary margin-versus-increment stability |
| Family H as first written | Treating `I` as a Hessian, and treating nested `do` as undefined in every SCM | Simultaneous ancestor-descendant interventions once truncation is specified |

## 3. Redundant examples

CE-02 and CE-03 are the two signs of one linear fact. CE-08 is the rank-zero case of the Blind theorem. They remain in the catalog because they are easy to check. They do not add independent mechanisms.

## 4. Unrealistic but legitimate systems

CE-18 is a lookup alignment. It is legitimate as an attack on an unrestricted map class (Sutter et al., 2025). It is not a Transformer. Shared linear `ρ` is outside that class and still fails to sign `R_c`.

## 5. Relevance to Transformer computation

Theorem Z on a doubling map is relevant as a warning, not as a block equation. Step 1A already said a pre-norm block is not a constant linear intertwiner. Families E, G, and H were the ones most relevant to residual addition, attention, and composition, and they were developed as finite maps. They were not developed as simplified Transformer blocks with LayerNorm, softmax, and an MLP. That is a coverage gap, not a hidden positive result.

## 6. Families treated too thinly, relative to this brief

Step 1A treated coordination, error correction, competing evidence, information bottlenecks, and repair-without-an-obstruction-objective only as remarks inside Families C, G, and H. Those are the families this step develops. Causal sufficiency was present as factorization through `ker d`, which is one special case and the one that idles `D`.

## 7. Assumptions rejected without checking architecture

The energy-descent family was rejected because a block is a map, not an iteration. That rejection is still right for Transformers as usually implemented. It was not an exhaustive check of every possible implicit energy. No such energy is identified here either.

The claim that nested interventions are undefined was too strong. An SCM with truncation defines them. Step 1A's real point, which survives, is that `I` is then still not `D`.

## 8. Analogy versus derivation

Calling a residual a "conflict" or a "germ" was analogy. The derived statements were the identities for `O`, `D`, Lipschitz bounds, and `∇L`. This step keeps that distinction.

## 9. "Impossible" versus "not identified" versus "no proof"

Step 1A's impossibility result, after review, is: no function of `(O,D)` equals the recomputed sign for every final head, and exact intertwiner residuals vanish on recomputed traces. That is not "no distributed relationship can constrain a response." It is "this score, in this class, cannot." File 02 of Step 1A said the latter after the reviews. The verdict label `ONLY_GENERIC_BOUNDS_SURVIVE` was easy to misread as the former. This step does not read it that way.

## Mandatory corrections

### A. Affine offset

A Jacobian is a derivative. For `f(h)=h^2` at `h=1`, `J=2` and `Jh=2 ≠ f(h)=1`. Absolute-state linear transport through the origin is a different operator from the affine map `f(h_0)+J δ`. PDF equation (11) is a derivative transport of projected states. Treating it as an equality of activations is a scope error. Test: `test_affine_offset_is_not_the_jacobian_times_the_state`.

### B. Factorization for output bounds

Theorem Lip bounds `|L(s)-L(Π s)|` only if `L` is a function of the cochain `s`. If the logit reads a coordinate that `P` dropped, a small distance in `z=Ph` does not bound `|ΔL|`. Test: `test_lipschitz_bound_requires_factorization`. Step 1A stated the Lipschitz hypothesis on the cochain space; it did not always repeat that the Transformer logit has to factor through that cochain. This step treats factorization as an extra assumption, not as a default.

### C. Simultaneous interventions

A structural causal model can clamp an ancestor and a descendant together. The rule used here is truncation: a clamped node is not recomputed; its descendants are recomputed from the clamp. Under that rule, `do(parent=p, child=c)` is defined, and the four-term interaction is defined. Nested interventions are therefore not rejected. What remains false is that this interaction equals a cochain residual of the unintervened state. Tests: `test_truncation_defines_nested_interventions`, `test_interaction_is_not_a_function_of_the_unintervened_residual`.
