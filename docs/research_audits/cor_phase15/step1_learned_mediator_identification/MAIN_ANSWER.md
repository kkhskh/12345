# Did we identify a computation in a learned Transformer that uses an independently defined source-consistency relationship to causally influence candidate selection?

**No.** This is the corrected answer, after the review round in `ADJUDICATION.md` and the deciding experiment REL-XOR3.

We did identify, causally and without output information in the selection step, the computations that drive these models' relation-sensitive decisions. In every case the identified computation encodes one of two things:

- **the target variable itself**, or
- **per-source content.**

In no model did it encode agreement as a computed intermediate that differs from the target. The rest of this document states exactly what was found, what failed, and why.

## 1. What was found

| task | identified computation at a primary cut (frozen protocol) | what it encodes |
|---|---|---|
| REL-XNOR (target = agreement) | **TF2:** a 1–2-dimensional subspace U at `resid_mid1@R`, with interchange 1.000, random subspaces ≤ 0.013 (mean), necessity 0–0.15, and content `≈ c·sign(ab)` (R² ≈ 1). **TF1:** source-additive at the pre-MLP cut in 5 of 6 models. | **The target.** On these pairs agreement and target coincide. U is the model's logit direction (cosine with the readout 0.79–0.97), and the site is readout-equivalent: deleting the final MLP leaves accuracy at 0.999–1.0. |
| REL-MIS | Interactive subspace (TF1 6/6; TF2 6/6) | A target-specific interaction. Its R² by `sign(ab)` is ≈ 0, so it is not agreement. |
| REL-GATE, K3-GATE | Source-additive subspace (0/24 relation-computing at R sites) | Per-source support. **No relation is computed**, consistent with the threshold-logic result (T-RF). |
| **REL-XOR3**, the deciding test: `y = sign(ab)·p`, so agreement ≠ target on decisive pairs | TF1 (the 3 models that learned the task) and TF2 (5/5): **TARGET_ONLY**. Patched decisions follow the source's target on 1.000 of decisive pairs, and agreement on 0.000. DAS cannot realise agreement (0.0–0.48). TF2-AO layer-0 output: follows the agreement hypothesis on 0.92–1.0, but its content is **source-additive** (R² ≈ 1). | **The target** (late sites), or **the two source signs, carried separately** (early AO sites). Agreement is never computed as an intermediate. |
| AGG-LIN, K3-BAYES, AGG-NL | No relation-sensitive behaviour on default matched pairs | Aggregate. Near the AGG-NL decision boundary the model's z-threshold shifts slightly with the spread, by up to 0.032. |

**Located computation (exact, Theorem T-ATT).** In one-layer models, every cross-source interaction at the readout position before the MLP is created by the attention softmax normalizers. Freezing the patterns removes it exactly (interaction share 0.000; normalizer-only freezing gives ≤ 1e−28).

**Reproducibility.**

- The 90 recovered checkpoints and all new checkpoints are committed.
- Retraining reproduces the historical parameters bitwise.
- An independent reimplementation reproduced every checked number.

## 2. What failed, and why

1. **In REL-XNOR and REL-MIS, the relation and the target are confounded on matched pairs** (review R_causal F1). An "output-free" relation contrast there is a target contrast. The positive-looking XNOR result is therefore a **learnability control**: a localised target variable, created by the attention normalizer.
2. **When agreement is an intermediate distinct from the target (REL-XOR3), the models do not compute it.** They compute the target interaction directly: by the pre-MLP cut in TF1 and by `resid_mid1` in TF2. Otherwise they carry the per-source signs and combine them later.
3. **Identifiability limit.** When the output depends only on `sign(a)·sign(b)·p`, an interchange that carries the two source signs cannot be distinguished, at the output, from one that carries their agreement. Only a content (additivity) test separates them. The frozen addendum rule did not include that test, and labelled such source-sign codes AGREEMENT_INTERMEDIATE in 3 of 5 TF2-AO models.
4. **Remaining caveats.**
   - Interventions are evaluated on patched states that are often off the natural manifold.
   - Mediation is not unique: there are redundant routes in TF2, and TF1 XNOR computes its interaction across the MLP and direct paths.
   - The REL-MIS content verdict for TF2 is threshold-sensitive.
   - With genuinely Gaussian matched pairs, one TF2 XNOR model fails necessity (reviewer result).

**Classification of the failure** (mandate §G):

- It is **not** caused by optimisation variability: every model learned its task, except two TF1 XOR3 models, which are excluded.
- It is **not** caused by an inadequate localisation method: the method finds the decisive subspaces, and they pass interchange, ablation and content tests.
- It is a **task-specific implementation** result. The learned computation encodes whatever variable the target factors through most directly, and agreement is never that variable unless it *is* the target.
- There is also an **identifiability limit**: agreement and per-source sign codes are inseparable at the output.

## 3. Does a consistency-based explanation add anything?

- **Mechanism.** No. The computations are ordinary: attention-normalizer interactions and per-source threshold logic. Matched-capacity MLPs, standard interaction models and causal-abstraction (DAS) descriptions account for them fully (`DISTINCTIVENESS.md`).
- **Restriction.** The task-level pattern matches threshold-logic representability (no relation computed for fixed-prior deferral rules). That is a property of the rule class, not of "consistency". It was not preregistered here.
- **Identification.** Aggregate-matched contrasts give a closed-form subspace equal to the DAS subspace (cosine 0.87–0.99). But when the relation coincides with the target, as in XNOR and MIS, this is a target contrast, and a plain difference of means without matching gives the same subspace (R_counterexample). It is no methodological advantage over standard tools.

## 4. Strongest valid theorem

See `DECISION_THEOREM.md`.

- **T-ATT (exact):** in one-layer models, all pre-MLP cross-source interaction at the readout position comes from the attention normalizers.
- **T-MED (conditional, corrected):**
  - It requires a missing hypothesis: the additive part of U must not by itself separate the decisions within a pair.
  - Only some models satisfy it.
  - On the support it shows that a decision is mediated by U's non-additive coordinate. It cannot show that the coordinate is agreement rather than the target, unless the task separates them, and in the task that does (XOR3), the coordinate is the target.

## 5. Where the evidence is

| what | where |
|---|---|
| frozen protocol and addendum | `PROTOCOL.md`, `PROTOCOL_ADDENDUM_XOR3.md` |
| all results, including failures | `INTERVENTION_RESULTS.md`, `XOR3_RESULTS.md` |
| mechanisms | `MECHANISMS_AND_CAUSAL_GRAPHS.md` |
| theorem | `DECISION_THEOREM.md` |
| comparisons | `DISTINCTIVENESS.md` |
| reviews and their adjudication | `reviews/`, `ADJUDICATION.md` |
| corrections to earlier claims | `ERRATA_PREVIOUS_PACKAGE.md` |
| F-matrix update | `FAILURE_MATRIX_UPDATE.md` |
