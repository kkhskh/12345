# Did we identify a computation in a learned Transformer that uses an independently defined source-consistency relationship to causally influence candidate selection?

**Yes, in a restricted and well-delimited sense.** The answer separates in three: what was found, which tasks it was found in, and whether "consistency" adds anything beyond ordinary nonlinear computation.

## 1. What was found

In small learned Transformers trained on REL-XNOR, whose target is agreement itself, the mediator is a one- or two-dimensional subspace U at a complete cut. In TF2 the cut is the last-token residual before the final MLP: k = 1 in 4 of the 5 TF2 models, k = 2 in 1. The subspace has four properties:

1. **It was identified without any output information.** U consists of the top singular directions of the state differences between matched input pairs with **identical aggregate** `z = a + b` and opposite agreement. The procedure was frozen before evaluation (`PROTOCOL.md`, commit `ff180da`).
2. **It is causally used, with controls.**
   - Interchanging only U between matched states transfers the model's decision with interchange-intervention accuracy (IIA) 1.000.
   - Random subspaces of the same size give mean IIA 0.000–0.013 (best single draw 0.256).
   - An equally decodable subspace orthogonal to U gives 0.000.
   - Ablating U to the pair mean removes the relation-sensitive decision difference: 0–15.2% of pairs still differ.
   - Restoring U alone, into an otherwise counterfactual state, keeps the decision: sufficiency 1.000.
3. **Its content is the independently defined relation.**
   - Its cross-source (non-additive) part is explained by `sign(ab)`, the agreement indicator, with R² 0.997–1.000 in every TF2 relation-computing model at that site (E1, E2).
   - Replacing U's content with its best source-additive fit, i.e. a function of each source separately, destroys the decision: accuracy falls to 0.49–0.57, and matched-pair decision differences fall from 1.0 to 0.01–0.37.
4. **The computation that produces it is located.**
   - In TF1 models, the interaction is created **exactly** by the attention softmax normalizer. Freezing the layer-0 attention pattern makes it vanish (interaction share 0.000; Theorem T-ATT, `DECISION_THEOREM.md`).
   - In TF2 it comes mainly from layer-0 attention. One seed (s7) produces it through either attention layer, i.e. redundantly. A small remainder comes from the layer-0 MLP acting on the mixed state.

**Replication.** This held in 5/6 TF2 models on the training distribution (historical seeds 2–4 and new seeds 7–9) and in 5/6 under a new Gaussian evaluation distribution. An independent reimplementation reproduced every checked number (`reviews/independent_verification/REPORT.md`).

## 2. Where it was found, and where it was not

| task (target) | relation-computing mediator at a primary cut | interpretation |
|---|---|---|
| REL-XNOR (target = agreement) | TF2: 5/6; TF1 pre-MLP: 1/6 (in the other 5 the site's content is source-additive and the interaction is computed downstream across the MLP and direct paths, X1); TF2-AO layer-0 output: 0/6 (computed in layer 1) | Learnability control: the network computes the relation because the target **is** the relation. |
| REL-MIS (agreement misleading) | TF1: 6/6, train and gauss; TF2: 6/6 on train, 1/6 on gauss (borderline content criterion: the additive surrogate costs 5–6 accuracy points on train, 3 on gauss) | A genuine cross-source interaction, but **not** the agreement relation: R² by `sign(ab)` is ≈ 0. It is a target-specific interaction (partial R² with `sign(ab)·sign(a+b)` of 0.04–0.31). |
| REL-GATE (agree → follow, else prior) | 0/12 at R sites. The mediator is causally identified but **source-additive** (TF1 6/6; TF2 5/6) | The relation is **not** computed. The model implements threshold logic on per-source support, as T-RF predicts. |
| K3-GATE (3 candidates) | 0 relation-computing; source-additive or not identified | Same as REL-GATE. |
| AGG-LIN, K3-BAYES, AGG-NL (aggregate-only targets) | the model's decisions differ on **0 of 500** matched pairs in every model | No relation-sensitive behaviour, even though the states retain information beyond z (contrast-energy ratio up to 1.39). AGG-NL margins have 423–450/465 order reversals while being functions of z (within-z spread ≤ 3% of the SD). **Behavioural order reversals do not indicate relational computation.** |

## 3. Does "consistency" add anything?

**As a mechanism, no.**

- The identified computation is an ordinary cross-source nonlinearity: the attention normalizer, read by an MLP and the readout.
- It appears in the form of agreement only when agreement is the training target.
- When agreement is only a gate on the decision (REL-GATE, K3-GATE), the learned models do not compute it.
- When agreement is misleading (REL-MIS), they compute a different, target-specific interaction.
- A matched-capacity MLP, the standard source-interaction account, and a causal-abstraction description, i.e. interchange intervention (DAS), all describe these mechanisms equally well (`DISTINCTIVENESS.md`).

**As an identification tool, a modest yes.**

- Defining the relation independently and contrasting **aggregate-matched** pairs gives a closed-form, output-free mediator subspace.
- That subspace coincides with the target-trained DAS subspace (cosine 0.87–0.99) and transfers decisions as well.
- It beats the output-gradient (direct-sensitivity) subspace in 9 of the 22 relation-computing sites (IIA 0.0–0.15 against 1.0).
- This is a counterfactual difference-of-means construction, a known technique. Its contribution is the **matching on the aggregate**, which is what makes it identify relation content rather than source content.

**As a restriction, a modest yes.** The task-level pattern matches the threshold-logic prediction from the previous package (T-RF):

- no relation is computed for fixed-prior deferral rules;
- an interaction is computed for XNOR-type and misleading rules.

This prediction was not written into this protocol as a hypothesis, so it is consistent with the result but **not preregistered** confirmation. It also comes from representability theory, not from "consistency" as such.

## 4. Limits of the claim

1. **Learnability control.** The relation-shaped mediator appears only where the target is the relation (XNOR). This does not show that Transformers favour consistency.
2. **Agreement vs spread.** At fixed z, `s = 1[|a−b| < |z|]`. "Uses agreement" and "uses the spread thresholded at |z|" are the same function on this input family, so no experiment here can separate them.
3. **Off-manifold interventions.** Many successful interchanges produce states far from natural activations. For the five headline XNOR TF2 mediators, the nearest-neighbour ratio is 18–189; across all models it is about 1–200, reported per model. Sufficiency and necessity are measured on those patched states, so they show what the downstream computation does **with** the U-coordinate, including off-manifold values. They do not show what happens on natural inputs alone. The on-manifold evidence is the input-level matched-pair contrast together with U's content analysis.
4. **Uniqueness.** Mediation is not unique.
   - In TF2, R-only cuts work, but the layer-0 cut across all positions often fails the necessity gate (redundant routes).
   - TF2-AO's layer-0 output is source-additive; the interaction forms inside layer-1 attention, where no single cut was tested.
5. **Scope.** The results cover small synthetic models (d = 16, 1–2 layers), two-source tasks plus one K = 3 family, and inputs in [−3, 3]². They say nothing about pretrained models.
6. **Borderline content verdicts.** The relation-computing classification of TF2 REL-MIS depends on the 0.05-accuracy threshold.

## 5. Where the evidence is

| what | where |
|---|---|
| protocol | `PROTOCOL.md` |
| per-model results, including every failure | `INTERVENTION_RESULTS.md` |
| mechanisms and causal graphs | `MECHANISMS_AND_CAUSAL_GRAPHS.md` |
| theorem | `DECISION_THEOREM.md` |
| comparisons | `DISTINCTIVENESS.md` |
| corrections to earlier claims | `ERRATA_PREVIOUS_PACKAGE.md` |
| F-matrix update | `FAILURE_MATRIX_UPDATE.md` |
| reproducibility | `README.md` and `RESEARCH_LOG.md` |
