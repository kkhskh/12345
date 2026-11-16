# Open COR foundational questions after XIII-D

COR Phase 1.5, Step 1 (~25%). This list says what remains open after XIII-D. It does not claim COR has been validated, and it says nothing about the O/D operators.

Every item below follows from a specific XIII-D result. The mandate's completion gates are each answered in MAIN_RESEARCH_SYNTHESIS.md, with a result or a documented blocker. They are not deferred here.

## Mechanistic questions

1. **Is there any low-order structure in which fresh initialisations lead to the early organisation?**
   - What is established:
     - XIII-C's readout-map cause is falsified (P1, P10).
     - D does not change the class.
     - Single-factor changes away from an early diagonal were non-early in 4 of 4 cells in uP160/uP161, and in 1 of 2 cells in uP164.
     - In uP163, W22 gives early with L21 and with L22, but W20 does not.
   - What is open: the data fit "a strongly interacting function of parent × L × W" (review 2, F1). Base rates are 7 early, 6 mixture and 4 hybrid diagonals; off-diagonal cells are early in 3 of 11, against 7 of 17 diagonals (Fisher p = 0.69).
   - Discriminating next step: several random re-draws of W (or L) on early and on mixture diagonals, to estimate P(early | change) on each side; then the step-0 alignment between the fresh layer-2 key/query maps and W_A.

2. **What is the sensitive window, and is the sensitivity route-specific?**
   - What is established: a block of either route during steps 0–100 permanently redirects the organisation (n = 1 per direction).
   - What is untested:
     - blocks shorter than 100 steps;
     - a sham-mask control;
     - a severity-matched late block. The late ro0 block drove the loss to chance, while the late l2q block left a stable dual-route hybrid.

3. **Source of the early-class Bayes-invariant (MOVE) error.**
   - What is excluded:
     - positional asymmetry of the layer-1 readout rows (P7 confirmed the negative);
     - the parametric decision families of XIII-C.
   - The remaining candidates are untested:
     - the layer-1 readout MLP's nonlinearity;
     - interference from the residual skip of non-sufficient statistics.

4. **Key/query division of the relational carrier.**
   - The carried relation always acts through the joint layer-2 key–query score (≥ 0.94).
   - Its split between keys and queries is model-specific: keys 0.89 in C-C1, queries 0.76 in C-C2.
   - What decides the split is unknown.

5. **Specificity of the eligibility subspace.**
   - In one confirmation unit (uP165 s20, E3 family), the source-free interchange carries more of the eligibility effect than DAS (P6 falsified).
   - Open: which non-source tokens carry the E3 eligibility signal in mixture models.

## Mathematical questions

6. **The two-colluder posterior and identifiability are unchanged from XIII-C:** Proposition S, the ρ non-identifiability proof, E identification, and the exact LOCALQ gap of 3.6 ± 0.07 points. The open mathematical question is the same one as before: a characterisation of which *implementations* are reachable under gradient descent from a given initialisation. XIII-D's results show this depends on joint initialisation properties, not on capacity.

## Implications for the wider COR programme (stated as limits, not claims)

- **Why a decision algorithm emerges.** In this setting the choice depends on the fresh initialisation, interacting with the pretrained parent. It is sensitive to route availability in the first ~1% of the learning-rate schedule. No simple low-order law was found.
- **What information is used, versus what is present.** The two can differ: the early route's information stays present and unread (P2c). Interchange along signal (pattern) directions transmits a relation, while interchange along decoder filters does not. That is expected under correlated noise and is not a model property (P3c, reread).
- **Whether an algorithm can be changed or reused.** It can be changed by:
  - retraining a small downstream readout, if the unused representation is intact;
  - supplying the missing sufficient statistic, given enough optimisation;
  - retraining layer 0.

  At lr 3e-3 it cannot be changed by retraining layer 2 + readout. At lr 1e-3 it cannot be changed by retraining layer 1 alone at matched capacity (P5). The post-review arm shows it cannot at lr 3e-3 either (0.021).

All of this concerns a small synthetic generator family and 2- to 3-layer transformers fine-tuned from six pretrained parents. Nothing here is evidence about COR's operators.
