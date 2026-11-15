# Open COR foundational questions after XIII-D

COR Phase 1.5, Step 1 (~25%). This list says what remains open after XIII-D. It does not claim COR has been validated, and it says nothing about the O/D operators.

Every item below follows from a specific XIII-D result. The mandate's completion gates are each answered in MAIN_RESEARCH_SYNTHESIS.md, with a result or a documented blocker. They are not deferred here.

## Mechanistic questions

1. **What makes an (L, W) pair "early-compatible"?**
   - What is established:
     - In uP160 and uP161 the early organisation needs the early diagonal's readout map together with its layer-2 initialisation (freeze-2 P1, falsified as stated).
     - In uP163, the readout map alone is sufficient.
     - P10 (freeze 3b) tests a third parent.
   - What is open: no step-0 feature of W_A alone predicts the outcome across parents. That leaves the question of which joint property of the fresh layer-2 and readout-map initialisations, relative to the pretrained layers 0–1, makes the early route win the first ~100 steps.
   - The natural next measurement is the step-0 alignment between the fresh layer-2 mixture-head query/key maps and W_A's input-side singular vectors, together with the early-route gradient that each produces. It was not measured in XIII-D.

2. **Why is the mixture organisation an attractor and the early organisation fragile?**
   - Observations:
     - Blocking the early route for 100 steps, or for steps 546–1146 after it has formed, permanently gives mixture (F1a, F1e).
     - Blocking the mixture route late gives early only while the block lasts (F1f).
   - A quantitative account would compare the two routes' loss landscapes. One example is the loss reachable by retraining only the readout under each block; the reactivation measurement is one point of that comparison.

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

- **Why a decision algorithm emerges.** In this setting it is decided in the first ~1–3% of training. The decision depends on joint initialisation properties of the freshly initialised layers, and it is asymmetric:
  - one organisation (mixture) is the default and absorbing;
  - the other (early) is fragile.
- **What information is used, versus what is present.** The two can differ: the early route's information stays present and unread (P2c). And the directions from which a relation is best decoded are not the ones the model uses (P3c).
- **Whether an algorithm can be changed or reused.** It can be changed by:
  - retraining a small downstream readout, if the unused representation is intact;
  - supplying the missing sufficient statistic, given enough optimisation;
  - retraining layer 0.

  It cannot be changed by retraining layer 2 or layer 1 alone at matched capacity (P5).

All of this concerns a small synthetic generator family and 2- to 3-layer transformers fine-tuned from six pretrained parents. Nothing here is evidence about COR's operators.
