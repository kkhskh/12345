# Open COR foundational questions after Investigation XIII-C

Phase 1.5, Step 1 (provisional ≈ 25%). This is not a closure document: it states no thesis, makes no claim that COR is validated, and makes
no claim about the original obstruction/discrepancy (O/D) operators, which XIII-C did not investigate.

## Open problems, by foundational value

1. **Why does the readout-map initialisation decide the organisation?**
   - R1b showed that the fresh 64 × 3 readout map, not the fresh layer 2, carries the init effect. A WA × data interaction is not yet
     excluded.
   - Next: (a) vary WA alone with L and D fixed. (b) Find what about WA biases the early race. A hypothesis is the gradient it sends into
     the readout tokens, which favours early readout of candidate evidence; it is testable by the step-0 gradient alignment of WA with the
     layer-0 readout heads. Step-0 statistics of layer 2 did not predict the class (|r| ≤ 0.41).
   - Why it matters: it locates the selection between implementations in the **output map**, not in the computing layer.
2. **What implements the early class's "tempering"?** By T1/T2, ρ < 1 is one function with two equivalent stories. Routing causally shifts
   ρ (R3), but the internal step is unknown, and ρ explains only 1–4% of the early class's KL.
   - Next: path-specific interventions on the layer-2 MLP at the readout token, rescaling its input along the candidate-evidence
     directions.
   - Characterise the 55–64% "non-invariance" error, the dependence on information Bayes ignores (MOVE edits cost most).
3. **Availability versus computability at layer 2.** (The per-group-format hypothesis was withdrawn post-review: a group-agnostic supply works at 3× steps or 3× learning rate.) Text before review: supG learns the joint relation slowly (0.28 at 1000 steps, 0.90 at 3000), while a
   group-agnostic supply fails even at 3000 steps (exploratory).
   - Open: is this a statement about attention gating in general?
   - Test the dependence on supply position and encoding, on more units.
4. **The constructed carrier.** The early-layer repair builds a low-dimensional causal subspace that is orthogonal to the linearly decodable
   statistics (R5).
   - Open: is it a non-linear function of m, a key-space code read by the frozen mixture heads, or something else?
   - Next: align DAS subspaces with the mixture heads' key projections; nonlinear probes evaluated causally.
5. **Learning-transition predictor.** No internal predictor of the behavioural transition was found. The organisation predictor
   (step-546 mixture mass) failed near the class boundary (R2c, 2/5 failures).
6. **Hybrids.** In the hybrid units the rows carry 24–81% of the effect, other heads up to 33%, and route O up to 20%. Hybrid organisation
   is not decomposed per unit; the hybrid residue is partly data-sensitive.
7. **Early-class layer-2 repair** is inconsistent across units (J slope 0.07–0.63). What sets it is unknown.

## Experiments not done in XIII-C

| item | status |
|---|---|
| full 4000-step determinism replay | not done; bitwise at 50 steps and resume-bitwise only |
| development L2/WA split cells and development l2q runs | descoped |
| development adapt3 on uP160 and on uP161 s22 beyond the L2 arm | descoped |
| MLP-capacity and architecture manipulations for B3 | not done |
| learning-rate search for the failing repair arms | only lr 1e-3 / 3e-3 and 1000 / 3000 steps on one unit [expl]: L2 fails under all of them; supG succeeds at lr 3e-3 or 3000 steps |
| GATE comparison; TRANS; adaptive sources; unknown group structure | not run |

## T4-specific versus plausibly general

**T4-specific:**
- the routes and their proportions;
- the ρ values;
- the readout-map result;
- the per-group sufficient statistic m_{H(−g)}, which is specific to G = 3, C = 2 shared copying.

**Plausibly general (hypotheses, not results):**
- an output-map initialisation can select among internal implementations through an early race between pathways;
- input–output decision laws can be causally tied to internal routing yet stay non-identifiable as internal operators (T2);
- information can be present at a layer's input yet unusable by that layer unless it is in a usable per-unit form;
- decodable statistics need not be the causal carriers (R5);
- source-free patches create base-rate artefacts, so specificity must be measured as excess over them.
