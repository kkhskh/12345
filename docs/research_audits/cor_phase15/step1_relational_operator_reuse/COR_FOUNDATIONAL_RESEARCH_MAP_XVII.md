# COR foundational research map after Investigation XVII

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). **W1:** Proposition S is replaced by S′. **W3–W5:** transfer does not identify a shared computation; the layer locus and the H4 verdict are withdrawn. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%, unchanged. No thesis is selected, no phase is closed, and the O/D operators are not validated.

## What XVII adds

1. **O/D is silent on positional reuse** (ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md, committed before neural results).
   - The original definitions contain no symmetry, sharing or acquisition structure.
   - Source-level discrepancy cannot express any GATE decision [proof + num].
   - This extends the Phase 1.5 failure record of the source-level bridge (zero-weight and copying sources) to relational gates.
2. **In the GATE Transformer, "position" is only a slot code** [proof, Theorem P]. The network is exactly slot-permutation equivariant.
3. **A relational computation that any slot can run, accessed through a code-keyed early-layer transformation** [frozen, 6/6 fresh models, both arities].
   - Exchanging only the slot code in layers 0–1 transfers gate use to an unacquired slot.
   - Attention routing alone does not.
   - *[withdrawn, W4]* The readout layer is largely code-agnostic.
4. **Slot competences are learned and maintained by each slot's own training signal** [frozen + expl].
   - Targeted training helps the targeted slot only, and erodes the others.
5. **Slot-staggered acquisition is reliable at arity 3 and inconsistent at arity 2.** The long partial window of XVI's C_s202 is rare [frozen P0 MIXED].

## Open territory, in rough order of value [judgement]

1. **What makes a code "engaging"?** Codes are near-orthogonal random vectors that barely move. Why the shared early layers work for some codes first, and how training extends them to the rest, is unexplained. It needs geometry of the early-layer weights as seen through each code; a cheap analysis on existing checkpoints.
2. **Architecture control** (mandate Experiment E, not run). A model with a single shared code, or with tied codes, would remove slot keying. Does it learn the gate faster, slower, or not at all?
3. **Earlier heterogeneity (XIV).** Different seeds implemented GATE with different layouts. Whether those differences are code-keyed in the same way was not tested.
4. **T4.** The Transformer's stall at the source stage (XV) is untouched.
5. **TRANS, ADAPT, hidden groups, trust uncertainty:** unchanged since XVI.
6. **O/D.** Still unvalidated. XVII gives no route to validation: the framework is silent on the questions where the neural evidence is now strongest.

## What the next investigation should not do

- Another training-speed or arity sweep.
- Attaching COR terminology to the slot-code results.
- Fitting alignment maps between slots. The architecture already supplies the canonical identity correspondence.
