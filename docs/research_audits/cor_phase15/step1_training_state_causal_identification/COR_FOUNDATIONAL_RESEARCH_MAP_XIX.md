# COR foundational research map after XIX

**Status.** Phase 1.5, Step 1, provisional ≈ 25%. No phase closure, no thesis selection. O/D is unvalidated.

## What XIX adds to the map

| question (since XIII) | status after XVIII | status after XIX |
|---|---|---|
| Does a 100-step route block durably select the organisation? | yes (XIV), within one parent | reproduced on 6/6 fresh pairs [frozen anchors] |
| Which component of the block matters? | unresolved: content, gradients and share confounded | state level: parameters, not moments; E layer 0 sufficient and necessary; M L0+L1 and L0+L2 each sufficient, L0+L2 restoration prevents, L0 alone insufficient, single-partner necessity not established [frozen] |
| Is it generic optimization disturbance? | untested | **no**, in the tested forms: matched-norm displacement (functionally inert at step 100) and label-noise loss disturbance (matched on E, 4.7–12.4× on M) [frozen] |
| Content against value learning | confounded (vrg) | M: removing V+O learning through T suffices, with content and per-step routing gradient intact (vsgo) [frozen]; vsg alone too [expl: the partial prediction was falsified]; a routing-state change following from it is not excluded. E: not established (vsgo PARTIAL); content loss suffices |
| Routing learning | gsg / gsv ambiguous | routing-parameter freeze (frzp) partial on E and nil on M [frozen]; the frz/frzp contrast is consistent with a moment artefact but not established (PARTIAL) |
| Function against route | weak behavioural gate | route-free error structure follows the causal class in all 86 non-hybrid models [frozen-addendum]; causal implementation reorganisation with a precision difference; no different rule [interp] |
| Parent generality | 1 of 5 parents bistable | not tested in XIX |

## Open questions, in order

1. The E remainder. Test a projected routing-gradient stop combined with vsgo.
2. Transport to a second bistable parent (uP163).
3. Why the M step-100 states, which are functionally indistinguishable, carry different futures.
   - The selection lives in latent parameter coordinates (L0 plus a partner) before it becomes functional.
   - A tangent-kernel or linearised-dynamics analysis of the continuation from those states is the natural next mathematical object.
4. Whether the threshold-like dependence on the I − B direction ([dev]: half the layer-0 difference does nothing) reflects a basin boundary of the continuation map.
   - Instability and interpolation analysis in the style of Frankle et al. (ICML 2020; ICLR 2020) is the relevant prior method. Juneja et al. (ICLR 2023) and Lubana et al. (ICML 2023) already describe basins that correspond to distinct strategies from one parent.

## Appendix: relation to the original O/D mathematics

This appendix is **speculation**, kept separate from the results.

The results show that early training can lock in which of two approximations a network uses, by a latent, not yet functional state difference. A *dynamic* obstruction/discrepancy hypothesis would have to predict, from the step-100 state alone, which basin the continuation enters. No such derivation exists. The original O/D framework does not predict any XIX result. XIX does not validate O/D, and O/D was not used to design or interpret XIX.
