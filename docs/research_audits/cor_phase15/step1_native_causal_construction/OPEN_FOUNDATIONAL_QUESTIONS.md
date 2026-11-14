# Open foundational questions after Investigation XI

Investigation XI does **not** close Step 1 and does not select a thesis. By the program plan, Step 1 is roughly 25%
complete. The questions below are open. Each is stated with what XI did and did not settle.

## Mathematics of abstraction

1. **Realizability beyond projections.**
   - Settled: Proposition A2 characterizes projection-realizability (direct-sum decomposition).
   - Open: there is no characterization of R_lin. I2-VS is linear but not projective. When does a network's natural
     intervention class lie between R_proj and R_any, and can such interventions be searched without trivializing
     identification (a sufficiently rich map realizes anything on a finite support)?
2. **Legitimate intervention sets.** VS is an inconsistent high-level intervention (§4 of the math audit).
   - Open: a principled criterion for which interventions on a high-level causal model are admissible tests. For example,
     interventions that remain inside the image of the input map, versus "soft" interventions that are realizable but
     never produced by inputs. How should failures on inadmissible interventions be scored?
3. **Uniqueness up to what?**
   - Settled: E is shared across seven development pairs and is linearly mappable between independent parents (mapped
     gain ≈ 1).
   - Open: is the E-subspace unique up to an invertible linear map, given the decoy results? Or are there other
     subspaces, carrying other variables, that pass preservation? No formal identifiability theorem connects preservation +
     interchange + mapped transfer to uniqueness.

## Mechanism

4. **The arithmetic of the conjunction.**
   - Settled: the conjunction is formed in layer 1, by attention and MLP1 in model-dependent proportions. It is
     necessary (P2) and restorable through 6 dimensions.
   - Open: the actual circuit. Which query/key features let a layer-1 head's attention depend jointly on perfectness and
     agreement? Is MLP1 a thresholded AND, or a smooth product? The weight-level QK/OV analysis was not done.
5. **Why layer 1 and not layer 0?**
   - Open: is it because perfectness (an aggregate over the five other questions) is only available after layer-0
     attention has gathered it? This would make the depth requirement a theorem about the dependency structure. It has
     been argued (decodability: perfectness is linearly available only after L1 attention), not proved.
6. **Redundancy.**
   - Settled: within layer 1, the attention-only and MLP-only removals are each partial, and together complete. The
     two routes are therefore serial/overlapping, not independent backups.
   - Open: whether any model keeps a dormant alternative route that appears only under removal (compensation). This was
     not tested.

## Learning

7. **Why does the representational conjunction precede the behavioural switch?**
   - Settled: P6 shows the ordering in the development models (5/5); see Part II for confirmation.
   - Open: the cause of the delay. Is it a readout (layer-2) learning lag, a magnitude threshold, or competition with the
     A1/A2 heuristics? This requires layer-2 readout analysis along the trajectory.
8. **Why does C1 (frozen layers 0–1) fail to compute the conjunction?** The frozen pretrained layers carry agreement
   and perfectness additively; the conjunction is absent (P1c). It is not established whether this is a capacity
   limitation of a fresh layer 2 alone, or an optimization one.

## Generalization

9. **Off-support behaviour (S = 0) is under-determined.** Models differ: the β that fits best ranges 0.5–0.8, and no
   β fits within KL 0.1. Open: what inductive bias decides it, and is it predictable from the mechanism (for example,
   graded rather than thresholded eligibility in MLP1)?
10. **Rule versus variable generalization.** Under C = 2, the eligibility *variable* generalizes but the *combination
    rule* does not. Open: is this a general dissociation for learned relational computations? This needs other tasks.

## COR program links

11. **T4 / GATE / accessibility.**
    - T4: the results concern T4's relational eligibility rule. The conjunction e = perfect ∧ agree(q) is the T4
      collusion test.
    - GATE: the layer-1 conjunction is the computational site a GATE-style intervention would target; the restoration
      result (§NATIVE) shows that a 6-dim site is causally sufficient for the decision change.
    - Accessibility: the variable is linearly accessible after layer 1, but the conjunction is not linearly accessible
      before it. This makes accessibility layer-specific and computation-dependent. Accessibility claims based on
      decodability alone are insufficient, because AND is linearly separable.
    - **No O/D connection was found or tested.** None is claimed.
12. **What would close Step 1?** At minimum:
    - a weight-level circuit for the layer-1 conjunction;
    - newly trained (not pre-existing) confirmation models;
    - a second relational task.
