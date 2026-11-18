# Open COR foundational questions after Investigation XVI

**Status.** COR Phase 1.5, Step 1, provisional ≈ 25%, unchanged. No thesis is selected, no phase is closed, and the obstruction/discrepancy (O/D) operators are not validated.

## What XVI changed

1. **The XV arity-3 censoring was a finite-optimisation effect in condition C** [frozen]. Every arity-3 model given a substantially larger effective learning budget acquired the gate: 4/4 fresh at a constant rate and 3/3 restarted XV failures. This happened at total exposures of 1.0–3.9, against XV's 2.0.
2. **In condition D (strong agreement cue, lower gate value) a persistent finite-budget barrier exists for some seeds** [frozen, selected seeds]. 2 of 3 restarted XV failures stayed at the gate-free optimum through 4000 further full-rate steps while still making substantial parameter updates. XV's development model D_s100 behaved the same way.
3. **On every plateau measured there is no first-order expected gradient toward the gate.** It appears only a few hundred steps before the behavioural onset [expl, validated estimator].
4. **Gate information beyond the agreement cue appears together with its causal use,** at 25–50-step resolution. XV's apparent "information before use" was the cue [expl].
5. **Value against cue.** With the competing cue's law held exactly fixed, a 1.74× change in gate value produced no consistent change in timing (P3 INCONCLUSIVE). XV's arity-2 value effect, measured with the cue changing, was consistent [frozen + expl].
6. **The agreement stage survives corrected, floor-conditioned criteria** where the agreement cue is strong or the gate is slow (P5, 12/12). **It does not** in XV's fast-gate, weak-cue condition B (1/4) [frozen + expl].
7. **Arity 4 can be built at exactly matched value, intermediate gain and Bayes NLL relative to arity 3** [argument + num]. That makes a clean complexity contrast available for the first time.

## Open questions, in order of value

1. **What sets the plateau escape time?**
   - The signal is zero on the plateau, escape times vary 3× between fresh seeds (1100–3500), and some D seeds do not escape.
   - Candidates: gradient-noise-driven escape, slow drift of a representation the gate computation needs, and higher-order curvature.
   - Tests:
     - vary the batch size and learning rate at fixed exposure (noise scale);
     - measure second-order quantities, such as the Hessian along the gate direction, on the plateau;
     - restart the persisting D models with different noise.
2. **Is the D barrier competition-driven?**
   - D differs from C in gate value (0.138 against 0.277) **and** in cue strength (agreement benefit 0.077 against 0.025).
   - The P3 null result and the cross-study pattern point to the competing agreement rule.
   - A design that varies ν with gate value and difficulty held fixed needs a richer generator. Fresh D-condition models at a constant rate would also measure how common the barrier is.
3. **Arity-4 scaling at exact value matching.**
   - Arity 3 is now known to be learnable in C, so an arity-4 experiment in the 7-source format with arity-3 baselines in the same format would be diagnostic (about 20 lane-hours here).
4. **The intermediate stage as cause or bystander.** Does the agreement stage help or hinder gate acquisition? It could be tested by ablating or freezing the agreement pathway during the plateau.
5. **Restart against moment reset.** Re-derive the optimizer state by deterministic replay and restart only the learning rate (about 0.7 lane-hours per model).

## Broader landscape (carried over, unresolved)

- **T4.** DeepSets learns it from scratch (XV); the Transformer under XV's recipe stalled at the source stage. XVI's finding that constant-rate training rescues arity-3 GATE suggests the T4 Transformer failure deserves the same extended-optimisation test **[spec]**. It was not run.
- **GATE:** now characterised across arity 2 and 3, two cue strengths, two value levels and three schedules.
- **TRANS:** composition is not learned. Untouched.
- **Adaptive sources and position dependence (ADAPT):** untouched.
- **Architecture dependence:** one Transformer recipe in XVI. DeepSets only for T4 in XV.
- **Unknown or hidden groups; trust uncertainty:** noisy gates were analysed mathematically only in XV.
- **O/D framework:** untested. XVI says nothing about the original obstruction and discrepancy operators.

## Most valuable next direction [judgement]

**Question 1, the escape-time mechanism,** combined with fresh D-condition runs. It is the one place where XVI turned a finite-budget observation into a sharp, testable puzzle: a plateau with zero expected gate gradient that some seeds leave and others do not. It is also cheap: the plateau checkpoints and the validated estimator already exist.
