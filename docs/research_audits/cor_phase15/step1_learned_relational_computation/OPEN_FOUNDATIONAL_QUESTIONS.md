# Open foundational questions

Each entry gives the question, why it is still open after Investigation VIII, and the smallest experiment or proof that
would settle it.

1. **Can the P-count representation itself be made to support the relation?** All successful repairs add a learned
   relational module over the raw ingredients. The native modifications that would answer this (unfreeze layer 1, LoRA on
   layer 1, two fresh blocks) were deferred for compute; any that did run are reported in `SELECTIVE_LEARNED_REPAIR.md`.
   *Smallest test:* unfreeze1, lora1 and deep2 on 4 P lineages under the same 4000-step budget, with the same
   preservation rules.

2. **Does the downstream block use the adapter as an m-like correction or as a parallel posterior?** *Test:* the
   post-hoc adapter ablation (`scripts/repair_ablate.py`), and an interchange of the adapter's output between R1 pair
   members. The ablation is in the exploratory queue.

3. **Why does the separable phase come first, and how long does it last?** No theorem; documented [N]. *Candidate
   analysis:* a linearised (NTK-style) or two-timescale argument in which the separable component has larger gradient
   signal, because 91.6 % of decisions have a single perfect group. Its prediction, that reweighting toward two-perfect
   states shortens the phase, can be tested directly.

4. **Is "decodable but unused" one phenomenon across the frozen P-count states, the failed MLP and auxm?** *Test:* the
   same causal predictor (interchange share of the m-probe subspace) on all three kinds of failure. It is not yet
   defined for MLPs and frozen Transformers in a comparable way.

5. **Cross-model causal-equivalence transport** (`CAUSAL_IDENTIFICATION.md` §3): not run.

6. **Condition X at full budget.** The from-scratch token Transformer was run only at 6000 steps (the 16k run was
   stopped for compute). Whether ordinary token training reaches the relation without a relational inductive bias at
   the reader is therefore only partly answered.

7. **Generator variations.** Only one generator (T4: K=3, Q=6, three groups of two) was studied. Holdouts within it are
   in `COMPOSITIONAL_GENERALIZATION.md`. Larger K and more groups were not tested, so it is open whether the relational
   readers' advantage grows with the number of groups.

8. **What is the right notion of "selective" repair?** On these models the original transfer is wrong on many families.
   A correct relational module therefore improves all of them, and "selectivity" reduces to "no family made worse". A
   setting where the original is right everywhere except R1 is needed to separate a surgical repair from a global one.
