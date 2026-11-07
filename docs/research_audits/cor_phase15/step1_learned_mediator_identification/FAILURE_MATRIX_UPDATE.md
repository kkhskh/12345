# Update to the ten original failures

**Historical verdicts are unchanged.** This table rates only the new evidence from this package.

| id | failure | new evidence | update |
|---|---|---|---|
| F1 | missing noncircular causal bridge | **Learned** XNOR models: generator relation → output-free aggregate-matched subspace → interchange, necessity and sufficiency on the total decision → content ≈ `sign(ab)` → computation located (the attention normalizer, T-ATT). This chain is noncircular: no output information is used in selection. | **progress** (learned), restricted to XNOR-type targets and to small synthetic models. For deferral rules (GATE): **no** relation is computed. |
| F2 | lower discrepancy can worsen a candidate decision | REL-MIS: the learned models compute an interaction that makes agreeing evidence point to the opposite answer. | unchanged (sharpened negative) |
| F3 | consistency ranking does not determine candidate preference | XNOR's mediator is candidate-symmetric (`sign(ab)`), and its decision is not a candidate preference over sources. In GATE, preference follows per-source support without a relation. | unchanged (sharpened negative) |
| F4 | no distinctive prospective prediction | The task pattern (no relation for deferral rules; interaction where the rule is not threshold-representable) matches T-RF, but it was not preregistered here. | **partial**; needs a preregistered prediction |
| F5 | representation and map non-identifiability | U is basis-free and covariant under exact gauges. Agreement and spread are provably inseparable at fixed z. Mediation is not unique (redundant routes in TF2; distributed computation in TF1 XNOR). | **partial**, with a proved identifiability limit |
| F6 | output optimization hidden in repair objectives | S_rel uses no output information. Output-using comparisons (S_das, S_grad) are labelled as such. | **progress** (procedural) |
| F7 | outcome-dependent tuning | The protocol was frozen before evaluation, with new seeds and a new input distribution. Development used dev seeds 0–1, documented. | **progress** (procedural) |
| F25 | same consistency, different decisions | GATE models: identical agreement with different decisions, driven by per-source support and the prior. | unchanged |
| F29 | no demonstrated advantage over existing methods | The mechanism is ordinary. The identification method equals DAS and beats output-gradient directions, but it is a known construction (counterfactual mean differences). | essentially unchanged; a modest methodological point only |
| F30 | excessive unfrozen choices | The protocol thresholds (0.99, 0.95, 0.2, 0.95; interaction 0.10 / 0.05) were frozen. The TF2 MIS content verdict depends on the 0.05 threshold. | **partial** |
