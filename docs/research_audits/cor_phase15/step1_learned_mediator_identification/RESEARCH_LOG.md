# Research log: learned causal mediator identification (mandate 7885c5cb)

- Base: branch cor-phase15-step1-mechanism-identification @ 61c23ae. New branch cor-phase15-step1-learned-mediator.
- A: 90 historical Transformer checkpoints converted pkl -> npz (round-trip exact; MANIFEST with both hashes).
  Retraining 4 models with the vendored code reproduced the historical parameters BITWISE (logs/retrain_check.json).
- B: generators.py; aggregate-only (z,p) ceilings: REL-XNOR 0.754, REL-GATE 0.877, REL-MIS 0.754; AGG-* 1.0.
- Exploratory development on historical dev seeds 0 and 1 only (scripts/dev_probe.py, dev2.py, dev3.py, smoke runs):
  found S_rel (matched-pair contrast subspace) and the content-surrogate test; changed IIA to model-relative
  counterfactuals and the interaction criterion to a like-for-like surrogate comparison BEFORE freezing.
- Protocol frozen: see commit below. New models (AGG-NL s0-4; seeds 7-9) were trained before the freeze but not analysed.
