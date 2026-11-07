# Research log: learned causal mediator identification (mandate 7885c5cb)

- Base: branch cor-phase15-step1-mechanism-identification @ 61c23ae. New branch cor-phase15-step1-learned-mediator.
- A: 90 historical Transformer checkpoints converted pkl -> npz (round-trip exact; MANIFEST with both hashes).
  Retraining 4 models with the vendored code reproduced the historical parameters BITWISE (logs/retrain_check.json).
- B: generators.py; aggregate-only (z,p) ceilings: REL-XNOR 0.754, REL-GATE 0.877, REL-MIS 0.754; AGG-* 1.0.
- Exploratory development on historical dev seeds 0 and 1 only (scripts/dev_probe.py, dev2.py, dev3.py, smoke runs):
  found S_rel (matched-pair contrast subspace) and the content-surrogate test; changed IIA to model-relative
  counterfactuals and the interaction criterion to a like-for-like surrogate comparison BEFORE freezing.
- Protocol frozen: see commit below. New models (AGG-NL s0-4; seeds 7-9) were trained before the freeze but not analysed.
- PROTOCOL FROZEN at commit ff180da (pushed). No evaluation set analysed before this commit.
- Evaluations E1-E4, locus, exploratory X1/X2, AGG-NL reversals, independent reimplementation (agrees) done; docs written; pre-review commit next.
- Review round on a899a7c: independent reimplementation (agrees), R_causal, R_math (incl. re-review of earlier post-review results), R_counterexample.
  Key finding: relation and target confounded in XNOR/MIS; TF2 resid_mid1 readout-equivalent; sufficiency == reverse interchange.
- Addendum PROTOCOL_ADDENDUM_XOR3.md frozen at c7bcd2c BEFORE training REL-XOR3 models (seeds 0-6; dev 0-1 code check only).
- XOR3 results: late sites TARGET_ONLY (TF1 3 learned models, TF2 5/5); AO early sites follow H_AB but are source-additive (content test, E5).
  Headline changed from 'yes, restricted' to 'no'. ADJUDICATION.md records all dispositions.
