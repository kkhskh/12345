# Investigation XX — development notes (DEVELOPMENT ONLY; not evidence; never mixed with confirmation)

All development runs use the XIX development parent uP160 and data order D801 (or the XIX D801 anchors), streams 91_900_000 /
20_000_001, on CPU in the build session.  Confirmation uses fresh data orders D2001-2006, fresh streams 91_1xx_xxx-91_43x_xxx and
other parents.  Nothing here is a verdict.

## rsg (new routing-gradient intervention) — dev/runs, dev/rsg_measure
| run (step 1211) | KL | acc | nec_E | nec_M | causal class |
|---|---|---|---|---|---|
| E base rsg own0 (L21, D801) | 0.016 | 0.849 | 0.227 | 0.386 | HYBRID (phi vs E801 anchors ~ 0.58) |
| M base rsg own2 (L23, D801) | 0.013 | 0.854 | 0.031 | 0.674 | MIXTURE (B-like) |
| XIX anchor E801 fullB | 0.031 | 0.854 | 1.178 | 0.005 | EARLY |
| XIX anchor E801 fullI | 0.012 | 0.855 | 0.001 | 0.807 | MIXTURE |
| XIX anchor M801 fullB | 0.013 | 0.837 | 0.021 | 0.605 | MIXTURE |
| XIX anchor M801 fullI | 0.028 | 0.842 | 1.166 | 0.012 | EARLY |
Caveat: the rsg runs are trained from step 0 (window then continuation in one run), the anchors through the composer branch; at
CPU these are bitwise the same trajectory for the plain parts (XIX check), but n = 1 per base.  Used only to set HA8M / HA8E.

## Track B — rule-mixture fits (dev/trackb_dev_rules*.json) and interchange (dev/trackb_ic)
- argmax disagreement with Bayes on 12k natural episodes: local 0.24, noelig 0.62, nocount 0.09, maprole 0.21 -> on-support
  discriminating inputs exist for every candidate rule (B2 design-gate requirement met before compute).
- matched-KL rule profiles similar between classes (MIXTURE @600 vs EARLY @1211) -> frozen prediction 'not B1'.
- other-question interchange IIA at the layer-2 input (OQI_2): EARLY 0.44 / 0.49, MIXTURE 0.24 / 0.20 (n = 2 + 2) -> HB_d can
  discriminate implementations.
