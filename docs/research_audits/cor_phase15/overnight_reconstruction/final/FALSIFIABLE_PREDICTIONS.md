# Falsifiable predictions

## Rejected conditional prediction

Under the kernel hypothesis of Theorem K, `R = 1_{t = r} w·δ`.

Falsifier: kernel products zero and recomputed `R` different from the skip dot. The constructed point does not meet the falsifier. The prediction is still rejected as a research program because its hypothesis set, on injective `QKV` weights with full gains, contains only constants, where the shift identity already applies.

No other prediction is issued.

## Predictions that were not made

- `D ↓ ⇒ L ↑`. Falsified by the zero cochain against `(1,0)`.
- `D_c < D_{c'} ⇒ L_c > L_{c'}`. Falsified by two readouts on one cochain.
- `Dm = 0 ⇒ R = 0`. Falsified on seed 0 at step `0.2`.
- Value-only sign equals total sign. Falsified on seed 16.
- A dropped coordinate determines the sign. Falsified on seed 16, feature 3.
- A weight-free finite bound on centered `R`. Falsified by the scaling family.
- A Lipschitz certificate strictly below the Cauchy number. Contradicts differentiability whenever the certificate is valid.
- Final-LN logits obey the raw constant-shift response. Falsified by applying LayerNorm to `H2`.

## What a later prediction would have to publish before any run

Model class, outcome, measured quantity, intervention family, budget (B0, B1, or an explicit larger budget), assumptions that do not mention `sign(R)`, the predicted relation, the falsifier, and the competitor among direct `R`, `Dm`, a Hessian bound, a path-specific effect, and an interchange or abstraction test. A notation for the Jacobian fails that list. This package does not publish a survivor and does not schedule the run.
