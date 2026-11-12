# Frozen protocol, addendum A1: the successful-vs-failed predictor

This addendum was committed after Part I and **before any confirmation run started**. `checkpoints/compose_conf/` and
`checkpoints/conf/` did not exist at the time of commit. It adds one prediction and changes nothing in Part I.

## Development finding behind it (`logs/predictor_probe.json`, `scripts/predictor_probe.py`)

The candidate predictor was the linear decodability of the eligibility vector m from the representation just before
each Condition-R reader's readout map. It was measured as held-out multinomial-probe accuracy, averaged over groups,
against a majority baseline of 0.667.

| reader (dev) | probe acc | E_R1 | outcome |
|---|---|---|---|
| mlp R label 16k s0 / s1 | 0.988 / 0.986 | −0.46 / −0.42 | **fail** |
| mlp R label 48k, mlp R Bayes | 0.993 / 0.997 | +0.08 / +0.01 | success |
| attn R label 16k s0 / s1 / 48k | 0.877 / 0.840 / 0.844 | −0.08 / −0.12 / −0.04 | success |
| attn2 R label 16k | 0.769 | +0.00 | success |
| relnet R (four runs) | 0.976–0.980 | −0.01 … +0.14 | success |

The two failed readers decode m *better* than every successful attention reader does. Decodability therefore does not
predict success, and it was rejected as a predictor on development data alone. What separates success from failure in
development is whether m is *used* with the correct multiplicative form. The causal measures in P3 assess this for the
attention readers; no architecture-agnostic causal predictor was frozen.

## Prediction P7 (frozen)

On the confirmation readers (`PHASE=conf scripts/predictor_probe.py` on the ten C-A readers, run after C-A):

- (a) Every mlp R label 16k confirmation reader that fails (E_R1 ≤ −0.20) has probe accuracy ≥ 0.90.
- (b) The mean probe accuracy of the failed mlp readers is at least as high as the mean over the successful attn R
  readers.

That is, linear decodability of the relation does not separate learned success from failure. If no mlp reader fails, P7
is undecidable and is reported as such.
