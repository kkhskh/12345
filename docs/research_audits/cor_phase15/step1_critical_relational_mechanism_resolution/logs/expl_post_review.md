# Post-freeze exploratory experiments run after review 3 (not scored; they do not change any frozen verdict)

## M5: is "per-group format" needed?

Unit uP161 L20/W20/D23 (confirmation late unit), adapt3_expl.py. It is identical to adapt3.py except that its checkpoint directory carries
an AD_OUT suffix, so confirmation checkpoints are never overwritten.

| arm | 1000 steps, lr 1e-3 (frozen) | 3000 steps, lr 1e-3 | 1000 steps, lr 3e-3 |
|---|---|---|---|
| L2 (plain) | 0.721 / −0.001 | 0.727 / −0.002 | 0.729 / −0.001 |
| supL2 (m group-agnostic) | 0.730 / 0.028 | **0.795 / 0.931** | **0.797 / 0.902** |
| supG (per-group) | 0.753 / 0.277 | 0.790 / 0.895 | 0.791 / 0.934 |

Entries are accuracy / J slope. Bayes accuracy is 0.795.

**Conclusion.**
- The per-group-format claim is **withdrawn**. Supplied in either form, the joint statistic is learnable by a layer-2 repair once it is
  given 3× the steps or 3× the learning rate.
- The plain layer-2 repair, which must compute the relation itself from the frozen layer-2 input, fails under every setting tested.
- What separates success from failure is whether the relation is **available** at the layer-2 input, either computed upstream (L0only
  suffices) or supplied. Capacity and these simple optimisation changes do not explain the failure.

## M1: the readout map, or the readout map × data?

The split cell uP163 L21/W22/D21 has a readout-map seed that differs from both the layer-2 seed and the data seed. Result below.

**Result.** uP163 L21 / **W22** / D21: final v = (lik_RO1 0.818, mix_mass 0.168), confirmation stream 88_100_000.

| reference | v | distance |
|---|---|---|
| s22 (the WA seed) | (0.820, 0.246) | 0.08 |
| s21 (the L2 and data seed) | (0.377, 0.908) | 0.86 |

The organisation follows the **readout-map seed against both the layer-2 seed and the data seed**. This excludes the readout-map × data
interaction and the "majority of seeds" alternative for this parent. It is one exploratory run in one parent.
