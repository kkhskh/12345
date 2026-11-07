# Preregistration addendum (after review; before any C′ code)

Review R1 (finding S7) found that Mechanism C has no cross-layer attention rereading. This addendum registers predictions for a new construction **C′**, written and committed before its code. C′ was added in response to review. It is not part of the original preregistration.

## C′: conditional rereading through attention

Positions: prior token P, source A, source B, read.

- **Block 1.** A uniform head copies A's and B's evidence to the read token. A GELU MLP with the Mechanism-A pairing writes the agreement signal `q` onto `e_3`.
- **Block 2.** A head whose **query reads `e_3`**, and whose keys are `+1` at source positions and `−1` at P. When the sources agree, the read token attends to the sources and copies their votes. When they disagree, it attends to P, which carries a prior vote for `c'`. No MLP. No final LN.

Predictions:

- **PR1.** In agreeing cells, `sign m = sign a`. In disagreeing cells, `sign m = −1` (the prior wins), *whichever source is stronger*.
- **PR2.** Block 2's attention mass on the source positions is strictly increasing in `q̂`. Its mass on P is strictly decreasing.
- **PR3 (one-layer impossibility).** In a single attention layer, `log(α_A/α_P)` does not depend on `b`. Checked numerically by running block 2's head as a first layer on the raw tokens.
- **PR4 (mediation).** Interchanging block 1's MLP output at the read position from an agreeing run into a disagreeing run switches block 2's routing toward the sources and flips the decision. Ablating block 1's MLP makes block 2's routing independent of agreement.
- **PR5 (strength control).** Among disagreeing cells, increasing the stronger source's magnitude does not change `sign m`.

Corrections also registered here, before rerunning:

- **PB6** is rescored on both of its preregistered parts: additivity *and* relation-blindness.
- **PA11** is reclassified as **non-discriminating**, because its threshold could not fail (R1, S5).
- Mechanism C's matched-sum analyses are recomputed with pairs matched **at the block-2 reader** (`Ŝ`), as R1 S6 and R2 S2 require. The original numbers stay in the log.
