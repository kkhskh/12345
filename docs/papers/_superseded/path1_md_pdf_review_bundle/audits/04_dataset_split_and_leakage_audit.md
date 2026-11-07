# Dataset split and leakage audit

## Frozen generation seeds

| Split | Seed | Role |
|---|---|---|
| Train games | 0 | Probe, PCA, ridge, and rowspace map fitting |
| Validation games | 1 | 08R2 validation, FIBER, SPEC, COST, NST causal pairs |
| Test games | 2 | Not used for Path 1 causal claims |
| FIBER pair census | 1082 | Excluded from later SPEC matching |
| SPEC pairs / wrong / random | 2182 / 2282 / 3182, 3282, 3382 | 48-pair L2 and later JS studies |
| NST pairs | 4182 | New unused-val 48 pairs |
| Leftover SYM | 7182 | Infeasible |
| FRESH pilot | 8182 / 8282 | Engineering only; excluded from scientific FRESH |
| FRESH scientific | 9182 / 9282 / 9382 / 9482 | 32-triplet switching |
| Anchor-audit bootstrap | 9582 | Read-only |
| VNF | 10182–10782 | Independent factorial cohort |
| Confirmation | sealed `confirm_` / 08r2_08r3_confirmation | **Not opened** |

## What is independent of what

The 48 SPEC pairs were used for SPEC (equal L2), the original COST run, the COST forensic, and the restored fixed-JS study. Those four analyses are **not** independent confirmations of one another. The restored fixed-JS result is a protocol-compliant re-analysis of already inspected pairs.

The 48 NST pairs exclude FIBER and SPEC games by construction. They are a new validation cohort relative to SPEC, but they remain unused **validation** games, not the sealed confirmation set.

STATE-SEMANTICS ablated the frozen NST map on the **same 48 NST pairs**. Its letter `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY` is exploratory and is not independent confirmation of NST.

The leftover unused-val symmetric-target census was infeasible (`SYMMETRIC_TARGET_CENSUS_INFEASIBLE`). It contributes no causal Y.

FRESH (seed 9182) generated 400 new games. Sequence-hash collisions with prior splits were dropped. The census reported **24 identical board+player prefixes** that already occurred in train/val/test/pilot. Games and move sequences were still disjoint. Board-state reuse is therefore real and must be disclosed: game-level independence is not state-level uniqueness.

VNF (seed 10182) is a separately generated 400-game cohort. It does not reuse the FRESH triplet hash.

## Probe, representation, and evaluation leakage

- The 08c probe was fit on training activations. Causal evaluation used validation or later cohorts.
- The NST 64-dimensional map (PCA on probe-null residuals + ridge from symbolic features) was fit on training activations. Hyperparameters \(k\) and \(\alpha\) were selected by training-game grouped cross-validation residual \(R^2\), not by held-out Y.
- Oracle donor activations \(h_D\) for NST evaluation targets were loaded only after learned directions were frozen. “Donor-free” means donor-free at **evaluation-time direction construction**, not training without natural activations.
- Symbolic features for the learned map include the simulator legal-move mask of the intended target. The method is therefore supplied target legality; it does not infer Othello rules from an unseen board at evaluation.
- Editor A optimization, where used, sees the recipient activation and a board-decoding objective. It is not a donor-free method.

## Sealed confirmation

All runners call `require_no_confirmation`. No Path 1 claim uses the sealed 400-game confirmation split. Absence of confirmation is a limitation, not a hidden extra result.

## Repeated inspection

SPEC/fixed-JS, NST/SEM, and FRESH/VNF form three evaluation layers of decreasing reuse. A 95% interval on a later analysis of the same 48 pairs is not an independent replication. The manuscript treats FRESH and VNF as the only prospective cohorts generated after the representation was frozen.
