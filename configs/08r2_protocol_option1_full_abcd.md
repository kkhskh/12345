# 08R2 Protocol Option 1 — Full A/B/C/D

This is a **candidate** protocol for review. It does not replace `08r2_preregistration.md` until you choose it.

Use only if the validation census shows enough genuine B and C pairs after category-aware search.

## When this is eligible

- B: at least 8 recipient-game-unique pairs with Hamming ≤ 8 and changed legal sets.
- C: at least 8 recipient-game-unique pairs with unchanged legal sets and Hamming > 0, under the **strict** same occupancy-bucket constraint. Do not count slack-only C as fulfilling this option.
- A and D remain as in the original preregistration.

## Scientific gain

Tests local-board specificity (B) and same-legal invariance (C) in addition to legal-set counterfactuals (A) and wrong-target controls (D).

## Scientific cost

If B or C is scarce, forcing this protocol requires weakening matching. That would change the estimand. Do not do that silently.

## Selection

Same representation and causal floors as the original preregistration, plus KL not exceeding 1.5× same-budget random-edit KL. Compare only on the intersection of pairs where both methods executed.
