# 08R2 Protocol Option 2 — Narrower A/D

This is a **candidate** protocol for review. It does not replace `08r2_preregistration.md` until you choose it.

Use if B is too scarce for a reserved local-edit analysis or C is unavailable under strict occupancy matching.

## What is tested

- A: simulator-valid recipient/target pairs with different legal sets.
- D: wrong-target controls for the same recipients.

## What is not claimed

- No claim that edits are local/near-identical board interventions.
- No claim of same-legal-set invariance.

## Scientific cost

The experiment can still ask whether a decoder-constrained residual edit moves decoded board state and legal-set probability mass together, and whether that beats wrong-target, random, and probe-direction controls. It cannot support Section 5.3 claims that require local germs or legality-preserving board changes.

## Selection

Same floors as the original preregistration on A pairs only, plus KL ≤ 1.5× same-budget random-edit KL. Donor-access and model-logit methods are reference baselines, not selectable primary editors.
