# 08R2 Preregistration Amendment ABD_v1

**Amendment ID:** `ABD_v1`  
**Amends:** `configs/08r2_preregistration.md` (original document preserved; this file does not replace it)  
**Frozen before real-model smoke or validation outcomes**  
**Does not use model outcomes to change pair-selection rules**

Official labels remain unchanged:

- Original 0A: **C / PARTIAL**
- 08R1 execution: **COMPLETE**
- Section 5.3: **PARTIAL**
- `OBSTRUCTION_READY = NO`

This amendment is a protocol freeze. It is not scientific evidence that a mathematical state editor works.

## Why the original preregistration is amended

The original preregistration (`configs/08r2_preregistration.md`) defined four pair categories:

- **A.** Different legal-move sets.
- **B.** Similar valid boards with changed legality (Hamming ≤ 8).
- **C.** Different boards with unchanged legal-move sets.
- **D.** Wrong-target control pairs.

A category-aware census of the validation prefixes, performed before any 08R2 editor outcomes were used to change pairing, found:

- many genuine A pairs,
- a smaller but real set of genuine B pairs,
- **zero genuine C pairs** under the investigated constraints (strict same occupancy-bucket and ±1 occupancy-bucket slack).

Category C is therefore **unavailable**. This amendment does not relabel A or B as C. It does not fabricate near-identical same-legal boards. It does not silently reweight pairing after seeing method performance.

Candidate protocol option 1 (`configs/08r2_protocol_option1_full_abcd.md`) required genuine C and is ineligible. Candidate protocol option 2 (`configs/08r2_protocol_option2_narrow_ad.md`) dropped B. The supported design keeps genuine B and drops only unavailable C.

## Frozen A/B/D design

| Role | Category | Frozen validation count |
|---|---|---|
| Primary, legal-set counterfactual | A | 24 recipient-game-unique pairs |
| Primary, similar-board legal change | B | 8 genuine recipient-game-unique pairs |
| Control | D | 32 matched wrong-target controls (one per A/B pair) |
| Same-legal invariance | C | **Unavailable. Count = 0. Not substituted.** |

Primary sample size remains `n_val_pairs = 32` (= 24 A + 8 B). Deterministic pairing, same player-to-move, occupancy/stage matching, no identical-target primary pairs, confirmation `game_id`s excluded.

Smoke-test sampling (`--smoke --n-pairs 4`) is a technical subset of this design. It must include **at least one A pair and one B pair** plus their D controls. Default smoke quotas are 2 A + 2 B + 4 D. Smoke sampling does not change the frozen validation quotas.

## What A/B/D can and cannot establish

A/B/D can test:

1. whether a state-constrained edit moves decoded board state and legal-set behavior on legal-difference pairs (A),
2. whether that also occurs on similar-board legal-change pairs (B),
3. whether those effects exceed matched wrong-target controls (D).

**A/B/D cannot establish same-legal-state invariance.** That estimand required genuine category C pairs: different boards, unchanged legal-move sets. Those pairs were not found. Absence of C is a limitation of the design, not a null finding about invariance.

Do not interpret a pooled A+B success, or a D contrast, as evidence that the editor is specific to board identity rather than legal-set change.

## Reporting rule (frozen)

Report three separate outcome blocks. Do not collapse them into a single headline number.

1. **Category A only** — representation floors, `ΔP_gain`, `ΔP_loss`, `S_legal`, D contrast on the A-matched wrong targets.
2. **Category B only** — the same metrics on the 8 genuine B pairs and their D controls.
3. **Pooled A+B** — descriptive combined summary only, with this caveat:

   *Pooled A+B mixes legal-difference and similar-board pairs. It is not a same-legal invariance test. Category C remains unavailable.*

Editor-selection and hyperparameter rules are not updated from smoke-test results. Pair-selection rules are not updated from any 08R2 model outcomes.

## Editor-failure accounting (frozen)

For Editors A, B, and C, every attempted edit remains in the denominator. Failures are not deleted.

Required counts:

- total attempted edits
- successfully executed edits (`valid_edit = 1`)
- optimization failures
- fit failures (Editor C train-map)
- near-zero but valid edits
- state-target successes
- joint state-and-behavior successes

Required rates:

- **A. Unconditional success rate** = joint state-and-behavior successes / all attempted edits
- **B. Conditional effect size** = mean `S_legal` among valid executed edits

## KL and norm comparison rule (frozen)

Random-baseline KL threshold:

1. Collect same-budget, same-slice `random_same_norm` KLs on valid edits.
2. Compute the mean of those finite KLs.
3. If that mean is missing, the KL gate is `KL_BASELINE_MISSING` and the candidate does not pass the KL gate.
4. If the mean is below `RANDOM_KL_FLOOR = 1e-3`, replace it with `1e-3` and record `RANDOM_KL_NEAR_ZERO_FLOOR_APPLIED`. A near-zero random KL is not a free pass.
5. Threshold = `1.5 × max(mean random KL, 1e-3)`.
6. A candidate is `kl_ok` only if its mean KL is finite and `≤` that threshold.

Also record, for **all** methods, effect-versus-KL (`S_legal / mean KL`) and effect-versus-actual-L2 (`S_legal / mean ‖δ‖`). Donor-access methods, large-KL methods, and methods with different pair coverage may not be declared superior on raw `S_legal` alone.

Smoke-test outputs may compute these comparisons. They may not select the official editor, budget, learning rate, step count, `λ`, rank, or floors.

## Confirmation and later work

The 400-game confirmation manifest remains sealed. This amendment does not authorize confirmation, 08R3, or a change to the official 0A letter.
