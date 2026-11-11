# P6 mechanism protocol — frozen after the behavioural result, before any confirmation-cohort mechanism data

Frozen in the commit that adds this file. The behavioural result (`logs/confirmatory.json`) was known at freezing: P1(b), P2 (D-R1),
P3 and P4 confirmed. Development basis: `scripts/mechanism.py dev` on T2, P3, T0, T1, cT61, cT62, P0, P1, cP61, cP62 (stream 15_500_000;
logs/mech_dev/). Confirmation units: the 16 Investigation VI parents at transfer step 4000, on **independent episodes, stream
16_000_000** (3000 episodes; decoder fitted on stream 16_200_000, 4000 episodes). Unit of inference: parent model; exact permutation
over C(16,8) and model bootstrap, one-sided T1 > P where directional. Script: `scripts/mechanism.py conf 16000000 …`, analysis
`scripts/p6_analysis.py` (hashes in `logs/FREEZE_HASHES_P6.txt`).

Intervention site: the input to layer 2 (`resid_pre2`) = output of the frozen, curriculum-pretrained layers 0–1. The edit's effect is
patched from the after-episode into the before-episode (definitions in `scripts/mechanism.py`).

| id | prediction | criterion |
|---|---|---|
| P6a | localisation: the decision site (eqv.site_positions) carries the model's 4→5 response | median regression share ≥ 0.8 in each arm |
| P6b | the curriculum difference in the R1 excess is reproduced by site-only patching | B_site-R1: T1 − P > 0, p < .05 and CI lower > 0 |
| P6c | … and by patching only the decoder (count) subspace of the edited group at the site | B_sub-R1: same rule |
| P6d | specificity: F0 (same token, counts unchanged) site patches are small | mean F0 site \|Δm\| < 0.2 × mean F45 \|Δm\| |
| P6e | redundancy: removing the site count-subspace change from the after-episode removes most of the response | median remaining share < 0.5 |

Exploratory (no criterion): decoding accuracy of n^{(−q)} at the site after the edit (T1 vs P); shares of non-site and sub-complement patches.

Interpretation fixed in advance (THEOREMS_AND_COUNTEREXAMPLES.md T5): P6a–e speak to *where* (representation at the site and its
use by layer 2) — conditions 1, 2, 4, 5 of T5(c). Condition 3 (transferring the downstream transformation across curricula) is not
tested: the two arms' residual spaces are not aligned. A confirmed P6 therefore cannot say whether the curriculum difference lies in
how the frozen representation encodes the 4→5 change or in how the transfer-trained layer 2 reads it; partial evidence on that split
comes only from the exploratory decoding comparison.
