# Adjudication of the independent agent reviews (Investigation XIII-C)

There were three separately initialised agent reviews (not human peer review), and each could reject the main interpretation. **No frozen
verdict was re-scored.**
- Corrections to the documents are made in place, and each correction is marked.
- The math corrections are also kept as boxes in `logs/NOTES_MATH_*.md` and in DECISION_LAW.
- Experiments run after review 3 are labelled post-freeze exploratory.

## Review 1, mathematics (`REVIEW_1_MATH.md`, scripts in `review1_scripts/`)

| finding | disposition |
|---|---|
| The LOCALQ gap "4.4 points" comes from an unconverged 48-draw Monte-Carlo estimator (median ESS 4.7); the exact value is 3.6 ± 0.07 points, as in XIII-B | **Accepted.** Corrected in NOTES_MATH_C and JOINT C1; the exact script is part of the verification |
| B5 prose contradicts its tables (COMB > CAL in uP161 s22; "nearly all" = 72–79%; "about half" holds for early units only) | **Accepted.** Corrected |
| T3: on-support, ρ never cancels exactly (≥ 2 weighted groups always) | **Accepted** |
| T2 scope: holds for any weights; needs equal source counts and uniform wrong answers; not for γ ≠ 0; "COMB" is equally "q-tempering" | **Accepted** |
| Corollary P: "per-group format is needed" is an implementation hypothesis; supG vs supL2 is not a pure format contrast | **Accepted.** Wording changed |
| "E is the only rule" holds only against A1 and A2; graded rules untested | **Accepted** |
| 8-resample bootstrap; lapse-clip flat regions; NPOE is a weighted geometric mean; b5 log-mean cell model; vacuous J-pair support at β = 0.6 | **Accepted (disclosure)** |
| T1, T2, T4, T5, Proposition S, Corollary P coefficient, β = 1 sufficiency, edit families, b5 cell sufficiency | **Confirmed by the reviewer** |

## Review 2, experimental integrity (`REVIEW_2_INTEGRITY.md`)

| finding | disposition |
|---|---|
| Chronology: population before training; freeze excludes confirmation outputs; atime shows no pre-freeze reads; hashes unchanged; preflight reproduces | **Confirmed by the reviewer** |
| R3 can hide a failing correlation arm if no causal arm is diagnostic | **Accepted (disclosure).** Did not occur: all three causal arms were diagnostic |
| R4 fail-predicted arms would pass if training did nothing | **Accepted.** Checked: all trained (accuracy 0.47 → 0.72–0.73) |
| R5b does not check decodability; R6 missing OTHER defaults to 0; malformed inputs crash; non-diagnostic stored as passed=True; stricter-than-text diagnostic gates; "pure mixture" defined differently across R4/5/6/8 | **Accepted (disclosure).** Decodability is reported (R² 0.70–0.82); OTHER was present; no crash occurred |
| Loose git blobs from staging and unstaging confirmation files during the freeze commit | **Accepted (disclosure).** No amend; reflog clean |
| The full 4000-step determinism replay was descoped; the bitwise claim rests on 50 steps (+ resume) | **Accepted.** CAUSES §A1 wording is limited accordingly |
| Statistical wording ("data rejected", "do not overlap", 12 non-independent points) | **Accepted.** Rewritten; see review 3 M2 |
| Lane names C1/C2 vs unit IDs; a misattributed descoping reason; "no development combination" untrue for C7–C9 without the intervention; development streams also 76_/87_ | **Accepted (disclosure)** |
| The `unit_metrics_conf` read at 12:02 | That read was post-freeze inspection by the authors (allowed); recorded here |
| Residual risk: any later descoping of a confirmation job would count as a fail | Noted. **C8 was silently skipped by a claim-key collision** with a descoped development line; it was detected at 12:32, logged and run. No confirmation job was descoped |

## Review 3, mechanism and causality (`REVIEW_3_MECH.md`)

| finding | disposition |
|---|---|
| **M1:** "the readout map carries the init effect" is not established. In the split cells WA and data share a seed, so WA × data or "majority of seeds" fits as well. It is a post-hoc reading of a falsified prediction | **Accepted.** Downgraded to "WA and/or data (jointly), not the layer-2 init". The discriminating run uP163 L21/W22/D21 was run post-freeze; result in §Post-review experiments |
| **M2:** "data effect ≤ 0.02" is development-only; in confirmation data moves the hybrid residue | **Accepted.** Rewritten: the binary head event follows the initialisation; the continuous residue is partly data-sensitive |
| **M3:** the uP161 l2q run is under-trained; lik ≥ 0.4 is the default early state | **Accepted.** That arm is excluded from interpretation of R2b and R3; R2b rests on the competent uP160 run |
| **M4:** ρ_KL mostly indexes distance from Bayes (corr −0.93 with log KL; partial corr lik–ρ given KL −0.49); with lapse the uP160 l2q shift is −0.07 and corr −0.64 | **Accepted, reproduced:** PML shifts +0.118 / −0.071; corr(ρ, log KL) −0.93; partial −0.49. R3 stays CONFIRMED as frozen, but its meaning is narrowed to "organisation changes move the fitted exponent together with overall accuracy". It is not evidence that routing specifically changes the combination law |
| **M5:** the format claim lacks supL2 at the larger budgets; "optimisation excluded because the same optimiser" is contradicted | **Accepted.** That sentence is removed. supL2 at 3000 steps and at lr 3e-3 run post-freeze; result in §Post-review experiments |
| **M6:** R7 at ALL2 reproduces the model's own E margins; the site includes the edited tokens; the source-free excess is only part | **Accepted.** R7 is described as "the model's E-following behaviour is carried by a low-dimensional subspace of the full layer-2 input, beyond source-free and A2-decoy controls". It is not a localized mediator |
| R2 shows path dependence and lock-in after release, not active competition in unmasked runs; a release-time sweep is needed | **Accepted.** Wording changed to "path-dependent early race; masking one side changes which state locks in" |
| R5: probes are decoder directions on pooled tokens; encoding directions, decode-from-DAS, ablation of DAS and null-pair tests are missing | **Accepted (open).** R5 narrowed to "the decodable linear decoder directions are not the used directions" |
| R8 additivity is expected under local linearity; the informative part is the class difference | **Accepted** |
| SHUFD should also be scored against the shuffled pair's own change | **Accepted (open)** |
| The D2 text "exactly 0.0 in every unit" | **Corrected** (≤ 1e-6 on the row path; exact on the DAS path) |
| Deterministic training: no perturbation-sensitivity estimate | **Accepted (disclosure)** |

**Unresolved disagreements.**
- Whether ρ is a decision law or a deviation-from-Bayes index. The authors consider both readings compatible with the data; M4 favours the
  latter.
- Competition versus lock-in.
- The readout-map claim and the format claim: see the post-review results below.

## Post-review experiments (post-freeze, exploratory)

See `logs/expl_post_review.md`.
- **M1:** uP163 L21/W22/D21 followed the WA seed (distance 0.08) against both the L2 and the data seed (0.86). The readout-map claim is supported in this parent; it is one exploratory run.
- **M5:** supL2 succeeds at 3000 steps (0.931) and at lr 3e-3 (0.902). The **format claim is withdrawn**; availability at the layer-2 input decides, and plain L2 fails under every setting.
