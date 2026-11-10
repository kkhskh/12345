# Review R-stat: statistical support, experimental design, prospective-prediction integrity

Reviewer: independent adversarial reviewer (did not produce the package). Scope: freeze integrity, verdict recomputation,
the VI bootstrap, threshold/calibration design, Outcome classification, power. Checks: `reviews/checks/r_stat_verdicts.py`,
`r_stat_vi.py`, `r_stat_bayes.py`; outputs `logs/review_checks/r_stat_*.out` (all < 1 min CPU total).
Severity: **Major** (changes what a headline may claim), **Moderate** (claim needs qualification), **Minor** (bookkeeping).

## Findings

**1. Freeze integrity holds. (No defect.)**
- `git diff 579a7a2 8c6363e -- FROZEN_PREDICTIONS.md` is a pure append starting after the `# OUTCOMES` line; no frozen line changed.
- Freeze commit 05:47:05 UTC. First confirmation artefact `logs/conf/episodes.npz` 05:47:17; COMPUTE_LOG "conf queue start
  (after freeze commit)" 05:47:15; all `logs/conf/*` 06:09–06:39; `logs/site/site_eval.json` 06:29. The freeze commit contains
  no conf/site output. S6/S7 checkpoints were still training at freeze (as disclosed); none was evaluated before 06:29.
- All 8 hashes in `logs/freeze_hashes.txt` match the current files; `git diff 579a7a2 HEAD` on `scripts/`, `src/` shows only
  the addition of `scripts/transport_posthoc.py` (correctly labelled post hoc, run 06:39–06:49, after outcomes; not used in any verdict).
  `site_lane.sh` is not in the hash list but is in the freeze commit and unchanged.
- Calibration/confirmation separation verified in the json and in the `SETS` tables of `equiv_run.py`/`necessity.py`:
  cal = P0,P1,T0,T1 on streams 12.10M/12.20M/12.25M; conf = P2–P4,T2,T10,T11 on 12.30M/12.40M/12.45M. Disjoint.
- Minor bookkeeping: (a) `logs/cal/episodes.npz` (created 05:31, pre-freeze) was committed only in 8c6363e; (b) the docstring of
  `analyze.py` says "2000 replicates"; the code and the frozen text say 300; (c) `responses.npz` stores the *Bayes residual*
  for `random_support` (equiv_run.py modifies `rr` in place before saving), undocumented; raw responses are recoverable as
  residual + beta·dB (done in r_stat_bayes.py, reproduces r_with_bayes exactly). **Correction:** fix docstring; note (c) in README.

**2. All eleven OUTCOMES rows reproduce from the json. (No defect in arithmetic.)** See verdict table. Additional facts the
table omits:
- II(a) fails on both fidelity metrics (r-based: P donors |diff| 0.17–0.42); only P3-single-IIA passes (0.097, by 0.003).
- II(b) was tested only on T donors (no P donor reached decoder 0.95). Thus the Outcome-C trigger "II(b) fails" could not be
  fired by any P-count donor in confirmation; II(b)'s pass carries no information about P→T transport. **Moderate.**
- I(b) T10: one token alone gives adjusted 0.55, i.e. it would pass I(a)'s sufficiency threshold by itself. I(b)'s Δ criterion
  passes (0.43), but the frozen gloss "one token is not sufficient" is false for T10. **Minor; correct the gloss.**
- V: the only S model meeting the S part is S6, whose source denominator is 0.008 (adjusted 4.39 is meaningless). On raw
  IIA − floor, readout ≥ source in S6 and S7. FAILED is correct; "partly not evaluable" is fair.

**3. Outcome C is correctly applied, but the explanation of *why* is incomplete and the "read without that rule" section
re-labels in substance. Major.**
- Rule check: IV(a) failed (P4 marg 0.677 < 0.70; T11 rotated 0.386 ≥ 0.30) ⇒ C. Correct.
- But A and B both require "I, **II**, IV(a) pass". II(a) failed for every P donor. So even with IV(a) passing, the
  pre-registered classification would have been "other combination", **not B**. CAUSAL_EQUIVALENCE_RESULT.md says C holds
  "because IV(a) failed (narrowly, in two of six models)", which invites the reading "B but for a near-miss". That is wrong.
- The Direct answer ("same variable, different downstream computation") is exactly Outcome B's description (I, IV(a), VI-sep).
  Presenting it as what the evidence shows "read without that rule" re-labels a pre-registered failure (of II(a) *and* IV(a))
  as the partial-abstraction success. Individual items are graded [F] honestly, but the synthesis is not graded.
- **Correction:** state that A/B were excluded by II(a) independently of IV(a); grade the Direct-answer synthesis [E]
  (post-hoc synthesis over failed and passed criteria); drop "narrowly" from the classification sentence.

**4. "Narrow" is applied asymmetrically; per-model thresholds carry no uncertainty. Moderate.**
- T11 rotated 0.386 vs < 0.30 is 29 % over threshold — not narrow (COUNTEREXAMPLES lists it as narrow).
- Narrow *passes* are not flagged: P2 marg 0.7006 (passes by 0.0006, closer than P4's failure), P3 II(a)-single 0.097,
  T2 I(a) with adjusted-scale SE ≈ 0.11 (denominator 0.164).
- Uncertainty I could estimate: VI-bayes P2 r 0.486, episode CI [0.438, 0.530], P(r<0.5) 0.73; P4 restoration 0.785,
  rough SE ≈ 0.05. KL-reduction SEs cannot be computed (no per-decision data logged). P2's pass and P4's failure on IV(a) are
  plausibly indistinguishable.
- **Correction:** list narrow passes next to narrow failures; say explicitly that pass/fail at these margins is within
  sampling noise in both directions; log per-decision KL/accuracy so SEs can be given.

**5. VI bootstrap: valid for the six fixed models, not for curricula. Moderate.**
- 300 replicates is adequate here: Monte-Carlo sd of the 97.5 % bound 0.003 (vs 0.15 margin to −0.10); 1000-replicate CI
  [−0.315, −0.247] matches the logged [−0.316, −0.249]. Per-pair r SE 0.023–0.032 (frozen claim "0.02–0.03" confirmed).
- Pairs share models, so the 15 pair-r's are not independent; the episode bootstrap handles episode dependence across pairs
  correctly (same resample for all pairs) but treats models as fixed.
- Model-level checks (r_stat_vi.out, π-preserving abstract): exhaustive curriculum-label permutation (20 splits): the true
  split is the most extreme, p = 1/20 = 0.05 — the minimum attainable with 3+3 models (symmetric statistic: 2/20).
  Two-way bootstrap (models within curriculum × episodes): cross − within_T CI [−0.412, **−0.102**] — the upper bound sits
  on the threshold. Leave-one-model-out: −0.275 to −0.350, except drop T11 → −0.149.
- **T2 is not the driver**: dropping T2 strengthens the contrast (−0.350); min within-T pair (T2~T10 0.243) > max cross pair
  (0.182). T2's prior exposure (one number) does not affect VI.
- **Correction:** keep VI-sep as frozen; add the permutation p and two-way interval; replace "curriculum-specific" and
  "H1 downstream: rejected" with "on these six models"; the evidence for a curriculum-level difference is p = 0.05 at best.

**6. VI-sep cannot distinguish H3 from "P seeds idiosyncratic". Minor (already partly disclosed).**
- within_P < cross in every family (π-abstract −0.074 vs 0.096). The frozen statistic only contrasts cross with within_T, so
  VI-sep is satisfied by "T coherent, P incoherent" — not two curriculum-specific maps. CROSS_MODEL_CAUSAL_TRANSPORT.md §3
  says this; the Direct answer's "curriculum- and seed-specific" is acceptable. An encoder artefact (P decoders 0.72–0.78)
  is largely excluded because π-preserving *input* edits, which use no encoder, show the same pattern (within_P 0.029,
  within_T 0.532).

**7. Secondary VI families are fragile; do not cite them as corroboration. Minor.**
- random_support: separation is entirely the T10~T11 pair (0.613); dropping T10 → +0.024, T11 → +0.005; permutation p 0.20;
  two-way CI [−0.476, 0.070].
- off_natural_support: a mixed split (P3,T10,T11) is more extreme than the true labels; p 0.10; P3~T10 0.688 exceeds both
  T2 within-T pairs. **Correction:** report these with CIs and the LOMO/permutation results, or drop them from the reading.

**8. Thresholds were not tuned to favour passes overall, with two exceptions. Minor/informational.**
- Risk-taking: II(a) (≤ 0.10) was frozen although calibration P1 already failed it (cond IIA |diff| 0.117). VI-sep threshold
  −0.10 vs calibration −0.685 is not edge-tuned. I(a) 0.5 vs cal 0.61–1.02, IV(a) marg 0.70 vs cal 0.77–0.89.
- Edge/low-information: rotated < 0.30 set just above cal max 0.268 (no margin; failed). IV(b)-T "< 0.8 in ≥ 2/3" predicts
  the calibration failure of the operator (cal 0.31, 0.60) — it could hardly fail and adds little evidence.

**9. Power statement correct but incomplete. Moderate.**
- Episode-level: SE of the group contrast 0.011–0.019, so 0.10 is detectable (MDE ≈ 0.05) — as frozen, for fixed models.
- Model-level: no α < 0.05 test is possible with 3 + 3 models (Finding 5). The frozen text says conclusions are about these
  six models; documents should keep to that.
- No power/uncertainty analysis exists for the per-model threshold criteria (I–V), and floor-adjusted IIA has small
  denominators (T2 single 0.164; S arm 0.008–0.178; P source sites 0.024–0.034), making adjusted values ±0.1 to unbounded.
  T2 cond adjusted 1.82 (encoder > "ceiling") shows the ceiling is not a ceiling; the document excludes T2 from its cond
  range and says so. **Correction:** report raw IIA − floor with binomial SEs whenever the denominator is < 0.2.

## Verdict table

| prediction | package verdict | recomputed | review |
|---|---|---|---|
| I(a) | PASSED | 0.95 / 0.98 / 1.03 | confirmed (T2 adj. SE ≈ 0.11) |
| I(b) | PASSED | Δ 0.99 / 0.43 / 1.13 | confirmed; T10 one token alone 0.55 (gloss false) |
| I(c) | PASSED | 0.95 / 0.84 / 0.84 | confirmed |
| II(a) | FAILED | P donors 0.19 / 0.18 / 0.33 (cond IIA) | confirmed; also fails on r metric |
| II(b) | PASSED | T donors ≥ 0.984 | confirmed; vacuous for P donors |
| III | PASSED | count 0.008–0.095; full 0.33–0.62 | confirmed |
| IV(a) | FAILED | P4 0.677; T11 rot 0.386 | confirmed; P2 passes by 0.0006; T11 not narrow |
| IV(b) | FAILED | P4 0.785 | confirmed; within ≈ 1 SE |
| V | FAILED | S: 1/3 (S6 degenerate) | confirmed |
| VI-sep | VI-sep | −0.281 [−0.316, −0.249] | confirmed for these six models; model-level p = 0.05, two-way upper −0.102 |
| VI-bayes | FAILED | P2 0.486 [0.438, 0.530] | confirmed; within noise |
| Classification | C | C (IV(a)) | correct; A/B also excluded by II(a) — document must say so (Finding 3) |

**Overall:** prospective integrity is sound and every verdict reproduces. Required corrections: Finding 3 (Major) and
Findings 2, 4, 5, 9 (Moderate) — all wording/reporting changes; no frozen verdict changes.
