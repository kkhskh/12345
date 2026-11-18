# Adjudication of the three Investigation XVI reviews

The three reviews are agent reviews, not human peer review:
- A: mathematics, 14 findings;
- B: learning dynamics, 11 findings;
- C: integrity, 11 findings.

**The frozen verdicts are not re-scored.** P1 BROAD, P2 CONT_BROAD, P3 INCONCLUSIVE and P5 STAGE_ROBUST stand as returned. All three reviewers independently confirmed the scorer output: C byte for byte, and B by recomputing 18/18 acquisition times. Corrections change interpretations, labels and disclosures.

**Re-checks I ran myself before accepting:**
- B1: C_s202's pooled F_G is 0.502 at every point from 400 to 2000 except one (0.499).
- B2: the 31 null z-scores have mean −0.10 and SD 0.55.

## Accepted: withdrawals and corrections of substance

**W1 (B2, major). "No first-order push toward the gate on the plateau" is withdrawn, and so is the inference that escape must be noise-driven or higher-order.**
- U is unbiased, but its jackknife SE is conservative: the null z SD is 0.55, not 1.
- Its plateau detection limit is ‖μ‖ ≈ 0.001–0.004, about 1–3% of the full expected gradient.
- Two pre-onset values I called "not detectable" exceed every null value: C_s204 @2000, z = 1.89, and s404 @3000, z = 1.18.
- Growth of 64× in 250 steps would make an ordinary first-order escape look like zero at earlier checkpoints.
- **Corrected statement:** on every plateau measured, the gate-attributable expected gradient is below a detection limit of about 1–3% of the expected gradient. It is clearly present (z ≥ 4) from 150–250 steps before onset in 3 of 5 transitions, and it grows by one to two orders of magnitude through the transition. The data do not distinguish first-order saddle escape from noise-driven or higher-order escape.
- **Planted control:** it tests only the arithmetic, not the estimator's sensitivity (B2), so it is relabelled as an arithmetic check.

**W2 (B3, major). "A finite-budget barrier persists in D, selectively (Explanation B)" is withdrawn.**
- An escape rate estimated from the D runs themselves (about 1 per 10,000–12,000 plateau steps) gives a 45–51% chance that both remaining D continuations stay censored for 4000 steps.
- Their plateau motion looks like that of C models before escape.
- **Corrected statement:** the two D continuations are right-censored at total exposure 5.9. They are consistent with late learners, and the data cannot distinguish a slow escape from a barrier.
- The Workstream A classification becomes: **condition C, A1 (extended learning succeeds broadly: 7/7); condition D, A4 (inconclusive, 1/3 with censoring)**.

**W3 (B4, A9, major/moderate). "Nothing indicates a restart effect beyond restored exposure" is withdrawn.**
- The frozen FAST_UNLOCK descriptor (a restart signature) fired, but only at the 0.5 threshold: it is false at F_G ≥ 0.51 (B1).
- On exposure since the restart, the restarted models were faster than the fresh ones (descriptive p = 0.11, n = 3 against 4).
- The frozen matched-exposure call (CONT_LOWER, Fisher p = 0.47) was not reported in the synthesis.
- **Corrected statement:** restart and duration effects cannot be separated with these data. The restart is also confounded with an Adam moment reset, which leaves a visible signature in the update norms.

**W4 (A1, A2, major). The competition-driven reading of P3 is withdrawn.**
- The competing cue is matched in its **law** and its **log-loss benefit** (0.66% against the order-2 rule, 1.75% against the exact mixture), but **not at the decision level**.
- The agreement rule changes 18.9% of decisions and gains 0.037 accuracy in the low condition, against 0% and 0 in the high condition. Its benefit relative to the gate is 0.94 against 0.54.
- Decision-level competition, relative competition and the difficulty difference all predict that the low-value condition is slower. So the frozen mapping (NO_EFFECT means competition-driven) did not follow from the accounts.
- **INCONCLUSIVE stands.** It is robust to 9 threshold variants (A2).
- **What P3 can say:** no consistent timing difference was detected between conditions that differ in gate value, decision-level competition and difficulty all at once. With 4 pairs, the 95% t-interval on the mean ratio is 0.54–2.64 (B7): **no effect size can be excluded.**
- The cross-investigation "competition fits better" paragraph is withdrawn.

**W5 (B1, A4, major/moderate). Slot-local acquisition.**
- The model reads out 6 question slots. F_G pools them, and the slots acquire the gate separately.
- C_s202's frozen t_A = 400 rests on F_G = 0.50187 against 0.5, about 6 of 3471 flips: 3 of 6 slots had the gate from step about 350, and the other 3 acquired it at about 2000.
- Its "regression" (11 flips) and its "1850-step gradual transition" are artefacts of pooling over slots. It is two sharp jumps.
- Per slot, every observed acquisition is sharp. F_G widths are lower bounds; G-based widths are 175–450 steps (A4).
- **Corrected statements:**
  - "15 of 16 abrupt, one gradual with a regression" becomes "acquisition is slot-local and abrupt per slot; one model (C_s202) acquired the gate in half its slots about 1600 steps before the other half".
  - The P2 count (4/6) is robust at every threshold; FAST_UNLOCK is not (W3).
  - This is reported as a new exploratory observation, the most informative mechanistic finding: the six slots share parameters yet acquire the gate separately [expl, B1].

**W6 (B5, B6, moderate). Probe timing.**
- The probes pooled over slots. With the slot fixed, gate information and causal use are already present at WB_high_s501 step 600, not 700.
- The "co-emergence" conclusion survives and is stronger slot by slot (B5), but the reported onset checkpoints shift.
- The held-out-x₀ artefact also appears at arity 3: C_s201 at the gate tokens before onset reads 0.44 and 0.38. "Arity 3 shows no artefact" is withdrawn.

**W7 (C1, major). Disclosure of access to confirmation files.**
- File access times show that 10 of 18 trajectory files were opened at about 19:42, 23:19, 01:08 and 03:04 UTC, before scoring.
- These were my progress checks. They ran `wc -l < traj.jsonl` (line counts only) and printed nothing but the number of lines; the commands are recorded in this session's command history.
- The reports' statement "no confirmation file was opened" is false and is corrected to "trajectory files were read only by line-counting `wc -l` for progress; no content was displayed".
- No frozen file changed afterwards (C).
- Lesson for future freezes: monitor progress through the training process (`done.json`, snapshot files), not through outcome files.

**W8 (C2, A9, moderate). The frozen record holds two P2 expectations.**
- The protocol says CONT_SELECTIVE, revised before the freeze after the development evidence. The hashed scorer docstring still says CONT_BROAD.
- Both are disclosed. Verdict rules are unaffected. Against the scorer's text the expectation was met; against the protocol's it was not.

## Accepted: moderate and minor corrections

- **A3:** the floor-conditioned stage certifies that "the gate is not yet decision-visible" (F_G ≤ 0.1), not that "the gate is not used". In C, D and WB-high, agreement use is visible only through the graded, confidence-normalised index. P5 survives a graded-gate floor (A).
- **A5:** t_first is not specific: gate-blind rules with stray sensitivity reach F_G of 0.2–0.34. t_A is protected by its KL condition.
- **A6, A7:** the references are named exactly (order-2 optimum); A_conf(reference) = 1.006 and 1.116 at arity 2.
- **A8:** the rejection of a decision-level agreement measure holds for the order-a and order-1 optima on the diagnostic set in the arity-3 conditions and WB-high. In WB-low, 54 of 162 swaps would have allowed one.
- **A10, A11, A13:**
  - Proposition 2's upper-bound wording is corrected (it uses g ∈ {0, 1}).
  - The lower bound, Proposition 3 and the "structural" difficulty confound are upgradable to proofs (A gives the arguments).
  - V5 and V6 are identities, relabelled.
  - Reviewer A verified the minimal order a + 1 also at the decision level by an LP.
- **A12:** the inert-continuation proof is correct, but a real no-restart control is still missing.
- **A14, B (exposure):** the 2–9× complexity figure compares a cosine with a constant schedule. It is relabelled as a cross-schedule exploratory comparison.
- **B7:** P3 uncertainty is restated (W4). **B8:** the XV-B "stage skipped" correction depends on a KL floor margin of a few thousandths of a nat in the weak-cue condition, so it is downgraded to "not established either way".
- **C3:** the preflight covered 8 of 13 verdict branches, not "all eight". The actual P3 outcome (INCONCLUSIVE) was not among them; Review C checked that branch by hand.
- **C4–C9:**
  - the second container restart (about 04:06Z) is added to the deviations; the first came after 01:04:32Z;
  - the development stop time is 16:19:23, and C_const_s150 ran to 2050;
  - no memory log is archived (the top readings are anecdotal);
  - resumed runs' recorded seconds exclude pre-restart time; WB wall times are 45–49 min;
  - the optimizer and confidence table in EXTENDED §5–6 is regenerated by `scripts/xvi_optimizer_summary.py`, and its numbers are replaced by that script's output;
  - the learning-signal estimates were made on 27 checkpoints (not 25); C_s201's frozen first response after the restart is 1200.
- **C10:** the arity-4 pilot condition depended on my decision to end the session. Disclosed as such.
- **C11:** the protocol file is not in the hash list; it is covered by the pushed freeze commit (C verified that it is unchanged since `737393b`).

## Rejected

None.
