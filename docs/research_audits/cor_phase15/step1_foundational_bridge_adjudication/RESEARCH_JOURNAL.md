# Research journal

**Checkpoint 1: sources.** Main has no audits. The audits are on `cursor/overnight-reconstruction-b68c` (`680c5fd`), and the work branch was created there. Read: the Phase 1 core files and the verdicts of Steps 1A, 1B, 1C, 1C-R, 1D, 1E, and the overnight reconstruction. Diagnosis: every step kept a researcher-assigned state (cochains, projections, transports), or one synthetic block with a bound, and ended at generic mathematics. The identification failures (F5, F7, F30) were never removed at the root.

**Checkpoint 2: strategy.** Keep the question and change the object. Use only canonical variables: tokens, writes, attention weights, and the frozen readout. Then ask (i) where in the architecture another source *can* change a source's effect, and (ii) what fixes the sign there. Use measurement theory (conjoint additivity) to give "has a causal role" a metric-free meaning.

**Checkpoint 3: mathematics.** SS, NC, CI, RB, MIX, CORR. Construction D was built in a full block. Tests 21/21, then 22 with NEG. Literature verified by web search. First verdict draft: `CONDITIONAL_BRIDGE_IDENTIFICATION_OPEN`, with A4 open.

**Checkpoint 4: independent review.** Four separate-context subagents (B, C, D, F). Main findings:
- the D crossover is salience reallocation, not agreement (C);
- the "sub-consensus lowers the margin" prediction is false for substitutions (B, C, D);
- A4 is outcome-adjacent (C);
- writer alignment is the norm of the sum (B, C, D);
- RB is generic (C);
- the NC.3 "any path" clause is false (B);
- under RMSNorm, κ ≡ 0 and nothing flips (D);
- CI-5 is Rushing and Nanda's decomposition, and RB.6 is the Greco et al. necessary relation (F).

All were accepted after checking. Two were checked independently: the reallocation split in D (own-change term +0.009 against reallocation −0.489), and the two-flip path.

**Checkpoint 5: revision.** Proofs corrected (NC.3 clause; CI.3 scope; CI.3*; RB.2 scope; prior-work notes). The code was hardened: ties count as crossovers, and `rb_forces` refuses designs larger than 3×3. Six tests were added (26 in total). The verdict was changed to `STEP1_NECESSARY_CONDITIONS_ONLY`, with the earlier verdict's reasoning kept in the cross-examination record.

**Self-assessment.** The first draft repeated the failure pattern in a subtler form. It read an exact identity of ordinary normalization as a "consistency mechanism", and it located the open step at the decision arrow itself. The separate-context review caught this. The final verdict is the reviewers' recommendation, and the author agrees with it on the merits.
