# Ten-failure adjudication

Each row lists:

1. the original problem;
2. the strongest known counterexample;
3. the new mechanism relevant to it;
4. the precise surviving claim;
5. its assumptions;
6. whether those assumptions are independently identified;
7. what was proved;
8. what remains unproved;
9. whether the original status changes.

Theorem labels refer to `THEOREMS_AND_COUNTEREXAMPLES.md`.

**Standing preserved:** O/D has no established universal decision implication. Direction 1's Othello results do not validate obstruction. `OBSTRUCTION_READY = NO`. `PHASE_2 = NO GO`.

---

### F1 — Missing causal bridge
1. **Original problem:** consistency scores were never derived from the decision computation.
2. **Strongest known counterexample:** Phase 1 Attack A (same O/D, opposite heads); the previous package's A4 failure.
3. **New mechanism:**
   - A (MLP XNOR);
   - B (QK gate);
   - C (gated amplification);
   - C′ (conditional rereading through attention; depth is necessary, by C′2).
4. **Surviving claim:**
   - A1: in `𝓕_A`, `sign m = sign(xy)`.
   - B1: in `𝓕_B`, agreement raises the agreeing source's weight, with an exact reversal band.
   - C1: in `𝓕_C`, agreement amplifies the net-evidence candidate; for a stated prior interval it decides the candidate at fixed net evidence.
   - C′1: agreement routes to the sources; conflict routes to the prior, whichever source is stronger.
5. **Assumptions:** weight-structure conditions (A1–A4, B1–B4, the `𝓕_C` conditions), including no downstream re-reader.
6. **Independently identified?** Checkable from weights modulo gauges, and only for designed weights. No trained model was examined.
7. **Proved:** implemented-mechanism structural theorems: computation → relation → decision, inside the classes.
8. **Unproved:** that any trained model meets the conditions. Also that the conditions are more than "an agreement gate exists": check 5 **fails** (R2), and the bridge is rejected as explanatory.
9. **Status change:** **partial.** The bridge exists as an *implemented-mechanism existence theorem* in Transformer blocks. That settles architectural possibility, which the previous package wrongly implied was excluded. It is not a discovered decision theory. **UNRESOLVED** as a theory of trained-model decisions.

### F2 — Lower consistency discrepancy can worsen the candidate margin
1. **Original problem:** `ΔD < 0 ⇒ Δm > 0` is false.
2. **Strongest known counterexample:** Phase 1 Attack B.
3. **New mechanism:** A, B, C.
4. **Surviving claim:**
   - In `𝓕_A`, making the sources agree pushes toward `c` *whichever candidate the sources support* (A1.2). So more agreement can move the margin away from the evidence-supported answer.
   - In `𝓕_C`, agreement moves the margin toward the net-evidence candidate.
   - The mirrored circuit (`γ < 0`) does the reverse.
5. **Assumptions:** as in F1.
6. **Independently identified?** As in F1.
7. **Proved:** the direction of the agreement effect is set by the sign of the gate weights (`λ`, `γ`) and the readout. These are architecture parameters, not by consistency itself. Both directions are implementable in the same architecture.
8. **Unproved:** any reason a trained model would choose one sign.
9. **Status change:** **no.** The failure is explained and shown to be unremovable by architecture alone. A directional theorem exists only conditionally on a sign-fixing weight condition. `PROVED_IMPOSSIBLE` without that condition.

### F3 — Candidate consistency ranking need not match logit preference
1. **Original problem:** `D_c < D_{c'} ⇒ L_c > L_{c'}` is unsupported.
2. **Strongest known counterexample:** Phase 1 Attacks A and C.
3. **New mechanism:** B, C, C′ are candidate-relative, but only for a binary pair with scalar evidence (R1 S2: with three or more candidates the gate's relation is not agreement about the preferred candidate). A is candidate-symmetric.
4. **Surviving claim:**
   - B1.3–B1.4: unanimity fixes the preferred candidate; in conflict, A wins iff `|v_A| > k|ŷ|w(−λ|x̂ŷ|)`.
   - C1.3: a prior interval in which agreement decides the candidate.
5. **Assumptions:** pairwise margins on one shared logit scale; the class conditions.
6. **Independently identified?** Pairing and scale come from weights; candidate identity is the token index.
7. **Proved:** pairwise preference implications under explicitly identified aggregation rules (a gate on a bilinear score; a gate on `(Ŝ, q̂)`).
8. **Unproved:**
   - rankings over three or more candidates;
   - any per-candidate consistency *score* whose order matches the logit order. No such score is defined; consistency here is between sources, not per candidate.
9. **Status change:** **restricted.** `RESOLVED_CONDITIONALLY` for pairwise preference inside the designed classes. Unresolved for candidate-consistency ranking as originally posed.

### F4 — No distinctive prospective prediction
1. **Original problem:** nothing distinct from gradients, attribution, or geometry.
2. **Strongest known counterexample:** Phase 1 `12`; the previous package's withdrawn consensus prediction.
3. **New mechanism:** A4 (matched reader-input contrast); B1 reversal band; C1 interval.
4. **Surviving claim:** for a model whose reader is identified, at fixed `Ŝ`:
   - a relation-computing reader changes with `D̂`;
   - a net-evidence reader does not.

   Both are falsifiable.
5. **Assumptions:** an identified reader; a frozen design.
6. **Independently identified?** No in trained models.
7. **Proved:** A4.
8. **Unproved:** distinctiveness. A4 is an interchange test (BASELINES). The other predictions are those of an ordinary gating or composition account.
9. **Status change:** **no.** `UNRESOLVED`. Predictions exist, but none is distinctive.

### F5 — Nonidentifiability of states and maps
1. **Original problem:** `P`, `ρ`, metrics and graphs were arbitrary.
2. **Strongest known counterexample:** Phase 1 Attacks E and F; audit §4 head redistribution.
3. **New mechanism:** —
4. **Surviving claim:**
   - Identifiable modulo the stated gauge groups: input-level effects, reader compositions, summed same-layer writes, and patterns.
   - Not identifiable: per-head and per-unit quantities, Q/K/V vectors, raw residual coordinates, raw scores and patterns, raw preactivations, summed writes, and source polarity, which is fixed only through the outcome pathway (R3).
   - Equivalent implementations give different "consistency" values with identical decisions, in three ways (A2, C3, ID1/2).
5. **Assumptions:** the listed groups; designed weights.
6. **Independently identified?** —
7. **Proved:** ID1, ID2 (ten map types), A2 (no final LN), C3 (exact non-representability; non-identification from finite behavior).
8. **Unproved:**
   - identification of source directions in a trained model;
   - completeness of the gauge list.
9. **Status change:** **the previous package's F5 resolution is withdrawn.** `UNRESOLVED`. The three-level classification and the counterexamples are now correct.

### F6 — Margin reward embedded in a purported consistency-repair objective
1. **Original problem:** the mixed `D − βm` objective succeeds because of the `m` term.
2. **Strongest known counterexample:** Phase 1 `06` §3.
3. **New mechanism:** none uses a repair objective.
4. **Surviving claim:** MIX from the previous package: the gain splits as `βJ^TA^{-1}J ≥ 0` plus an unsigned consistency term.
5. **Assumptions:** local quadratic model, `A ≻ 0`.
6. **Independently identified?** n/a.
7. **Proved:** the decomposition.
8. **Unproved:** —
9. **Status change:** **diagnostic resolved.** It rescues nothing. The audit agrees.

### F7 — Outcome-dependent parameter tuning
1. **Original problem:** flexible choices could be tuned on outcomes.
2. **Strongest known counterexample:** Phase 1 `04` §4; 2D's `β = 2`.
3. **New mechanism:** —
4. **Surviving claim:** this step froze its constructions (`b85222b`) and predictions (`ddf935d`) before the first run, and the C′ predictions (`fdc87d8`) before its code. Original preregistration, rescored: 29 passed, 3 failed, 1 non-discriminating. Addendum: 4 of 5 passed. The failures are kept. Post-hoc analyses are labeled. The C1.3 prior is labeled as demonstrative. R1 notes that a two-minute gap between commits cannot by itself exclude interactive runs; the parameter sweeps show the predictions do not depend on the constants chosen.
5. **Assumptions:** git history as evidence.
6. **Independently identified?** —
7. **Proved:** procedure followed for synthetic work.
8. **Unproved:** that it is followed in any future trained-model study.
9. **Status change:** **procedure demonstrated, not validated empirically.**

### F25 — Equal consistency with opposite decisions
1. **Original problem:** equal `D`, opposite logits.
2. **Strongest known counterexample:** Phase 1 Attack A.
3. **New mechanism:** A, B, C.
4. **Surviving claim:**
   - In A, equal agreement goes with equal decisions (`m(a,b) = m(−a,−b)`), even though the sources support opposite candidates.
   - In B and C, equal agreement goes with *opposite* decisions when the sources support opposite candidates: B's vote sign is `sign b`; C's decision sign is `sign Ŝ`.
   - The ambiguity is resolved by the candidate-directed evidence: the sign of the source vote or of the net evidence. That information is not in the relation.
5. **Assumptions:** the classes.
6. **Independently identified?** As in F1.
7. **Proved:** A1.2, B1.2, C1.1.
8. **Unproved:** —
9. **Status change:** **confirmed and sharpened.** No relation between sources determines a candidate decision without candidate-directed evidence. `PROVED_IMPOSSIBLE` for a sign predictor that uses the relation alone, in these classes.

### F29 — No advantage over established alternatives
1. **Original problem:** the mathematics was already explained by known tools.
2. **Strongest known counterexample:** Phase 1 and 1A–1E; previous package.
3. **New mechanism:** A, B, C.
4. **Surviving claim:** each result has a complete standard explanation (BASELINES table; R4).
5. **Assumptions:** —
6. **Independently identified?** —
7. **Proved:** the reductions.
8. **Unproved:** —
9. **Status change:** **no.** `UNRESOLVED` (confirmed). No distinctive advantage.

### F30 — Excessive unfrozen design choices
1. **Original problem:** analytical flexibility.
2. **Strongest known counterexample:** Phase 1 `04`, `11`.
3. **New mechanism:** —
4. **Surviving claim:** synthetic designs are fully specified and frozen. For any trained-model study, the protocol in `FINAL_SCIENTIFIC_DECISION.md` §5 lists every choice to freeze: sources, levels, candidate pair, read position, reader identification rule, matched statistic and site, decision scalar.
5. **Assumptions:** freezing.
6. **Independently identified?** —
7. **Proved:** —
8. **Unproved:** enforcement.
9. **Status change:** **conditional** (`RESOLVED_CONDITIONALLY`).

---

| Row | Final |
|---|---|
| F1 | partial: implemented-mechanism existence theorems proved; UNRESOLVED as a trained-model theory |
| F2 | not resolved; direction set by gate sign, both signs implementable |
| F3 | RESOLVED_CONDITIONALLY for pairwise preference in designed classes; otherwise unresolved |
| F4 | UNRESOLVED |
| F5 | UNRESOLVED (previous resolution withdrawn) |
| F6 | diagnostic resolved |
| F7 | procedure demonstrated |
| F25 | sharpened; relation-alone sign predictor PROVED_IMPOSSIBLE in these classes |
| F29 | UNRESOLVED (confirmed) |
| F30 | RESOLVED_CONDITIONALLY |
