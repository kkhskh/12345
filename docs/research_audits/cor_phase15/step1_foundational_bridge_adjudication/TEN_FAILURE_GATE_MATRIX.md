# Ten-failure acceptance matrix (final, after cross-examination)

Columns follow mandate §20: 1 original statement · 2 original proof or counterexample · 3 strongest proposed resolution · 4 exact assumptions · 5 independent? · 6 identified in actual computation? · 7 mathematical evidence · 8 counterexample attacks · 9 baseline comparison · 10 what remains unresolved · 11 final status.

Theorem labels refer to `proofs/PROOFS.md`. "Phase 1" refers to `cor_phase1/`; "1A" refers to `cor_phase15/step1_causal_bridge/`. No row resolves the original `O`/`D` theory.

---

### F1 — No causal bridge
1. **Original.** Consistency scores were never derived from the decision computation.
2. **Original evidence.** Phase 1 `06` §4. 1A Theorems Sign and Descent.
3. **Strongest resolution.** Relations among canonical variables, followed through exact identities. Same-cut relations: inert (SS). Softmax: other sources enter only via reallocation × (DLA_j − DLA_head) (CI.3*). Final norm: temperature plus a pull toward `κ` (NC).
4. **Assumptions.** A1–A3. A4 for the decision.
5. **Independent?** A1–A3 yes. A4 no (outcome-adjacent).
6. **Identified?** Direct and final-layer terms, exactly. The decision arrow, no.
7. **Evidence.** SS, CI, CI.3*, CI-5, NC. Tests.
8. **Attacks.** D′. σ-channel reversals (591/4000). Self-repair.
9. **Baselines.** Head patching and the attention-weight gradient give the same quantity.
10. **Unresolved.** No consistency-specific causal role exists in the investigated classes. MLP-computed agreement and QK-composition are unexamined.
11. **Status.** `UNRESOLVED`. Candidates H1–H3 are rejected as consistency mechanisms. `PROVED_IMPOSSIBLE_IN_SPECIFIED_CLASS` for same-cut relations.

### F2 — Lower D can worsen the decision
1. **Original.** `ΔD < 0 ⇒ Δm > 0` is false.
2. **Original evidence.** Phase 1 Attack B.
3. **Strongest resolution.** Exact boundary: a relation can set the direction of a source's effect only if `m` has a nonremovable (independence-violating) interaction (RB.3). In a softmax head that interaction arises only through attention reallocation (CI.3*). Constructions D and LN realize "agreeing evidence lowers the margin" with designed weights.
4. **Assumptions.** Same as F1.
5. **Independent?** Yes.
6. **Identified?** The direct term at layer 0.
7. **Evidence.** RB.3, CI.3*, D, LN.
8. **Attacks.** Salience held equal removes D's crossover (review C). D′.
9. **Baselines.** Weight-space sensitivity gives the same sign.
10. **Unresolved.** No decision-level directional theorem for any consistency relation.
11. **Status.** `UNRESOLVED` at the decision level. The boundary results are `PROVED`.

### F3 — Consistency rankings do not determine preference
1. **Original.** `D_c < D_{c'} ⇒ L_c > L_{c'}` is unsupported.
2. **Original evidence.** Phase 1 Attacks A and C.
3. **Strongest resolution.** NC is a ranking theorem for norm-only edits within `C`: one `τ`; single crossing in `σ`; a flip iff `aκ < 0` and `σ* > σ_min`; the limit `κ`. It is not a consistency ranking: "alignment" is the norm of the sum. CI.4: unanimity of all attended positions fixes the head's direct preference.
4. **Assumptions.** LN with bias, `|C| < d`, and only the direct term (CI.4).
5. **Independent?** Yes.
6. **Identified?** Yes, from the weights and base state.
7. **Evidence.** NC proofs and tests, including the two-flip path and RMSNorm.
8. **Attacks.** RMSNorm: no flips at all. Tokens outside `C` can overtake. Vacuous for `|V| ≥ d`.
9. **Baselines.** The mechanism is known (Stolfo et al. 2024; Gurnee et al. 2024).
10. **Unresolved.** Any consistency-based ranking.
11. **Status.** `PROVED_IMPOSSIBLE_IN_SPECIFIED_CLASS`: same-cut relations; and RMSNorm norm-channel edits, which cannot change any pairwise order. Consistency-ranking otherwise `UNRESOLVED`.

### F4 — No distinctive prospective prediction
1. **Original.** Nothing distinct from gradients, attribution, or geometry.
2. **Original evidence.** Phase 1 `12`. 1A Q28.
3. **Strongest resolution.** RB.6: frozen training comparisons force held-out orders under the additive null. The consensus-locus prediction was proposed and then withdrawn: it is false for substitutions, and its corrected form equals head patching.
4. **Assumptions.** A3, plus freezing of the decision scalar.
5. **Independent?** Yes.
6. **Identified?** The design is.
7. **Evidence.** RB.6 proof and test.
8. **Attacks.** Random tables are relation-sensitive about 86% of the time, so a violation is uninformative about consistency.
9. **Baselines.** RB.6 is the necessary preference relation (Greco et al. 2008).
10. **Unresolved.** No consistency-specific prediction exists.
11. **Status.** `REJECTED_CANDIDATE` (the consensus-locus prediction). `UNRESOLVED` (the failure stands).

### F5 — Nonidentifiability
1. **Original.** `P`, `ρ`, metrics, and graphs were arbitrary.
2. **Original evidence.** Phase 1 `04`. Sutter et al. 2025.
3. **Strongest resolution.** Canonical variables: no fitted maps. The scalars and centered writes are invariant under function-preserving symmetries. RB is invariant to monotone transforms of a frozen scalar.
4. **Assumptions.** A frozen decision scalar and design ledger.
5. **Independent?** Yes.
6. **Identified?** Yes. Raw writes are not invariant, which is why only the scalars are used.
7. **Evidence.** RB.2, the symmetry argument, review B M3.
8. **Attacks.** Slot and level choices are tunable, so freezing is required.
9. **Baselines.** Interaction contrasts are scale-dependent.
10. **Unresolved.** The identified objects are not consistency relations with their own role (F1). Identification is solved; relevance is not.
11. **Status.** `RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS` (the identification problem itself).

### F6 — The repair objective embeds the output
1. **Original.** Mixed `D − βm` success is not evidence about `D`.
2. **Original evidence.** Phase 1 `06` §3. 2D `β = 2`.
3. **Strongest resolution.** MIX: the gain splits into `βJ^TA^{-1}J ≥ 0` (output) and `−J^TA^{-1}g` (unsigned).
4. **Assumptions.** Local quadratic and linear models, `A ≻ 0`.
5. **Independent?** Yes.
6. **Identified?** n/a.
7. **Evidence.** Proof and test.
8. **Attacks.** —
9. **Baselines.** —
10. **Unresolved.** —
11. **Status.** `RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS`.

### F7 — Outcome-dependent tuning
1. **Original.** Flexible choices could be tuned on outcomes.
2. **Original evidence.** Phase 1 `04` §4.
3. **Strongest resolution.** A four-compartment freezing procedure and a full ledger. The surviving objects have zero fitted parameters but about 12 design choices.
4. **Assumptions.** Preregistration.
5. **Independent?** Yes.
6. **Identified?** n/a.
7. **Evidence.** `OUTCOME_LEAKAGE_AND_FREEZING.md`.
8. **Attacks.** Crossover fishing is easy (86%).
9. **Baselines.** —
10. **Unresolved.** Enforcement.
11. **Status.** `RESOLVED_CONDITIONALLY`.

### F25 — Same consistency, different decisions
1. **Original.** Equal `D`, opposite logits.
2. **Original evidence.** Phase 1 Attack A.
3. **Strongest resolution.** Fiber analysis. Same-cut relations: the decision is constant on sum fibers, and equal relations allow different decisions (SS). The agreement indicator mixes signs (FIB.1). For a layer-0 substitution, the head's numerator sign is fixed by `w(v'−v) + Δw(v'−M)` (CI.3*).
4. **Assumptions.** Layer 0, direct term.
5. **Independent?** Yes.
6. **Identified?** Yes.
7. **Evidence.** Proofs and tests.
8. **Attacks.** The σ channel and D′ mix signs at the decision level.
9. **Baselines.** Head patching.
10. **Unresolved.** Decision-level fibers.
11. **Status.** `PROVED_IMPOSSIBLE_IN_SPECIFIED_CLASS` (same-cut relations; the bare agreement indicator). The direct term is `RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS`.

### F29 — No distinctive advantage over established methods
1. **Original.** The mathematics was already explained by sensitivity or causal tools.
2. **Original evidence.** Phase 1 `12`. 1A, 1C-R, 1D, 1E.
3. **Strongest resolution.** None survives. Every result reduces to established work at equal information: residual linearity, weighted means and influence functions, LN self-repair and entropy neurons, conjoint measurement and robust ordinal regression. Possibly new: the NC flip criterion (minor), and removability as an interpretability criterion (an application).
4. **Assumptions.** —
5. **Independent?** —
6. **Identified?** —
7. **Evidence.** Review F.
8. **Attacks.** —
9. **Baselines.** Equal information was matched in each comparison.
10. **Unresolved.** The failure stands.
11. **Status.** `UNRESOLVED`. Confirmed in the investigated classes.

### F30 — Too many unfrozen choices
1. **Original.** Analytical flexibility.
2. **Original evidence.** Phase 1 `04`, `11`.
3. **Strongest resolution.** Complete ledger. No continuous fitted parameters. Discrete design choices are listed and frozen.
4. **Assumptions.** Freezing.
5. **Independent?** Yes.
6. **Identified?** Yes.
7. **Evidence.** The ledger.
8. **Attacks.** Slot and level fishing.
9. **Baselines.** —
10. **Unresolved.** Enforcement.
11. **Status.** `RESOLVED_CONDITIONALLY`.

---

## Summary

| Row | Final status |
|---|---|
| F1 | UNRESOLVED (candidates rejected; same-cut class PROVED_IMPOSSIBLE) |
| F2 | UNRESOLVED (decision level); boundary results PROVED |
| F3 | PROVED_IMPOSSIBLE_IN_SPECIFIED_CLASS (same-cut; RMSNorm norm channel); otherwise UNRESOLVED |
| F4 | REJECTED_CANDIDATE / UNRESOLVED |
| F5 | RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS |
| F6 | RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS |
| F7 | RESOLVED_CONDITIONALLY |
| F25 | PROVED_IMPOSSIBLE_IN_SPECIFIED_CLASS / direct term RESOLVED_UNDER_IDENTIFIED_ASSUMPTIONS |
| F29 | UNRESOLVED |
| F30 | RESOLVED_CONDITIONALLY |

The methodological failures F5, F6, F7, and F30 are resolved: the project now has a way to define, freeze, and test relations without fitted maps. The substantive failures F1, F2, F4, and F29 are not. Within the investigated classes, the reason is identified: the only relation-sensitivity a residual Transformer has is ordinary normalization (softmax reallocation and the final norm), and it carries no consistency-specific information. Step 1 is not "complete" in the sense of a surviving bridge. It ends with `STEP1_NECESSARY_CONDITIONS_ONLY`.
