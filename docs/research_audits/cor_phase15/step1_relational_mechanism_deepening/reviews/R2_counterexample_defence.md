# R2 review — counterexample construction and positive-theory defence (corrected claims only)

Scope: only text marked [R: …] and the claims it changed. Independent reviewer; I did not produce the work.
New check: `reviews/checks/r2_null.py` → `logs/review_checks/r2_null_ref_T1p1.out` (ref_T1p1, 600 eval episodes,
2000 fit episodes, same seeds/streams/site/pairs as `src_enc.py`; 1393 informative pairs, identical to the adjudicator's
re-run; 12 s CPU). Freeze integrity checked with `git diff --numstat` against the freeze commits.

## Null controls run for this review (ref_T1p1, per-slot = 36 separate linear encoders at resid_pre2)

| variant | IIA (n = 1393) |
|---|---|
| per-slot leave-q counts (reference; reproduces 0.805 → 0.799) | **0.799** |
| per-slot, encoder fitted to counts of a **permuted episode pairing** | 0.073 |
| per-slot, counts under the **wrong source pairing** (0,2),(1,4),(3,5) (same capacity, fit and apply) | 0.075 |
| per-slot, **model's own answer label** one-hot (maximally answer-correlated feature) | 0.080 |
| per-slot, **argmax-group** one-hot (coarse 3-valued function of the right counts) | 0.452 |
| true encoder, counterfactual counts taken from a **random other pair** (wrong target) | **0.459** vs true cf (0.581 any flip) |
| full-source patch | 0.833 |
| geometry, per-slot delta vs true patch at the 36 source tokens | norm ratio 0.42, cosine 0.38 (medians 0.43 / 0.39) |

## Findings

**R2-1 (major; positive defence of CA1, partly rebutting the "too expressive" worry).** A per-slot encoder is *not*
able to align arbitrary features. With identical capacity (36 × 21-or-fewer features), the wrong-pairing counts, a
permuted-pairing fit and — most tellingly — the model's own answer label all give 0.07–0.08. The answer-label null shows
the per-slot class cannot simply "steer toward the target answer" from any answer-correlated input; the R1 reviewer's
P-count control (per-slot at P-count's source site 0.22/0.24 ≈ that site's own patch level) is a second capacity null.
The delta the count encoder writes is small (0.42 of the true patch norm, cosine 0.38) and still reproduces 0.80. So
"T1's source tokens carry a low-dimensional, causally effective, count-indexed code" is legitimate and stronger than the
corrected text says (§3 of the synthesis still states "no permuted-count null encoder run"; it now has one).

**R2-2 (major; the IIA scale has a high floor).** Writing the true encoder with the *wrong* counterfactual counts already
matches the true counterfactual answer on 0.459 of informative pairs. Reason: own reports X_q are fixed under `cond`
pairs, so any intervention that moves the belief off the currently favoured group lands on one of few alternative
answers, usually the right one (79 % of flips). The coarse argmax-group variable (0.452) is at this floor, i.e. carries no
alignment beyond "move belief somewhere". The meaningful quantity is excess over this floor: T1p1 (0.799 − 0.459) /
(0.833 − 0.459) = 0.91 of the patch's excess — strong — but raw "IIA 0.66–0.82" and "93–96 % of the full-site
interchange" overstate the absolute level. The same floor has not been measured for P-count's 0.60–0.72 or for ref_T1p2;
cross-model IIA comparisons (P-count vs T1) are not interpretable without it.

**R2-3 (major; ref_T1p2's per-slot 0.657 > its own full-source patch 0.563).** The "full-source patch" is not a ceiling
for an *additive fitted* intervention. Writing the regression-mean count direction can override conflicting
counterfactual-irrelevant content (and the unpatched non-source positions) more effectively than the model's own
counterfactual activations do. Implications: (i) for p2 the IIA measures *steerability* along a count direction, not the
fraction of the model's mechanism the variable captures; the recovery-ratio language (93–96 %) gives 117 % for p2 and
must not be used there; (ii) only 0.563 of p2's counterfactual behaviour is carried by its source tokens at all, so "T1
stores the variable at the source tokens" is a partial-site statement for p2; (iii) "causally used" for p2 rests on
sufficiency of an injected direction plus the (T1p1/T1 s0 only) necessity result. The OPEN map lists this as open; the
synthesis and PCOUNT A3 quote the 0.56 without comment. Severity: major for the "3 parents" count.

**R2-4 (major; "the same abstract variable" over-identifies).** The per-slot data do not distinguish the claimed
variable from coarser or different count functions: own-group count only 0.795 vs full 3-group vector 0.805 (T1p1),
0.650 vs 0.657 (T1p2), 0.765 vs 0.821 (T1 s0, R1); full-episode counts ≈ leave-q on T1p1 (0.807); a non-additive lookup
is equivalent. A counter-model consistent with all T1 evidence: each source token stores only *its own group's* leave-q
agreement count (a per-source reliability tag — the pre-review "per-source reliability" story of Answer 4), while P-count
stores all three counts at a readout in a position-shared code. That is a projection of P-count's variable, not the same
variable. Defensible wording: "both curricula carry causally effective group-agreement counts (in T1 mainly the token's
own group's count, slot-specific; in P-count all three groups, position-shared, at the readout)". MAIN Q23 "site (not
code)" over-corrects: the code also differs (shared vs slot-specific; P-count's per-question encoder adds nothing, T1's
shared encoder fails). "The curricula are interchangeable at the level of this variable" (ADJUDICATION §4.1) is not tested
— no cross-model interchange was run.

**R2-5 (major; stale pre-review text left uncorrected, contradicting the headline).** Not struck or marked:
PCOUNT A4 ("Necessity by mean-replacement is not specific … no necessity claim"), PCOUNT A5 ("T1's is not [effective] at
either site … do **not** implement the same intermediate variable"), CAUSAL_IDENTIFICATION §4 bullet 2 ("T1's source
tokens do **not** carry counts … **not** interchangeable"; only the file header mentions the reversal),
RELATIONAL_COMPUTATIONAL_CLASSES l. 45–47 ("non-count code (0.04–0.08 …)"). THEOREMS T3 body (l. 30–32: "with exactly
one of either, the composite is restricted to 2 candidates") and "Consequence for learning" (l. 38–39: "must be at
chance") and T1a/T2 (no product-prior qualifier) are corrected only in the header; the bodies still state the falsified
versions (FM2, FM3).

**R2-6 (minor; numeric).** PCOUNT A3: "the slot's own-group count alone gives 0.77–0.81" — logs give 0.765 (T1 s0),
0.795 (p1), **0.650** (p2): range 0.65–0.80. The "IIA 0.66–0.82 (3 parents)" mixes p1/p2 (adjudicator, 600 episodes) with
T1 s0 (R1 reviewer, 1000 episodes, ~2300 pairs, not re-run by the adjudicator).

**R2-7 (moderate; CA3 necessity over-corrected).** Count-subspace ablation at T1 source removes 1.65–1.74 of the relational
gain (pooled-mean version 1.77 — the per-position fix barely changed the count number; it changed the control). A drop
> 1 means the model falls below own-question processing: collateral, off-manifold damage, so "specific vs controls" ≠
"necessity of the count variable". R1 itself noted P-count seeds 1–4 at 1.08–1.37. The adjudicator's verification was on
ref_T1p1 only; the P-count readout figure (0.63 vs 0.32, seed 0) is unreplicated. Suggest: "specific relative to
per-position controls; magnitude > 1 indicates additional damage; necessity not separated from it".

**R2-8 (moderate; GATE correction lacks its operator).** MAIN l. 23 "≥ 86 % of the gate effect survives ablating both"
is a mean-ablation result. Under resample ablation the same pairs give 0.374 (s0, R1 §4) / 0.391 (s1, `gate_chk_s1.json`), far *below*
the gate-blind 0.890 — because these heads carry evidence values (R→E attention 0.93–0.97), not only the gate. Neither
operator isolates gate necessity; the accurate statement is "under mean ablation the pairs contribute (8–14 %); under
resample ablation they are necessary for the answer, not demonstrably for the gate". The GATE header "reverses
single-head results" is right only for s0 L1-h2 (1.000 → 0.623); s1 single heads stay 0.98–0.99. Otherwise the GATE
corrections (CA4–CA6, CA8, CA9) are accurate; the "abstract structure strengthened" part is well supported.

**R2-9 (moderate; Q3-D correction is asymmetric).** SD7's parent-sharing argument was applied to SAT only, but it
applies to every within-TF predictor on the same 6 models. Q3-D3 (A2, ρ 1.00) ties the parent lookup (ρ 1.00) and is
still labelled [C]. Defence: A2 beats the lookup in MAE (0.019 vs 0.042), and A2 is measured on the held-out model itself,
so a narrower [C] is legitimate — "A2 is more accurate than parent heritage in MAE; its rank association is not
distinguishable from it". MAIN §1 "no structural signature predicted extrapolation on genuinely new units" and §2 Q30
"[C: Q3-D1, D3]" need this joint qualifier. Q3-D1 (DeepSets seeds 6–15) is not affected.

**R2-10 (minor; TRANS slight over-correction).** "Much of its accuracy needs no composition (0.429)" is true but omits the
positive part: masking τ (→ 0.416) and masking σ (→ 0.432) each remove the increment above the no-τ Bayes level, and the
model attains (0.562 − 0.429)/(0.591 − 0.429) = 82 % of the restricted Bayes rule's composition-dependent increment. That is
evidence of within-support joint use of σ and τ evidence (lookup-style), still not generalising composition (held-out
0.246). "Map following 0.64 equals the rule's own rate (0.65–0.74)" — 0.64 is just below that range; say "close to".

**R2-11 (verified; no finding).** Theorem corrections themselves are correct: OD2 K = 2 counterexample (10,4,4,4) checked
by hand (NSR holds since any 3-source majority outweighs its complement; COND fails with P = {0}, R = {1,2}); AD2's
restriction to i.i.d., source-independent copying is the right repair; the FM3 restatement in MAIN Q21 is right.

**R2-12 (verified; freeze integrity).** PROSPECTIVE_PREDICTIONS.md: `git diff --numstat` gives 12+/0− (freeze 1 → freeze 2),
41+/0− (→ pre-review commit), 14+/0− (working tree vs HEAD). No frozen block was altered; the post-review notes are appended
under a separate heading. The Q3-A2 relabel and Q3-E2 reference note are interpretation-only, as they say.

## Verdict table

| corrected claim | verdict |
|---|---|
| CA1: T1 source tokens carry a causally effective count code (per-slot) | **upheld**, strengthened by capacity nulls (R2-1); absolute IIA must be read against a 0.46 floor (R2-2) |
| CA1: "same abstract variable (additive one-hot leave-q counts)" for P-count and T1 | **over-corrected** — not identified beyond "group-agreement counts"; T1 fits own-group count equally; code differs (R2-4) |
| "3 parents, IIA 0.66–0.82" | **qualify** — p2 exceeds its patch; partial site; mixed protocols (R2-3, R2-6) |
| "no permuted-count null encoder run" | **outdated** — run here: 0.073 (and wrong-target floor 0.459) |
| CA3 necessity "supported" | **partly over-corrected** — specific vs controls, but drop > 1 (R2-7) |
| GATE: pairs contributing, ≥ 86 % survives | **accurate under mean ablation only** (R2-8) |
| Q3-D: SAT a parent-shared replication | **accurate**; same caveat owed to D3 (R2-9) |
| TRANS: largely without composition | **slightly over-corrected** (R2-10) |
| AD2, T3, OD2 K = 2 | **correct**, but THEOREMS bodies not updated (R2-5, R2-11) |
| Stale pre-review statements | **inconsistent** in PCOUNT A4/A5, CAUSAL §4, RCC l. 45–47, THEOREMS T1a/T2/T3 (R2-5) |
| Frozen predictions | **intact**, append-only (R2-12) |
