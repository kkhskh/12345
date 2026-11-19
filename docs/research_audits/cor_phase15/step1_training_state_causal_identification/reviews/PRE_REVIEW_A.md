# XIX pre-experiment review A: mathematics and identifiability

**Reviewer.** Reviewer A (mathematics and identifiability), mandate §3.4 / §13. This is an AI agent review, not human peer review.

**Object under review.** `DESIGN_XIX.md` (draft), `logs/conf_spec19_draft.json`, `scripts/compose19.py`, `scripts/score19.py`, `scripts/conf_setup19.py`. Evidence used: `logs/dev/dev_table19.json`, `logs/dev/compat_dev.json`, `logs/dev/s100nec_dev.json`, `logs/dev/distance_L21.json`, `logs/dev/distance_L23.json`, `logs/dev/state_diff_*.json`, `logs/graph_table19.csv`, `logs/DEV_NOTES19.md`, `logs/COMPUTE_LOG.txt`.

**What I did.** I read the files listed and ran json reads only. I ran no training and no measurement, and I changed no file other than this report.

**Question.** Can the frozen predictions distinguish the explanations they claim to distinguish (X1–X8, plus the training-time X3/X4/content readings)?

**Short answer.** The core design is identifying, on two conditions: the claims have to be narrowed, and several scorer and labelling defects have to be fixed. Three structural problems need fixing before the freeze:
- **P7 cannot fail.** It cannot be CONTRADICTED as coded.
- **P6 on M is confounded.** The M control is opposite-targeted and gives no loss disturbance, and the dev data already show it.
- **Pooling across bases hides failures.** Pooled "both" hypotheses can mask a failure on one base, and the per-base minimum of pairs that the design promises is not implemented.

None of these needs more training compute.

---

## 0. Classification of every frozen prediction in §6

Key: (a) guaranteed by construction; (b) re-measures the same causal quantity as another prediction (double counting); (c) does not discriminate (all live explanations predict the same outcome); (d) admissible.

| id | class | what it can legitimately reject (after the corrections below) |
|---|---|---|
| P1_E_L0_sufficient | **d** | On E only, at state level: "the non-L0 step-100 state is necessary". By L2 this is the same observation as "B←I (non-L0) is not necessary". It does **not** support X1 as a process (§2 below, obj. 10). |
| P2_E_L0_necessary | **d** | On E: "the non-L0 groups are jointly sufficient" (by L2, I←B L0 ≡ B←I{emb,L1,L2,WA,final_LN}). Together with P1_E it also rejects symmetric "mismatch" accounts (obj. 9). |
| P1_M_L0_insufficient | **d** | By L2 this is the M **necessity** test for the non-L0 complement (I←B{emb,L1,L2,WA,LN}). It must be labelled that way. |
| P2_M_L0_necessary | **d, mislabelled** | It rejects "the non-L0 complement is sufficient on M". It does **not** test "layer 2 carries the M switch": that is P1_M_L0L1's job. |
| P1_M_L0L1_sufficient | **d** | "The I-state of L2 (the blocked/target layer) is necessary on M." This is the only confirmation test of X1 on M. |
| P1_M_L0L2_sufficient | **d** (one family with L0L1) | Together with L0L1: "a specific partner layer is required". The two are one claim (substitutability) and count once. |
| P3_param_carries + P3_opt_alone_nil | **d**, one family | X6 only, and only beyond an equivalence margin of 0.25 φ. X1–X5 and X8 all predict the same outcome for both arms, so P3 discriminates nothing except X6. |
| A7_randdir_L0_nil (E) | **d, weak** | Only the *isotropic, norm-matched* form of "any L0 displacement switches". It is not a disturbance-matched control (obj. 7). |
| A7_randdir_L0_nil (M) | **c** | Already implied by P1_M_L0_insufficient on the same units: the *actual* L0 displacement does not switch M, so a random one of the same norm cannot discriminate anything. |
| P6_matched_ctl_nil (E) | **d** | X8 in the "natural training-time disturbance, parameter-distance-matched" form. The dev E control is strong: step-100 KL excess is 2.1× the block's and the L0 per-group ratios are 0.75–1.20. |
| P6_matched_ctl_nil (M) | **c / confounded** | Nothing, as constructed. The control shares the target's softmax row and pushes the network *away* from the switch: dev φ −0.17, step-100 nec_M 0.086 vs B 0.023. Its step-100 loss disturbance is about 0 (KL 0.432 = B's 0.432; the block's is 0.444). See obj. 8. |
| P4_vsg_partial | **d** (the key P4 test) | Pure X3, pure content-loss, and "content available but not learned". All mixed accounts predict it too. |
| P4_vrs_switch | **d, narrow** | Only "a literal stop-gradient on the values is required". It does not separate X3 in its natural form (absence of *informative* value learning) from content-loss, because both predict a switch (obj. 5). |
| P4_vrg_switch | **c** | Every explanation still open after XVIII predicts a switch, except X4. vrs and frz already test X4. Already 5/5 in XVIII sensitivity, plus the dev result. |
| P5_frz_nil | **d for X4a, c for X4b** | X4a only: "absence of routing learning at the target layer switches". It cannot reject the state-level or relative-competition form (obj. 6). |
| P5_gsg_exceeds_frz | **d, conditional; partly b** | It works only under a stated monotonicity assumption. It shares the frz measurement with P5_frz_nil: if frz ≈ 0 it reduces to "gsg ≥ 0.25", which replicates XVIII. |
| P7_E_L0_functional | **unfalsifiable as coded**; same run as P1_E | Once fixed: "Claim 1 only" on the E transplant. |
| P7_vrs_functional | **unfalsifiable as coded**; same run as P4_vrs | Once fixed: "Claim 1 only" on vrs. |

**Findings from the table.**
- **(a) None is guaranteed by construction.** The full-state anchors correctly carry no hypothesis.
- **(b) No exact L2 duplicates.** I checked all nine transplant arms: none composes the same (θ, m, v) state as another.

---

## 1. Lemma check (§2)

### Objection 1. L1 wrongly lists paramB_optI as a machinery check. MANDATORY.

L1 is correct as a statement about S_100. I checked that:
- Φ is a deterministic function of (θ, m, v, c, r);
- `hist` is only logged (`xix_train.py` l.216–245) and never enters the update;
- the schedule depends on c alone.

But the sentence "The same holds for fullB, and for paramB_optI = fullB when the moments do not matter" places paramB_optI among the construction-guaranteed checks. "The moments do not matter" is the very hypothesis P3_opt_alone_nil tests. Nothing in L1 guarantees it.

Two further points:
- L1's "machinery check" is real only if the composed fullB/fullI step-100 (θ, m, v) are bit-identical to the native S_B/S_I. That costs no training.

**Correction.**
1. Delete the paramB_optI clause from L1.
2. Add a fail-closed pre-continuation assertion: for each pair, the (θ, m, v) arrays of `tp_*_fullB` equal those of S_B^100, and those of `tp_*_fullI` equal S_I^100, bit for bit.
3. Fix the `compose19.py` docstring. It says the loss history is "identical to I's", which is false: B's and I's loss histories differ. The difference is harmless because `hist` is never used.

### Objection 2. L2 is correct, but "the scorer treats them as one unit" is not implemented. MANDATORY.

The duality is correct. B←I_G and I←B_{Gᶜ} with moments following give identical (θ, m, v), and c and r are verified equal. The composed files can still differ in `hist`, which affects nothing.

`score19.py`, however, contains no code that detects or merges dual states. The design's sentence claims a safeguard that does not exist. It holds today only because the arm list happens to contain no duals.

The design also fails to use the duality where it matters for interpretation:
- P1_M_L0_insufficient *is* the necessity test for the non-L0 complement on M;
- P2_E_L0_necessary *is* the insufficiency test for the non-L0 complement on E.

**Correction.**
1. Either implement a fail-closed check in `conf_setup19.py`/`score19.py`, or delete the sentence. The check: within each pair, hash the composed (θ, m, v) arrays only, excluding `hist`, and refuse the design if two hypothesis-bearing arms share a hash.
2. Add to §6 the dual reading of each transplant prediction, as in the table in §0 above.

### Objection 3. L3 is correct, but its dev compatibility evidence is misstated and points the other way. MANDATORY.

"All E801 composed states lie between the two source states (KL 0.459–0.540 against sources 0.459 and 0.536)" is false:
- `tp_E801_BfromI_L0` has KL 0.540, above both sources;
- its accuracy is 0.418, below both sources (0.560 and 0.443).

So the sufficient composite is the most off-trajectory E state measured at step 100. That supports L3's caution. It also makes the step-100 disturbance-matching of the X8 controls more important (obj. 7, 8).

A second point concerns KL. On E, step-100 KL separates every switching composite from every non-switching one:
- switch: ≥ 0.534;
- none: ≤ 0.483.

Step-100 KL is therefore fully confounded with "I-like content" across the transplant arms. Only training-time ctl breaks the confound on E: it has KL 0.621 and does not switch.

**Correction.**
1. Rewrite the [dev] bullet with the true numbers.
2. State that "between in KL" is not evidence of being on-manifold.
3. Report the step-100 KL/acc/nec of every confirmation composite as a predeclared compatibility diagnostic. `compat19.py` already does this, so no training is needed.

### Objection 4. L4's time-scale argument is incomplete. MANDATORY wording; the margin part is part of this item.

**The second moment.** The m-term decays in about 10 steps (β1 = 0.9), but the preconditioner v does not. At c = 100, v_100 ≈ (1−0.999^100)·ḡ² ≈ 0.095·ḡ². Its share of v_{100+k} is about:

| k | share of v_100 in v_{100+k} |
|---|---|
| 10 | 0.90 |
| 100 | 0.48 |
| 300 | 0.21 |
| 1111 | 0.045 |

So any v-mismatch steers per-coordinate step sizes for about 100 post-transplant steps, not "tens". A 100-step window is exactly the length already shown to be able to select the organisation.

**Moment differences.** L4 quotes only the μ differences (0.3–1.95). The ν differences are smaller:
- E: 0.06–0.27, largest on L0_QK;
- M: 0.05–0.21.

That smallness is the real reason a nil P3 is plausible.

**Margin.** P3's band [0.75, 1.25] would call an optimizer contribution of up to 0.25 φ "nil". The one group-level dev datum on E is B←I L0 with B's L0 moments: 0.74 against 0.91 with I's moments. That is a 0.17 drop, which this band would pass as no effect.

**Correction.**
1. Restate L4: the m-mismatch lasts about 10 steps; the v-mismatch lasts about 100 steps, with a relative size of 0.05–0.27 (quote ν).
2. State P3 as an equivalence test with margin 0.25. The report may only say "no optimizer-state contribution ≥ 0.25 φ at full-state level". It must never say "no optimizer effect".
3. Report the E801 L0-level 0.17 difference as an open, unconfirmed group-level moment effect.

---

## 2. Discrimination problems in the §3 tables and §6 labels

### Objection 5. vrs does not "keep the value gradient" in the sense X3 needs. MANDATORY relabel.

Under vrs, the gradient path through the values exists, but the values read through T come from other episodes (`graph_table19.csv`: forward changed; WV gradient changed). The value learning through T is therefore uninformative, not intact.

X3 in its natural form is "the switch occurs because no *informative* value learning happens through T". In that form X3 predicts a **switch** for vrs, not "partial" as the §3 table says. vrs and vrg then predict the same outcome under X3 and under content-loss. The only arm that separates them is vsg: content intact, value gradient removed.

**Correction.**
1. In the §3 training table, change X3/vrs to "switch (informative-learning form)", or split X3 into X3-literal and X3-informative.
2. Relabel P4_vrs_switch as rejecting only "a literal stop-gradient is required".
3. State that P4's discrimination between X3 and content-loss rests on P4_vsg_partial alone.
4. State that a vsg hybrid rejects only the *pure* forms. It is predicted by every mixed account, so it eliminates no mixture.

### Objection 6. P5_frz_nil cannot reject X4 as the mandate states it. MANDATORY relabel; ADVISORY arms.

frz freezes WQ/WK of the whole target layer: all four heads and every edge, T and complement alike. It therefore removes routing learning non-specifically. Two forms of X4 have to be kept apart:
- **X4a.** "Absence of routing-parameter learning at the target layer produces the switch." frz tests this: X4a predicts a switch, so a nil result rejects it.
- **X4b.** "The block's specific change to the routing state, i.e. T de-recruited *relative to* competing edges, carries the switch." frz keeps T and its competitors at their initial relative balance, so the prediction is not determined. The direct test of X4b is the state-level QK transplant pair, which the dev data already have on E:
  - B←I L0_QK = 0.10 (not sufficient);
  - I←B L0_QK = 0.95 (not necessary).

  Neither arm is in the confirmation.

P5_gsg_exceeds_frz rejects "gsg acts by removing routing learning" only under a monotonicity assumption: less T-recruitment signal means a larger switch. That assumption is doubtful. gsg is not a scaled-down frz. It zeroes the T-edge contribution to the WQ/WK gradient while keeping the complement contribution (C1 in the audit), so it changes the *direction* of routing learning, not only its amount.

P5_gsg_exceeds_frz also uses the same frz runs as P5_frz_nil. The two must not be counted as two independent eliminations.

**Correction.**
1. Label P5_frz_nil as rejecting X4a only.
2. State that X4b is not tested in confirmation, unless the arms in advisory obj. A1 are added.
3. Write the monotonicity assumption into P5_gsg_exceeds_frz. Limit its claim to "gsg's effect is not explained by the *amount* of routing learning it removes".
4. Count P5 as one family.

### Objection 7. randnorm_L0 is matched in parameter norm only, and is redundant on M. MANDATORY.

**(i) Disturbance size on E.** Random isotropic directions of a given norm usually cause far less functional disturbance than a structured Δ of the same norm. Mandate §7 D3 rules out controls whose disruption is negligible. The step-100 KL of randnorm_L0 is not in `compat_dev.json`; only all-groups randnorm is, at KL 0.465 against 0.540 for B←I L0. A7 can therefore reject only "any L0 displacement of this *norm*", not a disturbance-based account.

**(ii) Redundancy on M.** P1_M_L0_insufficient already says the *actual* L0 displacement fails to switch M. "A random one of the same norm also fails" adds nothing. The control M actually lacks is one for the sufficient sets L0+L1 and L0+L2.

**Correction.**
1. On M, replace `randnorm_L0` with a randnorm on {L0, L2}, per-group norm matched to ‖θ_I − θ_B‖. This is the same arm count, and `compose19.py --params-from-I L0,L2 --perturb` already supports it.
2. Before the freeze, compute with `compat19.py` the step-100 KL/acc of each randnorm composite (no training), and label A7 "norm-matched, not disturbance-matched".
3. Do not cite A7 against X8 unless the randnorm step-100 KL excess is at least half of the matched transplant's.

Optionally, scale the random direction until its step-100 KL matches. That is a small change to compose19 and adds no runs.

### Objection 8. P6 on M is confounded by construction, and the dev data show it. Pooling hides this. MANDATORY.

**Both controls act on the target's softmax row.** Both are readout-row interventions on the same layer and the same query rows as the target:
- E: `msk oq0` vs target `own0`;
- M: `gof oq2` vs target `own2`.

Down-weighting the other-question keys renormalises the softmax and **raises** T's share. The control is therefore partly an anti-intervention on T. Mandate §7 D2 requires checking effects on complementary routes, and here T *is* the complementary route of the control.

**On M the confound is visible in dev:**

| quantity | ctl | B | block |
|---|---|---|---|
| step-100 nec_M | 0.086 | 0.023 | −0.003 |
| step-100 KL | 0.432 | 0.432 | 0.444 |
| final φ | −0.17 (away from the switch) | — | — |

So the M control moves the network the opposite way and produces no loss disturbance. It is weak (D3) and targeted (D2) at the same time. Matching on total parameter distance (ratio 1.08) does not repair either problem.

The M control's per-group distance record is also missing. `distance_L23.json` has no `gofoq2b1.0` entry; the 1.08 appears only in DEV_NOTES.

**On E the control is usable.** Its step-100 KL excess is 2.1× the block's, its L0 per-group ratios are 0.75–1.20, and its step-100 nec_E is halfway to I (0.133 vs 0.266 / −0.003). It still does not switch (φ −0.005). That is real evidence against X8 on E.

**Pooling.** With base "both" and 6 pairs, three M failures plus three E passes score PARTIAL, not CONTRADICTED, and vice versa.

**Neither control is in the audit.** The graph audit (`graph_table19.csv`) has no row for `ctl` or `frz`, although mandate §3.1 requires one for every proposed intervention.

**Correction.**
1. Split P6 into P6_E and P6_M.
2. Add ctl (both bases) and frz (both bases) rows to the graph audit, including T's attention share during the window at steps 0 and 50. This needs no training.
3. Predeclare a second matching diagnostic: cumulative window training-loss excess relative to B, from `hist`. Report it next to the distance ratio.
4. For M, choose one of:
   - (a) construct in development, before the freeze, a control that does not share T's softmax rows and whose loss excess is non-negligible; or
   - (b) remove P6_M as a hypothesis, keep the run as descriptive, and declare X8 **unresolved on M** (mandate §7 D3).
5. Write the M gof b=1.0 per-group distance record into `logs/dev/`.

Both options keep the arm count unchanged.

### Objection 9. The foreign-donor row in the §3 table has no arm, and the argument that actually controls for mismatch is missing. MANDATORY.

The §3 table lists "foreign baseline donor of the target layer" as the arm that discriminates X8 (mismatch). The arm is not in `conf_spec19_draft.json`. The design should instead state the argument it actually relies on, which is stronger.

**On E the reciprocal pair already rejects symmetric mismatch.** B←I L0 and I←B L0 are mismatched by the same ‖Δ_L0‖ in opposite directions. A mismatch account predicts the same outcome for both. Dev gives I-class for the first and B-class for the second.

**On M it does not.** For the L0-only pair, both outcomes are B-ish (0.22 / 0.28), which a "mismatch → B" account also predicts. The sufficient composites L0+L1 and L0+L2 have no reciprocal in the confirmation; dev I←B L0L2 = 0.12 is not carried forward.

**Correction.**
1. Remove the foreign-donor row from §3, or add the arm.
2. Write the reciprocal-asymmetry argument into §2/§3 as the E mismatch control.
3. State that on M, the claim "L0+partner carries I-specific content, not mismatch" is controlled only by the randnorm of obj. 7 (isotropic), unless advisory A4 is adopted.

### Objection 10. Claim boundary: transplants identify state-level location, not mechanism. MANDATORY.

What a SUPPORTED transplant prediction establishes: in the composed box between S_B^100 and S_I^100, imposing I's coordinates on group G sends plain continuation to the I basin, or restoring B's prevents it. It says nothing about the process that wrote those coordinates during the window (L3). It is also bound to the tensor-group basis: a mediator spread across groups would only show up as an "interaction".

Three consequences the design must state:
- **(i) The E localisation is close to tautological with respect to what is measured.** The E route own0 lives in layer 0, and nec_E measures that route. Transplanting L0 transplants the route's own parameters: the composite's step-100 nec_E is already 0.025, I-like. X1, X3, X4b and an X8 concentrated in L0 all predict P1_E. P1_E/P2_E therefore discriminate only against carriers outside L0 (X2/X5 at state level). The §3 entry "X1: switch" must not be read as support for X1.
- **(ii) The M result is the informative one, and it goes against X1 on M.** The carrier is the upstream L0 plus a substitutable partner, and the blocked layer L2 is not necessary. Dev I←B L2 = 1.00 already contradicts X1 on M. The confirmation tests this only through P1_M_L0L1, and has no B←I L2 or I←B L2 arm. The design should say plainly that on M, the X2-type state-level reading (a non-target carrier) is what is being *confirmed*, not rejected.
- **(iii) "X5 downstream adaptation" must be defined at state level.** "The step-100 downstream state carries the switch" is testable. "L0 causes later downstream adaptation during the continuation" is part of Φ and cannot be tested by any transplant.

**Correction.**
1. Add a "what is and is not identified" paragraph (mandate §10.10) containing (i)–(iii).
2. Change the "rejects" column of P1_E to "non-L0 state carriers (X2/X5, state level, E only)".
3. Add X7 (interactions) explicitly to the §3 table. The E within-L0 dev result (no tensor class necessary or sufficient alone) is X7 evidence, but n = 1 and development only.
4. Unless advisory A1 is adopted, state that every within-layer tensor-class claim (mandate §4 A4: QK vs OV vs MLP) is development-only and exploratory.

### Objection 11. Admissibility of "replications of a single development pair" (§11), and how to label them. MANDATORY.

**My ruling.** These replications are admissible. Mandate §11 bars predictions whose outcome is "already known from development", meaning known for the units on which they are scored. Here the dev unit (D801) and the confirmation units (D821–823) are disjoint, the stream is fresh, and the instrument is frozen. With a deterministic Φ and n = 1, the dev result fixes the *direction* of the prediction. It does not establish that the result holds across data orders, which is what §10.9 requires.

**Conditions.**
1. **Labelling.** Every such prediction carries the label "FROZEN REPLICATION of a development observation (dev: 1 pair, D801, measure18 / stream 86 100 000) on 3 fresh data-order pairs per base, 1 parent". P5_gsg_exceeds_frz and the replaced M randnorm (obj. 7) are labelled "NEW on these units". No replication may be described as an independent discovery.
2. **D801 is never pooled** into any confirmation statistic.
3. **Unit counting.** Report 3 independent units per base and 1 parent. "Both"-base hypotheses do not make 6 units of evidence for a base-specific mechanism (see obj. 12).
4. **Eliminations.** An explanation counts as eliminated only on SUPPORTED. PARTIAL eliminates nothing. Predeclare, per prediction, what a CONTRADICTED result reinstates.
5. **Fix the last bullet of §6, which is wrong.** P5 and P6 are themselves replications, not the only eliminators of XVIII-open explanations. The XVIII-open confounds and their eliminators are:
   - confound 1 (vrg confounds content and value learning): **P4_vsg_partial**;
   - confound 7 (the step-100 state was never examined): **P1/P2**;
   - confound 5 (no non-targeted control): **P6_E**;
   - X6: **P3**.

   The success condition of mandate §19 should be argued through those, each with its narrowed claim from the objections above.

---

## 3. Scorer defects that affect identifiability

### Objection 12. "Both"-base hypotheses are pooled, and the per-base minimum of pairs is not implemented. MANDATORY.

§6 says "min 2 evaluable pairs per base". In `score19.py`, `n` counts evaluable pairs over *both* bases, and `min_pairs` = 2 is checked on that total. Two consequences:
- A "both" hypothesis can be SUPPORTED with zero evaluable M pairs.
- A complete failure on one base (3/6 outside the band) scores PARTIAL, not CONTRADICTED, because CONTRADICTED needs more than n/2.

The explanations in §3 are base-specific anyway, since the target layer differs (L0 vs L2). This affects P3, A7, P6, P4 (×3), P5 (×2) and P7_vrs.

**Correction.**
1. Replace every "both" hypothesis with an _E and an _M hypothesis, each with min_pairs = 2 inside its base.
2. If a pooled statement is wanted, define it as "SUPPORTED on both bases" (a logical AND), never as pooled bands.

This changes no compute.

### Objection 13. P7 can never be CONTRADICTED, and it does not test what §6 says. MANDATORY.

The P7 hypotheses use the band [−10, 10] together with `need_functional: "switch"`. In `score19.py`:
- SUPPORTED requires `functional_ok` for every pair;
- CONTRADICTED requires a strict majority of v outside [−10.1, 10.1], which never happens.

So a pair whose functional class stays at B (the "Claim 1 only" outcome P7 is meant to detect) gives PARTIAL at worst. The prediction cannot fail.

Also, §6 says "category BOTH, not CAUSAL_ONLY", but the code checks only the functional class. It does not require a causal switch, so a FUNCTIONAL_ONLY or HYBRID pair with a switched functional label passes.

**Correction.** Score P7 on the scorer's `category`:
- SUPPORTED: every evaluable pair is BOTH;
- CONTRADICTED: a strict majority are CAUSAL_ONLY or NEITHER;
- otherwise PARTIAL;
- INCONCLUSIVE(·) pairs are counted and reported, not silently passed.

Add a planted case to `scorer_planted19.json` for each outcome.

---

## 4. Process

### Objection 14. The development record is still changing. MANDATORY.

`COMPUTE_LOG.txt` shows dev continuations still being composed and run at the time of review. Examples:
- `tp_E801_interp050_L0` and `tp_E801_interp025_L0` (α-interpolation, B's moments);
- `tp_E801_BfromI_L0_QKMLPLN`.

None of these is in `dev_table19.json` or in §4.

**Correction.**
1. Freeze only after every started dev run is either in §4 (with its result) or listed as abandoned with its reason.
2. Do not adjust any band after seeing these runs without re-review.
3. Report the interpolation results as an A7 diagnostic (mandate: "appropriate interpolation diagnostics").

---

## 5. Advisory items

**A1 (ADVISORY, net zero runs). Swap vrg for a state-level test of X3 vs X4b on E.**
- **Cut:** the vrg continuations (−6). vrg carries no discriminating hypothesis (obj. 0/5). Its comparison with vsg/vrs is covered by the dev result and XVIII's 5/5.
- **Add:** I←B L0_QK and I←B L0_OV on the three E pairs (+6). Dev values are 0.95 and 0.95.
- **Gain:** a frozen state-level test of "QK state necessary" (X4b) and "OV state necessary" (state-level X3). That is the mandate §4 A4 question the confirmation currently omits.
- **Caveat:** mandate §5 B2 lists the content-disruption-with-controlled-backward-rule comparison (vrg) as required. If the adjudicator reads B2 as needing fresh vrg runs, keep vrg as a descriptive factorial cell, drop P4_vrg_switch as a hypothesis, and accept the narrowed X4 claim instead.

**A2 (ADVISORY, measurement only). Re-check the bands with measure19.** All dev φ values and bands come from measure18 on stream 86 100 000. The confirmation uses measure19 on 94 700 000. The one comparison available (vsg E: nec_E 0.57 vs 0.66) shows the instruments differ. Re-measure the D801 dev checkpoints with measure19 on a non-confirmation stream and check that every dev φ still falls in its band. This takes minutes and no training, and it guards against an XVIII-type transfer failure.

**A3 (ADVISORY, zero compute). Bridge the training arms to the transplant result.** For each training arm's step-100 state, report:
- the cosine between its L0 Δ (from B) and the block's L0 Δ;
- its L0 per-group distance ratio.

This is the only available link between "where" (transplant) and "how" (training arms). It should be reported as a diagnostic, not as mediation.

**A4 (ADVISORY, +3 runs, about 20 min on 2 lanes). Add I←B L0L2 on M.** This is the reciprocal of the sufficient M composite (dev 0.12). It closes the symmetric-mismatch gap of obj. 9 on M. If it is not added, the M combination claim must carry the label "sufficiency without a direct reciprocal; mismatch controlled isotropically only".

**A5 (ADVISORY). Rename the M "necessary" predictions.** The bands for P1_M_L0_insufficient and P2_M_L0_necessary reach φ = 0.5. Call them "restoration reduces φ to ≤ 0.5" (mandate P2: "prevents or meaningfully reduces"), not "necessary".

**A6 (ADVISORY). Predeclare the response to the P6 matching gate.** The gate can drop pairs, and the M ratio is steep in b. Predeclare what is reported if fewer than two pairs pass. The answer should be "X8 unresolved", not a re-tuned control.

**A7 (ADVISORY). NaN values score PARTIAL silently.** A NaN φ fails both the in-band and the out-of-band tests in `score19.py`, so it yields PARTIAL. Make NaN drop the pair with a reason, like a non-competent arm.

---

## 6. Budget

**Mandatory items.**
- Training cost: zero extra continuations. Objection 7 replaces one arm on M; objection 8(b) keeps the run descriptive, and 8(a) only replaces the M control after development.
- Other work: scorer edits, labels, compat/graph-audit diagnostics and the bit-identity assertion. None of it needs training.

**Advisory items.** A1 is net zero. A4 adds 3 continuations (about +3% of the 90).

The total stays within about 10.5 h on 2 CPUs.

**Still out of scope.** Head-level transplants, and transport to a second parent.

---

## 7. Mandatory items, in brief

1. L1: remove the paramB_optI clause; add a bit-identity check for the fullB/fullI anchors (obj. 1).
2. L2: implement the dual-state dedup check or delete the claim; write the dual readings into §6 (obj. 2).
3. L3: correct the compatibility statement; report step-100 KL/acc/nec for every composite (obj. 3).
4. L4: add the v time-scale and ν figures; make P3 an equivalence test with margin 0.25 (obj. 4).
5. P4: relabel vrs; give P4_vrg_switch no hypothesis; vsg carries the X3-vs-content discrimination (obj. 5; class table §0).
6. P5: frz rejects X4a only; X4b is untested; state gsg's monotonicity assumption; P5 counts as one family (obj. 6).
7. A7: on M, replace randnorm_L0 with randnorm{L0,L2}; report randnorm step-100 KL; label "norm-matched" (obj. 7).
8. P6: split by base; add ctl/frz rows (with T share) to the graph audit; add a loss-excess diagnostic; on M, a new control or "X8 unresolved on M" (obj. 8).
9. Foreign donor: remove it or add it; write the reciprocal-asymmetry argument (obj. 9).
10. Claim boundary: state level only; E localisation nearly tautological; M contradicts X1; X5 at state level; X7 explicit; A4 tensor-class claims dev-only (obj. 10).
11. §11 labelling of replications, unit counts, the elimination map, and the corrected §6 last bullet (obj. 11).
12. Per-base hypotheses, with min_pairs inside each base (obj. 12).
13. Rewrite the P7 scorer on `category` so it can be CONTRADICTED, with planted cases (obj. 13).
14. Freeze only once the dev record is complete (obj. 14).

CLASSIFICATION: Approved with mandatory corrections

---

## Round 2

**What I read.**
- `DESIGN_XIX.md` v2 and `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.
- `logs/conf_spec19.json` and `logs/conf/design19.json`: 32 hypotheses, 2 gates.
- `logs/dev/dev_table19.json`, `logs/dev/dev_bandcheck19.json`, `logs/dev/s100diag_E801.json`, `logs/DEV_NOTES19.md`, `logs/graph_table19.csv`.
- `scripts/score19.py` v2 and `scripts/conf_setup19.py`.
- The new harness modes in `scripts/xix_train.py`: vsgo (l.151–154), frz/frzp (l.172–196), pnz (l.182–186, 264–265), lbl (l.255–259).

I ran no training and no measurement, only json/csv reads and arithmetic on them.

### R2.0 Status of the 14 round-1 mandatory items

| # | status | evidence / remainder |
|---|---|---|
| 1 L1 | **Resolved** | Clause removed. `score19.py` l.147–154 asserts that composed fullB equals native B (θ, moments, rng) and that fullI's θ equals I's. Remainder: see ADV-R2-5 (fullI moments). |
| 2 L2 | **Resolved (report-only)** | `duplicate_composed_states` is reported (l.155). It does not fail closed, but no current arm pair is a dual. Remainder: see ADV-R2-5. |
| 3 L3 | **Resolved** | §2 L3 gives the true numbers. |
| 4 L4 | **Resolved** | ν figures and the ≈100-step second-moment influence are stated. P3 is an equivalence test with margin 0.25. Moment effects are quoted (0.74 vs 0.91; frz vs frzp). |
| 5 P4 / vrs / vrg | **Resolved** | vrs relabelled; vrg hypothesis and arm removed. New vsgo issues: R2-1, R2-2. |
| 6 P5 | **Resolved** | gsg contrast withdrawn. frz and frzp form one family. X4a is explicitly *not* eliminated on E. X4b is now tested through the E QK restoration. |
| 7 A7 | **Resolved** | M uses randnorm{L0, L2}, labelled "norm-matched only". The X8a double count is addressed in R2-5. |
| 8 P6 | **Mostly resolved** | Same-row controls removed; P6 split by base; audit rows added for ctl, frz, frzp, pnz, vsgo. **lbl has no audit row**, and the lbl gate has a denominator problem on M: R2-3, R2-4. |
| 9 foreign donor | **Resolved** | Dev only, E φ −0.02. The reciprocal argument is stated, and I←B L0L2 is added on M. |
| 10 claim boundary | **Resolved** | §3 says transplants localise and do not identify mechanism, that the E localisation is near-tautological, and that M contradicts blocked-layer X1. A4 wording: R2-7. |
| 11 labelling | **Resolved, with a remainder** | Replication labels and elimination-only-on-SUPPORTED are in place. "What a CONTRADICTED result reinstates" is still not predeclared per family: ADV-R2-6. |
| 12 per base | **Resolved** | The scorer refuses any base other than E or M (l.67). |
| 13 P7 | **Resolved** | Category hypotheses with a CONTRADICTED rule, requiring RPC-concordant anchors. The status of P7 is in R2-6. |
| 14 dev record | **Resolved in the files** | All runs are in `dev_table19.json`. v2 dropped the v1 §4 dev table, so dev results without a hypothesis are no longer summarised in the design: ADV-R2-4. |

### R2.1 New arms: are they identifying?

**vsgo.** The implementation matches its description.
- The T-term head output is `cT · stop_gradient(WO)` with `cT = (a ⊙ XM) · stop_gradient(v)`.
- The attention weights `a` are not stopped, so the T-logit routing gradient is d/da = WO·v, exactly as in plain training.
- The audit row confirms it: forward max difference ≤ 1.4e-6, d loss/d bias_T equal to plain at steps 0 and 50, 7 (E) / 33 (M) tensors with changed gradient.

The elimination it supports is a **necessity** statement. If E_/M_P4_vsgo_switch is SUPPORTED, then neither removal of forward content through T nor exogenous removal of the routing gradient is *necessary* for a full switch. That reading is valid. The [dev] E step-100 diagnostics support it:

| arm | T-share drop (fraction of the block's) | outcome |
|---|---|---|
| vsgo | 0.27 | switches (0.83) |
| lbl | 0.80 | no switch |
| frz, frzp | 0.97 | no switch / 0.57 |

So T de-recruitment at step 100 is neither necessary nor sufficient on E801.

The design text claims more than this. See R2-1 and R2-2.

**lbl.** It is a legitimate control for **X8b in its uniform form**: loss disturbance or slower learning applied to every route alike. Its dev E disturbance is 0.88 of the block's (gate 0.5), so it is not a weak control under D3. It is, however, **not route-neutral in effect**:
- it replaces task-relevant information (the labels), which D2 asks about explicitly;
- [dev] E step-100: T share 0.154 (B 0.235, I 0.134), nec_E 0.027, L0 cosine with the block 0.76.

Its nil therefore says two things: uniform slowing does not switch, and a block-like step-100 route state without the block does not switch. The second is a useful dissociation, but it is a statement about route state, not about "non-targeted disturbance". See R2-3.

**pnz.** A legitimate control for X8a only.
- [dev] E: KL excess ratio 0.01, T-share drop 0.03, cosine 0.01.
- It is a training-time version of randnorm, so it overlaps with A7. See R2-5.

**frzp vs frz.** These are identical step-100 θ states with different moments: [dev] E −0.01 vs 0.57. That is a clean demonstration that Φ is sensitive to the moments at fixed θ. Consequences:
- E_P5_frz_nil is now an **artefact check**. It eliminates none of X1–X8.
- P3's nil is specific to the *block's* moments, not a general insensitivity of Φ to moments.

The design §2 L4 states both points correctly.

### R2.2 Remaining objections

**R2-1. vsgo: the attribution sentence over-claims, and the vsg/vsgo bands cannot carry the "+O" attribution. MANDATORY.**

§5 says that vsgo SUPPORTED together with vsg PARTIAL "would attribute the effect to the combined V+O learning through T". Three problems:

- **(a) Adjacent bands.** The bands are vsg ∈ [0.25, 0.75] and vsgo ∈ [0.75, 1.25]. Both can be SUPPORTED on a pair with vsg 0.74 and vsgo 0.76, a contrast of 0.02. The part attributed to WO's T-learning is the *contrast*, and no hypothesis tests the contrast.
- **(b) Not unique.** The evidence shows sufficiency, not attribution. vrs (content rolled) is also sufficient. On E, frzp (routing parameters frozen while V and O learn normally) is partially sufficient at 0.57. Mandate §4 A5 says a sufficient manipulation may be one of several routes.
- **(c) What vsgo leaves learning.** Downstream readers of T's content (on E: L1, L2 and the readout; on M: the readout) still learn from it. Trajectory-level routing also changes endogenously: [dev] E T-share drop 0.27 of the block's.

**Correction.**
1. Add a per-pair contrast hypothesis per base: `phi['vsgo'] - phi['vsg'] ∈ [0.2, 1.0]`. Declare {vsg, vsgo, contrast} one P4 family. Or drop "+O" from the attribution.
2. Rewrite the sentence as: "removal of target-layer V+O learning through T is sufficient for a (near-)full switch with the forward pathway intact and no exogenous routing-gradient removal. Forward-content removal and routing-gradient removal are therefore not necessary. Attribution is not unique (vrs; frzp partial on E). Downstream learning from T content and endogenous routing change (step-100 T-share drop, reported per base) are not excluded."
3. Compute `s100diag` for M801 as well. It needs no training, and the claims are made per base.

**R2-2. Mandate §5 B3 categories are not applied to the training arms. MANDATORY.**

B3 makes it mandatory to separate natural forward perturbations, natural gradient consequences, synthetic gradient interventions, parameter freezes and optimizer-state manipulation. v2 labels only the P3 transplants as "synthetic".

**Correction.** Add a column to the §4 arm table with these classes:

| arm | class |
|---|---|
| vsg, vsgo | synthetic gradient interventions; the forward is natural |
| vrs | natural forward perturbation with its natural gradient |
| frz | parameter freeze plus optimizer-state manipulation (moments held) |
| frzp | parameter freeze |
| pnz | synthetic parameter perturbation |
| lbl | data perturbation |
| transplants | synthetic state composition |

Interpretation sentences must use these classes. Example: "an intervention on the optimisation algorithm, not removal of information from a normal network" (B3).

**R2-3. lbl: missing audit row, and an inaccurate description. MANDATORY.**

`graph_table19.csv` has no lbl row, although mandate §3.1 requires one for every proposed intervention. The §4/harness description "forward, gradients' routing structure and all edges unchanged" is wrong for gradients: replacing labels changes every gradient, including the T-logit and value gradients. Only the forward pass and the edge structure are unchanged.

**Correction.**
1. Add lbl rows (E and M, steps 0 and 50) to the audit, with the same columns as the others.
2. Change the description to "non-edge-targeted data perturbation: forward unchanged; every gradient changed; not route-neutral in effect", and quote the [dev] E step-100 numbers: T-share drop 0.80 of the block's, nec_E 0.027, L0 cosine 0.76.
3. State the elimination as "X8b in the uniform-slowdown form". Note that lbl's nil is also predicted by relative-competition accounts (X2), so it discriminates X8b from the targeted explanations, not among them.

**R2-4. The lbl gate (kl100) has no denominator rule, and on M the denominator is about 0. MANDATORY.**

`matching()` computes r = (KL_C − KL_B)/(KL_I − KL_B) with no guard. On M, the [dev] block excess is 0.012 (0.444 against 0.432). On a fresh M pair it can be ≤ 0:
- if the denominator is negative and lbl's excess positive, r < 0 and the gate fails. Pairs are dropped, and the design then reports "X8 unresolved on M" exactly when the block causes no loss disturbance for X8b to explain;
- if both are negative, r > 0 and the gate passes on the wrong sign.

**Correction.** Predeclare the rule in the scorer and in §4:
1. If KL_I − KL_B < 0.005 for a pair, X8b is "not applicable" on that pair: the block causes no step-100 loss disturbance to explain. Report it as such, not as "unresolved".
2. Otherwise apply the ratio rule.
3. The gate also fails if KL_C − KL_B ≤ 0.

Add planted cases for all three branches. On M, state the [dev] fact that the block's loss excess is 2.8% of KL_B, which already makes X8b implausible there a priori.

**R2-5. X8a is eliminated twice, by A7 and by P6_pnz. MANDATORY.**

§5 lists A7 → X8a and P6 pnz → X8a. They are different observations, a step-100 isotropic displacement and a training-time random walk, of the *same* explanation. Mandate §11/§19 forbids counting such results as independent confirmations.

**Correction.** Declare {A7, P6_pnz} one X8a family per base. X8a is eliminated once, only if both members are SUPPORTED. If they disagree, report it as such.

**R2-6. E_P7 arm chosen after seeing the dev result; dev column wrong; P7's status. MANDATORY.**

- `dev_bandcheck19.json` scored E_P7_vsgo_concordant as PARTIAL (E801 vsgo `INCONCLUSIVE(causal_SWITCH)`, RPC f = −0.013, ambiguous). The E P7 arm was then switched to vrs.
- The design table gives E_P7_vrs_concordant's dev value as "none (—)". The bandcheck arms show E801|vrs = BOTH.
- RPC correlates 0.91 with necessity (C3), so P7 is nearly a re-measurement of the band hypothesis on the same run.

**Correction.**
1. Set the E_P7_vrs dev column to "BOTH".
2. State in §5 that the E P7 arm was moved from vsgo to vrs because the E801 vsgo RPC was ambiguous.
3. Report E vsgo's category descriptively in the confirmation.
4. Label every P7 hypothesis "same-run concordance diagnostic; not an independent confirmation; eliminates nothing", consistent with §5 "P7: RPC concordance only".

**R2-7. A4 wording is ambiguous. MANDATORY (wording only).**

"A4: the QK state alone, or the OV state alone, carries the E selection" reads as a sufficiency claim. The arms test **necessity**: I←B L0_QK ≥ 0.75 means I's L0_QK state is not necessary. By L2 that is the same observation as "B←I (everything except L0_QK) is sufficient". Sufficiency of QK/OV (dev B←I 0.10 / 0.00) is development-only.

**Correction.** Write: "A4 rejects that the I-state of L0_QK (respectively L0_OV) is necessary for the E switch. Together with the dev-only non-sufficiency, this is X7 (redundant or joint carriage)."

### R2.3 The 0.75 lower edge of E_P4_vsgo_switch (dev 0.83)

**Keep 0.75. Do not lower it.**

The edge is not a fit to the dev value. It is the scorer's switch threshold (PHI_HI), and the claim attached to this hypothesis is sufficiency for a *full* switch. Lowering it to make dev 0.83 sit more comfortably would merge "full switch" into the hybrid region already occupied by vsg (0.35 / 0.44). That would destroy the vsg/vsgo distinction R2-1 depends on.

The class condition (need_class I) is robust on E801: nec_E 0.037, nec_M 0.584. The instrument is stable too: 0.826 with measure18, 0.825 with measure19.

The cost is power, and it must be accepted openly. Under the all-pairs rule, with a 0.08 dev margin and an unknown between-order SD, a PARTIAL is quite likely. Illustration: for SD 0.1, P(all 3 ≥ 0.75) ≈ 0.5.

**Requirements (folded into ADV-R2-1).**
- Predeclare that a PARTIAL here means "a large partial effect without content removal; full sufficiency not established". Like every PARTIAL, it eliminates nothing.
- Report the per-pair values and the Clopper–Pearson bound.
- Add a frozen sensitivity reading at 0.70, alongside phicut70, that never replaces the primary.

### R2.4 Advisory items

- **ADV-R2-1.** As in R2.3: predeclared PARTIAL reading and a 0.70 sensitivity edge for both vsgo bands.
- **ADV-R2-2.** Label E_P5_frz_nil and M_P5_frz_nil as "harness/moment artefact check; eliminates no X". On M, frzp freezes L2_QK while the state-level carrier is L0 (L2 not necessary), so M_P5_frzp_nil addresses only the target-layer X4a as defined. Declare M_P2_L0 and M_P2_L0L2 (nested restorations) one family.
- **ADV-R2-3.** `dev_bandcheck19.json` shows M801|randnorm_L0L2 as "NONCOMPETENT" with φ NaN and f = 0, but no such run exists in COMPUTE_LOG or the dev measurements. The bandcheck turns a missing arm into "arm non-competent". Make it report "missing". The confirmation scorer is unaffected: it requires exactly one record per run.
- **ADV-R2-4.** Restore a short dev-results summary in the design, or a pointer to `dev_table19.json`, for dev results with no hypothesis:
  - interpolation α = 0.25 / 0.5 on L0 → φ −0.01 / −0.01; full L0 with B's moments gives 0.74. This is a strongly threshold-like dependence, relevant for reading A7;
  - B←I L0_QK+MLP+LN 0.46;
  - gsg / vrg;
  - same-row controls;
  - foreign donor;
  - randnorm (all groups).
- **ADV-R2-5.** Make duplicate composed states fail closed for hypothesis-bearing arms. Extend the fullI bit-identity check to the moments.
- **ADV-R2-6.** Predeclare, per family, what a CONTRADICTED result reinstates. Examples: E_A4 CONTRADICTED reinstates "QK (OV) state necessary"; P6 lbl CONTRADICTED reinstates X8b.

### R2.5 Double counting and construction check (v2 hypothesis set)

- **Construction.** None of the 32 is guaranteed by construction. The fullB/fullI anchors still carry no hypothesis.
- **L2 duals.** No two arms compose the same state. I checked IfromB_L0_QK, IfromB_L0_OV, IfromB_L0L2 and the randnorm arms against all the others.
- **Same-run pairs.** E_P1 and E_P7_L0; E_P4_vrs and E_P7_vrs; M_P4_vsgo and M_P7_vsgo. Handled by R2-6.
- **Same-explanation pairs.** A7 and P6_pnz (R2-5). P5 frz/frzp, P3, and the M partners are already declared as families.

**Budget.** 99 continuations, about 11 h. All mandatory items above add no training: they are scorer rules, an audit row computed at fixed parameters, labels, and diagnostics computed without training.

CLASSIFICATION: Approved with mandatory corrections

---

## Round 3 (verification of R2-1 … R2-7 only)

**What I checked.**
- `DESIGN_XIX.md` v2.1 and the Round 2 table in `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.
- `logs/conf_spec19.json` and `logs/conf/design19.json`: 35 hypotheses, identical id lists in both files.
- `logs/dev/s100diag_M801.json` and the `lbl` rows of `logs/graph_table19.csv`.
- `scripts/score19.py`: the kl100 gate (l.175–181) and P7 eligibility (l.224).

I ran no training.

| item | status | verification |
|---|---|---|
| R2-1 vsgo attribution | **Resolved** | `E_/M_P4_vsgo_exceeds_vsg` (`phi['vsgo'] - phi['vsg']` ∈ [0.2, 1.5]) exist per base. [dev] values: E 0.48, M 0.55. P4 is read only jointly and worded as sufficiency / "not necessary" / "not a unique attribution" (vrs, frzp). The downstream-reader limitation is in the arm table. M801 diagnostics exist. The 0.70 reading is labelled a sensitivity reading that never replaces the primary, and the PARTIAL meaning is predeclared. |
| R2-2 B3 categories | **Resolved** | DESIGN §4 categorises every training arm. frz is correctly classed as freeze plus optimizer-state manipulation. |
| R2-3 lbl audit | **Resolved** | lbl rows on E and M at steps 0 and 50: 44/44 tensors with changed gradient, forward unchanged. The description now says "every gradient changes" and that lbl's route state partly resembles the block's. |
| R2-4 kl100 gate | **Resolved** | Control excess ≤ 0 → gate fails. Block excess < 0.005 → loss matching not applicable; the control passes only if its own excess is ≥ 0.005. Otherwise the ratio rule applies. The sign-flip pass is no longer possible. |
| R2-5 X8a family | **Resolved** | "A7 + P6-pnz (one family, X8a) … counted once per base". |
| R2-6 P7 | **Resolved** | The swap is disclosed. P7 is labelled "same-run RPC concordance check, eliminates nothing". It is scored only on causal-switch pairs, and CAUSAL_ONLY contradicts it. Reading NEITHER as not a contradiction is correct once eligibility is restricted to switch pairs. |
| R2-7 A4 wording | **Resolved** | "Restoration (necessity only) … says nothing about sufficiency". |

**No new defects found.**

**One ADVISORY note** (one sentence, no change to scoring). [dev] M801: vsgo's step-100 T-share drop is 0.67 of the block's (E: 0.27). On M, therefore, the routing *gradient* is intact but the routing *state* changes substantially through the endogenous trajectory. The P4 reading "routing-gradient removal not necessary" is correctly worded. It must not be extended to "routing change not involved", and a sentence saying so should be added to §5 P4.

CLASSIFICATION: Approved
