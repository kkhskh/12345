# Adjudication of review round R1

Four independent reviewers (subagents that had not produced the work) examined the final results:
`R1_formal_math.md` (formal mathematics, counterexamples, positive-theory defence), `R1_causal_architecture.md`
(causal inference, Transformer architecture, representation/invariance), `R1_statistics_dynamics.md` (statistical
inference, learning dynamics, freeze integrity) and `R1_composition_novelty.md` (composition, novelty, baselines).
Their check scripts are archived unchanged in `reviews/checks/` (reviewer temp paths `/tmp/r1/`, `/tmp/r1chk/`).

**Verification rule.** A finding that changes a headline claim was re-run by me from the archived script on the committed
checkpoints before it was accepted; outputs are in `logs/review_checks/`. Smaller findings were accepted after reading the
script and the relevant log. Disagreement is preserved below where I accept a finding only in part.
Corrections in the documents are marked **[R]** and cite the finding (e.g. [R: CA1]). Pre-review text that is superseded is
left in place only inside clearly marked "superseded" notes; frozen prediction blocks are untouched.

Codes: FM = formal math, CA = causal/architecture, SD = statistics/dynamics, CN = composition/novelty.

## 1. Findings that change headline claims (all re-run and confirmed)

| id | finding | my re-run | verdict |
|---|---|---|---|
| CA1 (critical) | T1's "no count code at the source tokens" is an artefact of a source encoder shared across all 36 slots. A per-slot linear one-hot encoder (same alignment class otherwise) gives high IIA. | `src_enc.py` on ref_T1p1: shared 0.060, per-slot 0.805, own-q tokens only 0.803, full-source ceiling 0.833 (1393 pairs). On ref_T1p2 (not checked by the reviewer): shared 0.035, per-slot 0.657, full-source patch 0.563. Reviewer's ref_T1 s0: 0.821 vs 0.878. | **Accepted.** 3/3 parents. T1 *does* carry the leave-q count variable, additively and causally used, at the source tokens (slot-specific direction). The "different variable / curricula not interchangeable" conclusion is withdrawn. |
| CA3 | Ablation "non-specificity" came from pooled cross-position means. | `abl_pos.py` on ref_T1p1, source site, per-position means: count subspace 1.65 vs variance-matched 0.00, random 0.02. | **Accepted.** Necessity by count-subspace ablation is specific once the control is per-position. Answer 2 changes from "not established" to "supported in this operator class (mean-replacement, per-position)", still not natural necessity. |
| CA4 | GATE pairs are not jointly necessary; the 0.883/0.936 agreement is at/above the gate-blind level 0.890; resample ablation reverses single-head results. | `gate_chk.py` on gate_s1: gate-blind 0.890; pair mean-ablation 0.936; flip-following with both heads ablated 0.924, with all layer-2 heads 0.928; resample pair 0.391. | **Accepted.** "Jointly necessary pair" withdrawn: the pairs *contribute*; ≥ 86 % (s0) / 92 % (s1) of the gate effect survives. In s1 the remaining route is the pre-layer-2 readout residual. |
| SD7 | Held-out Transformers share parents with development models; a parent-lookup predictor gives Spearman 1.00, MAE 0.042. | Own computation from `logs/extrap/dev.json` and `heldout_outcome.json`: ρ 1.00, MAE 0.042. | **Accepted.** SAT's held-out association is a parent-shared replication, not a new prospective confirmation; never frozen as a positive prediction; exact null p 0.068. Removed from "prospective confirmations". |
| SD8 | The GCD "sign reversal" is an artefact; on the same six parents in development GCD ρ is already −0.66. | Script read; consistent with dev.json. | **Accepted.** The sign-reversal narrative is withdrawn. |
| FM6 (critical for AD2) | "Any copier with constant per-question copying probability has the honest law" is false (Markov copier; agreement-triggered copier). | `r1_chk2.py`: Markov λ=0.5 gives P(agree twice) 0.491 vs honest 0.436; trigger-on-agree joint law differs. | **Accepted.** AD2 restricted to *independent* per-question copying (i.i.d. over questions, independent of other sources). |
| FM3 | T3: one non-chaining AB and BC pair leaves 4 candidate composites, not 2; certainty possible with one chained pair each. | `r1_chk1.py`: (False, 4), (True, 2). | **Accepted.** T3 restated. |
| FM2 / CN4 | T1a ("composite at chance without BC→C information") needs the product prior; under the 24-pair training prior, Bayes without τ evidence scores 0.429 on C-only. | `r1_chk1.py`: no-C-in-BC train-restricted C 0.429; `r1_prior_only.py`: prior + own C reports 0.392. | **Accepted.** The holdout model's masked 0.416 is compared with 0.429, not chance. The masking verdicts (frozen Q3-C) still stand as frozen; their interpretation as "composition" is weakened: much of the 0.562 needs no composition under the 24-pair prior. |
| CN1 | Probe ceilings for the holdout run used the 36-pair prior; under the matching 24-pair prior they are 0.694 / 0.703 / 0.540. | `r1_stat_checks.py` prints the prior-matched ceilings (holdout-prior rows). | **Accepted.** "σ at ceiling in every run" withdrawn for the holdout run (it is below all three ceilings). |
| CN2 | Aux run: σ/τ decodability at the episode token is guaranteed by the auxiliary head; composite decodability is the probe composing. At decision tokens τ ≤ 0.44, composite at chance. | Script read (`aux_layers.py`). | **Accepted.** "Represented but not composed; failure lies in combining" → "maps decodable at the supervised token only; routing vs combining failure not distinguished". |

## 2. Accepted with refinement or partial disagreement

- **CA2 (alignment class favours P-count).** Accepted as a caveat: P-count was pretrained with a linear one-hot count head,
  so a linear one-hot encoder is its native code. After CA1 the class also fits T1, so the bias no longer drives a
  conclusion; it is recorded.
- **CA5 / CA6.** Accepted. Conditional sufficiency is informative only for seed 0; seed 2's ≈ 0.66 under ablation is
  collapse to one gate state, not "the gate's share".
- **CA8, CA9 (supportive).** Accepted and added: held-out-x₀ probe 0.98–1.00 at the evidence tokens; readout attention mass
  to E 0.93–0.99, to G ≤ 0.006 (re-run on s1: L2h1 0.969, L2h3 0.933). Seed 0 has a ≈ 14 % path not via E.
- **CA10.** Accepted: "relation variable at source tokens" holds for answer-supervised curricula (T1, GATE), not for
  P-count, which stores the same count variable at the readout. It is curriculum-dependent, not family-invariant.
- **CN3 / CN5.** Accepted with a refinement I add: the holdout model following map interventions at ≈ 0.64 is at the
  training-prior Bayes rule's own following rate (0.65–0.74), so it reads as *full* use of evidence in the way that rule uses
  it, not "partial causal use". 78 % of held-out-episode counterfactuals land on training pairs, so that row is not a
  held-out test.
- **SD9.** Accepted: under the frozen signed wording GCD complied with Q3-D2; SAT alone failed it. The verdict FAILED is
  unchanged. The post-hoc "predicted-by-hypothesis directions" sentence is withdrawn.
- **SD11.** Accepted: the "half the pretraining steps" efficiency framing is withdrawn (T1 at 2000 steps transfers 0.732 >
  every P-count seed). One run; the T1 transition is sharp.
- **SD12, SD13, SD14.** Accepted: long fine-tunes are one seed with T1 at ceiling; "P-local cannot use it" is a 2000-step
  protocol statement; A6 supports the first ordering arrow only. *Partial disagreement with SD13's third point:* the
  reviewer notes the probed leave-q code is "one T1 does not causally use"; after CA1 it is used (per-slot), so the probe
  measures a causally relevant variable — but the probe still did not move before transfer, so the conclusion (second
  arrow unsupported) stands.
- **SD14 ("sufficient: yes").** Accepted as "partially sufficient (0.60–0.72)". No permuted-count null encoder was run; open.
- **SD-Q3-A2.** Accepted: relabelled "not evaluable as frozen (2 of 3 seeds run; both fail)".
- **SD-Q3-E2.** Accepted: references added (static 0.519, best count rule 0.566, 4 % exact ties). Verdicts unchanged.
- **SD15.** Accepted: from-scratch ADAPT is one seed and step-unmatched; Q3-E′ thresholds were set after exploratory values.
- **FM5.** Accepted: OD2 at K = 2 closed by counterexample (weights 10,4,4,4: NSR holds, COND fails), so NSR ⇔ COND is false at K = 2.
- **FM9.** Accepted: EX1 shows on-support behaviour does not *determine* extrapolation; it does not exclude on-support
  features *predicting* it among trained models.
- **FM10.** Accepted: RC's interchange threshold is now stated (IIA ≥ 0.5 on informative pairs; seed-12 T4 at 0.47 is
  therefore "no"), and "TRANS: no" is re-graded "not shown" (condition 2 untested for one-hop σ mediation).
- **FM7, FM8, FM11, CN7, CA11, SD16, SD17.** Accepted as wording/numerical fixes (AD4 "dip" not "lag"; AD1 spread is a
  max deviation; small-cell chance ranges; argmax tie bias; equivalence 0.0 in logs; k = 4 not 2-D; relabel rates per log;
  selectivity 1.7 % is seed 0 only; P-local drop fractions undefined; landscape table qualifiers).
- **CN6.** Accepted: one seed per TRANS regime, regimes confounded with steps and difficulty.
- **CN8.** Accepted: citations added (Wang et al. 2024 grokked implicit reasoning; Balesni et al. 2024 two-hop curse;
  Sanford et al. 2024 depth for k-hop; Hewitt & Liang 2019 control tasks; ICL plateau work). "Represented-but-uncomposed"
  is downgraded to an instance of known phenomena.

## 3. Rejected

None. No finding was rejected; two were accepted with the refinements stated above (CN3, SD13).

## 4. Net effect on the account

1. *(Restated after R2 — see ADJUDICATION_R2: "count variables at different sites and codes"; "same variable" not established.)* **Workstream A is reversed in its mechanistic conclusion.** Both P-count and T1 transfer models carry the same abstract
   variable (leave-q per-group agreement counts, additive one-hot, causally used); they differ in **site** (readout vs
   source tokens) and in slot-specificity. The curricula are interchangeable at the level of this variable. The earlier
   claim of "different variables" is preserved as a superseded pre-review result.
2. **GATE**: abstract structure (gate state computed and stored at E, read by R→E attention) is strengthened; redundancy
   claims are weakened to "contributing heads plus a residual route".
3. **Q3-D**: no prospective structural predictor succeeded on genuinely new units; the held-out test is a parent-shared
   replication. Q3-D verdicts as frozen are unchanged.
4. **TRANS**: no evidence of learned composition anywhere; the restricted-prior model's success is largely achievable
   without composition.
5. Theorems: AD2 and T3 narrowed; OD2 K = 2 closed negatively.

A second round (R2) on the corrected statements: `R2_counterexample_defence.md`, `ADJUDICATION_R2.md`.
