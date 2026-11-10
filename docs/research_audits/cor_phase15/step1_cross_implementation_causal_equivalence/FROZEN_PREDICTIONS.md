# Frozen predictions — Investigation IV

Frozen after calibration (development) runs on P0, P1, T0, T1 (streams 12_100_000 / 12_200_000 / 12_250_000;
`logs/cal/`) and **before** any confirmation run. The confirmation models and streams below have not been evaluated by any
script at freeze time; the site-test arm S5 has been trained but not analysed (only its training log was read: transfer accuracy
0.631, pretraining count accuracy 0.516). Everything below the line "OUTCOMES" is appended after evaluation; the text
above it is not to be edited. Code at freeze: `scripts/equiv_run.py`, `necessity.py`, `info_structure.py`, `analyze.py`,
`site_eval.py`, `src/eqv.py` (hashes in the freeze commit).

## Fixed design

- **Confirmation models.** P-count transfer seeds P2, P3, P4 (Investigation III `ft_pcount_s{2,3,4}`). T1 transfer parents
  T2, T10, T11 (`ref_T1p{2,10,11}`). T2 was used once in Investigation III's R1 encoder check (one number seen: per-slot IIA
  0.657); T10, T11 have never been analysed mechanistically. Site-test arm: S5, S6, S7 (`ft_pcsrc_s{5,6,7}`). Readout-arm
  comparison: P2–P4.
- **Data.** Encoder/decoder fit stream 12_300_000 (4000 natural T4 episodes). Evaluation stream 12_400_000 (1000 episodes,
  all six decisions per family). Necessity evaluation stream 12_450_000. Info-structure held-out stream 12_300_001.
- **Model-pair selection.** All pairs: 30 ordered transport pairs, 15 unordered response pairs (3 within-P, 3 within-T, 9 cross).
- **Sites (fixed a priori from Investigation III).** P: readout token 36+q, encoder `P_full` (one-hot n^{(-q)}, 18).
  T: question q's six source tokens, encoder `T_own` (token (q,s) ← one-hot of n^{(-q)}_{g(s)}, 6).
- **Informative pairs.** Interchange accuracy is computed on decisions whose argmax changes under the natural counterfactual
  (frozen criterion). Unconditional agreement and Pearson r of centred-logit changes are always reported alongside it.
- **Floors and ceilings.** Floor = same encoder with counterfactual counts from another episode (mismatched); ceiling =
  full-site patch of the same positions; adjusted = (IIA − floor)/(ceiling − floor).
- **Support.** cond: X_{-q} resampled from P(X_{-q} | X_q) (marginal support, not same world in ≈ 39 %). single-group edits:
  accepted only if in marginal support (exact check); same-world fraction reported. π-preserving edits: exact Bayes invariance
  (verified to 2e-15). Abstract V-interventions are classified natural-support (P(X_q, n′) > 0) or off-natural-support.
- **Statistics.** 300 bootstrap replicates over episodes for group means of response correlations. With ≈ 800 episodes the
  per-pair r has SE ≈ 0.02–0.03, so differences of 0.10 between group means are detectable. Only three models per curriculum
  are available: every conclusion is about these six models, not about a population of training runs.

## Competing hypotheses and what each predicts

| | I local components | II transport | III count-preserving | IV specific contribution | VI beyond-Bayes response |
|---|---|---|---|---|---|
| H1 shared abstract variable, shared V→decision map | pass | pass | pass | pass | no curriculum separation |
| H2 different projections needing reconstruction | T local writes fail or depend on other groups | cross-curriculum transport fails | – | – | – |
| H3 shared information, different downstream computation | pass | pass (Theorem C) | pass | pass | **separation** |
| H4 intervention artefact | adjusted IIA ≈ 0 | fidelity ≈ floor | – | marginalisation not toward own-question posterior / nulls as large | – |

## Predictions

**I — Full vector vs local components (single-group edits).** (a) In each T model, writing the changed group's component
only at its two source tokens of question q gives adjusted IIA ≥ 0.5. (b) In each T model, writing at only one of the two tokens
gives adjusted IIA lower than (a) by ≥ 0.20 (one token is not sufficient: distributed realization). (c) In each P model the
single-component readout write gives adjusted IIA ≥ 0.5.

**II — Cross-model transport (cond and single families).** (a) Theorem C: transport fidelity depends on the donor's
decodability, not on the recipient's curriculum: for every donor, the cross-curriculum mean fidelity and the
same-curriculum mean fidelity (other recipients) differ by ≤ 0.10. (b) For every donor with per-component decoder accuracy
≥ 0.95, every cross-curriculum fidelity (IIA-based) ≥ 0.85.

**III — Count-preserving controls (relabel family).** In every model the count channel (self-decoded-then-encoded patch)
reproduces ≤ 0.20 of informative natural changes (IIA ≤ 0.20), while the full-site patch reproduces ≥ 0.20: the effect of
count-preserving edits is carried outside the identified variable.

**IV — Specific causal contribution.** (a) On-support marginalisation reduces KL(own-question Bayes ‖ model) by ≥ 70 % from the
clean model in every model; the matched-capacity wrong-variable null and the norm-matched rotated control each reduce it by
< 30 % in every model. (b) Restoration after span ablation recovers ≥ 0.8 of the lost accuracy in every P model. For T models
the calibration runs showed span ablation damaging own-question evidence (restoration 0.30, 0.60), so the frozen prediction is
**< 0.8 in at least 2 of 3 T models** (the span-ablation operator is not clean for T; necessity is to be judged by (a)).

**V — Training-objective site.** In ≥ 2 of 3 S models, source-site (`T_full`) adjusted cond IIA ≥ 0.5 and exceeds readout-site
(`P_full`) adjusted IIA by ≥ 0.20; in all of P2–P4 the readout-site value exceeds the source-site (`T_full`) value by ≥ 0.20.
Known confound at freeze: the source-supervised pretraining reached count accuracy 0.516 (S5) vs 1.0 for the readout arm in
the same 3000 steps, so the arms are not matched in optimisation success.

**VI — Decision-response equivalence beyond Bayes (the H1/H3 discriminator).** Family: π-preserving abstract interventions at
the frozen sites (exact Bayes response 0). Statistic: mean pairwise Pearson r of centred-logit responses, per group.
- **VI-sep (H3 / partial):** the upper 95 % bound of (cross − within_T) is < −0.10.
- **VI-shared (H1):** the upper bounds of both (cross − within_T) and (cross − within_P) are ≥ −0.10.
- Secondary families, reported with the same statistics but not used for the verdict: random natural-support V-interventions
  (residual from per-model-scaled Bayes response), off-natural-support V-interventions, π-preserving input edits.
- **VI-bayes:** every model's response to random natural-support V-interventions correlates with the exact Bayes response at
  r ≥ 0.5.

**Pre-registered outcome classification.**
- Outcome A (shared causal relational variable, shared downstream map): I, II, IV(a) pass and VI-shared.
- Outcome B (partial abstraction): I, II, IV(a) pass and VI-sep.
- Outcome C (equivalence fails): II(b) or IV(a) fails.
- Any other combination is reported as such, without reclassification.

---
# OUTCOMES

Evaluated on the confirmation models and streams after the freeze commit 579a7a2 (`logs/conf/results.json`, `analysis.json`,
`necessity.json`, `info_structure.json`; `logs/site/site_eval.json`). Adjusted = (IIA − floor)/(ceiling − floor).

| prediction | criterion | values | verdict |
|---|---|---|---|
| I(a) T local write | adjusted ≥ 0.5 each T | T2 0.95, T10 0.98, T11 1.03 | **PASSED** |
| I(b) one token of pair | ≥ 0.20 below I(a) each T | T2 −0.04 (Δ 0.99), T10 0.55 (Δ 0.43), T11 −0.10 (Δ 1.13) | **PASSED** |
| I(c) P single component | adjusted ≥ 0.5 each P | P2 0.95, P3 0.84, P4 0.84 | **PASSED** |
| II(a) fidelity independent of recipient curriculum | \|cross − same\| ≤ 0.10 per donor | T donors: ≤ 0.012 (pass); P donors cond: P2 0.19, P3 0.18, P4 0.33; single: 0.12, 0.10, 0.40 | **FAILED** (P donors) |
| II(b) decodable donors | every cross fidelity ≥ 0.85 for donors with decoder ≥ 0.95 | T2, T10, T11 → P: 0.984–1.006 | **PASSED** (no P donor qualified: decoders 0.72–0.78) |
| III count-preserving | count channel IIA ≤ 0.20, full site ≥ 0.20, every model | count channel 0.008–0.095; full site 0.33–0.62 | **PASSED** |
| IV(a) specificity | marg KL-to-own reduction ≥ 0.70; null and rotated < 0.30, every model | marg: P2 0.701, P3 0.830, **P4 0.677**, T2 0.803, T10 0.878, T11 0.860; null 0.005–0.020; rotated −0.17–0.28 except **T11 0.386** | **FAILED** (P4 marg; T11 rotated) |
| IV(b) restoration | P ≥ 0.8 each; T < 0.8 in ≥ 2/3 | P2 0.973, P3 0.979, **P4 0.785**; T 0.68, 0.35, 0.37 | **FAILED** (P4) |
| V site | ≥ 2/3 S: source adjusted ≥ 0.5 and ≥ readout + 0.2; all P: readout ≥ source + 0.2 | S5 0.15 vs 0.12; S6 4.39 vs 0.83 (ceiling 0.231 ≈ floor 0.223: degenerate); S7 0.34 vs 0.33; P2 0.85 vs 0.92 (degenerate source ceiling), P3 0.82 vs −0.26, P4 0.76 vs undefined | **FAILED** (and partly not evaluable: degenerate floors/ceilings) |
| VI-sep / VI-shared | upper CI of cross − within_T < −0.10 | cross − within_T −0.281 [−0.316, −0.249]; cross − within_P +0.170 [0.142, 0.200]; within_P −0.074, within_T 0.377, cross 0.096 | **VI-sep** |
| VI-bayes | r with Bayes ≥ 0.5 every model | P2 **0.486**, P3 0.716, P4 0.571, T2 0.788, T10 0.824, T11 0.732 | **FAILED** (P2) |

**Pre-registered classification: Outcome C** (rule: II(b) or IV(a) fails; IV(a) failed). It is reported as C without
reclassification. How close the failures were and what the rest of the evidence shows is set out in CAUSAL_EQUIVALENCE_RESULT.md.

Notes added at evaluation:
- The V criterion used floor-adjusted IIA; when the full-site ceiling of a site is within a few points of its floor (S6 source,
  P2 source, P4 source), the adjusted score is numerically meaningless. This weakness of the frozen statistic was not anticipated.
- Prediction II's decoder pooled the six decision positions. A post-hoc rerun with per-position decoders
  (`scripts/transport_posthoc.py`, written after the outcomes) gives the same pattern (P decoders 0.73–0.79; P-donor
  same-curriculum fidelity 0.37–0.66, cross 0.55–0.89; T donors 0.98–1.01), so the failure is not a decoder-pooling artefact.
- The S arm's pretraining reached count accuracy 0.516 (S5), 1.000 (S6), 1.000 (S7).
