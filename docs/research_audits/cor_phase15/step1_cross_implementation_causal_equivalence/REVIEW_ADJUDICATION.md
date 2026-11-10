# Review adjudication

Three independent adversarial reviewers examined the final results after all major experiments (pre-review commit 8c6363e):
`reviews/R_formal_equivalence.md` (formal causal equivalence, theorems; 13 findings), `reviews/R_mechanistic_interventions.md`
(interventions, encoder capacity, collateral damage; 12 findings), `reviews/R_statistics_design.md` (statistics, design,
prediction integrity; 9 findings). Their check scripts are in `reviews/checks/`, outputs in `logs/review_checks/`.
**Verification rule:** each finding that changes a headline statement was re-run by me from the reviewer's script, or tested
with a new script of mine, before acceptance (outputs in `logs/adjudication/`, `logs/review_checks/adj_*`). Corrections are
marked **[R: id]** in the documents (FE = formal, ME = mechanistic, ST = statistics). The pre-review text is preserved in commit
8c6363e. Frozen predictions and their verdicts are unchanged.

## Findings that change headline statements

| id | finding | verification | verdict |
|---|---|---|---|
| ST1 | Outcomes A and B both require II; II(a) failed, so A/B were excluded independently of IV(a). The "read without that rule" answer stated Outcome B's content, re-labelling a failure. | read against the frozen classification | **accepted.** The result now states both exclusions and grades the descriptive synthesis [E]. |
| FE12 (critical) | The main answer overstates sharing; P-count sites may hold ≈ π (single-count decoding from π alone 0.727 ≈ P decoders 0.72–0.79). | **tested** with `scripts/adj_pi_vs_n.py`: P readouts decode the within-π-class rank at 0.87–0.91 (class-aware baseline 0.50) but the π-class only at 0.29–0.39; ME6 (independent) finds P decoding errors almost never π-equivalent | **accepted in part.** The "≈ π" mechanism is refuted (P sites carry more than π, lossily, and not π-respecting); the overstatement is accepted and the wording is restricted. |
| ME5 | P transport failed because the decoder read only the readout; with the readout plus q's source tokens, P2 decodes n at 0.973 and transports at 0.97–0.98. | re-run P2→P3 (0.982); extended to P3 (decoder 0.87; →P4 0.81, →T10 0.90) and P4 (0.95; →P2 0.92, →T11 0.96) | **accepted.** P-count's count state is distributed over the readout and q's source tokens; transport fidelity tracks the union decoder's accuracy, as Theorem C1 predicts. The "asymmetric transport" conclusion is withdrawn. |
| ME1 | T2's encoder absorbs own-evidence correlates of the counts; refitted with X_q covariates, T2 cond IIA 0.648 → 0.516 (adjusted 0.73); other models barely change. | re-run on T2 (identical numbers) | **accepted.** T2 reported as 0.52 / 0.73; range 0.73–0.95 including T2. |
| ME4 / FE11 | T's "attention frozen = 0" is architectural (random-direction edits also give ≤ 1.4e-6); the MLP split does not separate curricula. | re-run (identical) | **accepted.** E-Comp separation withdrawn; no pathway evidence for different computations remains. |
| ME2 | Natural input marginalisation (averaging outputs over real resampled X_{-q}) already gives drop fractions 1.02–1.14 in all models; T overshoot is mostly natural (T11 ≈ 0.25 excess; T2's is encoder leakage). P's rotated control removes 0.59 (P3) and 1.14 (P4) of the gain in accuracy terms. | script read; consistent with logs | **accepted.** "Collateral damage" statements rewritten: P's "no collateral" claim withdrawn; T overshoot is largely the models' own natural marginal behaviour. |
| ME3 | Neither frozen site is complete: union ceilings P 0.77–0.86, T 0.94–0.96; T readouts carry 12–39 points. | re-run on T2 (site 0.549, site+readout 0.936) | **accepted.** All "site" claims qualified as "main site". |
| ST3 | VI-sep holds for these six models only: model-level permutation p = 0.05 (minimum possible), model-and-episode bootstrap bound −0.102; leaving out T11 weakens to −0.149. | script read | **accepted.** VI-sep stated as a six-model result at the edge of model-level resolution. The input-level π-preserving family (no encoder) shows the same pattern (ST6), which argues against an encoder artefact. |

## Accepted corrections without change of conclusion

- FE1: A3's formula restated with the joint P(X_q, Y_q | R) weighting (code was correct; no number changes).
- FE2: B2/D3 restricted to own-group components (a token can carry all of n: P readout; T2's readout decodes n at 0.93).
- FE3: D5 restated: an additive one-hot code constant on π-classes can only encode the colluder set H; the practical risk is
  approximate fit by a π-like code, now tested (FE12 row).
- FE4: D4 completed with the other direction (sign(V¹V²): ablation effective, single-copy interchange ineffective).
- FE5: B3 marked as definitional. FE6: wording. FE13: D2's learned "instance" relabelled as an illustration (off-support responses
  differ, but natural-support identity of the models is not established).
- FE7: tolerance ε is now the frozen thresholds; E-Int is measured as an average over pairs, not per pair; "nested" removed
  (P1–P4 satisfy E-Int with lossy readout E-Info).
- FE8: C2 weakened to a sufficient condition; P self-transport with lossy decoders is a learned counterexample to the converse.
- FE9: C1's bound is at the activation level; no output-level Lipschitz bound is claimed.
- FE10: C3's corollary is conditional on exact E-Int, which fails for π-preserving abstract interventions in T2 (and partly T11);
  the input-level π-preserving family carries the VI conclusion jointly with the abstract family.
- ME (one token of pair): doubling the one-token edit recovers much of the effect in T2 and T10; consistent writing is clearly
  required only in T11. Prediction I(b) passed as frozen; its interpretation is weakened.
- ME (relabel): the count-channel number in III is decoder noise around a structural zero; III's conclusion is unchanged.
- ME (T2 redundancy): the "no redundant copy" statement for T2 relied on the leaking encoder; withdrawn for T2.
- ME (site test): the site test cannot detect a variable that moved but stayed distributed; S-model union ceilings are 0.82–0.99.
- ME (P4): P4's 0.677 reflects its site — natural input marginalisation at that site reaches 0.711.
- ME (code): within-model and transport floors are different nulls (both called "mismatched"); site_eval.py's docstring claimed
  pathway fractions it does not compute — corrected.
- ST2: "narrow" is now used symmetrically (P2's 0.7006 passes by 0.0006; T11's rotated 0.386 is not narrow); no per-model KL SEs.
- ST4: II(b) was vacuous for P donors (none qualified).
- ST5: floor-adjusted scores with denominators 0.008–0.17 flagged as meaningless wherever cited.
- ST7: secondary VI families (random-support residual, off-support) are fragile and not cited in support of VI-sep.
- ST8: threshold-setting assessed (II(a) frozen despite cal P1 failing it; rotated 0.30 just above the cal max 0.268; IV(b)'s
  T part restated a known calibration failure) — recorded.
- ST9: analyze.py docstring (2000 → 300 replicates), random_support residual storage documented; cal episodes.npz committed
  after the freeze (no bearing on confirmation).

## Rejected

None. FE12 was accepted in part: its proposed mechanism (P sites ≈ π) is contradicted by a direct test.

No second review round was run; the post-review statements are restrictions of claims, plus three re-runs and one new test
(`adj_pi_vs_n.py`), all listed above.
