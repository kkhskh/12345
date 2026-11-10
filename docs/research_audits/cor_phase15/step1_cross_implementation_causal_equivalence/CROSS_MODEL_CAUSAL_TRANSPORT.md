# Within-model baselines, cross-model transport, decision-response equivalence, pathways (mandate §6)

Confirmation models P2, P3, P4, T2, T10, T11; streams 12_300_000 (fit) / 12_400_000 (evaluation, 1000 episodes × 6 decisions).
`logs/conf/results.json`, `analysis.json`; calibration in `logs/cal/`. IIA on informative decisions; "adj" = floor-adjusted.

## 1. Within-model causal baselines (E-Int)

Abstract-variable interchange at each model's frozen site, generator-true counterfactual counts, floor = mismatched
counterfactual counts, ceiling = full-site patch.

| model | cond IIA (floor, ceiling) adj | single-group adj | π-preserving adj | relabel: count channel / full site |
|---|---|---|---|---|
| P2 | 0.642 (0.411, 0.682) 0.85 | 0.95 | 0.94 | 0.04 / 0.33 |
| P3 | 0.732 (0.406, 0.805) 0.82 | 0.84 | 0.78 | 0.10 / 0.33 |
| P4 | 0.609 (0.392, 0.677) 0.76 | 0.84 | 0.63 | 0.04 / 0.40 |
| T2 | 0.648 (0.428, 0.549) 1.82* | 0.95 | −0.29 (T_full 0.42) | 0.05 / 0.50 |
| T10 | 0.818 (0.442, 0.838) 0.95 | 0.98 | 0.87 | 0.02 / 0.62 |
| T11 | 0.768 (0.465, 0.788) 0.94 | 1.03 | 0.56 (T_full 0.97) | 0.01 / 0.57 |

\* T2's full-source patch is below its count-variable interchange (as in Investigation III's R2 for the same parent): the full
patch is not a ceiling for an added fitted direction; raw IIA and Pearson r (0.85 vs 0.75) are the meaningful numbers there.
Unconditional r between patch and natural centred-logit changes, cond: P 0.77–0.90, T 0.85–0.95. Matched-capacity
wrong-variable encoders (cross-pair counts) give IIA 0.02–0.10 everywhere; wrong-group encoders in T 0.21–0.32 (below floor).
Off-manifold: median Mahalanobis distance of encoder-patched sites equals that of natural states within 3 % in every model.

**Reading.** Both implementations satisfy intervention equivalence for V under cond and single-group interventions, well above
floor. Under π-preserving edits (Bayes unchanged, counts changed) the models do respond, and the count channel carries most of
that response in P and in T10/T11 but not in T2 with the own-group encoder: T2's π-preserving response runs partly through
other-group dependence at its source tokens (§3 of COUNT_INFORMATION_STRUCTURE.md). Count-preserving relabels change 33–62 %
of informative decisions through the full site but ≤ 10 % through the count channel (Prediction III): both models use content
beyond V, carried outside the identified variable.

**Prediction I (local components).** In T models the changed group's two tokens suffice (adj 0.95–1.03); one token of the pair
does not (−0.10 to 0.55): the distributed realization must be written consistently (Theorem B1). In P a single component of the
readout code suffices (0.84–0.95).

## 2. Cross-model transport through V (Prediction II)

Donor D's decoder reads its own natural base and counterfactual states; the recipient R receives E_R(α̂_D(x′)) − E_R(α̂_D(x)) at
R's site; scored against R's natural counterfactual; fidelity = (IIA − floor)/(oracle − floor) with oracle = generator-true counts.

| donor (decoder acc.) | to other same-curriculum recipients (cond / single) | to other-curriculum recipients (cond / single) |
|---|---|---|
| P2 (0.72) | 0.59 / 0.38 | 0.78 / 0.51 |
| P3 (0.76) | 0.66 / 0.45 | 0.84 / 0.55 |
| P4 (0.78) | 0.53 / 0.28 | 0.86 / 0.68 |
| T2 (1.00) | 1.00 / 1.01 | 1.00 / 1.00 |
| T10 (1.00) | 1.00 / 1.00 | 1.00 / 1.00 |
| T11 (1.00) | 1.00 / 1.00 | 1.00 / 1.00 |

- **T → P and T → T transport is exact** (fidelity 0.98–1.01, also in r). This is what Theorem C1 predicts for an exactly decodable
  donor, and by C2 it does not show that T's downstream computation resembles P's.
- **P → anything is partial**, limited by P's lossy site code; **P → P is worse than P → T** (Prediction II(a) failed). Self-transport
  in P is exact (0.97–1.05), so the decoded soft state is consistent with the donor's own encoder but does not carry the donor's
  count state into another P seed. Per-position decoders (post hoc, `transport_posthoc.json`) change nothing (P→P 0.37–0.66,
  P→T 0.55–0.89).
- Calibration showed the same split (P0, decoder 1.00, transported exactly; P1, decoder 0.78, at 0.57–0.85).

## 3. Decision-response equivalence beyond Bayes (Prediction VI; Theorem C3)

Abstract interventions with input fixed; response = change of centred logits at q. Mean pairwise Pearson r (95 % bootstrap CI).

| family | within P (3 pairs) | within T (3 pairs) | across (9 pairs) | cross − within T |
|---|---|---|---|---|
| π-preserving abstract (Bayes response 0) — frozen | −0.074 [−0.101, −0.051] | 0.377 [0.344, 0.414] | 0.096 [0.075, 0.114] | −0.281 [−0.316, −0.249] |
| π-preserving input edits | 0.029 | 0.532 | 0.208 | −0.325 |
| random natural-support (residual from Bayes) | 0.063 | 0.301 | 0.166 | −0.135 |
| off natural support | 0.348 | 0.672 | 0.506 | −0.165 |

Correlation with the exact Bayes response on random natural-support interventions: P2 0.49, P3 0.72, P4 0.57, T2 0.79, T10 0.82,
T11 0.73. Calibration showed the same ordering (π-preserving: within T 0.81, within P 0.18, across 0.12).

**Reading.** Beyond what Bayes requires, the three T1 parents share part of their response to count changes (r 0.24–0.46 per pair),
while the P-count seeds do not share it with each other (−0.22 to 0.03) or with T (−0.03 to 0.18). The separation is
curriculum-asymmetric: it is not that P-count implements a second, different shared map; P seeds' extra-Bayesian count
responses are idiosyncratic. Off natural support (where Bayes imposes nothing; Theorem D2) all models agree more (0.35–0.67),
so part of the extension is shared. On these six models, decision-response equivalence holds only to the extent the models track
Bayes (P less than T), not in their approximation structure.

## 4. Downstream pathway (E-Comp, Theorem C4)

Fraction of the abstract cond-intervention effect (|Δ centred logits|) remaining when the readout's layer-2 attention output, or
its MLP output, is frozen at its base value:

| | P2 | P3 | P4 | T2 | T10 | T11 |
|---|---|---|---|---|---|---|
| attention frozen | 0.45 | 0.81 | 0.91 | 0.00 | 0.00 | 0.00 |
| MLP frozen | 0.57 | 0.48 | 0.73 | 0.44 | 0.73 | 0.72 |

In T the variable reaches the decision only through layer-2 attention from the readout to q's source tokens (necessarily so for a
site off the readout). In P it acts in place, with a seed-dependent split between direct residual/MLP and attention paths. Under
this mediator decomposition the two implementations are **not** computationally equivalent, and P seeds differ among themselves.

## 5. Controls requested in §6.4

- Agreement behaviourally but different internal responses: §3 (all models ≈ 0.70–0.75 accurate; beyond-Bayes responses differ).
- Same count information, different downstream transformation: §4.
- Count-preserving change affecting the model while leaving Bayes unchanged: relabel (33–62 % of informative decisions) and
  π-preserving abstract interventions (mean |Δ centred logit| 0.14–0.35, against 0.71–0.97 for natural cond counterfactuals).
- Engineered aggregate-only computation approximating Bayes: the exact Bayes-from-counts map g_B (`eqv.bayes_from`) is such a
  computation; by construction its π-preserving response is 0, every model's is not.
- Matched-capacity encoder transferring an incorrect variable: cross-pair counts (IIA ≤ 0.10) and wrong-group counts (below floor).
