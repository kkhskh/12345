# Observable boundary responses

These are model responses measured directly on supported, same-world source-level edits: candidate log-probability ratios and
probabilities. No fitted parameter enters any number here.

**Data**
- Confirmation cohort: 16 independent parents (T1 seeds 70–77, P-count seeds 80–87) at five transfer snapshots.
- Episodes: 6000 natural episodes (stream 15_100_000), one decision each.
- Scripts: `scripts/boundary_run.py` and `scripts/summary_tables.py`.
- Logs: `logs/conf/` and `logs/summary_tables.json`.
- Pair counts (oriented, natural):

  | family | pairs | R1 | R2 | R3 |
  |---|---|---|---|---|
  | F45 | 1346 | 435 | 89 | 822 |
  | F34 | 1475 | 500 | — | — |
  | F23 | 1132 | 384 | — | — |

  F0 uses the same pairs as F45.

## 1. Raw responses at the end of transfer (step 4000)

Mean over the 8 parents of each arm, oriented log-ratio Δm toward c_g.

| family / regime | exact Bayes Δm_B | T1 Δm_M | P-count Δm_M | parents with Δm_M < 0 (T1 / P) |
|---|---|---|---|---|
| F45, R1 (c_g agrees with the colluder candidate) | **+0.234** | **+0.193** | **−0.505** | **0/8 / 8/8** |
| F45, R2 (c_g agrees with the third group) | −2.380 | −1.820 | −1.725 | 8/8 / 8/8 |
| F45, R3 (c_g agrees with no group) | −2.038 | −1.702 | −1.937 | 8/8 / 8/8 |
| F34, R1 | +0.422 | +0.388 | +0.351 | 0/8 / 0/8 |

**The central observable.** When g reaches 5/5 and its vote agrees with the existing colluder candidate, exact Bayes *raises* c_g.
Every T1 parent also raises it, by a slightly smaller amount. Every P-count parent *lowers* it. This is a sign reversal in the
decision update. It is not a matter of magnitude, and no fitted parameter is involved.

One step below the boundary (F34, R1), both curricula raise c_g by close to the Bayes amount.

## 2. Bayes-corrected excess E = Δm_M − Δm_B (step 4000, mean of 8 parents)

| | T1 | P-count | T1 − P (frozen test) |
|---|---|---|---|
| B45, pooled | +0.229 | −0.134 | +0.36, p = .018, 95 % CI [0.09, 0.66] |
| B45-R1 | −0.041 | −0.739 | **+0.70, p = 7.8e-5, CI [0.59, 0.81]** |
| B45-R2 | +0.56 | +0.66 | −0.10 (n.s.) |
| B45-R3 | +0.34 | +0.10 | +0.23 (n.s.) |
| B34-R1 | −0.034 | −0.071 | — |
| B34, pooled | −0.159 | −0.238 | — |
| D-R1 = B45-R1 − B34-R1 | −0.007 | −0.668 | **+0.66, p = 7.8e-5, CI [0.52, 0.80]** |
| D-pooled = B45 − B34 | | | +0.28, p = .090 (n.s.) |
| B0 (F0 control) | max \|B0\| over all 16 models = 0.013 | | |

**Probability space (frozen, sign only).** In R1 the model-minus-Bayes change in p(c_g) is higher for T1 by +0.058
(CI [0.043, 0.075]). The sign agrees with the log-ratio result.

**F45 vs F34 vs F23.**
- The curriculum difference in R1 is concentrated at 4→5. Below the boundary, P-count models end transfer almost Bayes-matched
  (B34-R1 = −0.07, B23-R1 = −0.08, vs T1 −0.03 and −0.09).
- Pooled across regimes the specificity contrast is not significant (D-pooled). In R2/R3 the excesses mostly reflect both
  arms under-discounting, by similar amounts.

## 3. Argmax and information vs decision

- Exact Bayes reverses the argmax on 35.7 % of oriented F45 pairs in this evaluation set. The 22 % in EXACT_BAYES_BOUNDARY_THEORY.md counts all valid edits, oriented or not.
- The models reverse it on 31.6 % (T1) and 30.9 % (P).
- The curriculum difference therefore lies in *how* the R1 update is signed and sized, not in how often a decision flips.
  This matches the information-vs-decision distinction: both arms detect the 4→5 change (T2), and they draw opposite
  conclusions from it in R1.

## 4. What the observable says, and what it does not

**What it says.** P-count-pretrained models treat a group's reaching perfect agreement as evidence *against* its vote even in
the regime where, under the generator, it is evidence *for* it. Agreeing with the existing perfect group makes "g is the
reliable group, h* is the colluder" and "h* is reliable, g colludes" both consistent, and both put weight on c_g. T1 models
preserve the regime distinction.

**What it does not say.**
- It does not by itself say how the computation differs. See CAUSAL_EVIDENCE_UPDATE_MECHANISM.md.
- It does not say the difference holds at other boundaries or other relational tasks.
