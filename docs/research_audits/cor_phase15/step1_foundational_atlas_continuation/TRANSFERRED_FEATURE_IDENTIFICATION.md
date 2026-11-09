# Transferred-feature identification: PP3 follow-up (mandate §4)

**Status.**
- The pretraining tasks were designed before training.
- The transfer and probe predictions P-B were frozen before the P-count, P-group, probe and dynamics results (commit `07f7752`).
- Interpretation beyond P-B is post hoc.
- One seed per arm.
- The withdrawn atlas "barrier" explanation is **not** restored here.

**Code and logs.**
- Tasks: `src/tasks_ext.py`. Training: `scripts/run_job.py` (pre_*, ft_*, ref_*).
- Probes: `scripts/pp3_probes.py`. Dynamics: `scripts/pp3_dynamics.py`.
- Logs: `checkpoints/{pre,ft,ref}_*.json`, `logs/pp3_probes.json`, `logs/pp3_dynamics.json`.

**Protocol.** Every arm uses the same frozen-feature fine-tune:
- 2000 T4 steps;
- layers 0–1 and the embeddings frozen, layer 2 re-initialised;
- evaluation on 20,000 episodes of stream 6,300,000;
- success threshold 0.6595 = own + ½(Bayes − own).

The atlas found its T1 transfer localised to layer 0 plus the embeddings. Compute cuts are logged in `logs/COMPUTE_LOG.txt`: pretraining 3000 steps for P-count and P-group, fine-tune 2000 steps instead of 3000.

## 1. Discriminating pretraining tasks

| arm | target during pretraining | what it can teach | pretraining result |
|---|---|---|---|
| P-local | the answer, with the reliable group resampled **per question** | within-question pair parsing and agreement; cross-question sharing is useless | acc 0.649 (own-question level) |
| P-group | agreement of each group in question (q + 1) mod Q | reading one other question's pair agreement; no counting, no answer | aux acc 1.000 |
| P-count | leave-q agreement count of each group (0–5) | agreement detection + aggregation over all other questions; no reliability → answer routing | aux acc 1.000 |
| P-reliability | T1 (atlas T1 s0) | all of the above + reliability-weighted answering | T1 acc 0.745 |
| P-control | none (random frozen network) | nothing | — |

## 2. Which feature transfers

| arm | frozen-feature T4 accuracy | agree_q | count_loo (majority 0.379) | perfect_loo (majority 0.622) | P_coll R² | outcome |
|---|---|---|---|---|---|---|
| P-control | 0.554 | 0.666 | 0.379 | 0.621 | 0.00 | fail |
| P-local | 0.576 | 0.876 | 0.422 | 0.642 | 0.13 | fail |
| P-group | 0.669 | 0.780 | 0.397 | 0.631 | 0.08 | ambiguous (+0.009) |
| P-count | **0.723** | 0.893 | 0.448 | 0.724 | 0.24 | success |
| P-reliability (T1) | **0.742** | 0.997 | 0.895 | 0.978 | 0.71 | success |

**Probe setup.** The probes are linear, at every source token of the last-layer input (the output of the frozen layers), on natural T4 episodes. agree_q has majority baseline 0.666.

**Findings.**
1. **Within-question pair parsing is not sufficient.**
   - P-local makes within-pair agreement decodable at 0.876, well above chance, yet transfers nothing: 0.576, the own-question level.
   - This separates "source parsing" from what the curriculum supplies.
2. **Cross-question agreement aggregation transfers most of the effect, without any answer or reliability supervision.**
   - P-count was trained only to count other questions' pair agreements, and transfers to 0.723. That is 88% of the T1 gain over own.
   - P-group (one other question only) gives a marginal 0.669.
   - So the transferable ingredient scales with how much *cross-question agreement aggregation* the pretraining demanded. This is the strongest identification here.
3. **Generic optimisation effects are excluded at this protocol.** A random frozen network gives 0.554.
4. **The linear source-token probe is not the transferred variable.**
   - P-count transfers with weak source-token count decodability (0.448).
   - So in P-count the aggregate is mostly not linearly present at source tokens. It may sit at readout tokens or in a nonlinear form.
   - This is why the frozen probe criterion P-B3 failed.
   - The T1 features are far richer at source tokens (counts 0.895; P_coll R² 0.71) and transfer only slightly better (0.742 vs 0.723).

**Mandate question 4 ("what early-layer feature makes curriculum learning effective").**
- **Answer on present evidence.** Early layers that *aggregate within-pair agreement across questions*.
- It need not be linearly readable at source tokens.
- It needs no reliability or answer supervision.
- It is not provided by within-question parsing.
- Its exact representation is not identified.

## 3. Learning dynamics (§4.3)

**Setup.** The atlas T1 s0 pretraining was re-run with snapshots, reproducing the stored checkpoint exactly (max parameter difference 0.0). Each snapshot was given a 1000-step frozen-feature T4 fine-tune.

| T1 step | T1 acc | transfer to T4 | agree_q | count_loo | perfect_loo | P_coll R² |
|---|---|---|---|---|---|---|
| 250 | 0.633 | 0.551 | 0.667 | 0.379 | 0.621 | 0.00 |
| 500 | 0.638 | 0.550 | 0.667 | 0.378 | 0.621 | 0.00 |
| 1000 | 0.650 | 0.562 | 0.736 | 0.388 | 0.618 | 0.04 |
| **2000** | **0.735** | **0.733** | **0.928** | 0.603 | 0.889 | 0.55 |
| 4000 | 0.742 | 0.742 | 0.993 | 0.878 | 0.973 | 0.71 |
| 6000 | 0.745 | 0.741 | 0.997 | 0.895 | 0.978 | 0.71 |

**Findings.**
- **Transferability emerges abruptly, in a single window (1000–2000 steps).** In that same window:
  - T1 accuracy leaves the own-question plateau (0.650 → 0.735);
  - agreement detection becomes decodable (0.74 → 0.93);
  - the colluder-relevant "perfect leave-q agreement" feature appears (0.62 → 0.89), even though T1 contains no colluders.
- **Before the window, nothing transfers.** After it, nearly everything does: 0.733 at 2000 steps against a final 0.741.
- **The counts at source tokens keep improving after transfer has saturated** (0.60 → 0.90). Linear count decodability at source tokens is therefore not the limiting quantity, consistent with §2 finding 4.
- **P-B4** (the first successful snapshot coincides with agree_q ≥ 0.90) **passed**. One run; the 1000-step resolution cannot order the features *within* the window.

## 4. What remains open

- **Learning dynamics at finer resolution.** Which of agreement detection, aggregation and saturated-agreement features comes first inside the 1000–2000 window? This needs snapshots every 100 steps.
- **Where the P-count aggregate lives.** Probes at readout tokens, and nonlinear probes, were not run.
- **Seeds.** One seed per arm. P-group's marginal result needs replication.
- **Why curriculum and not end-to-end.** The atlas showed that T4 Transformers from scratch fail at this budget while DeepSets succeed. These results show *what* the T1 stage supplies. They do not show *why* T4 alone fails to discover it.
