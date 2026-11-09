# What P-count pretraining transfers (Workstream A)

Code: `scripts/run_job3.py` (pre_/ft_/ftlong_/ref_ jobs; pretraining by the continuation's `train_tasks.train`, transfer by
`train_ext.train` — identical recipes to the continuation), `scripts/pcount_causal.py`, `scripts/pp3_dyn3.py`.
Logs: `checkpoints/*.json`, `logs/pcount/`, `logs/pp3_dyn3.json`. Compute deviations: `logs/COMPUTE_LOG.txt`.

## A1. Protocol check
All arms: pretraining on T1-family episodes (P-local 6000 steps; P-count, P-group 3000; T1 parents 6000 from the atlas);
transfer = 2000 T4 steps with layers 0–1 and embeddings frozen and layer 2 re-initialised, AdamW 1e-3 cosine, batch 256,
T4 RNG seeded by the arm seed; evaluation 20,000 episodes of stream 6,300,000; threshold 0.6595 (own + ½(Bayes − own)).
One difference: P-count/P-group pretrain for half the steps of T1/P-local (continuation budget); this favours T1.

## A2. Replication across seeds (frozen Q3-A: all passed; P-local on 2 seeds only)

| arm | seeds | T4 accuracy | success |
|---|---|---|---|
| random frozen | 3 | 0.554, 0.552, 0.553 | 0/3 |
| P-local | 2 | 0.576, 0.576 | 0/2 |
| P-group | 3 | 0.669, 0.577, 0.571 | 1/3 (marginal) |
| **P-count** | 5 | 0.723, 0.724, 0.708, 0.725, 0.704 (mean 0.717, sd 0.010) | **5/5** |
| T1 parents | 5 | 0.742, 0.740, 0.739, 0.739, 0.738 (mean 0.740, sd 0.002) | 5/5 |

P-count recovers (0.717 − 0.576)/(0.740 − 0.576) = **86 %** of the T1 gain over P-local (5 vs 5 seeds). The residual gap
is stable at 2000 steps; with 6000 fine-tuning steps (asymptotic usability, seed 0) P-count reaches 0.734 and T1 0.742
(gap 0.008). P-group's earlier marginal success was not reliable.

## A3. Where the transferred information lives (5 P-count seeds, 3 T1 parents)

Site: input of the last layer (= output of the frozen transferred layers) in the transferred T4 models. Counterfactual:
cond pairs (own question kept, other questions resampled on the generator). IIA on informative pairs.

| model | full readout q | full source tokens | count-contrast subspace (k = 4) readout / source | **abstract count variable** readout / source |
|---|---|---|---|---|
| P-count s0–s4 | **0.67–0.82** | 0.15–0.29 | 0.43–0.78 / 0.09–0.20 | **0.60–0.72** / 0.02–0.09 |
| T1 p0, p1, p10 | 0.10–0.13 | **0.84–0.87** | 0.09–0.13 / 0.57–0.72 | 0.08–0.12 / **0.04–0.08** |
| P-group s0 | 0.72 | 0.23 | 0.66 / 0.18 | 0.24 / 0.04 |
| P-local s0 | 0.54 | 0.44 | 0.35 / 0.37 | 0.20 / 0.06 |

The location is seed-stable within each curriculum and opposite between curricula.
(1) **Explicitly stored?** In P-count: yes — the leave-q counts are stored at the readout token in an approximately
additive one-hot code; writing only the counterfactual change of that code reproduces 60–72 % of counterfactual decision
changes. (2) **In T1:** the cross-question information is at the source tokens but not in that code (0.04–0.08), though a
4-dimensional contrast subspace carries most of it (0.57–0.72) — consistent with the atlas's per-group reliability tag (a
nonlinear function of counts), i.e. **reconstructed into a different variable**. (3) Neither code is fully relabel-
invariant (models change 7–15 % of answers under count-preserving relabels where Bayes changes none): both use some content
beyond counts.

## A4. Necessity and sufficiency
- **Sufficiency (interchange)**: established for the readout count code (P-count) and the source-token contrast subspace
  (T1) at the levels above, with full downstream recomputation.
- **Necessity by mean-replacement is not specific** (seed 0): ablating the count subspace at the readout removes 0.65 of
  the relational gain, a random subspace 0.14, a variance-matched subspace 1.33; and the count ablation moves outputs away
  from the own-question posterior (KL 0.36 → 0.71). We therefore make no necessity claim from ablation.
- **Selectivity**: on count-preserving edits the count-subspace patch changes 1.7 % of P-count answers.
- **Not run**: path-specific interventions isolating the readout→logit path from the readout→attention path; restoring
  experiments in damaged models.

## A5. Functional interchange between implementations
Under one generator-defined alignment (linear additive one-hot count encoder, refitted per model and site), P-count's
readout variable is causally effective (0.60–0.72) and T1's is not at either site (≤ 0.12). The two curricula therefore do
**not** implement the same intermediate variable in this alignment class, although both transfer. A nonlinear alignment
(e.g. a reliability-posterior encoder for T1) was not tested.

## A6. When the information appears (unconfounded dynamics; T1 s0 re-run bit-identical to the stored run)

| T1 step | T1 acc | transfer (2000-step protocol) | agree_q src / readout | count_loo src / readout | perfect_loo src / readout |
|---|---|---|---|---|---|
| 250 | 0.633 | 0.552 | 0.66 / 0.66 | 0.37 / 0.37 | 0.63 / 0.62 |
| 1000 | 0.650 | 0.572 | 0.74 / 0.73 | 0.38 / 0.37 | 0.62 / 0.64 |
| 1400 | 0.650 | 0.574 | 0.81 / 0.78 | 0.40 / 0.38 | 0.62 / 0.65 |
| 1600 | 0.651 | 0.575 | 0.82 / 0.80 | 0.40 / 0.39 | 0.63 / 0.66 |
| 1800 | 0.679 | 0.643 | 0.83 / 0.82 | 0.41 / 0.43 | 0.69 / 0.73 |
| 2000 | 0.735 | **0.732** | 0.93 / 0.88 | 0.60 / 0.53 | 0.90 / 0.83 |
| 4000 | 0.742 | 0.741 | 0.99 / 0.90 | 0.88 / 0.60 | 0.98 / 0.87 |
| 6000 | 0.745 | 0.742 | 1.00 / 0.91 | 0.90 / 0.61 | 0.98 / 0.88 |

(majority baselines: agree 0.666, count 0.379, perfect 0.622; probes are independent of fine-tuning.)
- **Within-pair agreement information emerges first and gradually** (1000–1600 steps: 0.74 → 0.82) **without any usable
  transfer** (0.572–0.575 even with the full 2000-step protocol) and without T1 improvement.
- **Usable transfer emerges with the cross-question aggregate** (1600 → 2000: count and perfect-agreement decodability and
  T1 accuracy rise together; transfer 0.575 → 0.732). With the fine-tune protocol fixed at 2000 steps, the earlier
  "abrupt at 2000" reading is replaced by: agreement detection → (≈ 1700–2000 steps) aggregation → transfer.
- This matches A2's arm ordering: agreement alone (P-local) does not transfer; the aggregate (P-count) does.
- Resolution: snapshots every 100–200 steps (budget-cut from every 100); one run.

## A7. Are counts necessary or convenient? (partial)
Counts are not *necessary*: T1 transfers through a non-count code at source tokens, and T4 DeepSets solve T4 from scratch
by pooling. Explicit counting (P-count) gives a transfer nearly as good as full T1 at half the pretraining steps. Sample
efficiency, multi-task and context-length effects were **not** tested.

## Answers (questions 1–7)
1. **What transfers:** cross-question aggregation of within-pair agreement — in P-count as explicit additive leave-q
   counts at the readout tokens. 2. **Necessary?** Not shown by ablation (non-specific); shown sufficient by interchange.
3. **Sufficient under a defined intervention:** yes — the abstract count variable reproduces 60–72 % of counterfactual
   decisions (5 seeds). 4. **Why the location differs:** the P-count objective reads its targets at the readout tokens, so
   the aggregate is formed there; T1's objective is served by a per-source reliability variable that the last layer routes on
   (atlas), so it is formed at the source tokens — a hypothesis consistent with both objectives, not tested by moving the
   P-count targets. 5. **Survives seeds:** yes (transfer 5/5; location 5/5; T1 location 3/3). 6. **When available:**
   agreement information from ≈ 1000 steps; usable cross-question aggregate between ≈ 1700 and 2000 steps (one run).
   7. **Existence vs quick usability:** at 1000–1600 steps the agreement information exists (probes) but even a 2000-step
   fine-tune cannot use it; the earlier 1000-step protocol additionally under-rated usable features (P-count 0.654 at 1000
   steps vs 0.723 at 2000).
