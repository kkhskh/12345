# Prospective predictions — Investigation III

Rules (unchanged from the continuation): each block is frozen in a git commit before the evaluation it concerns; outcomes
are appended below the block; frozen text is never edited; development material is labelled post hoc.
Historical predictions (atlas PP1–PP5, continuation P-A, P-B) and their verdicts are preserved in their own packages.

**Freeze state (2026-11-09).** At freeze time: the four T1-parent reference transfers (`ref_T1p1, p2, p10, p11`) had
finished but were not inspected; no new P-count/P-group/P-local seed, no new GATE seed, no TRANS or ADAPT run, and no
held-out model for Workstream D had finished (`logs/progress.log`).  GATE seed 0 had been analysed (post hoc, logged in
`logs/gate/`); its mechanism motivates Q3-B.

---

## Q3-A — P-count transfer replication (Workstream A2)

Protocol: identical to the continuation (pretraining by `train_tasks.train`: P-local 6000 steps, P-count/P-group 3000;
frozen-feature transfer 2000 T4 steps, layers 0–1 + embeddings frozen, layer 2 reset; eval 20,000 episodes stream
6,300,000; threshold 0.6595).  Seeds: P-count 0–4, P-group 0–2, P-local 0–2, T1 parents 0,1,2,10,11, random 0–2
(seed 0 values are the continuation's).

- **Q3-A1.** P-count transfers (≥ 0.6595) in at least 4 of 5 seeds.
- **Q3-A2.** P-local fails (< 0.6595) in all 3 seeds.
- **Q3-A3.** Mean P-count transfer < mean T1-parent transfer (the residual 12 % gap is real), and every T1 parent ≥ 0.70.
- **Q3-A4.** Random frozen networks fail in all 3 seeds.
- No prediction is made for P-group (seed 0 was within 0.02 of the threshold).

## Q3-B — GATE mechanism replication (Workstream B4), seeds 1 and 2, `scripts/gate_circuit.py` + `gate_redundancy.py`

- **Q3-B1.** Both seeds reach Bayes level (accuracy ≥ 0.89 on the gate_circuit stream).
- **Q3-B2.** In both seeds, mean-ablating ALL layer-2 attention output at the readouts leaves agreement with Bayes on
  unique-argmax questions ≥ 0.98 (the salient last-layer attention is individually dispensable).
- **Q3-B3 (redundancy).** In both seeds there is a pair of heads in layers 1–2 whose joint mean-ablation lowers unique-argmax
  Bayes agreement by ≥ 0.05 while each head alone lowers it by ≤ 0.01.
- **Q3-B4 (stage 1).** In both seeds, patching the layer-0 attention output at the evidence tokens from a gate-flipped run
  transfers the flip in ≥ 0.5 of scored cases, and patching the layer-0 attention output at the gate tokens transfers ≤ 0.05.

## Q3-C — TRANS (Workstream C2–C6). Competing hypotheses, frozen criteria (no directional bet)

Composition is declared **learned** in a run iff all of:
(i) C-only accuracy ≥ 0.42 on 6000 held-out episodes (Bayes 0.487, chance 0.333);
(ii) masking the C reports of every BC question (removing τ evidence) lowers C-only accuracy by ≥ 0.05;
(iii) masking the A reports of every AB question (removing σ evidence) lowers C-only accuracy by ≥ 0.05.
For `trans_holdout_s1`, composition **generalises** iff (i)–(iii) hold on episodes drawn only from the 12 held-out (σ, τ)
pairs.  Convergence/stopping: runs are fixed-length (12,000 / 8,000 steps, cosine schedule, decided before training);
no run is extended after inspection unless recorded as a separate exploratory run.

## Q3-D — structure-to-extrapolation (Workstream D4)

Held-out models (trained before this freeze, evaluated only after it): 6 Transformers (`heldout3`, T4 curriculum seeds
40–45 from existing T1 parents 0,1,2,10,11,12 — **not independent of their parents**) and 10 DeepSets (seeds 6–15).
Target: the p_c = 0 local index (continuation definition, `pp2_sig.pc0_local_index`).
Predictors (code `src/graded_sig.py` sha256 928f3a7b…, fits `logs/extrap/frozen_fits.json` sha256 f79131f0…, fitted on the
14 earlier models — post hoc development):
- B-arch: architecture indicator; S-GCD: graded-count decodability on natural T4 (structural, on-support only);
  S-SAT: colluder decodability on natural T4 (structural, on-support); A2: the continuation's behavioural degraded-colluder
  probe LPI_deg (uses off-support inputs — a strong behavioural diagnostic, not structural); constant.
- **Q3-D1.** All 10 held-out DeepSets have local index < 0.30, and the mean held-out Transformer local index is > 0.30.
- **Q3-D2 (negative structural prediction).** Within the 6 held-out Transformers, neither on-support structural signature
  (GCD, SAT) has Spearman ρ > 0.5 with the local index, and neither beats the within-TF constant in MAE.
- **Q3-D3.** Within the held-out Transformers, A2 has Spearman ρ ≥ 0.6 and MAE below the within-TF constant.

## Q3-E — adaptive sources (Workstream F), `adapt_s0`

- **Q3-E1.** The trained model's accuracy on episodes WITH a change point exceeds the static T1-Bayes reference (0.660 on
  the theory stream) by ≥ 0.015, and its accuracy on unchanged episodes is within 0.02 of exact Bayes.
- **Q3-E2.** Count insufficiency is used: on pairs of episodes with equal per-group agreement counts and equal own
  reports but different exact-Bayes argmax (positional evidence), the model follows exact Bayes in ≥ 0.6 of cases
  (static-count reference: ≤ 0.5 by construction of the pairs being decided by position).

---

## Q3-E′ — adaptive sources under a T1 curriculum (frozen 2026-11-09 after the exploratory adapt_fromT1_s0 result; before the replication run below exists)

Exploratory finding that motivates it (post hoc, `logs/adapt/eval_fromT1_s0.json`): ADAPT fine-tuned 3000 steps from the
atlas T1 s0 checkpoint matches exact Bayes (0.706 vs 0.706) and passes the Q3-E1/E2 criteria, whereas the from-scratch
adapt_s0 (frozen Q3-E) fails.  Replication: the same recipe from a DIFFERENT T1 parent (main T1 s1, train seed 1).
- **Q3-E′1.** Q3-E1 holds (changed-episode accuracy ≥ static + 0.015; unchanged within 0.02 of exact Bayes).
- **Q3-E′2.** Q3-E2 holds (≥ 0.6 agreement with exact Bayes in count classes decided by position).
- **Q3-E′3.** History intervention (other questions' order permuted; counts unchanged): the model follows the new exact
  answer in ≥ 0.8 of questions whose unique exact argmax changes.

---

# OUTCOMES (appended after evaluation; frozen text above unchanged)

## Q3-A — P-count replication (`checkpoints/ft_*`, `ref_*`; seed 0 values from the continuation)
| arm | seeds | frozen-feature T4 accuracy |
|---|---|---|
| P-count | 0–4 | 0.723, 0.724, 0.708, 0.725, 0.704 (mean 0.717) |
| T1 parents | 0, 1, 2, 10, 11 | 0.742, 0.740, 0.739, 0.739, 0.738 (mean 0.740) |
| P-group | 0–2 | 0.669, 0.577, 0.571 |
| P-local | 0, 1 | 0.576, 0.576 (seed 2 not run: budget cut logged before it ran) |
| random | 0–2 | 0.554, 0.552, 0.553 |
- **Q3-A1 PASSED** (5/5 ≥ 0.6595). **Q3-A2 PASSED on 2 seeds** (deviation: the frozen text names 3; the third was cut
  before running — the claim is weaker than frozen). **Q3-A3 PASSED** (0.717 < 0.740; every T1 parent ≥ 0.70).
  **Q3-A4 PASSED** (3/3 fail). P-group (no prediction): the seed-0 marginal success did not replicate (2/3 fail).

## Q3-B — GATE seeds 1, 2 (GATE_FULL_CAUSAL_CIRCUIT §4)
- **Q3-B1 PASSED** (0.903, 0.900). **Q3-B2 FAILED** (layer-2 readout attention ablation: 0.939, 0.663).
  **Q3-B3 FAILED** (seed 1: L2-h1 + L2-h3 qualifies; seed 2: no qualifying pair — single necessary head).
  **Q3-B4 FAILED** (seed 2: layer-0 attention at the evidence tokens transfers 0.00; the gate is computed in layer 1).

## Q3-C — TRANS (TRANS_COMPOSITION_RESEARCH §2)
- `trans_full_s0` (12k): C-only 0.336 → **not learned**. `trans_aux_s2` (8k, map supervision): C-only 0.332 → **not learned**.
- `trans_holdout_s1` (8k): on training pairs C-only 0.562, masking τ evidence → 0.416, masking σ → 0.432 → **learned (on
  training pairs)**; on held-out pairs C-only 0.246 → **does not generalise**.

## Q3-D — structure-to-extrapolation, 6 held-out TF + 10 held-out DS (`logs/extrap/heldout_outcome.json`; hashes verified)
- **Q3-D1 PASSED** (all 10 DS < 0.30, max 0.215; TF mean 0.620).
- **Q3-D2 FAILED** (the *negative* structural prediction): SAT has within-TF Spearman **0.71** and MAE 0.123 < constant
  0.166; GCD has Spearman **−0.83** (strong, opposite to its frozen fit's sign; MAE 0.178 > constant). On-support structure
  therefore carried within-Transformer extrapolation information that the frozen prediction said it would not.
- **Q3-D3 PASSED** (A2 within-TF Spearman 1.00, MAE 0.019).
- Caveat: the 6 held-out Transformers share T1 parents with development models (new T4 trajectories only); n = 6.

## Q3-E — adaptive sources, from scratch (`adapt_s0`)
- **Q3-E1 FAILED** (changed 0.656 vs static 0.660). **Q3-E2 FAILED** (0.49).

## Q3-E′ — adaptive sources, T1 curriculum replication (`adapt_fromT1p1_s1`)
- **Q3-E′1 PASSED** (changed 0.691 vs static 0.660; unchanged 0.736 vs Bayes 0.735). **Q3-E′2 PASSED** (0.88).
  **Q3-E′3 PASSED** (history intervention follows new exact answer 0.88).
