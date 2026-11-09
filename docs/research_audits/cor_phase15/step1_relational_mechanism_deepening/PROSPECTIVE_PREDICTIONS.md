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
