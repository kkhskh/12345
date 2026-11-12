# Frozen confirmation protocol — Investigation VII

**Status: FROZEN** in the commit that adds this file. That commit comes before any T4 transfer, repair or evaluation of the
confirmation parents. The hashes of every confirmation script are listed in `logs/FREEZE7_HASHES.txt`.

**Disclosure.** Exactly one 20-step smoke run (`yP90 C1_injm`, untrained) was executed on a confirmation parent to test the
pipeline before this freeze. Its outputs were deleted, and no reported quantity comes from it.

## 1. Causal hypothesis frozen here

All of the following comes from development work on the Investigation VI parents, which are development data only.

**H7.** P-count transfer solutions get R1 wrong because the cross-group *eligibility* statistic m_g is not usefully accessible
to the one-block downstream learner. m_g counts the other groups that are perfect and agree at q. Theorem 1 shows it is
necessary for the Bayes R1/R3 sign pattern. Without m_g, the learner settles on a near-separable, perfect-agreement-discounting
rule, whose R1 and R3 signs must agree (Theorem 1).

H7 makes four predictions.
1. **Supplying m_g** at inference, computed from the episode by a fixed function, moves R1 toward the Bayes response, with
   R2/R3 and average quality preserved.
2. **Supplying separable count features** n_g, which are capacity-comparable and already present in the frozen state, does not
   help R1. It pushes R1 further toward the separable discount.
3. **A capacity-matched irrelevant feature** does nothing.
4. In the frozen P-count representation, "another group is perfect" is **not linearly accessible** at the q-source tokens.

**Development basis** (exploratory; parents P80, P81, P83 and T70; `logs/dev/`):

| condition (P80) | R1 excess E |
|---|---|
| original transfer | −0.895 |
| m-injection | +0.225 (P81: −0.716 → −0.232) |
| n-injection | **−1.980** |
| random injection | −0.888 |

On P80, m-injection also improved KL (0.104 → 0.062). Separately, a positive-control oracle reader with full count information
but without m failed R1 at 4000 steps (E −0.28), while matching Bayes on average (KL 0.002).

## 2. Cohort

- **Parents.** Eight new P-count parents, seeds **90–97**, pretrained under the Stage-0 freeze (commit a688902).
- **Independence.** They descend from no development parent.
- **T1 panel.** None. This is a resource decision; the confirmation question concerns P-count repair. It is not about the T1
  contrast.
- **Success rule** (inclusion; reported, not used to exclude): original-transfer (C1) step-4000 accuracy ≥ 0.6595 and finite
  losses.
- **Primary analysis.** Intention-to-train over all 8 parents. A missing or crashed run counts as a failure:
  - Δ = −10 for P1;
  - +10 for the comparators, which is worst-case for P2 and P4;
  - non-success for P3.

## 3. Conditions per parent

All conditions share the same 4000-step budget, AdamW warmup-cosine schedule and batch 256. The fresh layer 2, the transfer
data stream and everything else are matched to the original transfer of Investigation VI (fresh layer 2 from
`nets.init(seed + 7777)`; data stream `1_000_003·(seed+1) + 7919·4 + 93`). Training uses generator labels (cross-entropy) in
every condition. Code: `scripts/downstream.py` with `PHASE=conf`.

| condition | description |
|---|---|
| **C1** | original transfer: frozen embeddings and layers 0–1; inherited WA and final LN; fresh layer 2 |
| **C1_injm** (the repair) | as C1, plus a trainable, zero-initialised linear map from the one-hot eligibility vector m (3 groups × 3 classes, computed from X only) added to the six readout-token inputs of layer 2 |
| **C1_injn** | the same with one-hot leave-q counts (3 × 6). Count control |
| **C1_injrand** | the same with 9 fixed pseudo-random binary features of the episode. Capacity control |
| **W_bal** | C1 with per-decision loss weights that equalise the three natural decision classes in expectation (one perfect group; two agreeing; two disagreeing). Generic data-balancing baseline; **descriptive only, no frozen prediction** |

**Inference-time information.** The repaired model receives only X, from which m is computed deterministically. It never
receives the Bayes posterior, the answer, a regime label or the latent roles. Training supervision is the generator label
alone; no Bayes targets are used.

## 4. Evaluation

- **Set.** `logs/eval_conf.npz`: stream **17_900_000**, 6000 natural episodes plus all edit families, built by
  `scripts/build_eval.py conf 17900000`. It has never been used for development.
- **Statistics.** Taken at step 4000 (a step-1000 evaluation is recorded and is descriptive).

| statistic | definition |
|---|---|
| E_R1 | mean Bayes-corrected excess on F45-R1 pairs |
| absE_R23 | pair-weighted mean \|E\| over F45-R2 ∪ R3 |
| E_F34R1 | excess on F34-R1 pairs |
| F0abs | mean \|Δm\| on F0 pairs |
| KL, acc | over all natural decisions |

## 5. Frozen predictions and criteria

The unit is the parent. "Paired" means a per-parent difference against C1. Tests are exact sign-flip (2^8 = 256 assignments),
one-sided. CIs are from a parent bootstrap (20,000 resamples).

| id | prediction | criterion |
|---|---|---|
| **P1 (primary)** | m-injection raises E_R1: Δ = E_R1(injm) − E_R1(C1) > 0 | p < 0.05 **and** 95 % CI lower bound > 0 |
| **P2 (primary)** | the gain is specific to m, not to added capacity: Δ(injm) − Δ(injrand) > 0 | p < 0.05 and CI lower bound > 0 |
| **P3 (selective-repair score)** | per parent, success iff Δ(injm) ≥ 0.20 and all of: absE_R23 rises by ≤ 0.15; KL rises by ≤ 0.010; acc falls by ≤ 0.005; \|E_F34R1\| rises by ≤ 0.10; F0abs rises by ≤ 0.05 (each relative to that parent's C1) | ≥ 6 of 8 successes. Components reported per parent |
| P4 | separable count features worsen R1: Δ(injn) < 0 | p < 0.05 |
| P5 | replication on new lineages: the original C1 raw R1 response Δm_R1 < 0 | ≥ 7 of 8 parents |
| P6 | H2 ("two or more perfect groups") from the q-source tokens of the frozen state has held-out linear-probe accuracy < 0.96 (`scripts/conf_probes.py`) | ≥ 6 of 8 parents |
| P7 | in the C1 models, the readout-token-only input route carries a negative R1 effect (`scripts/conf_mech.py`) | ≥ 6 of 8 parents |

**Multiplicity.** P1 and P2 are the primary pair. Both must hold for the causal-repair claim (an intersection–union test, so no
alpha adjustment is needed). P3–P7 are secondary and reported individually.

**Outcome mapping (frozen).**

| result | interpretation |
|---|---|
| P1, P2, P3 all hold | selective behavioural rescue by the specified relational feature, specific against capacity |
| P1 and P2 hold, P3 does not | non-selective rescue |
| P1 holds, P2 does not | capacity or generic effect |
| P1 does not hold | H7's repair prediction fails on new lineages |

W_bal is reported against injm descriptively. If W_bal matches injm, that is reported as evidence for a generic
learning-dynamics explanation (mandate outcome F); it does not enter the frozen criteria.

## 6. Execution

- **Runner.** `scripts/conf_lane.sh` on two lanes over `logs/confqueue.txt` (40 runs, ordered by parent).
- **Then:** `scripts/conf_mech.py`, `scripts/conf_probes.py`, `scripts/confirm7_analysis.py`.
- **Logs.** All starts and ends go to `logs/COMPUTE_LOG.txt`. Failed runs are kept.
- **No looks.** No confirmation outcome is examined before all 40 runs finish.
