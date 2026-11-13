# Native relational learning (Workstream B)

**Setup.** Every condition receives the raw token sequence only. No engineered counts, reports, eligibility values or
raw-input side channels are given at training or at inference. Training uses generator labels, AdamW warm-up/cosine to
1e-3, batch 256, and 4000 steps unless stated (`scripts/native_train.py`, `scripts/native_reader.py`). Metrics are
per-episode absolute errors (absE, primary) and signed means (E), as defined in `src/native.py`.

## 1. What ran

| condition | parents | status |
|---|---|---|
| C1: original transfer (frozen 0–1, fresh layer 2) | xP80, xP81 (dev); **yP130–135 (conf)** | ran |
| keep3: all pretrained layers frozen, fresh layer on top | xP80, xP81 (dev) | ran |
| **full_c1: full fine-tuning in the original architecture** | xP81 (dev; a first run was lost at step 1803 and rerun, reproducing it exactly); **yP130, 131, 132, 134 (conf)** | ran |
| native-state-only readers on frozen states (lin_ro, mlp_ro, mlp_q, ds_src, bilin_src, pair_src, attn_q, ds_all, attn_all) | xP80 (all, resid_pre2; 6 at final states); xP81 (mlp_q, ds_src, pair_src); **yP130–135 (mlp_q, conf)** | ran |
| decode-then-compose (linear probes + exact formula) | xP80, xP81 (dev); **yP130–135 (conf)** | ran |
| full_keep3 | xP81 | **lost at mid-run, not rerun** |
| LoRA on layer-1 attention / MLP / both; unfreezing layer 0 only; keep3x2; raw-token scratch3; C1 at 16k; C1 with Bayes targets; full_c1 at 16k | xP81 | **not run** (queued in `logs/queue_dev_C2.txt`; compute was spent on confirmation) |
| curriculum with a separately specified relational objective | — | **not run** |

Investigation VIII's single-parent results for unfreeze1 (−0.31), lora1 (−0.44) and deep2 (−0.55) on xP80, an
incomplete parent, remain the only LoRA and partial-unfreezing evidence.

## 2. Confirmed results (independent pretrained parents; `PROSPECTIVE_PREDICTIONS.md` Part II)

| parent (complete) | C1 KL / absE_R1 / E_R1 / absE_R23 | **full_c1** KL / absE_R1 / E_R1 / absE_R23 / absE_F34 | native mlp_q (frozen states) absE_R1 / E_R1 |
|---|---|---|---|
| yP130 | 0.043 / 1.05 / −1.02 / 0.72 | **0.002 / 0.21 / −0.09 / 0.18 / 0.14** | 0.52 / −0.42 |
| yP131 | 0.076 / 1.01 / −0.91 / 1.07 | **0.003 / 0.30 / −0.18 / 0.22 / 0.14** | 0.50 / −0.34 |
| yP132 | 0.032 / 0.81 / −0.76 / 0.54 | **0.003 / 0.21 / +0.14 / 0.23 / 0.17** | 0.44 / +0.00 |
| yP134 | 0.074 / 1.06 / −0.97 / 0.73 | **0.007 / 0.35 / +0.26 / 0.27 / 0.16** | 0.51 / −0.44 |

Exact Bayes has 0 on every metric.

1. **The original Transformer can learn the cross-group operation through its own layers (NP1, 4/4 parents).**
   - Full fine-tuning cuts the per-episode R1 error 3–5× and R2/R3 error 3–6×, with KL ≈ 0.002–0.007.
   - Residual errors remain: absE_R1 0.21–0.35, and one parent overshoots (+0.26).
   - This is **Type III** (modified native layers learn a useful new relational computation). No bypass is possible: the
     model receives raw tokens only.
2. **Unfreezing is what matters: the frozen prefix cannot propagate relational information across groups (NP3, 4/4).**
   In C1, patching the other groups' token states transmits ≈ 0 of the R1 response (share −0.02 to −0.04).
3. **A native-state reader of frozen states is not enough at this budget (CP4 falsified).** A q-token MLP on the final
   pretrained states improves over C1 on every parent but leaves the R1 mean wrong on 3 of 4. In development, the same
   reader with Bayes targets and 32k steps reached Bayes on xP81. Accessibility therefore holds (CP1), but cheap learned
   integration from frozen states is budget- or objective-dependent. This is a truncated-optimization result, **not** an
   impossibility.

## 3. Distinguishing native computation from bypass (mandate B2)

| type | evidence here |
|---|---|
| I: external bypass | Investigation VIII's radapt only |
| II: frozen native states are sufficient for a new computation | decode-then-compose: exact Bayes, linear, 4/4 complete parents. Development mlp_q with Bayes targets and 32k steps |
| III: modified native layers learn the computation | **full_c1, 4/4 independent parents** |
| IV: a causally identified native pathway | development parent only (other groups' q-tokens: necessary and sufficient); **did not recur** (NP2 0/4). What recurs is attention-dependence of the R1 credit (freeze, 4/4) and the absence of propagation in frozen prefixes |
| V: generalization plus prospective predictions | not reached for native models |

## 4. Failure modes recorded

- Incomplete pretraining (2 of 6 new parents, about a third of all lineages seen) leaves a group's count absent. Those
  parents fail on KL and R2/R3 under every frozen-state method (CP1, CP2).
- With frozen layers, a downstream block with exact counts available (keep3) takes the separable shortcut *more*
  strongly (dev: E_R1 −1.49).
