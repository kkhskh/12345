# Reconstruction of the XIV T4 pathway result (XVIII step A1)

**Status.** COR Phase 1.5, Step 1 (provisional ≈ 25%, unchanged).

**Evidence labels:**
- **[code]** read from the implementation;
- **[replay]** bitwise re-execution;
- **[XIV frozen]** from XIV's frozen scorer output;
- **[dev]** a new XVIII development measurement on XIV checkpoints, made after XIV's freeze. Development measurements are not confirmations.

## 1. What was run

**Model** [code: `step1_cross_family_relational_foundations/scripts/ft_factorial_c.py`, `src/native.py`].
- A 3-layer Transformer, d = 64, 4 heads, with 43 tokens:
  - 36 source tokens (index 6q + s);
  - 6 readout tokens (index 36 + q);
  - 1 episode token.
- Fine-tuning starts from a pretrained native parent (uP160):
  - embeddings and layers 0–1 are inherited, and remain trainable;
  - layer 2 and the readout map W_A are freshly initialised from seeds L and W;
  - the data stream comes from seed D.
- Optimiser: AdamW, with a warm-up of 200 steps followed by cosine decay, peak learning rate 1e-3 and weight decay 1e-4.
- Training lasts 4000 steps. Most measurements are taken at step 1211.

**Cohort** [XIV frozen].
- One parent and two organisational bases:
  - E: L = W = 21, the early-readout organisation;
  - M: L = W = 23, the mixture organisation.
- Three data orders per base (D = base seed, 801, 802), giving n = 6 base × order pairs.
- These are fine-tuning variants of one pretrained parent, not independent pretraining replications.

## 2. Exact intervention edge sets (A1: code against description)

Every edge set is a set of (query token, key token) pairs in one layer. A mask entry adds −1e9 to the attention logit for steps 0–99.

| intervention | layer | query rows | key columns | edges per readout row | source |
|---|---|---|---|---|---|
| **E real block** (`ro0`) | 0 | all 6 readout tokens | **all 36 source tokens** | 36 | `mask_for`: `m[36:42, 0:36] = NEG` |
| E off-route sham (`msk`, `oq0`) | 0 | readout q | sources of the other 5 questions | 30 | `maskfwd.edges("oq0")` |
| E location-matched noise (`lnz`, `ro0`, σ = 2.5) | 0 | all readout rows | all 36 sources | 36 | same edges as the block |
| **M real block** (`l2q`) | 2 | readout q | **own-question sources** (6q … 6q + 5) | 6 | `mask_for`: `m[36+q, 6q:6q+6] = NEG` |
| M off-route sham (`msk`, `rd2`) | 2 | all readout rows | readout and episode tokens | 7 | `maskfwd.edges("rd2")` |
| M location-matched noise (`lnz`, `own2`, σ = 1.25) | 2 | readout q | own-question sources | 6 | same edges as the block |

**Finding A1-1 (description against code).**
- The XIV reports describe the E sham as acting on edges "outside both routes". In the code, the E sham's edges (readout → other-question sources) are a **strict subset of the E real block's edges**: the block covers all 36 sources, and the sham covers the 30 other-question sources.
- The description is therefore correct only if the E route is defined as the own-question edges (6 per readout row). Under that definition the real block covers the route plus 30 more edges.
- **Consequence:** on E, the real-block − sham contrast isolates the 6 own-question edges per row, given that the other-question edges are already blocked. It does not compare "route" against "elsewhere".
- The M description matches the code.

**Finding A1-2 (own-question edges carry the E route's necessity)** [dev, XIV step-1211 checkpoints].
- At test time, blocking only the own-question edges at layer 0 (`own0`) raises KL nearly as much as the full block:
  - own0: +1.02 to +1.20;
  - full block (`ro0`): +1.18 to +1.38;
  - over the 6 early-organised XIV runs.
- This test establishes necessity in the trained model. It does not show which edges matter during training. Whether a training-time block of `own0` alone flips E is tested in XVIII development.

## 3. Bitwise replay

- XVIII's copy of the harness (`scripts/xviii_train.py`) reproduces two runs bitwise at steps 100 and 150, with max |Δ| = 0.0 over all parameter arrays [replay: `logs/replay_check.json`]:
  - XIV's E/D801 baseline (`uP160_fx_L21_W21_D801_stop1211_B_base`);
  - its real-block run (`…_ro0T100_stop1211_B_ro0100`).
- The XVIII modes add code that executes only when one of the new mode names is set.

## 4. The organisational diagnostic, and two independent checks

**XIV's frozen rule** (`score_I.py`, `cls1211`):
- **EARLY** if mix_mass < 0.5 and lik_RO1 ≥ 0.4.
- **MIXTURE** if mix_mass ≥ 0.8 and lik_RO1 ≤ 0.15.
- **HYBRID** otherwise.

Both quantities are route-structural:
- `mix_mass` is the maximum over layer-2 heads of the readout's attention mass on the own-question sources. On M these are **exactly the edges the M block manipulates** (XIV review B, A5).
- `lik_RO1` is a residual-patch fidelity at the readout token after layer 0. This is the site from which the E block removes source attention.

**Independent check 1: behavioural exponent ρ (attention-free)** [dev].
- ρ is the nested power-mean exponent fitted to the model's output distribution on 3000 natural episodes (XIII-B's family, `metrics.rho_fit`). It uses no internal quantity.
- On the 24 XIV step-1211 checkpoints:

| organisation by XIV rule | runs | ρ | KL |
|---|---|---|---|
| EARLY (3 E baselines, 6 E controls, 3 M real-block flips) | 12 | **0.48–0.58** | 0.024–0.036 |
| MIXTURE (3 M baselines, 6 M controls, 3 E real-block flips) | 12 | **0.68–0.81** | 0.010–0.013 |

- The two classes do not overlap: the largest early ρ is 0.577 and the smallest mixture ρ is 0.682.
- Every real-block flip moves ρ into the range of its new class:
  - E → mixture: ρ 0.71–0.78;
  - M → early: ρ 0.52–0.54.
- XIII-B's caveat stands. ρ's reading as "a different combination operator" is not unique: it is algebraically equivalent to tempered evidence plus re-sharpening (ledger N20). Here it is used only as a **behavioural discriminator**, not as a mechanism.

**Independent check 2: test-time route necessity (causal, not an attention statistic)** [dev].
- The test raises KL by blocking each route at test time on the trained model:
  - nec_E = KL rise when blocking the E route;
  - nec_M = KL rise when blocking the M route.

| organisation | nec_E | nec_M |
|---|---|---|
| early (12) | **1.18–1.38** | 0.001–0.017 |
| mixture (12) | 0.004–0.10 | **0.57–0.74** |

- This is a complete double dissociation, and every real-block flip follows its new class.

**Conclusion for XVIII.** On the XIV runs, the organisational change is corroborated by a behavioural and a causal diagnostic. Neither uses the manipulated attention statistic. So the XIV flips are changes in how the competent model combines evidence and which pathway it needs, not only changes in an attention score. XVIII's frozen classification rule (PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md) requires all three diagnostics to agree.

## 5. Loss disruption of the XIV interventions (existing data)

Immediate change in training loss when the intervention is applied to the baseline at steps 25/50/100 [XIV dev, `i_calib.json`]:

| | E real | E sham | E noise σ2.5 | M real | M sham | M noise σ1.25 / σ3 / σ8 |
|---|---|---|---|---|---|---|
| Δloss @ 25 | 0.045 | −0.005 | ≈ 0.04 | 0.006 | 0.005 | ≈ 0.00 / 0.014 / 0.046 |
| Δloss @ 100 | **0.548** | 0.031 | ≈ 0.18 | **0.011** | 0.040 | 0.001 / 0.034 / 0.082 |

**On M:**
- The real block's immediate loss disruption is no larger than the sham's, and it is smaller than strong noise on the same edges. Yet only the real block flips the organisation.
- The block's attention-TV is also the smallest of all M interventions: 0.14–0.17. That is because the fresh layer-2 attention is close to uniform early on, and 6/43 ≈ 0.14 of the mass sits on the own-question sources.
- **So "the M flip is caused by the size of the immediate loss or attention disturbance" is already contradicted by XIV's data** (Check A).

**On E:**
- The block's disruption (0.55 at step 100) exceeds every control tested, so general disturbance is not excluded on E by existing data.
