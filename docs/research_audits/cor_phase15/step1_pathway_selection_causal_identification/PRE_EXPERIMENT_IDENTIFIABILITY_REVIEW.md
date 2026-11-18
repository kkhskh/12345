# Pre-experiment identifiability review (XVIII design, before any confirmation run)

**Status.** COR Phase 1.5, Step 1 (provisional ≈ 25%, unchanged). This document is the design submitted to the two pre-freeze reviewers (mandate §2, §12). Their objections and my decision are in §9–§10.

**Labels:**
- **[proof]** a mathematical derivation;
- **[code]** read from the implementation;
- **[XIV]** existing frozen data;
- **[dev]** XVIII development data. Development data use XIV's data orders and the dev measurement stream; confirmation uses fresh data orders and a fresh stream.

## 1. The phenomenon, precisely

- **The block.** For steps 0–99 of fine-tuning, a set T of attention edges receives a −1e9 logit bias.
- **The effect.** On base E (T = layer-0 readout → all 36 sources) and on base M (T = layer-2 readout q → the 6 own-question sources), this switches the organisation that forms afterwards: E → mixture, M → early. It does so in 6 of 6 base × order pairs of one parent (uP160) [XIV].
- **The new organisation is real.** It is corroborated by two diagnostics that do not use the manipulated attention statistic: the behavioural exponent ρ and test-time route necessity (XIV_RESULT_RECONSTRUCTION.md §4) [dev on XIV checkpoints].

## 2. What the −1e9 block does, exactly [proof]

Write s_j for the logits of one query row, a = softmax(s), and o = Σ_j a_j v_j for the head output. With bias −B on j ∈ T and B = 1e9, a_j = 0 exactly in float32 for j ∈ T. The block therefore makes **three** changes during the window:

1. **(F) Forward deprivation.** o contains no contribution from T. The query's mass is renormalised over the complement C.
2. **(R) Routing-gradient removal.** For j ∈ T, ∂L/∂s_j = a_j (∂L/∂a_j − Σ_k a_k ∂L/∂a_k) = 0. The parameters that set the logits of T (W_Q of the readout rows and W_K of the keys, through those edges) receive no gradient through T. The route cannot be "recruited" while it is blocked.
3. **(V) Value-gradient removal through the head.** ∂L/∂v_j = a_j W_O^⊤ ∂L/∂o = 0 for j ∈ T in this head and row.

XIV's sham and location-matched noise changed neither (R) nor (V), and they *raised* T's share. So they could not separate these components (XIV review B, A1).

## 3. Lemmas that remove non-discriminating designs (Check A)

- **L1. Boosting the competitor is the block** [proof].
  - softmax(s + c·1_C) = softmax(s − c·1_T): adding +c to every non-target logit of a row is identical, as a function, to adding −c to the target logits.
  - So "reducing the route's contribution by boosting the alternative route" (mandate A3) is **the same intervention** as graded suppression whenever the boosted set is the whole complement.
  - A boost restricted to a strict subset C₁ ⊂ C differs only by **where the diverted mass goes** within C. Every hypothesis in §4 is stated in terms of T's quantities (its information, share or gradient), so none predicts that the destination matters, except H4 in a form we cannot state prospectively.
  - **Decision: no diversion arm.** It is either identical to graded suppression or a test without a stated competing prediction.
- **L2. Graded suppression cannot separate information from share or gradient** [proof].
  - A finite offset b multiplies T's attention odds by e^{−b}.
  - The forward contribution of T is a_T ō_T. Its signal-to-interference ratio against the complement's contribution scales as (a_T/(1 − a_T))² for fixed values.
  - The routing gradient on T scales as a_T. Both are strictly monotone in the same scalar a_T(b).
  - So every member of the offset family moves "information available through T", "T's share" and "T's routing gradient" together. H1 (graded), H2a and H2g in §4 therefore all predict a monotone dose–response, and they are **observationally equivalent** along this family.
  - **Decision:** graded suppression answers only the mandate's P1 ("is complete removal necessary?"). It is a secondary, descriptive arm, not a discriminating one.
- **L3. Measuring the effect of a finite offset requires achieved, not nominal, suppression.** The logits are learned during the window, so the network can partly undo a finite offset. Achieved share is measured from snapshots (steps 25, 50, 100) [dev].
- **L4. Existing data already exclude size-of-disturbance explanations on M** [XIV].
  - On M the real block's immediate Δloss is 0.006–0.011, no larger than the off-route sham's (0.005–0.040). Its attention-TV is 0.14–0.17, the smallest of all M interventions.
  - Strong same-edge noise, with larger Δloss (up to 0.082) and TV (up to 0.85), does not flip M.
  - "Size of the immediate loss or attention disturbance" is therefore **not** a viable H3 on M. It remains open on E, where the block's Δloss (0.55 at step 100) exceeds every tested control.
- **L5. A 2 × 2 factorial separates (F) from (R)** [proof + code + unit check].
  - Two new modes, implemented in `scripts/xviii_train.py`:
    - **gsg (gradient-only block):** s_T ← stop_gradient(s_T). The forward pass is unchanged, and (R) is removed.
    - **fst (forward-only block):** a = a_full + stop_gradient(a_blk − a_full). The forward pass is identical to the block, and the logit gradient is the unblocked softmax Jacobian, so (R) is present.
  - (V) follows the forward weight: (V) is present in plain and gsg runs and absent in fst and block runs. This is unavoidable, because a value read with weight 0 has gradient 0.
  - Unit checks (`logs/unit_modes.json`), on the parent-initialised models at step 0:
    - gsg logits equal the plain logits exactly;
    - gsg's gradient on the target logits is exactly 0 (against −0.0068 / −0.0056 in plain runs);
    - fst logits equal the block's to within 1.4e-6;
    - fst's parameter gradient differs from the block's by 7.8% (E) and 1.4% (M) in relative norm.

**Limitation of gsg** [proof]. Stopping the gradient on T's logits does not freeze T's *share*. T's share can still change through the complement's logits, which keep their gradient: lowering them raises T's share. gsg therefore removes the direct recruitment signal for T, not every change in T's share. The achieved share trajectory is measured under every arm (steps 25, 50, 100, 150, 200, 300, 600) and reported against the plain run's.

| | (R) routing gradient on T present | (R) removed |
|---|---|---|
| **forward through T intact (with V)** | plain baseline | **gsg** |
| **forward through T blocked (no V)** | **fst** | block (XIV real block) |

- **L6. Value resampling separates information from share, but not from routing gradient** [proof].
  - **vrs** keeps T's attention weights exactly. It replaces the value vectors read through T with those of another episode in the batch (`jnp.roll(v, 1, axis=0)`, on the T edges only).
  - Episode-specific evidence through T is destroyed, while the attention mass and the marginal distribution of the values are preserved.
  - Routing gradients still flow. They now carry the signal that T is uninformative, so they also push T's share down. vrs therefore separates H1 from H2a but not from H2g.

## 4. Hypotheses, sharpened

- **H1. Forward information deprivation.** The selection is caused by the absence (or degradation) of T's relational information in the forward computation during formation. Operationally: what matters is T's forward contribution o_T, removed by block and fst and destroyed in content by vrs.
- **H2a. Forward share or contribution competition.** What matters is T's attention mass, or the magnitude of its forward contribution, whatever it carries. Removed by block and fst; not by vrs (mass intact) or gsg (forward intact).
- **H2g. Learning-signal (routing-gradient) starvation.** What matters is that T cannot be recruited during formation: no gradient reaches the parameters that would grow T, while the competing organisation's parameters keep learning. Removed by block and gsg; *not* by fst, whose routing gradient is that of an unblocked softmax evaluated at the deprived forward state, so it actively says "attend to T".
- **H3. General disturbance.** Selection tracks the size of the training disturbance (immediate loss, gradient-norm change), whatever its route.
  - Already excluded on M by L4.
  - gsg has zero forward disturbance by construction (Δloss = 0 at every window step), so a gsg flip excludes H3 on that base.
  - The gradient-norm change is measured (`window_log.json`).
- **H4. Interaction.** Selection needs both (F) and (R): only the full block flips, or the components add sub-additively.
- **H5. Parent specificity.** The effect does not survive in independently pretrained parents.

## 5. Prediction table (Check B)

Outcome: the frozen three-diagnostic organisational class at step 1211 (§7). "Flip" means the class moves to the other organisation relative to the paired baseline. "—" means no flip.

| arm | H1 | H2a | H2g | H3 (E only) | H4 |
|---|---|---|---|---|---|
| block (reference) | flip | flip | flip | flip | flip |
| **gsg** (forward intact, no routing gradient) | — | — | **flip** | — (zero disturbance) | — |
| **fst** (forward blocked, routing gradient present) | **flip** | **flip** | — | flip if Δloss ≈ block | — |
| vrs (content destroyed, share intact) | flip | — | flip or partial | depends on Δloss | — / partial |
| graded b (secondary) | monotone | monotone | monotone | monotone | — |

**Distinct joint patterns over (gsg, fst):**
- (—, flip): H1 or H2a. vrs then separates them: flip → H1, — → H2a.
- (flip, —): H2g.
- (—, —): H4.
- (flip, flip): over-determined. Either component suffices, and the components are redundant.

Every hypothesis predicts a distinct joint pattern except H1 against H2a, which vrs splits. **This table is the core of the design.**

## 6. Development results that calibrate the design

(Filled in from `logs/dev/measure_dev.jsonl`; see §6a once the development runs finish. These are XIV data orders D = 801 on uP160.)

## 7. Organisational diagnostic and competence (frozen with the design)

- **Structural.** XIV's rule `cls1211`: mix_mass, lik_RO1.
- **Behavioural.** ρ ≤ 0.60 → EARLY; ρ ≥ 0.66 → MIXTURE; otherwise HYBRID. Development separation on 24 XIV checkpoints: ≤ 0.577 against ≥ 0.682.
- **Causal.** Test-time route necessity:
  - nec_E ≥ 0.30 and nec_M ≤ 0.10 → EARLY;
  - nec_M ≥ 0.30 and nec_E ≤ 0.15 → MIXTURE;
  - otherwise HYBRID.
  - Development separation: early nec_E ≥ 1.18 and nec_M ≤ 0.017; mixture nec_M ≥ 0.57 and nec_E ≤ 0.10.
- **Final class.** The common label if all three agree, otherwise HYBRID.
- **Competent.** KL ≤ 0.06 on the measurement stream. XIV runs are at 0.010–0.036.
- **Flip.** A competent run whose final class is the opposite organisation to its paired baseline's final class. A HYBRID result is reported as "partial", not as a flip.

## 8. Cohort, units and compute

**Within-parent confirmation (uP160).**
- Two bases (E: 21/21, M: 23/23) and three **fresh** data orders (D = 811, 812, 813).
- Arms: baseline, block, gsg, fst and vrs, plus two graded offsets (b = 2.3, 4.6) as a secondary arm.
- That is 7 arms × 6 = 42 runs, of about 10 minutes each, so about 3.5 h on 2 lanes.
- **Statistical unit:** the base × order pair (n = 6, within one parent). These are not independent pretraining replications.

**Independent parents (C).**
- Candidate parents: uP161–uP165, each independently pretrained.
- uP161 is recorded as having **no early base** under the same recipe (0/32 early in XIII-E). It is listed in the denominator as a non-evaluable parent.
- **Screening rule (predefined):** for each of uP162–uP165, train baselines L = W = s for s = 20, 21, 22, 23 (D = 811) and classify with §7. The parent is evaluable if at least one competent EARLY seed and one competent MIXTURE seed exist. The base is the first such s in that order. If a class is missing, s = 24 and 25 are added, and then the search stops.
- On each evaluable parent: both bases × 2 data orders (D = 812, 813) × {baseline, block, decisive arm(s) from §5}.
- **Unit:** the parent. Results are reported per parent and pooled with a parent-level (hierarchical) summary.

**Statistics.**
- Exact binomial counts, with Clopper–Pearson intervals, per base and per parent.
- A run that is not competent counts as **not a flip**: fail-closed.
- No p-value is computed across fine-tuning runs as if they were independent parents.

## 9. Pre-freeze reviews

(Reviewer A, mathematics and identifiability; Reviewer B, design, alternatives and novelty. Inserted when they are received.)

## 10. Design-review decision

(Written after §9.)
