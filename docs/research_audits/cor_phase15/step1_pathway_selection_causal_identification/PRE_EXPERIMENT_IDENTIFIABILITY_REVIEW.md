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

**Source.** `logs/dev/measure_dev.jsonl`, on uP160 with data order D = 801. This is one XIV data order and the development measurement stream 86_100_000. **One run per cell**, so these results are not confirmations.

| arm, base | lik_RO1 | mix_mass | ρ | nec_E | nec_M | KL | final class |
|---|---|---|---|---|---|---|---|
| baseline E (XIV) | 0.743 | 0.174 | 0.512 | 1.240 | 0.004 | 0.031 | EARLY |
| block E (XIV) | 0.002 | 0.955 | 0.779 | 0.011 | 0.753 | 0.011 | MIXTURE |
| **gsg E** | 0.257 | 0.945 | 0.570 | 0.157 | 0.385 | 0.017 | **HYBRID** (struct H, behav E, causal H) |
| **fst E** | 0.007 | 0.956 | 0.784 | 0.016 | 0.729 | 0.011 | **MIXTURE** |
| **vrs E** | 0.001 | 0.957 | 0.833 | 0.007 | 0.801 | 0.010 | **MIXTURE** |
| baseline M (XIV) | 0.050 | 0.956 | 0.705 | 0.033 | 0.599 | 0.013 | MIXTURE |
| block M (XIV) | 0.749 | 0.163 | 0.542 | 1.345 | 0.009 | 0.027 | EARLY |
| **gsg M** | 0.241 | 0.945 | 0.586 | 0.169 | 0.384 | 0.013 | **HYBRID** (struct H, behav E, causal H) |
| **fst M** | 0.743 | 0.184 | 0.556 | 1.241 | 0.008 | 0.025 | **EARLY** |
| **vrs M** | 0.738 | 0.166 | 0.545 | 1.299 | 0.008 | 0.026 | **EARLY** |

**Readings** (development only):
1. **The forward component alone suffices.** fst reproduces the block on both bases, and all three diagnostics agree. It does so even though fst's routing gradient is that of an unblocked softmax, which pushes towards T.
2. **The gradient component alone is partial.** gsg leaves the forward pass and the loss identical at every window step, yet it moves both bases about a third to two thirds of the way. Measured by test-time necessity, φ ≈ 0.72 on E and 0.27 on M.
   - Both gsg runs land in **the same hybrid state**, whichever base they started from: lik 0.24–0.26, mix_mass 0.945, nec_E 0.16–0.17, nec_M 0.385, ρ 0.57–0.59.
   - The diagnostics disagree in magnitude for hybrids. On E, ρ moves only 22% of the way; on M it moves 73%.
3. **Destroying content with the attention mass intact suffices.** vrs flips both bases.
4. **Mediator, not yet separated.** Across arms the outcome tracks how far T's share grew by the end of the window (unmasked measurement):
   - E, own-question share at layer 0 at step 100: baseline 0.234, gsg 0.188, fst 0.155, block 0.135, vrs 0.140, against 0.134 at step 25.
   - M, max-head own-question share at layer 2 at step 150: baseline 0.251, gsg 0.194, fst 0.171, vrs 0.146, block 0.142.
   - vrs therefore does **not** keep the share *growth* of the baseline. It keeps the share at its starting level: the content is useless, so the route is not recruited.
   - **So vrs separates "information content" from "attention mass as set by the intervention", but not from "route establishment (share growth) as a downstream consequence".** This is a design limitation, submitted to the reviewers.
5. **Window loss and gradient** (`window_log.json`). Training loss at steps 0 / 50 / 99:
   - gsg: identical to the plain run by construction;
   - fst: 1.43 / 1.10 / 1.08 (E) and 1.29 / 1.10 / 0.98 (M);
   - vrs: 1.33 / 1.09 / 1.07 (E) and 1.28 / 1.09 / 0.98 (M).
   - The gradient norm at step 0 is 3.9–6.8 in all modes.

**Graded offsets** (b = 2.3, 4.6) and the E own-question-only block were still running when the reviewers started. Their results are added to `logs/dev/measure_dev.jsonl`.

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

## 9. Pre-freeze reviews (summary; full text in `reviews/PRE_REVIEW_A_MATH_IDENTIFIABILITY.md`, `reviews/PRE_REVIEW_B_DESIGN_NOVELTY.md`)

Both reviewers are agents; this is not human peer review.

**Reviewer A (mathematics and identifiability): REJECT the primary comparison as specified (redesign).** Baseline and block anchors could start immediately. The main points:
- **A1. gsg does not remove (R).** Logit gradients sum to zero over each softmax row, so part of the share-recruitment signal sits on the complement logits, which gsg leaves untouched. gsg keeps a fraction n_T/(n_T + n_C) of the logit-level recruitment signal: 0.84 on E (ro0) and 0.14 on M [proof]; measured 0.72–0.92 and 0.03–0.48 [num]. On E with T = ro0 it also removes all own-question/other-question discrimination, a confound.
- **A2. fst is close to guaranteed.** Its forward pass equals the block's. Its push towards T collapses by step 50, and 75% of the window's learning-rate mass lies in steps 50–99. It refutes only a nominal H2g, so it should be dropped from confirmation.
- **A3. vrs is sound.** Episodes are independent, so there is no leakage, and the magnitude is preserved. vrs flips E with Δloss 0.126 at step 100, which is below the non-flipping XIV noise (≈ 0.18). That contradicts immediate-loss H3 and instantaneous-mass H2a on E. It cannot separate content from the share trajectory.
- **A4. §6 item 4 read a mediator after release (step 150).** At release, gsg-M suppresses the share at least as much as fst-M (φ_share 0.85 against 0.74), yet ends at φ_nec 0.27 against 1.00. So content matters beyond the share at release, on M and in one order. gsg's partial effect, with the forward pass intact, shows that content is not the whole story either.
- **A5. Content and establishment are cause and mediator on one path,** not rival hypotheses. The well-posed question is about controlled direct effects. Proposed fix: an adaptive, stop-gradiented per-head logit offset that clamps T's share to a schedule (the paired baseline's trajectory, or its initial level), with real or rolled content. Decisive cells: shc-hi-rolled and shc-lo-real; validity cell: shc-hi-real (dev).
- **A6. L1 does not cover the competitor route in the other layer.** Add xbo, or state that mandate A3 is not addressed.
- **A7. Observationally equivalent pairs to state explicitly:**
  - share-growth H2a and integrated H2g (merged as Hm);
  - forward content and the value learning it enables;
  - H4 and the additive account (separated by band only).
- **A8. Scoring.**
  - φ is undefined across diagnostics; use log-necessity φ_nec with bands F/P/N.
  - ρ must not drive the class of hybrids.
  - Incompetent runs should be scored as missing, not as "no flip".
  - Hybrids drift: continue P-band runs to step 2546.
- **A9. E should use T = own0.** The own0-only block flips E (preliminary, step 813).

**Reviewer B (design, alternatives, novelty): APPROVE WITH REQUIRED CHANGES** (R1–R10).
- **R1. The step-1211 class is not stationary.** gsg hybrids drift between steps 813 and 1211, and fst-E would be HYBRID at 813. Require stability between 1000 and 1211, or a 4000-step horizon.
- **R2. Add gsv** (forward intact, routing and value gradients through T removed) and split H2g into H2g-R and H2g-RV.
- **R3. Add a share clamp** to separate content from non-recruitment. A single scalar "T established by release" fits every dev arm.
- **R4. Outcome scoring.** Use a continuous primary outcome, add rows for partial/flip outcomes, and make the hybrid and incompetent rules symmetric.
- **R5. Dose.** Before the freeze, run gsg and gsv with a 200-step window.
- **R6. Primary content arm.** Make stop-gradient on the rolled values (vrg) the primary content arm, because vrs trains values to drop content across episodes.
- **R7. Diagnostics.**
  - Fit ρ with a lapse parameter (plain ρ is confounded with lapse-type error; KL alone separates the XIV classes).
  - Add a third-route check (rq1).
  - Competence: KL ≤ 0.045, with accuracy within 0.02–0.03 of the baseline.
  - A pair whose baseline is not unanimous is non-evaluable.
- **R8. H3.** Pre-register the vrs-based exclusion of H3 on E. Amend L4 for ledger N24 (strong layer-2-input noise flipped M, in degraded runs).
- **R9. Parents.** Name the decisive arms, give a per-parent rule, and report k/5 bistable parents.
- **R10. Logging.** Log per-tensor gradient norms in the window.
- **Not blocking:** an Adam-state swap and a parameter-subset transplant at release.
- **Novelty.** The broad phenomenon is established: Kleinman, Achille and Soatto (ICLR 2024) show a gradient-only (`.detach()`) early deficit decides a two-pathway race in deep linear networks; Chen et al. (ICLR 2024) suppress an attention structure early and see an alternative strategy learned; plus Singh et al. 2024/2025, Cloud et al. 2024, and work on competing algorithms. The only candidate contribution is the **forward-information against learning-signal decomposition**, and only if gsv and the share clamp are run.

## 10. Design-review decision (design v2; this replaces §4–§8 wherever they differ)

**Decision: REDESIGN, then freeze.** Every required change from both reviews is adopted, except those listed under "not adopted" with reasons. The revised design is below. Development runs that implement it (dose, gsv, vrg, the clamp's validity cell, own0 arms) were run before the freeze and are listed in §10.7.

### 10.1 Edge sets
T = own-question edges in both bases:
- E: layer 0, readout q → its 6 sources (`own0`);
- M: layer 2, the same edges (`own2`).

The E own0-only block flips E in development (step 1211, all three diagnostics: nec_E 0.004, nec_M 0.81). XIV's ro0 block is not re-run.

### 10.2 Arms (within parent; every arm on both bases × D ∈ {811, 812, 813})

| arm | forward through T | content through T | routing gradient on T | value gradient through T | share of T |
|---|---|---|---|---|---|
| **baseline** (with share log) | intact | real | yes | yes | free |
| **block** (`gof`, B = 1e9; logged; identical forward to −1e9 mask) | none | — | none | none | 0 |
| **vrg** (rolled values, stop-gradient) | intact mass | destroyed | yes | none | free (endogenous) |
| **gsg** | intact | real | none on T's logits (complement keeps n_T/(n_T+n_C) of recruitment) | yes | free |
| **gsv** | intact | real | none on T's logits | none | free |
| **sha-hi-rolled** (adaptive clamp to the paired baseline's per-head trajectory) | clamped mass | destroyed | yes | none | = baseline's |
| **sha-lo-real** (adaptive clamp to the paired baseline's step-0 share) | clamped mass | real | yes | yes | = initial |
| gof b = 2.3 (secondary, mandate P1) | reduced | real | scaled | scaled | reduced |

**Not adopted:**
- **fst.** Dropped from confirmation (A2, accepted). Its development result is reported as development.
- **xbo.** Not run. Mandate A3 ("reduced route contribution by an alternative mechanism") is addressed by sha-lo-real, which reduces T's aggregate odds by an adaptive offset rather than a fixed mask. The cross-layer competitor boost is **not addressed**, and that is stated as a limitation.
- **vsg.** Not run, so H1d is stated as "forward content through T, including the value learning it enables" (A7).
- **4000-step horizon.** Replaced by the stability rule (§10.4), on compute grounds: about 3.3× the cost.
- **Adam-state swap and parameter transplant.** Recommended and not blocking; deferred.

### 10.3 Hypotheses and their predicted bands (frozen)

| hypothesis | vrg | gsg | gsv | sha-hi-rolled | sha-lo-real |
|---|---|---|---|---|---|
| **H1d**: forward content through T during formation selects | F | N | N | **F** | **N** |
| **Hm**: route establishment (T's forward share trajectory / recruitment) selects | F | med | med | **N** | **F** |
| **H2g-R**: the routing gradient on T selects | N | F | F | N | N |
| **H2g-RV**: all learning signal through T selects | N | N | F | N | N |
| **H4**: content absence and low share are both required | F | N | N | N | N |
| **H1d + Hm additive** | F | P | P | P | P |

**"med" (mediator rule).** Hm predicts the band implied by the arm's own release mediator, φ_share = (A_base − A_arm)/(A_base − A_block). A is the unmasked latent mean own-question share at step 100, with the block's A taken at the same step. φ_share ≥ 0.8 gives F, ≤ 0.2 gives N, and anything between gives P.

**H3:**
- **M.** Size-of-disturbance H3 is excluded for logit-level disturbances on T by XIV data (L4). The scope is stated: ledger N24 records strong layer-2-input noise flipping M in degraded runs.
- **E** (frozen rule, after R8). H3 (size of the immediate disturbance) is **excluded on E** if two conditions hold: vrg is F on at least 2 of the 3 E pairs, and its window Δloss is ≤ 0.061. Window Δloss is the mean training loss over steps 90–99 minus the paired baseline's (from slog). The threshold 0.061 is XIV's E sham Δloss of 0.031 plus 0.03. XIV's E noise had Δloss ≈ 0.18 and did not flip. Otherwise H3 is not excluded on E.

### 10.4 Outcome, bands, stability, competence (frozen; Reviewer A §4)
- **Primary outcome.** φ_nec = ½ Σ_{r∈{E,M}} [ℓ(nec_r^arm) − ℓ(nec_r^base)] / [ℓ(nec_r^block) − ℓ(nec_r^base)], with ℓ(x) = log(max(x, 0) + 0.01). The anchors are the **paired** baseline and block (same base and order). It is measured at step 1211 and at step 1000.
- **Organisation class.** The class (cls2) needs unanimity of three diagnostics:
  - structural: XIV's cls1211;
  - behavioural: lapse-aware ρ_L and fitted lapse (`cls_behav2`; all 24 XIV runs classified correctly);
  - causal: necessity.
- **Bands at a given step:**
  - **F**: φ_nec ≥ 0.8 and cls2 is the opposite organisation to the baseline's;
  - **N**: φ_nec ≤ 0.2 and cls2 equals the baseline's;
  - **P**: anything else.
- **Primary band.** The band at step 1211. A run whose band at 1000 differs from its band at 1211 is flagged **unstable**.
- **Stability readout** (secondary). Every P or unstable run is continued to step 2546 on the unchanged learning-rate schedule and re-scored there. A P run that reaches a band edge is reported as a "transient hybrid". The primary verdict does not change.
- **Competence.** KL ≤ 0.045 and accuracy ≥ the paired baseline's accuracy − 0.03. A non-competent run is **missing**, not "no flip".
- **Evaluable pair.** The baseline is competent and its cls2 is unanimous (EARLY or MIXTURE), and the block is competent and F. Non-evaluable pairs stay in the denominator.
- **Arm band (pooled over 6 pairs).** The band held by at least 5 of the arm's evaluable, non-missing pairs. MIXED if no band reaches that count. **Inconclusive** if fewer than 5 evaluable pairs remain. Per-base results (3 pairs) are always reported.
- **Hypothesis verdicts:**
  - **CONTRADICTED** if any arm with a definite pooled band (F, P or N) differs from the hypothesis's prediction;
  - **CONSISTENT** if every arm's pooled band matches;
  - **INCONCLUSIVE** otherwise.
  - A P result contradicts every F or N prediction for that arm, and is never counted as partial support.
- **Minimum meaningful difference between two arms.** A band change, together with |Δφ_nec| ≥ 0.4 in at least 4 of 6 pairs.
- **Units.** The pair (base × order) within one parent: n = 6, effectively 2 clustered bases. Results are reported per base first. Clopper–Pearson intervals are reported on counts.

### 10.5 Independent parents (frozen rule; R9)
- **Candidates.** uP161–uP165. uP161 is non-evaluable: no early base in 32 XIII-E draws. Screening uses s = 20…23 (D = 811), then 24 and 25 if needed, classified with cls2 at step 1211. The base for each class is the first competent seed in order. **Bistable** parents are reported as k/5.
- **Arms per bistable parent,** for both bases and one order (D = 812): baseline, block, vrg, gsv, sha-hi-rolled, sha-lo-real.
- **Per-parent result per arm.** The band of the two pairs if both agree, otherwise MIXED.
- **Generalisation of the block effect (P4).** The block is F in both bases of a parent: "effect present". F in neither: "counterexample". Otherwise "partial".
- The component pattern per parent is compared with uP160's. Any claim is **scoped to bistable parents**.
- If compute runs out, the parents not reached are reported as not run.

### 10.6 Logging (R10)
- **Window log** (`window_log.json`, steps 0–99): the global gradient norm and the gradient norms per parameter group (embeddings, layer 0, layer 1, layer 2, W_A), plus batch loss.
- **Clamp arms:** the per-head offset β and the achieved share.
- **Baselines** (`slog.json`): per-head shares and loss.
- **Clamp validity rule** (prospective, Reviewer A §3.3):
  - In development, sha-hi-real must be N against its baseline.
  - |β| must exceed 1 on fewer than 10% of window steps.
  - Otherwise the sha arms are reported as uninterpretable.

### 10.7 Development runs made for v2 before the freeze
All on D = 801, uP160, development stream:
- gsg continued to step 4000 (both bases);
- E own0: gsg, gsv, fst, vrg;
- M: gsv, vrg;
- gsg and gsv with T = 200 (both bases);
- sha-hi-rolled, sha-lo-real and sha-hi-real (both bases).

Their results are in §10.8, added before the freeze commit.
