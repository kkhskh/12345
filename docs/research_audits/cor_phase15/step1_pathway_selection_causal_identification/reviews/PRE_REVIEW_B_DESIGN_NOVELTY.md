# Pre-experiment Review B: experimental design, alternative explanations, novelty (XVIII)

**Reviewer.** Agent Reviewer B (design, alternative neural and optimisation explanations, novelty). This is an agent review, not human peer review. No confirmation run had been made when it was written.

**Material read.**
- MANDATE_XVIII §2, §4–§6 and §10–§12.
- PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md (called "the design" below), XIV_RESULT_RECONSTRUCTION.md and ORIGINAL_COR_OD_RELEVANCE.md.
- `scripts/xviii_train.py`, `src/xviiim.py` and `logs/dev/measure_dev.jsonl`. The file had 10 lines; the graded and own0 dev runs were not yet measured.
- `logs/unit_modes.json` and `logs/COMPUTE_LOG.txt`.
- T4_FINAL_CAUSAL_CONTROLS.md and T4_FOUNDATIONAL_EVIDENCE_LEDGER.md.

**Labels.**
- [code]: read from the implementation.
- [num]: computed by me (scripts in `/tmp/preB18/`, run on existing dev checkpoints, no training).
- [lit]: from a fetched paper page, with URL.
- [argument]: reasoning.

**Verdict: APPROVE WITH REQUIRED CHANGES.** The required changes are substantial, and two of them change what the primary test is (§5).
- The forward/gradient factorial is a real advance over XIV.
- As written, however, the design cannot interpret its own dev pattern:
  - the observed (gsg partial, fst flip) cell is not in the prediction table;
  - the headline readings rest on a classification taken at a non-stationary step;
  - fst does not test "recruitment" in the sense the H2g column assumes.

---

## 0. New checks made for this review (no training)

**N1. The step-1211 classification is not stationary** [num] (`/tmp/preB18/stat.py`, dev stream 86 100 000, frozen `xviiim.measure_final`).

| run | step | lik_RO1 | mix_mass | ρ | nec_E | nec_M | KL | (struct, behav, causal) |
|---|---|---|---|---|---|---|---|---|
| gsg E | 813 | 0.346 | 0.914 | 0.512 | 0.243 | 0.242 | **0.055** | H, E, H |
| gsg E | 1000 | 0.272 | 0.930 | 0.527 | 0.171 | 0.344 | 0.034 | H, E, H |
| gsg E | 1211 | 0.257 | 0.945 | 0.570 | 0.157 | 0.385 | 0.017 | H, E, H |
| gsg M | 813 | 0.277 | 0.923 | 0.472 | 0.250 | 0.336 | 0.020 | H, E, H |
| gsg M | 1000 | 0.243 | 0.931 | 0.519 | 0.188 | 0.379 | 0.023 | H, E, H |
| gsg M | 1211 | 0.241 | 0.945 | 0.586 | 0.169 | 0.384 | 0.013 | H, E, H |
| **fst E** | **813** | 0.016 | 0.946 | **0.585** | 0.021 | 0.668 | 0.017 | **M, E, M → HYBRID** |
| fst E | 1211 | 0.007 | 0.956 | 0.784 | 0.016 | 0.729 | 0.011 | M, M, M → MIXTURE |

What this shows:
- **Both gsg hybrids are still moving towards MIXTURE:** nec_E falls, nec_M rises and lik_RO1 falls, on both bases.
- **On E, MIXTURE is the flipped class. On M, MIXTURE is the unflipped baseline class.** The "same hybrid from both bases" at 1211 may therefore be a transient snapshot. It may resolve to "gsg flips E, but not M", which is a base-dependent result and not the design's "partial on both".
- **fst E would have been scored HYBRID at step 813.** ρ goes from 0.585 to 0.784 between steps 813 and 1211. The behavioural diagnostic lags the route diagnostics by several hundred steps.
- At step 1211 the learning rate is still 8.35e-4, 83% of peak [num]: the cosine runs over 4000 steps and the run stops at 1211 [code: `warmup_cosine_decay_schedule(0, 1e-3, 200, 4000)` with FX_STOP]. "Final organisation" at 1211 is a mid-training state.
- XIV checked persistence to step 4000 only for real-block flips, and only on D = 801 (T4_FINAL_CAUSAL_CONTROLS §I-A). That check does not cover hybrids or the new modes.

**N2. What ρ measures, against an exact reference** [num] (`/tmp/preB18/rhob.py`, `rhob2.py`, frozen `metrics.rho_fit`).

| reference output | KL to Bayes | fitted ρ |
|---|---|---|
| Exact Bayes, any temperature (0.8, 1, 1.25) | 0 | 1.000 |
| Bayes with logit noise σ = 0.15 / 0.30 | 0.004 / 0.014 | 1.01 / 0.96 |
| Bayes mixed with uniform, ε = 0.02 / 0.05 / 0.10 / 0.15 | 0.001 / 0.005 / 0.016 / 0.030 | 0.93 / 0.85 / 0.76 / 0.69 |
| Bayes mixed with uniform, ε ≥ 0.20 | ≥ 0.046 | 0.001 (fit degenerates) |

What this shows:
- ρ is insensitive to unstructured error.
- ρ moves strongly with a **lapse-type** imperfection. A lapse at KL ≈ 0.016 alone already puts a model in the MIXTURE band. This is consistent with the ledger's N20 ("a lapse floor moves ρ by about 0.1").
- A pure lapse at the early class's KL (≈ 0.03) gives ρ ≈ 0.69, not ≈ 0.5. So the early class's low ρ is **not** just a lapse at matched KL, and ρ carries some organisation-specific information.
- But on the 24 XIV runs, **KL alone separates the classes as well as ρ does**: early 0.024–0.036, mixture 0.010–0.013 (XIV_RESULT_RECONSTRUCTION §4).
- Together with ρ's lag (N1), ρ is the weakest of the three diagnostics. Yet it is the one carrying the "independent of the manipulated edges" burden.

**N3. Gradient energy in the window** [num] (Σ‖g‖² over steps 0–99 from `window_log.json`):

| arm | E | M |
|---|---|---|
| gsg (= plain) | 218 | 204 |
| fst | **508** | 209 |
| vrs | 215 | 202 |

Learning-rate exposure:
- The window holds 2.3% of the cumulative learning-rate exposure up to step 1211: Σlr = 0.025 for steps 0–99 against 1.03 for steps 100–1211 [num].
- The learning rate at step 50 is 2.5e-4.
- In every arm, the step-25 and step-50 route shares equal the initial value (own0 ≈ 0.133) [num, `traj`].
- So the effective manipulation acts over about steps 50–99.

---

## 1. Alternative explanations for the dev pattern

Dev pattern: fst and vrs flip both bases fully; gsg gives the same hybrid on both bases at 1211.

| # | Alternative | Already controlled? | Required or proposed control |
|---|---|---|---|
| A1 | **Non-stationary endpoint.** Hybrids, and even fst's behavioural class, are transients at step 1211 (N1). | **No.** | **Required.** Continue every run that is HYBRID at 1211, and every baseline, to step 4000. This is bitwise possible: copy `resume.pkl` at 1211 as `opt_step1211.pkl` and use `FX_BRANCH`. Classify at 4000, and report 1211 as secondary. Cheapest form: the two dev gsg runs now (≈ 23 min each, both lanes). If this is unaffordable for all runs, require class agreement at 1000 and 1211 and call any change "unstable". |
| A2 | **fst is not a "routing gradient present, recruitment possible" condition.** fst's logit gradient exists, but V is removed and downstream layers never see T's output, so nothing can be co-adapted to read T. Empirically fst barely raises T's share: E own0 at step 100 is 0.155, against block 0.135 and plain 0.234. The H2g column predicts "—" for fst on the premise that its gradient "actively says attend to T"; dev data show it does not recruit. | **No.** The design notes V is "unavoidable" but keeps H2g's fst prediction. | **Required: add the missing cell, gsv** (forward intact, R *and* V removed): `s_T ← sg(s_T)` and `ctx = Σ a(1−XM)v + Σ a·XM·sg(v)`. This is the full learning-signal deficit with forward information intact. It is also the direct analogue of the deep-linear "gating deficit" (§4, Kleinman et al.). The factorial then becomes F × (R, V): fst = F off with R on and V off; gsg = R off; gsv = R off and V off. Restate H2g as two hypotheses: H2g-R (QK routing signal) and H2g-RV (all learning signal through T). |
| A3 | **"Route not recruited" as a common mediator.** Every arm that flips leaves T's share at about its initial level at release (E own0 at step 100 is 0.135–0.155; M at step 150 is 0.142–0.171). gsg is intermediate (0.188 / 0.194) and gives the intermediate outcome. A single scalar race ("T established by release, or not") fits all arms. On this reading, H1 vs H2 is a question of which manipulation retards T most, not of what kind of cause selects. | Acknowledged in §6.4 but unresolved. vrs cannot separate it, because its share also stays at baseline. | **Required: a share-clamp arm (shc)** with forward content intact. During the window, rescale the readout row's attention so that T's total mass is held at its step-0 level (≈ 0.134, matching vrs): a′_T = c·a_T/Σa_T and a′_C = (1−c)·a_C/Σa_C, with c fixed and gradients within T and within C intact. Predictions: H1 says shc gives no flip while vrs flips; "non-recruitment suffices" says shc flips. With existing plain and vrs this is a 2 × 2 of content (intact/destroyed) × share growth (free/clamped). In the current dev data the share is clamped in effect for vrs but free for plain, so content and share growth are confounded. |
| A4 | **Adam second-moment and state memory.** β₂ = 0.999 gives a time constant of about 1000 steps, so window gradients shape v until about step 1000. Adam also renormalises per parameter, so under gsg the QK tensors of the target layer still take full-size steps, driven entirely by non-T edges. gsg therefore **redirects** shared W_Q/W_K updates rather than merely removing a component. | Partly. Global gradient energy is matched for gsg, vrs and fst M (≈ 200–218, N3), so a global v difference cannot explain vrs-E flipping while gsg-E is partial. fst E (508) is not matched. There is no per-tensor information. | (a) **Required, cheap:** log per-tensor gradient and update norms for the target layer's W_Q, W_K, W_V and W_O, and for layer 2, during the window (extend FX_GLOG). (b) **Recommended:** an optimiser-state swap at release, which needs FX_SAVEOPT at 100 and a small branch variant: fst@100 parameters with plain@100 Adam state, and the reverse; 2 runs per base. If the class follows the parameters, Adam memory is excluded. |
| A5 | **Where the selection is stored at release** (mandate item 3: "redirecting optimisation toward an alternative route"). At step 100 the competitor is *not* ahead in share (block E own2 max-head 0.163 against plain 0.175). The deficit could lie in T's QK, T's OV, the downstream readers, or the competitor's non-attention parameters (layer-2 OV, W_A). | **No.** | **Recommended (high value, cheap, about 6 runs):** a parameter-subset transplant at release. Branch from fst@100 with one block replaced by plain@100's: (i) target-layer W_Q/W_K, (ii) target-layer W_V/W_O, (iii) layer 2 and W_A. This localises the selection and separates "T under-built" from "competitor pre-built". |
| A6 | **vrs cross-episode gradient coupling.** Episodes are i.i.d. [code: `bayes.sample_fast`], so the roll leaks no information. But the gradient from episode i's loss flows into episode i−1's values. The expected effect trains the T-read values (shared W_V and the upstream source representations) to be episode-invariant. vrs therefore also actively penalises carrying content, which is more than deprivation. | **No.** | **Required (one line):** `jnp.roll(jax.lax.stop_gradient(v), 1, 0)` on the T term. Pre-register vrs-sg as the primary content arm and keep the dev vrs as secondary. |
| A7 | **Misinformation against absence** under vrs. Downstream layers learn to ignore a channel that carries plausible but wrong evidence. | Mostly. fst (absence, no misinformation) also flips. | None beyond A6. State it as an interpretation limit. |
| A8 | **Window short relative to warm-up.** The manipulation is effective only over about steps 50–99, at lr ≤ 5e-4. Steps 100–200 carry three times the window's learning-rate exposure (N3). gsg's partiality may be a matter of **dose per learning-rate exposure**, not of kind. | **No.** | **Required on dev before freeze** (1 run per base): gsg and gsv with T = 200. If gsg(T = 200) flips fully, H1 against H2g is a potency question. The prediction table must then be stated in dose terms, or H2g marked "not excluded, weaker lever". |
| A9 | **gsg's removal reaches upstream.** sg on s_T also cuts the gradient into the source keys and readout queries, and through them into the trainable inherited layers 0–1 and the embeddings. On M this removes the signal that shapes the key-side reliability tag on source tokens (ledger N3: the tag lives on source-token keys). gsg therefore also manipulates upstream representation learning, not only "recruitment of T". | **No.** | Report it as part of H2g's operational definition. Optionally add a variant that stops the gradient only into this layer's W_Q/W_K (detach z before the projections for T's terms) to split "route parameters" from "upstream representations". |
| A10 | **gsg leaks share through shared parameters.** Under gsg, E's own0 share still rose from 0.133 to 0.188 by step 100, and src0 matched plain exactly (0.671 against 0.671) [num]. The design notes leakage only through the complement logits. Leakage through shared W_Q/W_K and through the inputs z is larger. | Measured (shares), not controlled. | Covered by shc (A3). Report the achieved share against arm in every analysis. |
| A11 | **A common hybrid attractor.** The XIV late blocks (E D802; XIII-E M) and gsg all end in a "both routes present" state. A third basin, or the slow coexistence regime, may be reachable by any partial retardation. Then φ ≈ 0.72 / 0.27 is not a fraction of an effect. | **No.** | A1 (4000-step horizon) decides it. Drop φ as a "fraction moved" until stationarity is shown. |
| A12 | **H3 on M is not as closed as L4 says.** Ledger N24: "strong early noise at the layer-2 input flipped the mixture base (2/2, degraded runs)". | L4 cites only same-edge logit noise. | Amend L4 to cite N24 and state the scope: "logit-level disturbances on T". |
| A13 | **H3 on E.** vrs-E flips with a window Δloss of about 0.05 at step 99 (1.066 against plain 1.020) and matched gradient energy (215 against 218). XIV's E noise (Δloss ≈ 0.18) did not flip. | Data exist, but the inference is not pre-registered. | Pre-register: "if vrs-sg flips E with Δloss(99) ≤ the E sham's 0.031 + 0.03 while lnz σ2.5 does not, H3 (size of the disturbance) is excluded on E". This replaces the missing confirmation-cohort mandated arm D cheaply. Otherwise **re-run** E lnz σ2.5 in the fresh orders (3 runs). |

---

## 2. Organisational diagnostics

**Independence.**
- The structural diagnostic (mix_mass on the own2 edges; lik_RO1 at the readout residual) and the causal diagnostic (necessity of the ro0 edges and the own2 edges) **are both read off the two manipulated edge sets** [code: `xviiim.necessity`, `RP.head_stats`, `MT.avail`]. They are two measurements of one latent variable, "which of the two manipulated routes the trained model uses". Their agreement is close to guaranteed whenever either is clean.
- This is not a flaw in itself: the intervention ends at step 100, and the measurement is taken at 1211 [argument]. But **the only diagnostic that is not a function of T is ρ**, and ρ lags (N1) and is confounded with lapse-like competence (N2).
- So "three independently informative diagnostics" (mandate B4) is in practice "route use, measured twice, plus a weak behavioural check".

**Required changes to the diagnostics.**
1. Make the behavioural arm stronger than ρ.
   - Add a lapse parameter to the ρ family, or report ρ only alongside a lapse-matched reference: Bayes with ε chosen to match the model's KL.
   - Add the existing counterfactual-edit slopes (slope_W, slope_e), or the D1/D2 relational edits. These read *how* evidence is combined, not how far the model is from Bayes.
   - Freeze their thresholds on the 24 XIV runs, as was done for ρ.
2. Add a **"no third route" check.** When nec_E and nec_M are both small, block the layer-1 readout route (rq1) at test time, so that a HYBRID label is not given to a model that uses an unmonitored route.

**The unanimity rule.**
- It is appropriate for claiming a **flip**: it is conservative.
- But it is **not neutral across hypotheses** in the factorial.
  - Defaulting to HYBRID ("partial", read as "—") biases gsg towards "—", which supports H1, and biases fst towards "—", which supports H2g.
  - fst E would have been HYBRID at step 813.
- **Required:**
  - a pre-registered *continuous* primary outcome per run: the causal index c = (nec_M − nec_E), normalised between the paired baseline and the paired block;
  - prediction-table entries in that unit, for example H1: c_gsg ≤ 0.2 and c_fst ≥ 0.8; H2g-RV: c_gsv ≥ 0.8, and so on;
  - explicit rows for **(partial, flip)** and **(flip, partial)**, which no row currently covers. The dev pattern lives in the first of these.

**Competence.**
- KL ≤ 0.06 is about 1.7 × the worst XIV value (0.036) and 5 × the mixture values. gsg E at step 813 (KL 0.055) would pass while still visibly disrupted.
- **Required:** KL ≤ 0.045, or ≤ the paired baseline's KL + 0.015, together with accuracy within 0.02 of the paired baseline.
- Report KL by class, because early runs are systematically higher.

**Missing rule.** The design does not say what happens if a fresh-order **baseline** is not unanimous.
- **Required:** such a pair is non-evaluable and stays in the denominator.

---

## 3. Model independence, cohort, statistics, compute

**Within-parent design.**
- With 2 bases × 3 fresh orders, data orders have never changed a uP160 base's class (XIV 6/6), so the six pairs are effectively **two clustered bases**.
- Clopper–Pearson on n = 6 treats them as exchangeable; that is acceptable only if it is reported per base (3/3 gives CP [0.29, 1]) as well as pooled.
- **Required:** report per base first. Say explicitly that n_eff ≈ 2 for "effect across organisations".

**Independent-parent screen.**
- The rule is prespecified and deterministic: s = 20…23, then 24 and 25, first EARLY and first MIXTURE, with uP161 counted as non-evaluable. Under the mandate's terms this is free of *outcome* selection bias [argument].
- Its remaining biases are scope biases, which must be stated:
  1. The claim becomes conditional on *bistable* parents. Report k/5 parents bistable.
  2. The screen classifies at step 1211 with the same unanimity rule. That is acceptable for baselines, which XIV showed stable to 4000, but add the A1 horizon check to at least one screened baseline per parent.
- Running the screen before the freeze is acceptable, because it uses baselines only and no intervention outcome [argument]. Log its timestamps against the freeze commit.

**Gaps in the parent plan.**
- "Decisive arm(s) from §5" is undefined. **Required:** name them now. My recommendation is block, fst and gsv (or gsg if gsv is not adopted), plus baseline.
- **Required:** define the parent-level outcome. For example, "effect present in a parent" if at least 3 of 4 competent evaluable pairs move in the predicted direction on c; "counterexample" if at most 1. Report k/n parents with a CP interval. "Hierarchical summary" without a model is not a pre-registration.

**Fail-closed.**
- Counting an incompetent run as "not a flip" is asymmetric across hypotheses (§2): it favours H2g for fst and H1 for gsg.
- **Required:** report incompetent runs as non-evaluable, and show that each conclusion survives both treatments (counted as not a flip, and excluded).

**Compute.**
- The measured cost is 10.8 min per 1211-step run with 2 lanes running concurrently (COMPUTE_LOG 16:41 → 16:52). The plan's 42 runs take ≈ 3.8 h, not 3.5 h.
- Screening takes 16–24 runs, ≈ 1.5–2.2 h. Parents take ≈ 20 runs each, ≈ 1.8 h each, so ≈ 7 h for 4 parents.
- 4000-step continuations cost ≈ 23 min each.
- Prioritisation is wrong in one respect. Graded suppression (12 of 42 runs) is shown non-discriminating by the design's own L2, yet it outranks controls that discriminate.
- **Required reorder:**
  1. Dev calibration first, before freeze: gsg-E/M to 4000 (A1); gsv, shc, vrs-sg and gsg(T = 200) on D = 801, one run per base, about 8 runs ≈ 45 min.
  2. Within-parent confirmation: baseline, block, fst, gsg, gsv, shc and vrs-sg × 6, ≈ 3.8 h, plus 4000-step continuations for hybrids.
  3. Parents: screen, then baseline, block, fst and gsv.
  4. Graded b = 2.3 only (P1), and the transplant/state-swap localisation (A4b, A5), if budget remains.

---

## 4. Novelty (Check C)

**Already established. These are not XVIII contributions.**

1. **Briefly suppressing an attention structure early in training changes which strategy is learned; the alternative strategy is learned instead; release timing matters.**
   - Chen et al. (ICLR 2024) suppress syntactic attention structure (SAS) with a loss regulariser on attention between syntactically linked tokens: λ > 0 for the first N steps, best N = 3k, then released.
   - They report that "SAS competes with other beneficial traits", that an "alternative strategy" (long-range semantic) is learned earlier under suppression, and that the worst time to release is during a phase transition.
   - The intervention acts **only through the gradient**; the forward pass is unchanged. No forward/gradient separation is made.
   - [lit] https://arxiv.org/abs/2309.07311v6, https://arxiv.org/pdf/2309.07311
2. **A temporary early deficit in one of two parallel pathways makes the other pathway win, in a winner-take-all race that depends on depth; late deficits have little effect.**
   - Kleinman, Achille & Soatto (ICLR 2024), deep linear networks.
   - Their "gating" deficit is implemented with `.detach()` on the deprived pathway's output: the forward pass is present and there is no learning signal. That is a **gradient-only** deficit, close to XVIII's gsv and stronger than gsg.
   - They *interpret* it as "blocking input information" without testing the separation.
   - This is the **closest prior experiment** both to XIV's phenomenon and to the H2g arm.
   - [lit] https://arxiv.org/html/2308.12221v2
3. **Critical periods.** Early deficits have permanent effects; deficits that preserve low-level statistics do not (Achille et al., ICLR 2019). Networks become stable to SGD noise early in training (Frankle et al. 2020).
   - [lit] https://arxiv.org/abs/1711.08856v3, https://arxiv.org/abs/1912.05671v1
4. **Training-time causal manipulation of attention and circuits in small transformers.**
   - Singh et al. (ICML 2024) "clamp" activations, including fixing attention patterns, during training to dissect induction-head formation, "while still allowing for proper gradient routing". Clamping all layer-1 heads slows learning severely.
   - Singh et al. (ICML 2025) use weight clamping and fixing to show competition *and cooperation* between ICL and CIWL. The forward and gradient effects of clamping are not separated.
   - [lit] https://arxiv.org/pdf/2404.07129, https://arxiv.org/html/2503.05631v2
5. **Gradient-only interventions as a tool to decide where computation forms:** gradient routing, i.e. data-dependent masks on gradients (Cloud et al. 2024). This is methodologically the gsg/gsv family.
   - [lit] https://arxiv.org/abs/2410.04332v2
6. **Competing algorithms, multiple solutions, and selection by initialisation or hyperparameters.**
   - Clock against pizza, including parallel implementations (Zhong et al., NeurIPS 2023).
   - Competing Fourier "circles" with fitted linear competition/cooperation dynamics; survival is predicted by initial signal (correlation 0.85) (Ding et al. 2024).
   - Four competing ICL algorithms with sharp phase shifts (Park et al., ICLR 2025).
   - [lit] https://arxiv.org/abs/2306.17844, https://arxiv.org/html/2405.17420v1, https://arxiv.org/abs/2412.01003v4
7. **Learning signal through one feature starving others:** gradient starvation (Pezeshki et al., NeurIPS 2021). This is the conceptual H2g. Multimodal "modality imbalance" corrected by on-the-fly gradient modulation (Peng et al., CVPR 2022) is a gradient-side lever on which pathway wins.
   - [lit] https://arxiv.org/abs/2011.09468v4, https://arxiv.org/abs/2203.15332v1
8. **Attention-based diagnostics are unreliable under training-time attention pressure.** Models penalised for attending to tokens keep using them, at little cost in accuracy (Pruthi et al., ACL 2020). This supports demoting mix_mass, as the design already does.
   - [lit] https://arxiv.org/abs/1909.07913v1

**What remains, stated narrowly** [argument].
- I found no study that **decomposes one training-time attention restriction into forward-information and learning-signal components** (straight-through forward-only against stop-gradient gradient-only) and scores the resulting organisation by causal necessity at test time. The closest interventions (Chen; Kleinman; Cloud) are gradient-side only. Clamping (Singh) conflates the two.
- This is absence in a focused search, not proof of novelty.
- If confirmed, XVIII's distinctive question is: *in a transformer that has two competent relational organisations, is the selecting lever the forward information through the restricted edges, the learning signal through them, or route establishment as such?*
  - The dev data hint at "forward suffices, routing-gradient-only is weaker". That would contrast informatively with Kleinman et al.'s gradient-only result.
  - But the hint holds only if (a) gsv is run (it is the like-for-like comparison with Kleinman) and (b) shc separates content from non-recruitment.
- Without (a) and (b), the result reduces to known findings: "early pathway deficits select the competitor" (Kleinman; Chen) plus "a straight-through block is as strong as a hard block". The second is close to guaranteed, since fst's forward pass equals the block's to 1.4e-6 and its gradient differs by 1.4–7.8% (`unit_modes.json`).
- **Do not claim** the broad phenomenon, "temporary restriction selects between competent computations", as new. Items 1, 2 and 6 establish it in spirit. The new elements could be only the component decomposition and the parent-level generality in this task.

**O/D assessment (ORIGINAL_COR_OD_RELEVANCE.md).**
- It uses only the original objects (O = ‖ds‖²_W, D = distance to ker d, the analyst-chosen complex, and corrected Proposition S′).
- It correctly notes that the definitions have no time, optimiser or intervention semantics.
- It keeps the repaired proposal out of scope.
- I agree with O3 (O2 only at task level) and with not running P6. No change required.

---

## 5. Verdict and prioritised required changes

**APPROVE WITH REQUIRED CHANGES.** Items R1–R5 must be met before the freeze. Without them the primary test does not identify what the design claims.

1. **R1 (horizon).** Classify at step 4000, or require stability between 1000 and 1211 with "unstable" as its own outcome. Do this first for the two dev gsg runs. Their 4000-step class can change the dev reading from "partial on both bases" to "base-dependent" (N1, A1).
2. **R2 (missing factorial cell).** Add gsv (forward intact, routing and value gradients removed through T). Split H2g into H2g-R and H2g-RV. Re-derive the prediction table over {block, fst, gsg, gsv} (A2).
3. **R3 (mediator).** Add the share-clamp arm shc to separate content from route establishment. State in the table what each hypothesis predicts for shc and for vrs (A3, A10).
4. **R4 (prediction table).** Use a continuous primary outcome (normalised causal index c) with a minimum meaningful difference. Add rows for (partial, flip) and (flip, partial). Make the inference rules symmetric across hypotheses for HYBRID and incompetent runs (§2, §3).
5. **R5 (dose calibration on dev).** Run gsg and gsv with T = 200 before the freeze. If either flips fully, restate H1 against H2g as a difference of potency (A8).
6. **R6.** Use vrs-sg (stop-gradient on rolled values) as the primary content arm (A6).
7. **R7 (diagnostics).** Add a lapse-aware behavioural measure and the counterfactual-edit slopes. Add the rq1 third-route check. Tighten competence to KL ≤ 0.045 (or baseline + 0.015) with accuracy within 0.02. Add the rule that a non-unanimous baseline makes the pair non-evaluable (§2).
8. **R8 (H3).** Pre-register the vrs-based exclusion of H3 on E, or re-run E noise in the fresh orders. Amend L4 for ledger N24 (A12, A13).
9. **R9 (parents).** Name the decisive arms. Define the per-parent success and counterexample rule. Report k/5 bistable parents, with the claim scoped to bistable parents (§3).
10. **R10 (logging).** Log per-tensor gradient and update norms in the window (A4a).
11. **Recommended, not blocking:** the Adam-state swap and the parameter-subset transplant at release (A4b, A5). Demote graded suppression to one offset (b = 2.3) at the lowest priority.
12. **Novelty statement.** Position against Kleinman et al. 2024 (gradient-only deficit, pathway race) and Chen et al. 2024 (early suppression, alternative strategy, release timing) as the closest prior work. Claim only the forward/learning-signal decomposition, and only if R2 and R3 are run (§4).

**Reproducing my checks:** `/tmp/preB18/stat.py RUN STEPS`, `/tmp/preB18/rhob.py`, `/tmp/preB18/rhob2.py`. They use the frozen `xviiim.measure_final` and `metrics.rho_fit`, with XLA single-threaded, about 12–25 s per measurement. No package file was modified except this review.
