# Investigation XII: main research synthesis

**Native relational algorithms: fresh causal replication, circuit reconstruction, computational identifiability and
generalization.**

- Branch: `cor-phase15-step1-native-relational-algorithm-validation`.
- Directory: `docs/research_audits/cor_phase15/step1_native_relational_algorithm_validation/`.
- **Phase 1.5, Step 1. Not closed.** The provisional estimate of ~25% is a judgment, not a measurement. No thesis
  is selected. The original COR (O/D) framework is **not** validated, and no connection to it was found or tested.

Evidence labels:
- **[proved]**: mathematics;
- **[conf]**: fresh, independently pretrained cohort, frozen predictions;
- **[dev]**: development models only;
- **[post hoc]**: added after the freeze, disclosed.

## 1. What was done

### 1.1 Fresh cohort

Seven new parents (P150–P156) were pretrained under a Stage-0 protocol frozen before any training.
- The original plan was five parents. Amendment 1 added two, decided from pretraining completeness only.
- **3 parents were complete** (P150, P154, P155) and **4 were incomplete**. Incomplete parents are reported
  separately.
- Fine-tuning used new seeds (10, plus same-parent replicates at seed 11 on P150 and P154), a new evaluation set,
  and new analysis streams.
- **Eight predictions** (P1–P8) were frozen with hashes **before any T4 fine-tune of the cohort started**.

### 1.2 Development work

On 11 existing models:
- weight-level circuit analysis;
- eight alternative intervention designs;
- compensation and direct-versus-total tests;
- learning-dynamics decomposition;
- a stage-separated generalization test;
- new mathematics.

### 1.3 Independent review

Two reviews were run by separately initialized agents with no shared context (mathematics; final results). Their
findings are preserved in `reviews/`, and the documents were corrected. This is not external human review.

## 2. Principal findings

1. **The layer-1 conjunction replicates in new models** [conf: P1, 3/3 complete and 4/4 incomplete].
   - Perfectness × agreement-at-q interaction is absent after layer 0 and present after layer 1: E-ratio 4.8–9.7,
     subspace-free ratio 3.1–5.0.
   - It is absent in frozen-prefix (C1) models: ratio 1.0–1.7.
   - The *allocation* within layer 1 is not invariant. P156 forms it almost entirely in MLP1.
2. **The causal result does not depend on XI's hybrid ablation** [conf: P2].
   - Layer 1 fed a natural input in which h does not agree at q (path_inP) moves decisions toward the "group
     ineligible" counterfactual: KL_e0 0.06–0.16 vs KL_B 0.44–0.50.
   - The path_inA variant gives a stronger shift (0.01–0.03 vs 0.59–0.71) but is not discriminating, because
     cf_e0 ≈ Bayes on xA.
   - Scrubbing of the 6-dim E-subspace with natural donor *states* (the patched site itself is synthetic, and DAS
     fitted the subspace) preserves decisions when the donor has the same eligibility
     vector (KL ≤ 0.022). It switches them when the donor has e_h = 0.
   - Mean ablation is non-specific.
   - Restoration works after natural removal too [dev, post hoc] and after hybrid removal [conf: P4].
3. **Weight-level operation** [conf: P3 a/c/d; dev details].
   - The conjunction is read from group h's **keys/values**. Queries play no role.
   - Attention scores change almost additively (interaction 2–8% of the main shifts). In most models the conjunction
     arises from **softmax normalisation** coupling an agreement-driven score shift with a perfectness-driven shift.
     This account rests on a reference-dependent decomposition, and P156 forms the conjunction mostly in MLP1.
   - MLP1 passes on and enlarges the decision effect of the attention interaction (total ≥ 1.5× direct). This is
     not compensation, though the test does not separate amplification from transmission.
   - The operation is **smooth**: threshold-AND units carry 12–34% of the MLP1 interaction. It is **not a Boolean
     gate**, and LayerNorm is not its source.
   - Methodological finding: "pattern vs value" attributions of an interaction inside attention are
     **reference-dependent**. They flip between the x00 and xAP references, and should not be reported as mechanism.
4. **Prediction P3(b) was falsified** [conf: 0/3].
   - Development models routed perfectness *redundantly* (patterns OR values sufficed; 10/11).
   - In the complete fresh parents, routing is **asymmetric and partially redundant**. Removing P's effect on
     patterns gives 40–60% of the full shift; values alone carry little.
   - Incomplete parents vary: redundant (P151, P152), or value-route and MLP1-dominant (P156).
   - The sub-head route is implementation-specific. The layer locus, necessity, mediation and amplification are
     the invariants.
5. **Same- vs cross-parent** [conf: P5].
   - Shared pretraining makes the *componentwise* organization more similar: head/MLP1 profile distance 0.28 within
     parent vs 0.62 across parents.
   - It does **not** change the transferability of the abstraction: mapped-interchange gain 0.98 within vs 0.94
     across. Maps were fitted on natural states only; R² was checked on a separate stream.
   - C1 → full transfer fails: 0.74, 0.33, −0.56.
6. **Identification** [proved, dev, conf: P6].
   - Interchange accuracy can separate E from the perfect-only rule A1 by at most δ ≈ 4 points on XI's distribution
     (Prop. XII-3).
   - Preservation *was* discriminating there, as corrected after the independent review.
   - On a *discriminating* pair design, repaired models' E-subspace mediates E, not A1: IIA_A1 rises over base by only
     0.01–0.09.
   - The C1 models' subspace moves decisions toward **A1** (+0.19–0.39) [post hoc runs; frozen script]. Calling it
     "E", as Investigations X and XI did, was a false positive of the label.
   - The frozen VS-linear arm **passed but is non-diagnostic**: a random-subspace control also passes. This was
     knowable before confirmation from a development control.
   - P6 as coded is passed even by the unpatched model. The E-specific
     effect is the gap (linear KL 0.32–0.47 in E vs 0.69–0.82 in a random subspace).
7. **Learning** [conf: P7, 3/3 and 7/7; revised after review].
   - The decision is *sensitive* to the conjunction from the checkpoint at which it emerges. The two measures share
     one double difference, so this is partly by construction.
   - Bayes-correct use lags by 1–5 checkpoints. That *is* a readout lag in the ordinary sense.
   - XI's "representation before behaviour" reflected a global behavioural classifier, A1 → A2 → B. The A2 → B
     calibration that follows is unexplained.
8. **Generalization** [proved + conf: P8].
   - Trained rule: degree-1 aggregation Σ_{h≠g} e_h.
   - Independent-copying C = 2: e_2 (an AND) [proved, Theorem XII-1].
   - The actual generator shares the copied value, so the exact posterior needs a **between-group** relation J_H that
     per-group e cannot express [proved, XII-2].
   - Native models apply the C = 1 rule (accuracy 0.34).
   - **Re-training one full block (layer 2) on the frozen layers 0–1 recovers exact Bayes** (0.80 vs 0.804). A fresh
     block does as well, and so does the same re-training with the 6-dim E-subspace mean-patched.
   - This shows that the needed information is in resid2 and that one block suffices. It does not show that the
     native failure lies only in aggregation, or that eligibility information is bypassed.

## 3. Corrections to the earlier record

1. **XI's check count.** It reported 67/72; the correct figure is **73/78** (re-derived from XI's scorer). XII's own
   count is 94/97, with the exact denominator listed.
2. **XI's "held-out" map R²** was in-sample. It is now fitted and evaluated separately; the in-sample inflation was
   ≤ 0.01.
3. **XI's "true C = 2 Bayes"** was genlib's independent-copying posterior. The exact accuracy is 0.80, not 0.75. For
   β < 1 the error is gross (0.50 vs 0.79).
4. **XI §4, "VS is illegitimate", is partially retracted.**
   - In SCM semantics do(S) is legitimate.
   - VE (accepted in XI) is equally non-input-realizable, because e is a function of (W, L) on the support.
   - Replaced by a five-class admissibility taxonomy.
5. **XI's Proposition A2** assumes product support, which T4 violates. Its restricted form constrains natural states.
6. **First-draft XII errors** caught by the independent mathematics review and corrected:
   - δ(E, A1) is 0.040, not 0.0035;
   - preservation discriminated A1 in XI;
   - a wrong quantifier in the admissibility rule.
7. **XII freeze defects** (disclosed):
   - the frozen scorer implemented an earlier P6 draft;
   - the C1 arm of the text P6 had no queued job;
   - the P6 rationale sentence used the wrong δ.

## 4. Hypotheses H1–H5 (mandate §3)

| hypothesis | verdict |
|---|---|
| **H1** shared causal relational algorithm | **Supported at the level of locus, necessity, mediation and transfer.** Not supported at the sub-head level: routing differs (P3b) |
| **H2** shared abstraction, different downstream rule | Not supported for the trained task: all models are Bayes-like and transfer interchangeably. Layer 2 was not dissected |
| **H3** intervention-dependent mechanism | **Rejected** for the layer-level claims (natural path patching, resampling, scrubbing). **Confirmed** for weight-level "pattern vs value" attributions, which are reference-dependent |
| **H4** redundancy / compensation | No compensation in the fixed forward pass. Sub-head redundancy is partial and model-specific: P3(b) is falsified in the complete parents, but present in two incomplete ones. Residual information outside E restores 19–34% of the decision effect [conf] |
| **H5** restricted structural generalization | **Supported.** The trained aggregation is degree-1; C = 2 needs a joint between-group relation. Readout-only retraining fixes it without E |

## 5. Answers to the final scientific questions

1. **Does the layer-1 mechanism replicate in new models?** Yes [conf]. All 3 complete and all 4 incomplete fresh
   parents show it. n = 3 complete parents does not support population-level claims.
2. **What weight-level operations combine the two signals?** Near-additive QK score shifts on group h's keys, coupled
   by softmax normalisation in most models; P156 instead relies on MLP1. The perfectness shift is carried mainly by attention *routing* in the complete fresh parents. MLP1
   then amplifies smoothly; there is no threshold-AND and LayerNorm plays no role.
3. **Does the mechanism survive alternative ablations?** Yes, under five natural-activation or donor-based designs,
   plus a non-specific control [conf: P2; dev for restoration after natural removal].
4. **Necessary, redundant, compensating, or partial?**
   - Necessary at layer 1, as a whole.
   - Not compensated: MLP1 amplifies.
   - Sub-head redundancy is model-specific.
   - **Partially characterised**: layer 2 was not dissected, and 19–34% of the effect can be restored from outside E.
5. **Do same-parent and cross-parent models differ in causal organization?** In component allocation, yes
   (0.28 vs 0.62). In transferable abstraction, no (0.98 vs 0.94).
6. **Does the abstraction transfer to new models without using confirmation outcomes?** Yes: maps fitted on natural
   states only give gains of 0.86–1.00 across new parents.
7. **When do the identification tests work, and when do they mislead?**
   - Interchange accuracy is bounded by the pair distribution's δ.
   - Preservation works only where the true variable changes on the preservation set.
   - Discriminating pairs are needed.
   - Mean-patch inflation must be controlled.
   - Restoration and necessity can fail for redundant encodings.
   - Uniqueness is never established.
   - Misleading cases found: XI's A1 decoy distribution (δ = 4%); the C1 "E" label; the frozen VS arm (passes on a
     random subspace).
8. **Why does the representation emerge before the decision rule?** Sensitivity to the representation is immediate;
   that is partly by construction of the measures. Correct use lags 1–5 checkpoints (A1 → A2 → B). The cause of that
   calibration lag is open.
9. **Does the eligibility representation generalize independently of the evidence-combination step?** The variable
   is still read out correctly under C = 2, but it is not sufficient for the real generator: proved, via J_H. The
   native combination applies the C = 1 rule. Re-training one block repairs it, with or without the 6-dim subspace.
10. **What survives the strongest tests?**
    - Theorems XII-1 and XII-2 (elementary).
    - The corrected X1′.
    - The δ bound and preservation-power criterion.
    - The taxonomy of admissible interventions.
    - The counterexamples: VS/VE non-realizability; projection-DAS false negative (partial, and weakened by the
      random control); redundancy defeating necessity.
11. **Which findings plausibly extend to other COR families, and which are specific?**
    - Plausibly general: relations read at source tokens via attention (shared with GATE's earlier finding); and the
      methodological results (natural-activation designs, discriminating pairs, mean-patch control, reference
      dependence).
    - Specific: the T4 eligibility conjunction, degree-1 aggregation, and the routing details.
12. **What remains unresolved?**
    - Layer 2's (S − e_g) implementation.
    - The A2 → B step.
    - What the non-E residual is.
    - Population-level replication (≥ 10 parents).
    - GATE/TRANS protocol transfer.
    - A non-product-support A2 proof.
    - Details are in `OPEN_COR_FOUNDATIONAL_QUESTIONS.md`.

## 6. Limitations

- **3 complete parents.** Primary arms required 3/3. Four of seven parents failed the readiness criterion. This
  completeness rate is itself a finding about the pretraining protocol.
- **Two freeze defects** (in the P6 code and the P6 queue) and **one amendment** (cohort extension). All are disclosed
  with commits and logs.
- Episode-level CIs are within-model; parent-level variability is shown as ranges.
- Layer 2 is not dissected. The sub-head mechanism is characterised by interventions, not by features.
- **No external human review.** Two independent-agent reviews were run. The final review's corrections, including
  the false amendment integrity statement and several overclaims, are applied and listed in
  `reviews/ADJUDICATION.md`.
- **Amendment 1 changed the protocol.** It reversed Stage 0's no-extra-parents rule, and its commit contained
  in-progress fine-tune outputs that were not read. The verdicts are unchanged if only P150 and P154 are used.
