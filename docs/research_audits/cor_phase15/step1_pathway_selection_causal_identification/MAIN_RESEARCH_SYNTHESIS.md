# Investigation XVIII: which component of a temporary pathway block selects the learned organisation?

**Status.**
- COR Phase 1.5, Step 1, provisional ≈ 25%, **unchanged**.
- No phase closure. O/D is **not validated**. The O/D assessment here is O3 (no applicable prediction): ORIGINAL_COR_OD_RELEVANCE.md, frozen before confirmation.
- All four reviews are agent reviews, **not human peer review**.
- The archive is **not standalone**: it needs the sibling packages in `logs/DEPENDENCIES.json`.

**Labels:**
- [proof]: mathematical result;
- [frozen]: frozen confirmation, frozen scorer;
- [sens]: the pre-registered sensitivity analysis without the behavioural gate, applied in full (written after the frozen output, `scripts/posthoc_nobehav18.py`);
- [dev]: development;
- [interp]: interpretation.

## The result in brief

**Background.** XIV showed that blocking one attention route for the first 100 training steps switches which of two competent relational organisations a small T4 transformer learns. XIV could not say what about the block does the work.

**What XVIII did.**
- Separated the block into forward content, learning signal (routing gradient and value gradient) and attention share, each set by its own intervention.
- Scored the outcome by **test-time route necessity**, not by an attention statistic.
- Had two agent reviewers attack the design before the freeze: one rejected version 1, and the design was rebuilt with both reviewers' changes.
- Froze a fail-closed scorer and ran:
  - 48 confirmation runs (uP160, 2 bases × 3 fresh data orders × 8 arms);
  - screening of 4 independently pretrained parents;
  - the decisive arms on the one bistable parent (uP163).

**Frozen outcome** [frozen].
- **Six of eight pre-registered hypotheses are CONTRADICTED:** H1d, Hm_window, H2g_R, H2g_RV, H4 and H1d_plus_Hm.
- **Two are INCONCLUSIVE:** Hm_release and H1d_plus_H2g.
- Only **2 of 6 pairs** were evaluable, so every arm-pooled band is INCONCLUSIVE and no hypothesis could have come out CONSISTENT. The behavioural gate in the evaluability rule, calibrated on development data, did not transfer to the fresh measurement stream.
- **Under the literal reading of an ambiguous clamp-validity rule** (Reviewer D, D1), Hm_window and H1d_plus_Hm become INCONCLUSIVE instead. That would leave four frozen contradictions.

**The pre-registered sensitivity analysis.**
- It was meant to drop the behavioural gate. My implementation dropped it from the bands but not from evaluability. **This is my error.**
- Applied as written [sens], it gives 5 evaluable pairs on uP160. The descriptive pattern below is Reviewer C's §3.6, adopted. φ_nec is the log-necessity shift: 0 = paired baseline, 1 = paired block.

| | what was done during steps 0–99 | result |
|---|---|---|
| 1 | **Destroy T's forward content** (another episode's values, value gradient stopped); T's attention mass free (vrg) or held on the baseline's own share trajectory (sha-hi-rolled) | **Switches the organisation: 5/5 evaluable pairs** (per base E 3/3, M 2/2; CP 95% [0.48, 1]). On uP163: sha-hi-rolled F on both bases; vrg F on M, P on E (φ 0.82). |
| 2 | **Remove all gradient through T's edges**, forward pass intact (gsv) | **M: switches fully** (2/2). **E: switches route necessity** (φ 0.81–0.85, just above the 0.8 band), **without mixture-like behaviour** (ρ_L 0.57–0.61). uP163: M F, E hybrid (φ 0.64). |
| 3 | **Remove only the routing gradient on T's logits** (gsg) | E: switches necessity in 2/3 pairs (φ 0.65–0.89), not behaviour. M: hybrid in 3/3 (φ 0.36–0.43). On M the value gradient through T (kept in gsg, removed in gsv) is the measured difference. |
| 4 | **Hold T's forward share at its initial level** while T's logits and values keep learning (sha-lo-real) | **Switches nothing:** 5/5 evaluable pairs (6/6 including the non-evaluable M\|813; uP163 2/2). T's latent share at release catches up with or exceeds the baseline's (release jump +0.12 to +0.14 on E). |
| 5 | Graded suppression, attention odds × 0.1 | Switches 5/5. In mechanism this is a near-block: ≈ 90% forward-share reduction that also scales both gradients. |

**What follows** [sens; interp].
- Neither forward-content removal nor learning-signal removal is **necessary**. Content removal is **sufficient**. Learning-signal removal is sufficient on M and switches route necessity on E.
- Holding the forward share low **while T keeps learning** is not sufficient.
- **No single attention statistic at release explains all arms.** Reviewer C checked mean share, max-head share and within-route concentration.
- I had proposed a unifying description ("the organisation switches when T receives no useful learning signal"). **It is withdrawn as a finding** (C2): taken literally it is contradicted by gsg on E and by fst [dev]; read loosely it is circular. Four alternative descriptions fit as well:
  - over-determination;
  - a direct logit-recruitment race;
  - a latent-share mediator;
  - value learning.

  They need a new pre-registered test.

**Limits.**
- One parent per confirmation, two bases (n_eff ≈ 2), 3 pairs per base.
- One bistable independent parent out of 5 candidates.
- Data orders and new-layer initialisations are shared between uP160 and uP163 (D3).
- Every positive count is [sens], not frozen.

## Answers to the 21 questions (mandate §15)

1. **What caused the XIV change?**
   - **Excluded** [frozen, 2 pairs; sens, 5 pairs; uP163]:
     - the attention-share *trajectory during the window* as the cause (Hm_window), within the limited contrast this design achieves: forward-share difference ≤ 0.10 on E and ≤ 0.03 on M, over steps ≈ 60–99;
     - forward-content absence as a *necessary* condition (H1d);
     - "only the routing gradient" and "any learning signal through T" as *necessary* conditions (H2g_R, H2g_RV);
     - the requirement that both content absence and low share be present (H4).
   - **Sufficient:** destroying T's content; on M, removing T's learning signal.
   - **Not identified:** a single mechanism. The latent-share mediator (Hm_release) and content-plus-signal (H1d_plus_H2g) are **not excluded**. Both are contradicted only in the sensitivity analysis, by gsg, in one parent, at thin margins.
2. **Is complete blockage necessary?** No [sens 5/5]. Attention odds × 0.1 switches as well. Mechanistically that is ≈ 90% share reduction, with both gradients scaled down (L2). (The development 2/2 on E used ro0, not own0.)
3. **Can partial suppression produce it?** At odds × 0.1, yes. Weaker offsets were not tested (L2: non-discriminating).
4. **Can a different intervention with comparable information reduction reproduce it?** Yes: content destruction with attention mass kept (vrg) [sens 5/5].
   - sha-hi-rolled is **not** an independent replication. Its release state and immediate Δloss are near-identical to vrg's (C11); it adds only the Hm_window test.
   - The cross-layer competitor boost (xbo) was not run, so mandate A3 is addressed only by the share clamp.
5. **Can generalised training disturbance explain it?**
   - **Immediate disturbance size: excluded on M** (XIV data; in-cohort vrg Δloss 0.011–0.013). Ledger N24 still stands: strong layer-2-input noise switched M in degraded runs.
   - **Immediate size on E:** excluded only [sens], against a 0.18 threshold from XIV's noise control, which was another cohort and another edge set. The **frozen H3 rule is not met** (1 evaluable E pair).
   - **Trajectory-loss size on E: not excluded.** Window Δloss orders the N arm below every F/P arm. This is confounded, because not building T raises loss.
   - **Gradient magnitude:** does not track the outcome on either base.
   - **No matched non-targeted disturbance arm** was run in XVIII.
   - The "identical loss at fixed parameters" of gsg and gsv is true by construction and is **not** evidence.
6. **Were the organisations established with independent diagnostics?** Partly.
   - Test-time necessity separates the pure classes cleanly. gsg runs, gsg E\|812, and uP163 vrg E and gsv E are mixed-necessity hybrids.
   - The behavioural diagnostic is the weakest. Plain ρ is confounded with lapse; the lapse-aware fit's thresholds did not transfer to fresh streams; and it lags training.
   - The structural and causal diagnostics read the same manipulated edges.
   - On E, gsg and gsv switch necessity **without** mixture-like behaviour.
7. **Did the models remain competent?**
   - Every confirmation run has KL ≤ 0.034.
   - On M\|813, the block and four arms fail only the accuracy criterion: 0.83 against the baseline's 0.87, because early-organised models are less accurate. So that pair is non-evaluable.
   - On uP163 M the arms' KL is 0.039–0.043, against a 0.045 ceiling.
8. **Does the effect occur in independent parents?**
   - In the one bistable parent (uP163): the block switches both bases; sha-hi-rolled is F on both; sha-lo-real is N on both [sens; frozen: non-evaluable].
   - Under the frozen per-parent rule, vrg and gsv are MIXED.
   - **Only 1 of 5 candidate parents is bistable.**
9. **Counterexamples.**
   - No bistable parent is a counterexample.
   - Four candidates cannot be tested: uP161 has no early base; uP162 has no competent seed; uP164 is early only; uP165 is mixture only.
   - Competent hybrid-necessity baselines occur without intervention in uP163–uP165.
10. **Stage of learning?** Not tested (Experiment D was conditional and was not run).
11. **XIV claims that became stronger.**
    - The block switches structural + causal class in **6/6** fresh pairs; 5/6 also pass competence.
    - The switch is corroborated by test-time necessity, so it is not an attention-statistic artefact.
    - It recurs in one independent parent.
12. **XIV claims to weaken or withdraw.**
    - **Withdrawn: "Blocking the information pathway" as the operative description.** A forward-intact learning-signal deficit switches M. A share clamp does not switch. A complete block is unnecessary.
    - **Withdrawn: "off-route sham" on E.** The sham's edges were a subset of the block's (A1-1).
13. **Which hypotheses are distinguished?**
    - Window-share (Hm_window) from content and signal: the sha arms, within the small contrast.
    - Content-necessary (H1d) from signal-sufficient: gsv on M.
    - H2g_R and H2g_RV from content: vrg switches with a routing gradient present. That is a *necessity* failure. The dose rule's potency wording (§10.11) concerns gsg/gsv sufficiency and does not rescue them.
14. **Which remain equivalent?**
    - The latent-share mediator (Hm_release) and content-plus-signal: neither is excluded.
    - gsg dissociates from the latent mediator on **both** bases, in opposite directions. The robust case is E: T's share recovers after release, yet the early route ends unnecessary.
    - Forward content and the value learning it enables (vsg was not run).
    - Uninformative content and an uninformative learning signal (the same manipulation in vrg).
15. **Already known.**
    - Kleinman, Achille & Soatto (ICLR 2024): an early gradient-only deficit decides a two-pathway race in deep linear networks.
    - Chen et al. (ICLR 2024): early suppression of an attention structure, after which an alternative strategy is learned.
    - Achille et al. (2019), critical periods.
    - Singh et al. (2024, 2025), training-time clamping.
    - Cloud et al. (2024), gradient masking.
    - Competing-algorithm work: Zhong et al. 2023; Ding et al. 2024; Park et al. 2025.
16. **Distinctive?** The decomposition itself, in a transformer with two competent organisations, scored by test-time necessity. And its descriptive outcome: content destruction suffices; learning-signal removal suffices on one base; a forward-share clamp with continued learning does not; no single attention statistic explains the arms. No prior study found in a focused search makes this decomposition. That is absence, not proof.
17. **Does O/D predict this?** No (O3).
18. **What is missing from COR?** Learning dynamics: a selection principle, a link from intervention to complex, and a time axis (ORIGINAL_COR_OD_RELEVANCE.md §4).
19. **A narrow publishable claim?** **Not yet.**
    - The frozen primary had 2 evaluable pairs.
    - The positive counts come from the sensitivity analysis.
    - Two bistable parents.
    - The most a re-run could support: "in this transformer, holding a route's forward share at its initial level during formation, while its logits and values keep learning, does not change the organisation; destroying its forward content does; removing the gradient through it does fully on one organisation and in route necessity on the other."
20. **Essential before publication.**
    - Re-run the frozen design with the causal diagnostic as the only gate, or a behavioural gate validated across streams, on fresh orders.
    - More bistable parents, which needs a recipe that yields both organisations.
    - The vsg arm.
    - A gradient-only rolled-content arm (Reviewer C §3.4).
    - An arm that forces the latent share independently of the learning signal.
    - A matched non-targeted disturbance arm.
    - A second task or architecture.
21. **Next foundational question** [interp]. Every robust T4 finding since XIII-D concerns what training *selects*, and COR's mathematics has no dynamics. The next question is whether a minimal two-route model of learning dynamics reproduces this pattern: content destruction selects; signal removal selects depending on the organisation; a share clamp with continued learning does not. Kleinman et al.'s deep-linear pathway-race analysis is the starting point. That would show whether the pattern is generic gradient-flow behaviour, before more transformer compute is spent.

## Run record

- **Repository.**
  - Branch `cor-phase15-step1-pathway-selection-causal-identification`.
  - Design commits `630905f`, `b184b88`, `ab6e6dc`, `0bf8dd6`.
  - **Freeze `1547ec9`** (2026-11-18T19:54:40Z).
- **Timeline** (UTC).
  - Start about 16:20 on 2026-11-18.
  - Confirmation 19:54–00:22.
  - Primary score at 00:24, after the H3 fix.
  - Container restart about 00:45, resumed 00:56.
  - Screening to 02:51, parents to 04:05.
  - Stability continuations from 04:07.
- **Machine.** 2 vCPU, 7.8 GB RAM. About 11 minutes per 1211-step run including measurement.
- **Deviations** (`logs/COMPUTE_LOG.txt`):
  - H3 script output path (fixed; output identical);
  - the container restart (resumed bitwise from resume state);
  - stability continuations deferred until after the parent stages.

## Post-review corrections (Reviewers C and D; full adjudication in REVIEWS_AND_ADJUDICATION.md)

- **C1:** withdrawn — "restricting a route's attention share does not change the organisation". Replaced by the forward-share-clamp-with-continued-learning statement.
- **C2/C15:** withdrawn as a finding — the "no useful learning signal" unifying description. Alternatives listed.
- **C3/C4:** the frozen verdicts are reported as frozen results. Hm_release and H1d_plus_H2g are "not excluded".
- **C5:** the gsg dissociation is stated for both bases. Q14 corrected.
- **C6:** H3 rescoped (trajectory-size H3 not excluded on E; no arm D). Same-θ loss identity withdrawn as evidence.
- **C7:** E gsg/gsv switch necessity, not behaviour. The "complete double dissociation on every run" claim is withdrawn.
- **C8:** counts corrected and made non-circular.
- **C9:** dose-rule wording applied.
- **C10:** frozen per-parent rule applied to uP163 (vrg and gsv MIXED). "Pattern matches uP160" withdrawn.
- **C11:** sha-hi-rolled not counted as an independent replication. The share arms' contrast is small.
- **C12–C14:** smaller corrections, applied.
- **D1:** the clamp β rule is ambiguous. Both readings are reported.
- **D2:** streams used once before by XIII-E (development only).
- **D3:** shared data orders and initialisations across parents disclosed.
- **D4:** post-fix hash recorded.
- **D5–D10:** applied.

## Stability readout (step 2546)

(Filled in from `logs/score_cont18.json` when the continuations finish. They were derived from the frozen, gate-inflated P list. Reported first for gsg/gsv on E and gsg on M.)
