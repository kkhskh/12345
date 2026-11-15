# Investigation XIII-C: critical resolution of relational decision mechanisms

## Main research synthesis

COR Phase 1.5, Step 1 (provisional ≈ 25%). This is not a phase closure. It states no thesis, does not validate COR, and makes no claim
about the original O/D operators.

**Evidence labels:**
- **[proof]** mathematical proof;
- **[num]** numerical verification;
- **[dev]** development;
- **[conf]** frozen confirmation (commit d6281c6) on new fine-tuning runs of three pretrained parents (not independent parents);
- **[expl]** post-freeze exploratory.

Reviews are agent reviews, not human peer review.

Topic documents:
- CAUSES_OF_NEURAL_SOLUTION_SELECTION.md (A);
- DECISION_LAW_AND_MATHEMATICAL_IDENTIFIABILITY.md (B);
- JOINT_RELATIONAL_FEATURE_AND_REPAIR.md (C);
- CAUSAL_CONTROLS_AND_MEDIATOR_VALIDATION.md (D);
- FROZEN_CONFIRMATION_AND_STATISTICS.md (verdicts);
- OPEN_COR_FOUNDATIONAL_QUESTIONS.md.

## 1. What changed

1. **Implementations are selected by the initialisation, through a race between pathways, not by data order.** [dev; conf]
   - The fine-tuning seed's three randomness sources were separated, with bitwise reproduction of XIII-B.
   - In four parent-level 2 × 2 designs, every crossed cell took the organisation of its *initialisation* diagonal (8/8 cells).
   - The init effect on early readout is 0.33–0.79. Data effects are 0.00–0.09: small, not zero. A hybrid's continuous residue moved with
     the data.
   - **The carrier inside the initialisation is the fresh readout map (WA, 64 × 3), not the fresh layer 2.**
     - The frozen a-priori prediction "layer 2 carries it" (R1b) was falsified in both split cells.
     - A post-review run with WA differing from both layer 2 and data followed WA (0.08 vs 0.86) [expl; one run, review 3 M1].
   - Mechanism: every run grows early readout at first. Mixture-destined runs form a q-mass layer-2 head early, after which early readout
     *declines* (suppression); hybrids form it late and keep both.
   - Masking one pathway for the first 600 steps flips the final organisation in **3/3 runs**, in both directions, and the flip persists
     after release [conf R2a, R2b]. One of these runs trained poorly.
   - Predicting the class from step-546 internals failed near the boundary (R2c).
2. **The decision-law difference is causal but not identifiable as an internal operator.** [proof; num; dev; conf]
   - The power-mean exponent ρ is *exactly* equivalent to the exact mixture applied to under-confident target-question evidence
     (α′(0.6) = 0.635), followed by output sharpening (T2). No input–output data can separate the two descriptions.
   - The fitted ρ depends on the loss and the family: 0.56 (MSE) against 0.71 (KL) for the same model.
   - Changing the routing by intervention shifts ρ_KL as predicted: +0.20, −0.13 and −0.37 (the last in the badly trained run)
     [conf R3].
   - **Narrowed by review 3 (M4):** ρ_KL tracks distance from Bayes (r = −0.93 with log KL; partial r(lik, ρ | KL) = −0.49). With lapse
     fitted, the shifts are +0.12 / −0.07. The data show that organisation changes move the fitted law together with accuracy, not a
     specific combination-law change.
   - **The early "pooled evidence" hypothesis is rejected.** Its readout state is group-resolved (SWAP transfer 0.99).
   - ρ explains only 1–4% of the early class's KL from Bayes. Of the early class's 2.5–6× larger error:
     - about 40% is a structured miscomputation of the right sufficient statistics;
     - about 60% is sensitivity to information Bayes ignores.
3. **E is distinguished from both A1 and A2, behaviourally and at the mediator level.** [num; dev; conf R7]
   - Four on-support edit families separate the three rules. Every class follows E.
   - XIII-B's early-readout negative was a **site artefact**: relational eligibility reaches layer 2 at the other-question tokens.
   - At the covering site, an 8-dim E-specific, source-specific subspace is identified in early units, with excess 0.31–0.36 over the
     source-free patch. An A2-trained subspace fails.
   - Narrowed by review 3 (M6): the site is all tokens and the margins mirror the behaviour, so this is a low-dimensional carrier of the
     E-following behaviour, not a localized mediator.
4. **The two-colluder repair: what is built, how it is used, and why final-layer repair fails.** [proof; dev; conf R4, R5; expl]
   - m (the pair copying count) is sufficient, and the per-group log-weight is linear in the complementary pair's m [proof].
   - Successful early repairs build a **low-dimensional causal carrier (3 dims ≈ 0.8)** that is orthogonal to the linearly decodable m
     and responsibility directions: those are decodable but carry about 0.01 when interchanged.
   - The **frozen layer 2 uses it through its mixture rows**: change in group attention ∝ change in the exact reliable-group posterior,
     r = 0.84–0.89 [conf].
   - The failure of final-layer repair is **not capacity** (two blocks fail).
   - **One block at layer 0 suffices; one at layer 1 does not.** Per-group supply makes the layer-2 repair possible but slow, so the frozen R4 failed:
     - 0.28 at the frozen budget;
     - 0.90 at 3000 steps and 0.93 at lr 3e-3 [expl];
     - plain layer 2 fails under both of those settings;
     - post-review, the group-agnostic supply also succeeds at those settings (0.93 / 0.90), so the format claim is withdrawn and
       availability is what decides.
5. **Controls rebuilt.**
   - Genuine no-ops are exact (DAS) or ≤ 1e-6 (row path). R6 failed on that strict criterion alone.
   - A displacement-matched natural decoy gives R² ≤ 0.01 against 0.90–0.97 for true rows.
   - **Slope-based decoy metrics were misleading:** unrelated rows give slope 0.4–0.5 with R² 0.03.
   - **Source-free patches carry base-rate artefacts:** own-share 0.3–0.7, so specificity must be measured as excess over them.
   - Mixture units use two **parallel, additive, non-redundant** routes (Q, R). Early units add a direct other-token route.
   - route_Q is a stable, initialisation-determined unit property (sd 0.03 across streams).

## 2. Predictions

| | verdict | key reason |
|---|---|---|
| R1 init vs data; L2 vs readout init | **falsified** | a: 4/4 pass; b: the readout init, not the L2 init, carries it |
| R2 pathway competition | **falsified** | ro0/l2q flips 3/3; the step-546 predictor failed 2/5 |
| R3 routing → ρ | confirmed | |
| R4 repair conditions | **falsified** | supG 0.28 at 1000 steps; early-unit L2 0.378 < 0.4 |
| R5 carrier ≠ decodable; rows | confirmed | |
| R6 row specificity | **falsified** | no-op 7.2e-7 ≠ 0 (all specificity criteria passed; one diagnostic unit) |
| R7 early E mediator | confirmed | one diagnostic unit |
| R8 parallel routes | confirmed | |

Tally: 4 confirmed, 4 falsified, 0 inconclusive. There was one protocol incident (C8 silently skipped, then run). See FROZEN_CONFIRMATION.

## 3. Answers to the mandate's 18 questions

1. **Initialisation**, not data order, decides the binary organisation in the three parents tested (8/8 crossed cells).
   - Within the initialisation, the **readout-map init** decides: R1b, plus one post-review run in which WA beat both layer 2 and data
     [conf; expl].
   - Data moves the continuous hybrid residue (0.06–0.18). [dev; conf]
2. **An early, path-dependent race:** early-readout growth against layer-2 q-mass head formation.
   - Blocking one side for 600 steps changes which state locks in (2 competent runs, plus 1 under-trained).
   - Active competition in unmasked runs is suggested by the trajectories, not shown (review 3).
   - A step-546 *predictor* is unreliable near the boundary. [dev; conf]
3. **Both:** a near-binary head-formation event (mix_mass bimodal) plus a continuous hybrid residue.
4. **Intervening on the routing shifts ρ (R3), together with accuracy** (review 3 M4: ρ largely indexes distance from Bayes). Internally, "ρ" is indistinguishable from q-evidence tempering. It is not caused
   by early pooling.
5. **Equivalent** reparametrisations: power mean ≡ tempered q-evidence + sharpening ≡ misstated α′ (T1, T2). **Distinguishable:** ρ from
   pure temperature; mixture-type from log-linear pooling; lapse in principle, though confounded under misspecification.
6. **Early class, in Shapley shares of KL:**
   - reliability weighting 7–8%;
   - lapse 6–7%;
   - calibration 2–4%;
   - ρ / tempering 1–4%;
   - eligibility coefficients 1–3%;
   - in total, the families capture about 20%;
   - non-parametrically, about 40% is structured and about 60% is non-invariance.

   Mixture models: tiny errors, mostly non-invariance (72–79%).
7. **A distributed low-dimensional carrier** of the copying-aware evidence, at the q source and readout tokens. It is **not** an explicit,
   linearly read m.
8. **Location:** q + readout tokens. **Evidence of use:** DAS interchange 0.80–0.88 on held-out pairs; probe directions give ≈ 0 under
   interchange and ablation; the frozen rows carry 0.82, with mass changes tracking the exact posterior. [conf R5]
9. **Not capacity, and not simple optimisation** (3× steps and 3× learning rate leave plain layer-2 repair at 0.00).
   - What decides is **whether the joint relation is available at the layer-2 input**.
   - Layer 0 can compute it ahead of the frozen routing; layer 1 cannot.
   - Supplied in any tested form (group-agnostic or per-group), a layer-2 repair learns to use it given more steps or a higher learning
     rate [expl, post-review].
   - The earlier per-group-format claim is withdrawn.
10. **Partially.** Organisation changes ρ (R3) and routes (R8). Its effect on repair ability is inconsistent in early units (0.07–0.63).
    Not established.
11. **Yes,** with rebuilt controls (R² against matched decoys) in the pure mixture units (1 confirmation unit + 4 development units). The
    frozen R6 failed on strict no-op equality.
12. **Yes, at the covering site** (all tokens, 8 dims) [conf R7, one diagnostic unit]. Not at single sites.
13. **Yes, behaviourally and at the mediator,** against A1 and A2. Graded rules were not tested.
14. **Genuinely separate routes exist** (Q and R, plus O in early units). They are **additive, not redundant**. XIII-B's "restoration" was
    serial.
15. **Early/hybrid:** MLP2 (0.39–0.52, largely redundant with attention, interaction −0.4), the direct other-token route O (0.39–0.50) and
    other heads (up to 0.33 in hybrids).
16. **No internal predictor of the behavioural transition** survived. The organisation predictor failed near the boundary.
17. **Survive across implementations:**
    - the target variables (E-gating, reliability weighting);
    - the input–output non-identifiability (T1, T2);
    - the m-sufficiency mathematics;
    - additive multi-route transmission.

    **Specific to this generator:** routes, ρ values, the readout-map result, G = 3 / C = 2 statistics.
18. **Strengthened:** causal selection mechanism; identifiability boundaries; E identification; accessibility versus capacity. **Revised:**
    - "seed-dependent routing" (it is init-determined);
    - Q5a (a site artefact);
    - the early pooling story (rejected);
    - "layer 2 decides" (the readout init decides);
    - "per-group format is necessary" (withdrawn: any supplied form works with more optimisation).

    **Open:** see OPEN_COR_FOUNDATIONAL_QUESTIONS.md.

## 4. Limitations

- Three parents; one run per cell. The predictions share units.
- Several arms had one diagnostic unit (R6, R7).
- One intervention run trained badly.
- The 4000-step determinism replay was not done.
- GATE was not run.
- Review findings and dispositions: `reviews/ADJUDICATION.md`.
