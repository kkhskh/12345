# Adjudication of the XVII reviews

**About these reviews.** All three reviewers are agents (mandate §15), and this is not human peer review. I adjudicated as the investigating agent. Reviewers' scratch scripts and outputs are archived in `reviews/scratch_A/`, `reviews/scratch_B/` and `reviews/scratch_C/` (access-time record and scorer copy).

**Rules.**
- **Frozen verdicts are not rescored.** `logs/score_XVII.json` is unchanged, and every call in it was computed by the frozen rules.
- **Interpretations can be withdrawn or qualified.** Reviewers can change what a verdict is taken to mean.
- **Labelling of reviewer numbers.** Numbers a reviewer produced after the freeze are labelled [expl, review]. They are not frozen evidence.
- **Spot checks.** Where cheap, I re-checked a claim myself before accepting it.

**Withdrawal IDs.** Withdrawals are numbered **W1–W12**. Each ID is referenced in MAIN_RESEARCH_SYNTHESIS.md, and `scripts/check_reports17.py` enforces this.

## Decisions on Review A (COR mathematics)

| finding | decision | action |
|---|---|---|
| F1 Proposition S false as stated; "0 of 729" vacuous | **accepted** | **W1** |
| F2 count-form D is not Definition 3; corrected S′ (quadratic ⇒ order ≤ 2; exact LP: no order ≤ 2 score reproduces Bayes at a = 2, 3) | **accepted** | S′ replaces S in ORIGINAL §4 and Q4/Q26 |
| F3 "order-1 optimum" column is the NLL-optimal rule | accepted | column relabelled |
| F4 numbering and omissions (Update Propositions 5–7) | accepted | ORIGINAL §2/§5 amended |
| F5 Case C(ii)/Q3: an equivariance axiom would give only cross-network invariance | **accepted** | **W2** |
| F6 Theorem P correct | no defect | — |
| F7 windows with lo > 0 are not code exchanges | **accepted** (same as B3) | **W4** |
| F8 P2 does not separate H2 from the mandate's H3 | **accepted** (same as B1) | **W3** |
| F9 P0, P1, P3, PL not hypothesis consequences; P5 does not exclude H1-with-binding | **accepted** | **W6**, **W8** |
| F10 "shared operation" not an identified object; equivalence is definitional | accepted | Q15 extended |
| F11, F12 wording | accepted | Q4, Q25 reworded |

**My check of F1.** `xvii_od_check.py` computes the flip statistic from `Pev` (evidence sources 3–5) only, as the reviewer says. The reviewer's `od.json` gives a 0.535 / 0.506 change rate for argmin D over all six sources. **I accept that the headline was vacuous.**

## Decisions on Review B (mechanisms)

| finding | decision | action |
|---|---|---|
| B1 P2 near-tautological under Theorem P; no H2/H3 discrimination; duplicated code does not transfer (0.08–0.22) | **accepted** | **W3** |
| B2 P3 is P2 rescored (complement windows); P1 is Theorem P; real count is 4 independent mechanism tests + P0 | **accepted, verified** | **W5**, **W6** |
| B3 layer locus not identified; layer-2-only swap gives 0.65–0.70 in A2_s704; restoring derived identity cuts transfer | accepted, with the reviewer's caveat (crude centroid shifts) | **W4** |
| B4 routing transplant content-mismatched; matched transplant 0.11–0.47; routing necessary (0.02–0.32) | **accepted** | **W7** |
| B5 PL interference = generic single-slot forgetting (acquired-slot-only control gives the same drop); Q17 "accelerates" unsupported | **accepted** | **W8** |
| B6 P5 rests on off-manifold states; random code 0.00–0.03 vs real-code exchange 0.32–0.40 | accepted | **W9** |
| B7 imminence: frozen checkpoints 20–220 steps before natural acquisition | accepted | **W10** |
| B8 baseline used as control for \|U\| = 1; random-code controls 0.00–0.07 | accepted; the new controls are recorded [expl, review] and transfer survives them | **W11** |
| B9 not a confidence artifact (KL, accuracy) | no defect; KL now quoted | — |
| B10 duplicated U→U controls; P0 at arity 2 unresolved at 20-step resolution | accepted | **W11**, **W12** |
| B11 "no activation moves" literally false | accepted | **W11** |
| B12 "H4 mixture" not identified | **accepted** | **W5** |

**My check of B2.** From `logs/analysis/*.json`, the mean donor value under [2,3) equals the mean recipient value under [0,2) to within 0.05 in every evaluable model:
- 0.70/0.68, 0.82/0.83, 0.79/0.79, 0.60/0.65, 0.90/0.90, 0.79/0.78;
- the pair-level values agree to within about 0.05.

This fits the complement identity, given that the episodes differ (they are content-swapped).

## Withdrawals and qualifications

- **W1. Proposition S, and "0 of 729 inputs change under a gate flip".** Withdrawn. Replaced by S′: every source-level O_c or D_c with single-source germs is a quadratic form, hence an order-≤2 score, and an exact LP shows that no order-≤2 score reproduces the GATE Bayes decisions at arity 2 or 3 [proof + num, Review A]. Argmin D over all six sources does respond to gate flips (about 51–54%), but it does not track the gate (F_G 0.14–0.23).
- **W2. "An equivariance constraint on ρ and P would yield a prediction" (Case C(ii), Q3).** Withdrawn. Such an axiom gives only invariance across (θ, θ_π), which is cross-network. It does not predict reuse within one trained network.
- **W3. P2 as evidence for "a common relational computation" and as rejecting H3** (Q12, Q14, Q18, Q23, Q25). Withdrawn.
  - Under Theorem P a slot *is* its code, and the mandate's H3 allows replacing implementation-specific components, which here can only be the code. P2 therefore does not discriminate H2 from H3.
  - The code also acts only as a unique binding label: duplicating it does not transfer.
  - What survives is descriptive. At partial checkpoints, an unacquired slot's own content is gated correctly when the acquired slot's label drives layers 0–1 [frozen P2]. The derived identity carried in the residual is then the donor's.
- **W4. Layer locus** ("the causal site is layers 0–1 for every slot", "late layer largely code-agnostic", "the code in layers 0–1 is necessary and sufficient"). Withdrawn.
  - Windows with lo > 0 change only the raw code vector. Derived slot identity remains, and is decodable at accuracy 0.92–1.00.
  - In A2_s704 an identity swap confined to layer 2 transfers 0.65–0.70 [expl, review].
  - What survives: re-adding another slot's raw code at layer 2, after the early layers have processed the original code, matters little.
- **W5. "Every frozen mechanism prediction held, 6/6", and "H4 mixture".** Withdrawn as counts of independent evidence.
  - P1 is a numerical check of Theorem P.
  - P3 is P2 measured on content-swapped episodes.
  - The independent tests are P2, P4, P5 and PL, plus P0.
  - H2, the mandate's H3 and H4 are **not identified** by any XVII intervention. The H4 verdict is replaced by the descriptive claims in MAIN.
- **W6. P0, P3 and PL as hypothesis tests.** These are replications of development regularities, not consequences of H1–H4. P0 also does not bear on H1, which allows temporary performance differences.
- **W7. P4 "routing alone does not carry it".** Qualified. The frozen transplant pasted the donor's content-specific attention onto different content, giving 0.05–0.17. A content-matched transplant gives 0.11–0.47, and forcing baseline attention during the relabel cuts transfer to 0.02–0.32 [expl, review]. Routing is **necessary**, and partly sufficient when matched.
- **W8. PL interference as evidence of slot-specific maintenance, and Q17's "targeted pressure accelerates one slot".** Withdrawn.
  - Training on an *already-acquired* slot's loss produces the same collapse of the other slots: A3_s804 0.87 → 0.23; A2_s705 0.96 → 0.61 [expl, review]. This is generic forgetting under a fresh AdamW restart.
  - Acceleration was not testable (ceiling), and checkpoints were 20–220 steps from natural acquisition.
- **W9. P5 as ruling out uniform reuse (Q23).** Withdrawn. P5 compares off-manifold label mixtures. A matched-norm random code gives 0.00–0.03, against 0.32–0.40 for another trained code [expl, review], which shows partial cross-code tolerance. P5 does not exclude H1-with-binding. It is reported descriptively.
- **W10. Generality of transfer across distance-to-acquisition.** Scope restricted. All frozen checkpoints lie 20–220 steps before the unacquired slots acquire naturally. Transfer far from acquisition rests on one development model (C_s202 step 1000, 0.96).
- **W11. Controls and wording.**
  - Q8's "controls of 0.01–0.12" is replaced by perturbation controls of 0.00–0.07 (frozen windows plus the reviewer's random-code controls).
  - U→U controls were duplicated per donor; pair counts are now reported net.
  - "No activation moves between slots" becomes "no slot's input content is copied to another slot, and no task-relevant evidence moves".
- **W12. P0 at arity 2.** Described as unresolved at 20-step snapshot resolution, not as non-staggering. The MIXED verdict stands as scored.

## What survives after review

1. Case D: O/D is silent on positional reuse [proof]. Corrected S′: source-level O/D cannot represent the GATE decision [proof + num].
2. Theorem P [proof, verified independently by Reviewer A].
3. Partial-slot states exist in fresh models (6 of 11 resolvable; all 3 acquiring arity-3 models) [frozen P0].
4. Relabelling an unacquired slot as an acquired one in layers 0–1 gives it gate use, 0.65–0.90. This is far above the perturbation controls (0.00–0.07), it improves KL to Bayes, and the reverse removes the donor's gate use [frozen P2; expl, review B9].
5. Transplanting another slot's content-specific attention is insufficient. Matched routing is necessary [frozen P4; expl, review].
6. Label mixtures between layers 0 and 1 degrade gate use after acquisition [frozen P5, descriptive].
7. Single-slot training, on any slot, degrades the other slots [frozen PL; expl, review].

## Decisions on Review C (integrity)

**Overall.** Reviewer C found integrity sound, with no critical or major findings. Specifically:
- the freeze chronology holds;
- all 27 freeze hashes match;
- first-read access times show no early reads outside the frozen queue;
- all 48 jobs exited 0;
- the scorer output is byte-identical.

| finding | decision | action |
|---|---|---|
| F1 an undisclosed development job on C_s202 ran 07:25–08:02, concurrent with blind training | **accepted** | **W13**: disclosed in MAIN and appended to COMPUTE_LOG.txt as a post-review note. It used XVI checkpoints only. |
| F2 "first read = score_XVII.py" is imprecise | accepted | **W14** |
| F3 arity-3 spreads are 60–100, not 80–160 | **accepted, verified** (s802 60, s803 80, s804 100) | **W15** |
| F4 other unacquired slots under targeted training end at 0.00–0.10 | **accepted, verified** | **W15** |
| F5 U→U controls conflated with baselines | **accepted, verified**: model means 0.04–0.06, pair-level 0.00–0.13; 0.01 and 0.12 were single-slot baselines | **W15** |
| F6 banners on frozen documents | accepted | **W16**: the frozen text is at `git show c419773:<path>` |
| F7 denominators | accepted | already given in MAIN's verdict table (pooled and per-arity k/n) |
| F8 trainer docstring is stale but hashed | accepted | left unedited because the file is frozen; noted here |
| F9 small sample (6 evaluable; Clopper–Pearson lower bound 0.54) | accepted | already disclosed |

**Checks the reviewer could not run.** These were blocked by a pending approval. Equivalent checks run by me are in `logs/VERIFY_CLEAN.txt`, but they are not independent of the investigator:
- **Bitwise snapshot replay:** step 15, max |diff| = 0.0.
- **Analysis re-run on one model:** steps 6–9, A3_s803 battery reproduced exactly.
- **Real-data scorer refusal:** step 12, rc = 2.

The 13-case scorer tamper test was not run by anyone.

**Concurrent editing.** The reviewer noted that other agents were editing the package during its review. Those were my adjudication edits after Reviews A and B, and they touched only reports, not frozen code or data. Reviewer C's hash checks were made against c419773.
