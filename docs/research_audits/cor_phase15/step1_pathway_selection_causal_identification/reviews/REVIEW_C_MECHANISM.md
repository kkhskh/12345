# Reviewer C (mechanistic interpretation): post-confirmation review of COR Investigation XVIII

**Reviewer.** An agent in the Reviewer C role (mandate §12, "AFTER confirmation"). This is **not human peer review**. The design review (Reviewers A and B, before the freeze) and this results review are separate activities.

**Scope.** I set out to decide which causal explanations the confirmation data support or exclude, and at what evidence level, and whether the reports overclaim.

**Labels:**
- **[frozen]**: output of the frozen scorer, `logs/score_XVIII.json`.
- **[post-hoc]**: the investigator's post-hoc files, `logs/posthoc_nobehav18.json` and `logs/posthoc_parents18.json`.
- **[num]**: my own recomputation from the raw measurement files or checkpoints. Scripts are in `/tmp/revC18/`. All of these were short read-only checks; I trained nothing.
- **[argument]**: reasoning, not data.

**Inputs read.**
- Mandate §3, §4, §9, §12 and §15.
- PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md §1–§10, with the main focus on §10.3–§10.11.
- FROZEN_PREDICTIONS_AND_RESULTS.md, PATHWAY_INFORMATION_VS_ATTENTION_COMPETITION.md, INDEPENDENT_PARENT_GENERALIZATION.md, ORGANIZATIONAL_MECHANISM_VALIDATION.md, PRIOR_WORK_AND_NOVELTY.md and MAIN_RESEARCH_SYNTHESIS.md.
- `scripts/score_XVIII.py`, `scripts/posthoc_nobehav18.py`, `scripts/posthoc_parents18.py`, `scripts/h3_immediate18.py`, `scripts/xviii_train.py` (window-log fields) and `src/xviiim.py`.
- `logs/measure_c18.jsonl`, `logs/measure_p18.jsonl`, `logs/XVIII_parents.json`, `logs/h3_immediate_c18.json` and the per-run `window_log.json` files.

**Stability continuations are absent.**
- `logs/measure_c18_cont.jsonl` and `logs/score_cont18.json` do not exist at the time of review. Stage 2b started at 04:07Z, and the first two continuation runs are on lanes K1 and K2.
- Nothing below uses step-2546 data.
- The continuation list (37 tags) was generated from the **frozen** primary bands, which include the behavioural gate. Most of its P entries are artefacts of the gate (§1.3), not hybrids.

---

## 0. Summary verdict

1. **The post-hoc analysis is a faithful application of the pre-registered sensitivity text. It is not a rescue** (§1).
   - It makes the result *less* favourable to every hypothesis, including the one the investigator later prefers.
   - The rescue risk lies elsewhere. It is in the post-hoc unifying description, which is essentially the pre-registered `H1d_plus_H2g` with its one failing prediction (gsg = P) quietly relaxed.
2. **Robustly excluded, at frozen level** (n = 2 pairs, one per base, one parent), and reinforced post-hoc (5 pairs; uP163 for the arms it ran):
   - **H1d as a necessary condition.** gsg and gsv are never N; φ_nec ≥ 0.36 in all 12 runs.
   - **Hm_window.**
   - **H2g_R and H2g_RV as necessary conditions.**
   - **H4.**
   - **H1d_plus_Hm.**
3. **Weakly contradicted, post-hoc only, and resting on one arm (gsg) in one parent:**
   - **Hm_release.** On M the contradiction sits on a mediator margin of 0.02–0.05 in φ_share, against a denominator of 0.031–0.034.
   - **H1d_plus_H2g.** Two E pairs sit at φ_nec 0.88–0.89, and the result depends on dropping the behavioural class *of the arm*, not only of the baseline.
   - These two should be reported as **not excluded**.
4. **"Route T receives no useful learning signal" is not supported as a mechanism** (§3).
   - Applied literally, it is contradicted by gsg on E (strong flip with a real value-gradient signal and substantial share recruitment) and by fst [dev] (flip with a content-bearing routing gradient).
   - If "useful" is defined by outcome, it becomes circular.
   - In vrg and sha-hi-rolled, "useless learning signal" is confounded with "useless forward content".
   - Several other descriptions fit equally well or better. The most defensible statement is descriptive (§3.6).
5. **There is a robust dissociation the reports miss: gsg and gsv on E** (§3.3).
   - The own0 share recovers after release, to φ_share 0.11–0.46 at steps 150–600, and gsg's max-head own0 share exceeds the baseline's.
   - Yet the early route ends unnecessary: nec_E_own0 0.02–0.19.
   - So T's attention share, measured anywhere from 100 to 600, is not what decides the E outcome.
   - The reports present gsg on **M** as "the one dissociation". That is the fragile one.
6. **H3 claims need rescoping** (§4).
   - On E, the in-cohort window Δloss perfectly separates the non-switching arm from every switching or partial arm (0.025–0.026 against 0.030–0.067).
   - The post-hoc exclusion on E rests only on a cross-cohort, cross-edge-set threshold applied to the same-θ immediate Δloss.
   - "gsg/gsv leave the loss identical" is guaranteed at the same θ, so it is not evidence. Their realised window Δloss is +0.03 to +0.05 on E.
7. **The narrow claim in MAIN Q19 contradicts the study's own gof b = 2.3 result**, and must be rescoped (§6, C1).

---

## 1. (a) Faithful application or rescue?

### 1.1 What the frozen text says

- **§10.8, B fix 4:** "Every verdict is also computed **without the behavioural diagnostic** (structural and causal only), as a pre-registered sensitivity analysis."
- Evaluability (§10.4) is defined through cls2, which contains the behavioural diagnostic. It is a precondition of every verdict.
- §10.8, B fix 5 already applies structural + causal classification to parent screening, so the spirit is consistent.

**Reading** [argument]. The most natural reading of "every verdict … without the behavioural diagnostic" removes it from evaluability too. The frozen code (`score_XVIII.py`, `judge("primary_nb", evp, …)` with `evp` built from behaviour-gated cls2) implements a narrower reading. The post-hoc script changes only the pair set; every other parameter is the frozen scorer's (`band_nb` was already computed by the frozen code for all pairs). I verified this line by line: `posthoc_nobehav18.py` reuses `S["pairs"][pk]["arms"][arm]["primary_nb"]` and the frozen mediator bands.

### 1.2 Direction of the change: not a rescue

| hypothesis | frozen (2 pairs) | post-hoc (5 pairs) |
|---|---|---|
| H1d, Hm_window, H2g_R, H2g_RV, H4, H1d_plus_Hm | CONTRADICTED | CONTRADICTED |
| Hm_release | INCONCLUSIVE | CONTRADICTED |
| H1d_plus_H2g | INCONCLUSIVE | CONTRADICTED |

The post-hoc analysis adds contradictions and rescues nothing. The two hypotheses it newly contradicts are exactly the two that the investigator's later "unifying description" resembles.

### 1.3 Evidence that the behavioural gate is noise at the boundary, not signal [num]

- **The gate changes class within one run between steps 1000 and 1211.**
  - The M|812 baseline is behaviourally HYBRID at step 1000 (ρ_L 0.640, lapse 0.008) and MIXTURE at step 1211 (ρ_L 0.828).
  - The M|811 baseline goes the other way: MIXTURE at 1000, HYBRID at 1211 (lapse 0.000 → 0.018).
  - So **one of the two frozen-evaluable pairs (M|812) is evaluable only because its baseline crossed the lapse/ρ_L boundary between steps 1000 and 1211.**
- **The frozen "unstable" flags are gate artefacts.** The flags on M|812 vrg, gsv, sha-hi-rolled and gof23 and on M|811 sha-lo-real follow from this baseline drift: each arm's own class is identical at 1000 and 1211. Neither FROZEN_PREDICTIONS nor PATHWAY reports these flags; they should be reported and explained.
- **The baseline lapses sit on the threshold.** They are 0.010, 0.017 and 0.019 on E, and 0.018, 0.014 and 0.012 on M, against a single threshold of 0.017.

### 1.4 But the post-hoc reading is not neutral for every arm [num]

Dropping the behavioural diagnostic from the **arm** classes (not only from the baselines) changes the gsg and gsv E results for a reason unrelated to the lapse-threshold failure that motivated the sensitivity analysis.

| E arm | ρ_L | lapse | behavioural class | structural / causal |
|---|---|---|---|---|
| block | 0.74–0.82 | 0.007–0.018 | mostly MIXTURE | MIX / MIX |
| vrg | 0.73–0.87 | 0.004–0.016 | MIXTURE | MIX / MIX |
| **gsg** | **0.46–0.54** | **0.000** | HYBRID (fails ρ_L ≥ 0.68 by about 0.15–0.2) | MIX / MIX (E\|812 HYB) |
| **gsv** | **0.57–0.61** | 0.000–0.006 | HYBRID | MIX / MIX |

- gsg and gsv on E end with **mixture-type route necessity but non-mixture behaviour**: an early-like exponent with zero lapse.
- The analysis variant that removes behaviour from the baseline and block classes only, but scores arms by all three diagnostics (`/tmp/revC18/variant.py`), gives:
  - gsg on E: P, P, P;
  - gsv on E: P, P, P;
  - H1d_plus_H2g on E: no longer contradicted by gsg. It is contradicted only by sha-lo-real P artefacts, which inherit the baseline's lapse-boundary problem.
- This variant is **not** the pre-registered one, and I do not propose it as the analysis. The point is that the post-hoc contradiction of H1d_plus_H2g depends on reading gsg on E through two diagnostics that both read the manipulated edges (ORGANIZATIONAL_MECHANISM_VALIDATION §B2: "two measurements of one latent quantity").

**Conclusion (a).**
- The post-hoc analysis is a faithful, literal application of the pre-registered sensitivity text.
- It is the right analysis to report alongside the frozen one, but it remains a **sensitivity analysis**. It may not replace the frozen primary in headlines (C3).
- The H1d_plus_H2g contradiction it produces is fragile, and must be labelled as such.

---

## 2. (b) What is excluded, at what level

### 2.1 Per-pair data, recomputed from `measure_c18.jsonl` [num]

I recomputed φ_nec, the bands without the behavioural gate, competence and φ_share independently (`/tmp/revC18/recompute.py`). Every value in FROZEN §3 reproduces to the third decimal.

Per-component detail the reports omit:

| pair | gsg φ (E part / M part) | gsv φ (E part / M part) | gsg φ_share@100 | denominator A_base − A_blk |
|---|---|---|---|---|
| E\|811 | 0.893 (0.88 / 0.90) | 0.823 (**0.72** / 0.93) | 0.496 | 0.104 |
| E\|812 | 0.647 (0.47 / 0.82) | 0.813 (**0.72** / 0.91) | 0.488 | 0.100 |
| E\|813 | 0.884 (0.85 / 0.92) | 0.849 (**0.77** / 0.93) | 0.551 | 0.099 |
| M\|811 | 0.426 | 1.015 | **0.818** | **0.0315** |
| M\|812 | 0.407 | 0.978 | **0.852** | **0.0337** |
| M\|813 (non-evaluable) | 0.360 | 0.955 | 0.850 | 0.0346 |

**Evaluability of M|813.** The M|813 **block** also fails the accuracy criterion (0.838 < 0.873 − 0.03). That is why the pair is non-evaluable, so counts such as "6/6" or "3/3" on M include a non-evaluable pair.

### 2.2 Per-hypothesis verdicts and their real evidence level

| hypothesis | frozen (2 pairs) | post-hoc (5 pairs) | what does the work | margin and robustness | uP163 |
|---|---|---|---|---|---|
| **H1d** (forward content selects; needs gsg = N, gsv = N) | CONTRADICTED | CONTRADICTED | gsg and gsv not N | **Robust.** gsg φ ≥ 0.36 and gsv φ ≥ 0.81 in every run; N needs ≤ 0.2. Independent of the behavioural gate. | gsv 0.64 / 0.94 (not N) |
| **Hm_window** | CONTRADICTED | CONTRADICTED | sha-hi-rolled F; sha-lo-real N | Robust in φ. **Weak manipulation on M** (§3.5): the forward-share contrast is ≤ 0.03 and confined to steps ≈ 60–99. | sha-hi F 2/2, sha-lo N 2/2 |
| **Hm_release** | INCONCLUSIVE | CONTRADICTED | gsg only, on both bases, in opposite directions | **Fragile on M:** mediator margin 0.018–0.052 against a pair-to-pair SD of about 0.04–0.05. On E the mediator is far from the band (0.49–0.55) but the outcome margin is 0.08. vrg, gsv, sha-hi and sha-lo all match. | gsg not run; M denominator 0.0048, uninformative |
| **H2g_R** | CONTRADICTED | CONTRADICTED | vrg and sha-hi-rolled F (prediction N) | Robust: φ ≈ 1.0 | sha-hi F 2/2 |
| **H2g_RV** | CONTRADICTED | CONTRADICTED | vrg and sha-hi-rolled F; gsg not N | Robust | as above |
| **H4** | CONTRADICTED | CONTRADICTED | gsv and gsg not N; sha-hi F | Robust | as above |
| **H1d_plus_Hm** | CONTRADICTED | CONTRADICTED | sha arms F and N, not P | Robust | as above |
| **H1d_plus_H2g** | INCONCLUSIVE | CONTRADICTED (E); INCONCLUSIVE (M) | gsg F on E\|811 and E\|813 only | **Fragile.** φ 0.893 and 0.884 against 0.8, and P under the three-diagnostic arm classification (§1.4). Every other arm matches in 5/5. | gsg not run |
| **H3, E** | not excluded | "excluded" | vrg immediate Δloss < 0.18 | Cross-cohort threshold; the in-cohort trajectory Δloss is *not* excluded (§4) | — |
| **H5** | P4 non-evaluable | 1/1 bistable parent shows the block effect | — | 1 replication; 3 of 4 screened candidates are not bistable | — |

**Comments on the evidence levels:**
1. **The frozen contradictions are legitimate under the frozen rule, and the reports undersell them.**
   - The frozen A2 rule (§10.8) needs only 2 definite mismatches to contradict, so six contradictions **are frozen results**, each on one E and one M pair of one parent.
   - The reports call the frozen primary "INCONCLUSIVE" without qualification. That applies to the **arm-pooled bands** (fewer than 5 evaluable pairs), not to the hypothesis verdicts.
   - Conversely, with 2 pairs the frozen analysis **could not** produce CONSISTENT for anything. "INCONCLUSIVE" for Hm_release and H1d_plus_H2g therefore means "not contradicted on 2 pairs", and nothing more.
2. **"Excluded" here means "excluded as the sole or necessary condition in its pre-registered band form".**
   - The sufficiency halves are supported: forward-content removal suffices (vrg, sha-hi-rolled, fst [dev]), and learning-signal removal (gsv) suffices or nearly suffices.
   - The reports should say so explicitly; MAIN Q1 partly does.
3. **Units.** n = 5 evaluable pairs in one parent is effectively 2 bases (n_eff ≈ 2). With 3/3 per base the Clopper–Pearson lower bound is 0.29. Pooled counts such as "5/5" must be shown per base.
4. **The pre-registered dose rule (§10.11) is not honoured in the text.**
   - It requires a CONTRADICTED H2g verdict at T = 100 to be reported as "weaker than content removal at the matched window, not excluded as a lever".
   - The reports instead say H2g_R and H2g_RV are "excluded" (PATHWAY §4; MAIN Q13).
   - The contradiction here comes from vrg and sha-hi-rolled flipping *with* a routing gradient, which is a necessity failure the potency phrasing does not address. So the frozen phrasing must be quoted and its inaptness explained, rather than silently dropped.

---

## 3. (c) Mechanism: is "no useful learning signal" supported?

### 3.1 What the description is, formally [argument]

> PATHWAY §4 / MAIN Q1: "the organisation switches when route T receives no useful learning signal during formation".

Arm by arm:
- The **pre-registered** `H1d_plus_H2g` ("content plus learning signal") predicts vrg F, gsg P, gsv P/F, sha-hi-rolled F and sha-lo-real N.
- The data match every one of these except gsg on E (F, P, F).
- The post-hoc description is H1d_plus_H2g with gsg's band left free and an explanation added afterwards ("partial where the base relies on recruiting T (M)").
- That is the standard shape of an accommodation. It must be labelled as **H1d_plus_H2g with its failing prediction relaxed**, not as a new account that "fits every arm".

### 3.2 The description, taken literally, is contradicted [num, dev, argument]

1. **gsg on E.**
   - gsg keeps T's real content in the forward pass, keeps the value gradient through T, and keeps part of the recruitment signal through the complement logits (§10.9 item 1: 0.5–0.7 on own0).
   - T's latent share at step 100 is half-way to the baseline's (φ_share 0.49–0.55). At step 150 it is near baseline (φ_share 0.11–0.15). The max-head own0 share exceeds the baseline's from step 150 to step 600.
   - So T receives a useful learning signal *and* is recruited, yet the E organisation switches (F in 2 of 3 pairs).
   - The investigator's gloss says gsg is "near-full on E" because E relies less on recruiting T. That runs opposite to their own mediator data: gsg recruits T **more** on E (φ_share 0.5) than on M (0.82–0.85), yet switches **more** on E.
2. **fst [dev, D = 801, one run per cell].**
   - The forward pass through T is blocked, and the routing gradient is the full softmax Jacobian with T's real values. That is a content-bearing signal that says "attend to T".
   - fst flips both bases (PRE_EXPERIMENT §10.10).
   - Under a literal "useful learning signal" account it should protect at least partly. The reports drop fst from the discussion; it must be included as development-level counter-evidence.
3. **If "useful" means "leads to T being recruited or established", the account is Hm_release** and inherits that hypothesis's gsg failures. If it means "leads to the baseline organisation", it is circular.
4. **In vrg and sha-hi-rolled, "uninformative learning signal" and "uninformative forward content" are the same manipulation.**
   - The description assigns their flips to the signal by argument. H1d's sufficiency half explains them equally well.
   - The only arms that separate the forward pass from the learning signal are gsg and gsv (forward intact, signal removed) and sha-lo-real (forward share reduced, signal intact). Two of these three are the arms the description handles least well.

### 3.3 The latent-share mediator: gsg on M is the fragile dissociation; gsg and gsv on E are the robust one [num]

**gsg on M, mediator robustness.**
- A_base − A_blk = 0.031–0.034 on M, against 0.099–0.104 on E.
- gsg's A100 is 0.1426–0.1451, within 0.005–0.006 of the block's.
- Pair-to-pair SDs across orders are 0.0013 (gsg), 0.0010 (block) and 0.0018 (baseline). Combined, φ_share has an SD of roughly 0.04–0.06 per pair.
- The observed φ_share of 0.818 and 0.852 is 0.4–1.3 SD above the 0.8 band edge.
- The result holds under max-head share (0.84 and 0.87) and under pooled anchors (0.84), but it is **step-sensitive**. At step 150 gsg on M has φ_share 0.64–0.69, which is band P and would *match* the observed P.
- The gsg-M "contradiction" is therefore a property of the frozen choice of step 100, made at the exact point after which gsg's route re-grows. This is the frozen rule, so the verdict stands formally. As mechanistic evidence it is weak, and MAIN Q14 rightly says so, but FROZEN §4 and PATHWAY §3.3 present it without the caveat.

**gsg and gsv on E: a robust dissociation of share from use.**

| E arm | φ_share at 100 / 150 / 600 | own0 max-head share at 600 (baseline 0.84) | nec_E_own0 at 1211 (baseline 1.17–1.41) | outcome |
|---|---|---|---|---|
| gsg | 0.49–0.55 / 0.11–0.15 / 0.26–0.35 | **0.91–0.93** | 0.02–0.19 | F, P, F |
| gsv | 1.01–1.03 / 0.43–0.46 / 0.39–0.42 | 0.73–0.75 | 0.04–0.06 | F, F, F |
| vrg | 0.92–0.94 / 0.94–0.97 / 0.97–1.02 | 0.36–0.41 | 0.001–0.002 | F |

On E, T is substantially attended after release under gsg and gsv, yet it ends unnecessary. No measure of T's attention share at steps 100–600 tracks the E outcome across arms.

The caveat is that shares are logged only up to step 600, while necessity is measured at 1211, where the structural lik_RO1 for gsg on E is 0.09–0.27. Attention may fall after step 600.

Even so, "the route is established at release" cannot be the mediator on E. PATHWAY §3.3 and §4 ("gsg on M is the one dissociation") must be corrected. **The dissociation exists on both bases, in opposite directions:**
- on E the outcome is more switched than the mediator predicts;
- on M it is less switched.

**Within-route concentration is not the mediator either** [num, `/tmp/revC18/sel.py`, pairs E|811, E|812, M|811, M|812].
- Measure: normalised within-T attention entropy over the 6 own sources at step 100. Lower values mean more selective attention.
- On E:
  - baseline 0.71;
  - sha-lo-real 0.62–0.63 (N);
  - **vrg 0.69 (F)**;
  - block 0.78;
  - gsv 0.79;
  - gsg 0.88 (min over heads).
- vrg develops the same concentration as the baseline and still switches. On M every arm lies at 0.83–0.86.
- So no attention-pattern summary I could compute at release separates all arms: mean share, max-head share and within-T concentration all fail.

### 3.4 Alternative descriptions that fit the data as well as, or better than, the investigator's [argument, on num data]

| description | vrg / sha-hi | gsv | gsg E | gsg M | sha-lo | fst [dev] | gof 2.3 | remaining problem |
|---|---|---|---|---|---|---|---|---|
| **D0, investigator: "no useful learning signal"** | ✓ (by argument) | ✓ | ✗ (useful V and recruitment present, F) | ✓ (stated after the fact) | ✓ | ✗ | ✓ | gsg E, fst |
| **D1: over-determination.** Forward-content removal and learning-signal removal are each (nearly) sufficient; neither is necessary; holding the forward share low with both intact is not sufficient. This is the "(flip, flip)" cell of the v1 design table (§5). | ✓ | ✓ | ✓ | P is a weaker case of signal removal | ✓ | ✓ | ✓ | It is a description, not a mechanism. Its only anomaly is that gsg is partial on M. |
| **D2: direct logit recruitment race.** The baseline organisation needs T's own logits to be driven up by a content-bearing gradient during formation. | ✓ (pushed down) | ✓ | ✓ (the share rise comes only from the complement) | ✗ unless the value gradient is added on M | ✓ (latent logits rise, release jump +0.12 to +0.14 on E) | ✓ (latent share only 0.155 in dev) | ✓ | Base-specific: on M gsg (V present) is P while gsv (V absent) is F |
| **D3: latent-share mediator at release (Hm_release)** | ✓ | ✓ | ✗ | ✗ (marginal) | ✓ | ✓ | ✓ | gsg in both directions; E share recovers later yet the route stays unnecessary |
| **D4: value learning or downstream consumers learn T's content** | ✓ | ✓ | ✗ (V present, F) | partly | ✓ | ✓ | ✓ | gsg E |

**Comparison of D2 and D0.**
- D2 fits E fully, including gsg E, which D0 does not. On M it needs one added component.
- The confirmation data give that component: on M, gsg (value gradient through T kept) is P at φ 0.36–0.43, while gsv (value gradient removed) is F at 0.98–1.02.
- On E the two are indistinguishable (gsg 0.65–0.89 against gsv 0.81–0.85; gsg is *higher* in 2 of 3 pairs).
- **The data-grounded base asymmetry is therefore "the value gradient through T partly protects the M organisation and not the E organisation".** It is not "M relies on recruiting T".
- This too is post-hoc, rests on n = 3 per base in one parent, and cannot be separated from forward content (vsg not run).

**What would discriminate** [argument]:
- **vsg** (value gradient stopped, routing gradient intact, real content). D0, D2 and D3 predict N or partial; D4 predicts F.
- **A gradient-only rolled-content arm** (forward real, routing and value gradients computed against another episode's values). D0 predicts F (useless signal); D1 and the H1d-sufficiency reading predict that the forward pass protects.
- **An E gsg arm with complement-logit recruitment also stopped.** This separates D2 from "residual recruitment".
- **Saving attention logs to step 1211 for gsg and gsv on E.**

### 3.5 Further interpretive problems with the share arms [num]

1. **sha-hi-rolled is not independent evidence from vrg.**
   - At release its latent share (0.1447–0.1482 on E) matches vrg's (0.1399–0.1422).
   - Its immediate Δloss equals vrg's to 1e-4 (β ≈ 0 at the baseline's parameters).
   - It differs from vrg only in the forward share during steps ≈ 60–99, and that difference is undone at release.
   - MAIN's table ("vrg; sha-hi-rolled … every evaluable pair, both parents") counts one finding twice. sha-hi-rolled adds only the Hm_window test.
2. **The forward-share contrast tests only the last ~40 steps.**
   - The baseline's forward share is 0.134 → 0.137 over steps 0–50 on E and flat at 0.142 on M. It grows only over steps 75–99: to 0.23 on E and 0.17 on M.
   - So sha-lo-real differs from the baseline's forward share by at most 0.10 on E and **0.03 on M**, in the last part of the window. The full block's contrast is 0.17 on M.
   - On M, Hm_window is therefore tested with a small manipulation: it excludes "the share must *grow* in the window", not "the share must be *substantial*".
3. **sha-lo-real is not "share held low" at release.**
   - The latent share at step 100 is *above* the baseline's on E (φ_share −0.20 to −0.31), because the routing gradient keeps raising T's logits under the clamp.
   - It therefore excludes only a forward-share account, not a latent-logit race (D2 and D3 both predict N).
   - PATHWAY §4's heading, "Forward share or attention competition during formation, as a cause", is too broad: latent competition is untouched.
4. **gof b = 2.3 (odds × 0.1, achieved latent share 0.1345–0.1349 ≈ block) switches 5/5.**
   - Attention-share restriction of that size *does* select. By L2 it cannot be separated from content and gradient scaling.
   - So "restricting a route's attention share does not change the organisation" (MAIN Q19, Q16) is contradicted by the study's own data. The correct statement is "holding T's forward share at its initial level while its logits keep learning does not change it".

### 3.6 What the evidence does support, stated descriptively [argument, on frozen and post-hoc data]

Within uP160 (2 bases, 5 evaluable pairs) and, for the arms it ran, uP163:
1. Destroying T's forward content for steps 0–99, with the value gradient stopped, switches the organisation. This holds whether T's forward share is free or held on the baseline trajectory. Its sufficiency is robust.
2. Removing all gradient through T's edges, with the forward pass initially intact, switches M fully. On E it switches route necessity, at margins of 0.01–0.05 above the F band, without producing mixture-like behaviour (ρ_L 0.57–0.61). On uP163 E it is a hybrid (φ 0.64).
3. Removing only the routing gradient on T's logits switches E necessity in 2 of 3 pairs (again without mixture behaviour) and gives a hybrid on M. The value gradient through T is the measured difference between gsg and gsv on M.
4. Holding T's forward share at its initial value while T's logits and values keep learning switches nothing. At release the latent share has caught up with, or overtaken, the baseline's.
5. **No single attention statistic at release explains all arms.** Neither forward content alone nor learning signal alone is necessary.

Any causal mechanism beyond this, the investigator's included, is a hypothesis for a new pre-registered test.

---

## 4. (d) H3 (general disturbance)

1. **Frozen:** H3 on E is not excluded (n_ok = 1). This is correctly reported.
2. **Post-hoc rule application** [post-hoc, verified num]:
   - vrg is F on 3/3 E pairs under the post-hoc evaluability. Its own behavioural class is MIXTURE, so this does not depend on §1.4.
   - Its immediate Δloss is 0.084–0.112 against 0.18.
   - So the frozen rule's text is satisfied under the sensitivity reading.
   - Limitations that must be stated:
     - The 0.18 threshold is "≈ 0.18" from XIV's E location-matched noise. That was a different cohort and different data orders, on XIV's ro0 edge set (36 sources), not XVIII's own0 (6 sources).
     - It is the only non-switching E control on that metric.
     - The mandate's arm D ("matched non-targeted disturbance", A6) was **not run in XVIII**. The confirmation cohort contains no non-targeted disturbance arm.
3. **In-cohort trajectory disturbance does not exclude H3 on E** [num, `window_log.json` plus the baseline slog]. Window Δloss (mean of steps 90–99 minus the paired baseline's):

   | E arm | window Δloss |
   |---|---|
   | sha-lo-real (N) | +0.025 to +0.026 |
   | gsg (F/P) | +0.030 to +0.033 |
   | sha-hi-rolled | +0.045 to +0.048 |
   | gsv | +0.050 to +0.054 |
   | vrg | +0.056 to +0.060 |
   | block, gof | +0.061 to +0.067 |

   - A threshold of about 0.028 separates N from every F or P arm perfectly.
   - This is confounded: an arm that does not build T has a higher loss, so trajectory Δloss is partly a *consequence* of the selection. But it means the in-cohort data do not exclude a trajectory-size H3 on E.
   - **On M** the ordering fails: gsv +0.001 to +0.002 (F), sha-lo-real +0.002 (N), gsg +0.003 to +0.004 (P). All values are ≤ 0.010.
4. **Gradient energy.** The window global gradient norm (mean 1.00–1.19) and Σ|g|² are within about 5% across all arms on each base, and do not order outcomes. Examples: on E sha-lo-real 235–257 against block 236–262; on M gsv 173–189 (F) against sha-lo-real 181–197 (N).
   - This is in-cohort evidence against a gradient-magnitude H3 on both bases.
   - The baseline has no gradient log, and gradient direction is not measured, so "gradient disturbance" in the mandate's sense is only partly addressed.
5. **"gsg and gsv leave the forward pass and loss identical at fixed parameters and still select"** (MAIN Q5; PATHWAY §5) is guaranteed by construction at the same θ. Pre-review A (§1.4, table at line 158) explicitly said it is "not evidence". Their realised window Δloss on E is +0.030 to +0.054, which is in the same range as the switching arms. This point must be withdrawn as H3 evidence.
6. **M:** L4 (XIV) plus the in-cohort vrg immediate Δloss of 0.011–0.013 (below XIV's non-switching M sham at 0.040) support excluding immediate-size H3 for logit- and value-level disturbances on T. Ledger N24 (strong layer-2-input noise switches M in degraded runs) must stay attached to that claim; MAIN Q5 omits it.

**Corrected H3 statement.**
- **Immediate-Δloss size** is excluded on M (XIV plus in-cohort) and on E (post-hoc, against a cross-cohort threshold).
- **Gradient-magnitude size** does not track outcome (in-cohort, both bases).
- **Trajectory-loss size** is not excluded on E.
- **No matched non-targeted disturbance** was run in XVIII.

---

## 5. Independent parents

1. **The frozen per-parent rule (§10.5)** is "the band of the two pairs if both agree, otherwise MIXED". Applied post-hoc to uP163 [num from `posthoc_parents18.json`]:

   | arm | E | M | uP163 result |
   |---|---|---|---|
   | vrg | P (φ 0.82; causal HYBRID, nec_E 0.150 at the 0.15 bound) | F | **MIXED** |
   | gsv | P (φ 0.64; HYBRID on both diagnostics) | F | **MIXED** |
   | sha-hi-rolled | F | F | **F** |
   | sha-lo-real | N | N | **N** |
   | block | F | F | F |

   INDEPENDENT_PARENT §C4 instead says "the component pattern matches uP160's" and "gsv … sits near the band edge on E in both parents". φ 0.64 with a two-diagnostic HYBRID is not "near the band edge". It is a partial result, and differs from uP160 E (0.81–0.85).
2. **uP163's M mediator is uninformative.** A_base − A_blk = 0.1375 − 0.1327 = 0.0048, so no φ_share statement about uP163 M is meaningful.
3. **uP163 M competence is marginal.** The arms' KL is 0.039–0.043 against the 0.045 ceiling, which should be stated.
4. **gsg was not run on parents** (by design). Every gsg-based statement, including "gsg strong on E, partial on M", is uP160-only.
5. **The parent side finding overreaches.** "The hybrid state that gsg produces in uP160 is therefore a state these networks can reach on their own" (INDEPENDENT_PARENT line 33) infers identity of state across parents and seeds from a shared necessity pattern. It should be weakened to "competent hybrid-necessity states also occur without intervention in uP163–uP165".
6. **What uP163 supports** [post-hoc]:
   - the block effect, and the sha-hi-rolled F and sha-lo-real N pattern, in one independent bistable parent, one data order, 2 pairs;
   - the vrg and gsv components only partly (MIXED).

---

## 6. (e) Required corrections, in priority order

**C1 (high). MAIN_RESEARCH_SYNTHESIS.md Q19 (line 102) and Q16 (line 95).**
- **Problem.** "restricting a route's attention share during formation does not change the organisation" is contradicted by gof b = 2.3, which switches 5/5 (§3.5.4).
- **Fix.** Replace with: "holding a route's *forward* share at its initial level during formation, while its logits and values continue to learn, does not change the organisation; destroying its forward content, or removing the gradient through it, does (fully on M; in route necessity but not behaviour on E)."
- **Also in Q16.** Replace "a useless or absent learning signal does" with "destroyed content (with the value gradient stopped) or absent learning signal does". Uninformative signal and uninformative forward content are not separated (§3.2.4).

**C2 (high). PATHWAY §4 (line 86), MAIN line 43 and MAIN Q1 (line 53): the "no useful learning signal" description.**
- Relabel it as `H1d_plus_H2g` (pre-registered) with its gsg prediction relaxed after the data.
- State that taken literally it is contradicted by gsg on E (useful V signal plus share recruitment, F) and by fst [dev] (content-bearing routing gradient, F), and that otherwise it is circular.
- List the alternatives D1–D4 (§3.4) as equally or better fitting.
- Remove "What fits every arm" (PATHWAY line 84) and "fits all arms" (MAIN line 41).
- Withdraw the gsg gloss "partial where the base relies on recruiting T (M)" (PATHWAY line 88): it contradicts the investigator's own φ_share data. Replace it with the measured contrast: on M, gsg (V kept) is P and gsv (V removed) is F; on E there is no difference.

**C3 (high). FROZEN §2 and §4, PATHWAY §3.1 and §4 (line 75), MAIN lines 26–31: evidence levels.**
- **(i)** State that the frozen hypothesis verdicts are 6 CONTRADICTED and 2 INCONCLUSIVE under the frozen A2 rule (n = 2 pairs, one per base). "INCONCLUSIVE" applies to the arm-pooled bands. With 2 pairs, CONSISTENT was unreachable.
- **(ii)** Keep the post-hoc analysis labelled as the pre-registered *sensitivity* analysis, not as the primary.
- **(iii)** PATHWAY line 75, "Excluded [post-hoc; on 2 evaluable pairs, frozen]", mixes levels. Give each bullet its own level: H1d, Hm_window, H2g_R, H2g_RV and H4 are frozen-contradicted; Hm_release and H1d_plus_H2g are post-hoc only.
- **(iv)** Report the frozen "unstable" flags and their cause: baseline behavioural drift between steps 1000 and 1211 (§1.3).

**C4 (high). Hm_release and H1d_plus_H2g must be reported as "not excluded; contradicted only in the sensitivity analysis, by one arm (gsg), in one parent, at thin margins"** (FROZEN §4 table lines 115 and 119; PATHWAY §3.3).
- For Hm_release, give the M denominator (0.031–0.034), the margins (0.018 and 0.052) and the step sensitivity (P at step 150).
- For H1d_plus_H2g, give the φ margins (0.084 and 0.093) and the dependence on dropping the arm's behavioural class (gsg on E: ρ_L 0.46–0.54, lapse 0.000).

**C5 (high). PATHWAY §3.3 (table) and §4 (line 93), MAIN Q14 (line 85): the gsg dissociation.**
- Replace "gsg on M is the one dissociation" with: gsg dissociates from the latent-share mediator on **both** bases, in opposite directions.
- The E dissociation is the robust one: the own0 share recovers after release, and the max-head share exceeds the baseline's, yet the route ends unnecessary (§3.3).
- The M dissociation is marginal.
- MAIN Q14 says the latent-share mediator and the learning-signal description are "observationally equivalent". That is inconsistent with FROZEN §4 listing Hm_release as CONTRADICTED. If they are equivalent, gsg on E contradicts both; resolve this.

**C6 (high). MAIN Q5 (line 64), PATHWAY §5 and pre-experiment §4 H3 ("gsg has zero forward disturbance").**
- Withdraw same-θ "identical loss" as H3 evidence. It is guaranteed by construction (Pre-review A).
- Report the realised window Δloss of every arm, and state that on E it orders N below every F/P arm (§4.3).
- Rescope the H3 conclusion as in §4: immediate size excluded (M in-cohort; E post-hoc against a cross-cohort, cross-edge-set threshold); trajectory size not excluded on E; no XVIII arm D.
- Attach ledger N24 to the M exclusion.

**C7 (medium). gsv and gsg on E are "F" by structural and causal diagnostics only** (FROZEN §3 table, PATHWAY §3.2, MAIN table line 36, MAIN Q12 line 78: "a forward-intact, learning-signal-only deficit selects").
- Add that their behaviour is not mixture-like (ρ_L 0.46–0.61 against 0.73–0.87 for block and vrg), that gsv's E component of φ is only 0.72–0.77, and that gsv on uP163 E is a two-diagnostic hybrid.
- Replace "selects" with "switches route necessity on E; switches fully on M".
- Fill ORGANIZATIONAL_MECHANISM_VALIDATION §B5 with this; it is currently empty. Also correct MAIN Q6 (line 66), "complete double dissociation on every run": it is false for the gsg runs (M: nec_E 0.26–0.28 and nec_M 0.25–0.28), gsg E|812, and uP163 gsv E and vrg E.

**C8 (medium). Counts.**
- "sha-lo-real N in 6/6" (FROZEN line 114; PATHWAY line 76; MAIN lines 37 and 50) should be 5/5 evaluable, or "6/6 including the non-evaluable M|813".
- "outcome only P in 3/3" (FROZEN line 115) should be 2/2 evaluable.
- MAIN Q11 (line 74), "the block effect replicates … 5/5 evaluable", is circular, because evaluability requires the block to be F. State "the block switches structural + causal class in 6/6 pairs; 5/6 also pass competence".
- Give per-base counts and Clopper–Pearson intervals throughout (frozen §10.4 requires per-base reporting).

**C9 (medium). The dose rule (§10.11) is not applied.**
- PATHWAY §4 and MAIN Q13 report H2g_R and H2g_RV as excluded. Quote the frozen reporting obligation.
- Explain that the contradiction here is a *necessity* failure (vrg and sha-hi-rolled switch with a routing gradient present), to which the potency wording does not apply. Do not drop it silently.

**C10 (medium). INDEPENDENT_PARENT §C4 (lines 57–59) and MAIN table line 35.**
- Apply the frozen per-parent rule: vrg MIXED, gsv MIXED, sha-hi-rolled F, sha-lo-real N on uP163.
- Withdraw "pattern matches uP160's" and "near the band edge on E in both parents".
- State that uP163 M's mediator denominator is 0.0048 and that M competence is marginal (KL up to 0.043).
- Weaken line 33 (§5.5).

**C11 (medium). sha-hi-rolled is presented as an independent replication of vrg** (MAIN line 35; Q4, line 58).
- State that their release states and immediate Δloss are near-identical. sha-hi-rolled informs only Hm_window.
- State the limited contrast of the share arms: forward-share difference ≤ 0.10 on E and ≤ 0.03 on M, confined to steps ≈ 60–99.
- State that sha-lo-real's latent share at release is at or above the baseline's. It therefore does not test a latent-logit race.

**C12 (low). MAIN Q2 and Q3.**
- State that odds × 0.1 is a ≈ 90% forward-share reduction that also scales the routing and value gradients (L2). "The route is still reachable" is true, but mechanistically it is a near-block.
- "dev 2/2" used ro0 on E, not own0.

**C13 (low). MAIN Q7.**
- State that the M|813 **block** also fails the accuracy criterion, which is why the pair is non-evaluable, not only the four arms.
- State the uP163 M KL margins.

**C14 (low). Stability.**
- State that no step-2546 data exist at the time of writing.
- State that the continuation list derives from frozen, gate-inflated P bands.
- When `score_cont18.json` arrives, report it for gsg and gsv on E and gsg on M first. These are the only runs whose band is mechanistically in doubt: the dev gsg hybrids drifted in ρ_L from 0.57 to 0.86 between steps 1211 and 4000.

**C15 (low). PRIOR_WORK §4** must not describe the contribution as "share restriction does not select; useless or absent signal does" (see C1 and C2). The defensible distinctive content is the decomposition itself and the descriptive pattern of §3.6.

---

## 7. Answers to mandate §15 items within Reviewer C's remit, as corrected

- **Q1 (what caused the XIV change).**
  - Not the forward-share trajectory during the window (frozen and post-hoc; uP163).
  - Not forward-content absence as a *necessary* condition (frozen and post-hoc), and not a learning-signal deficit as a *necessary* condition (frozen and post-hoc).
  - Each is (nearly) sufficient.
  - No mediator measured at release accounts for every arm.
  - The cause is **not identified beyond this disjunction** (§3.6).
- **Q5 (general disturbance).** As in §4.
- **Q13 (distinguished).**
  - Hm_window is distinguished from the content and signal components.
  - H1d-necessary is distinguished from H1d-sufficient.
  - H2g-necessary is distinguished from H2g-sufficient.
  - H4 and H1d_plus_Hm are excluded.
- **Q14 (equivalent or undistinguished).**
  - Forward content against value learning (vsg not run).
  - Uninformative forward content against uninformative learning signal (vrg and sha-hi-rolled).
  - D0–D4 in §3.4.
  - Hm_release and H1d_plus_H2g are not excluded at a robust level.

---

## Appendix: reproducible checks

All checks were run on 1 thread in under 20 s each, reading only frozen and logged files and saved checkpoints. They are in `/tmp/revC18/`:

| script | what it checks | section |
|---|---|---|
| `recompute.py` | per-pair φ_nec (with E and M components), φ at step 1000, bands without behaviour, competence, φ_share | §2.1 |
| `traj.py` | own0/own2 mean and max-head shares at steps 25–600, and φ_share at every step | §3.3 |
| `wlog.py` | window gradient norm, Σ\|g\|² and window Δloss (steps 90–99) against the baseline slog | §4.3–§4.4 |
| `variant.py` | evaluability without behaviour, arm bands with all three diagnostics | §1.4 |
| `sel.py` | within-T attention entropy at steps 100 and 150 from the saved checkpoints | §3.3 |
