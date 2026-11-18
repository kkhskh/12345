# Review A: mathematical validity (COR Investigation XVI)

**Reviewer:** an independent agent (Reviewer A, mandate section 11). This is an agent review, not human peer review.
**Scope:** the generator, the order structure, the Bayes and reference rules, the value and cue matching, shortcut elimination, the arity-4 propositions, the diagnostics' reference properties, the stage and acquisition definitions, and whether the frozen predictions follow from their assumptions.
**Method:** I read `src/xvitask.py`, `src/xvidiag.py`, `scripts/xvi_math*.py`, `scripts/xvi_value_search.py`, `scripts/score_XVI.py` and `scripts/xvi_train.py`, the logs (`xvi_math_verified.json`, `xvi_value_grid.json`, `xvi_math_arity4.json`, `score_XVI.json`), the 18 confirmation `traj.jsonl` files, and the reports. I recomputed the key quantities with my own scripts in `/tmp/revA16/` (`r1.py` to `r12.py`, `lp.py`). These use exact enumeration, LPs and reference-table lookups only. No model was trained, and no package file was modified.

---

## Findings

### 1. [MAJOR] The value-versus-cue confound is resolved only for the cue's *law* and its *NLL* benefit. At the decision level the competing cue is not matched at all.

**Evidence** (`r1.py`, `r2.py`, `r3.py`; exact enumeration over all 729 inputs):

| | low `2:0.9:0.55:0.05` | high `2:0.9:0.9:0.05` |
|---|---|---|
| agreement strategy's accuracy benefit, acc(order-2) − acc(order-1) | **0.0373** | **0.0000** |
| input mass on which the order-2 (gate-free) argmax differs from the order-1 argmax | **18.9%** | **0%** |
| gate's accuracy gain | 0.0196 | 0.0283 (ratio **1.45**, below the pre-set 1.5 tolerance) |
| competitor NLL benefit / gate NLL value | 0.075 / 0.080 = **0.94** | 0.076 / 0.140 = **0.54** |
| competitor benefit as a fraction of the order-1 excess loss | 0.48 | 0.35 |
| gate value (KL) carried by disagreeing-pair inputs / agreeing-pair inputs | 0.031 / 0.048 | 0.084 / 0.055 |

- In the high condition, and in both arity-3 conditions (C and D), the agreement strategy never changes a decision. It is a pure calibration refinement of the order-1 rule.
- In the low condition it changes 18.9% of decisions and is worth more accuracy (0.037) than the gate itself (0.020).
- The grid shows the same pattern throughout: at b1 = 0.9, ν = 0.05 the agreement accuracy benefit is exactly 0 for every b0 ≥ 0.8 and positive for every b0 ≤ 0.75 (`logs/xvi_value_grid.json`). In all 15 tolerance-satisfying pairs, the high member has an agreement accuracy benefit of 0 and the low member 0–0.076.
- The manipulation also changes *relative* attractiveness. The competitor's benefit relative to the gate's moves by exactly the value ratio (0.94 against 0.54).
- Most of the value change sits on the disagreeing-pair inputs: 0.031 against 0.084 nats. There the cue says "g is probably 0", and what changes is how much trusting the single source is worth.

**Assessment.**
- "The competing cue is held fixed" holds for (a) the joint law of the inputs, which is in fact identical (see survivor S1), and (b) the absolute NLL benefit of the order-2 rule over the order-1 rule.
- It does **not** hold for the cue's decision consequences, nor for its strength relative to the gate.
- The mandate frames the question in terms of decision value. The contrast is therefore clean only under one specific operationalisation (absolute NLL benefit). Under a decision-level or a relative-competition account, the low condition has the *stronger* competitor.
- `DECISION_VALUE_AND_COMPETING_CUES.md` §3 l.53 says the contrast "answers Review A's XV finding". That holds only in the NLL sense. The accuracy-ratio problem is disclosed (§5 Q12), but the decision-level competitor mismatch is not.

**Correction.**
- Add the two decision-level rows above to the §3 table, FROZEN_PROTOCOL §1 and MAIN Q9/Q12.
- State that the competitor's decision benefit is 0.037 against 0, and that the competitor-to-gate ratio differs 1.74×.
- Withdraw or qualify these statements:
  - MAIN l.141, "The value effect disappears when the cue is held fixed": P3 is INCONCLUSIVE, and the cue is not fixed at the decision level.
  - MAIN l.51, "leaves the competing agreement strategy ... as the leading candidate".
  - DECISION_VALUE §5 l.110–111, "fits the competition-driven account better". Under a decision-level or relative competition account, both conditions of XV and XVI moved the competitor and the value in the *same* direction.

### 2. [MAJOR] The frozen P3 alternatives do not follow from their stated accounts: every account considered predicts the same direction.

**Evidence.**
- FROZEN_PROTOCOL §5 P3 maps BENEFIT_SPEEDS to the benefit-driven account and NO_EFFECT to the competition-driven account ("with the competitor's law fixed, timing would then not follow the gate's value").
- That mapping is valid only if competition acts through the competitor's *law* or its *absolute NLL benefit*. As Finding 1 shows:
  - a relative-competition account (competitor/gate 0.94 against 0.54) predicts the low condition is slower;
  - a decision-level competition account (0.037 against 0) also predicts the low condition is slower;
  - the declared difficulty confound (Bayes accuracy 0.725 against 0.900) plausibly predicts the same.
- So BENEFIT_SPEEDS would not have discriminated the benefit account from the competition accounts. NO_EFFECT, or the observed INCONCLUSIVE, is not cleanly predicted by any of them.

**Correction.**
- Re-label P3 as a test of "absolute NLL value of the gate, with the cue's law and its absolute NLL benefit fixed, difficulty and relative competition confounded".
- Remove the "benefit-driven versus competition-driven" interpretation from the frozen alternatives in the synthesis.
- A discriminating design needs a parameterisation in which the competitor's decision benefit and its ratio to the gate value are both held fixed. That needs a fourth parameter, for example a separate reliability for P0 when g = 1, or an independent agreement-noise source.

The INCONCLUSIVE verdict itself is robust. I recomputed t_A from the trajectories under 9 threshold variants (F_G ∈ {0.3, 0.5, 0.7, 0.9, 0.95}; relative KL thresholds of 0.5, 0.75 and 0.9; absolute KL thresholds of 0.05, 0.02 and 0.01). The log ratios always stay near (+0.45, −0.35, −0.13, +0.68), so the verdict remains INCONCLUSIVE (`r6.py`).

### 3. [MODERATE] The floor-conditioned stage certifies "the gate is not yet decision-visible", not "the gate is not used". In 3 of 4 conditions the agreement stage is visible only through graded, confidence-dependent measures.

**Evidence** (`r9.py`).
- I took a geometric interpolation P_w ∝ (order-a optimum)^(1−w) · Bayes^w on the frozen diagnostic set (stream 99,700,000). It passes every stage condition (F_G ≤ 0.1, A_conf ≥ 0.5·A_ref, KL ≤ KL₁ − ½(KL₁ − KL_a)) up to:
  - w = 0.2 in C: 48% of the gate value already captured, G = 0.20;
  - w = 0.4 in WB-high: 71% captured, G = 0.40;
  - w = 0.3 in WB-low: 54% captured, G = 0.30.
- By Finding 1, the agreement strategy has no decision consequence in C, D and WB-high. There the stage cannot be decision-level, and A_conf is the only evidence of agreement use.
- A_conf is invariant to a *global* temperature (verified: constant to 1e-4 at T = 0.5 and 2). It is not invariant to other confidence changes. Label smoothing of the order-a optimum with ε = 0.1–0.5 lowers A_conf from its reference by 13–23% in C, 7–15% in WB-low and 17–21% in WB-high (`r5.py`).

**Empirical consequence (checked).** P5 survives a stricter definition. Adding |G_conf| ≤ 0.1, or |G| ≤ 0.05, to every stage point leaves all 12 acquirers with ≥ 9 consecutive points (`r12.py`).

**Correction.**
- Add a graded-gate floor (|G_conf| ≤ 0.1) to the stage definition in future protocols, and report the result above.
- Say explicitly that in C, D and WB-high the "agreement stage" is a calibration-level stage, with order-1 and gate-free decisions identical. A_conf is normalised only for global temperature.
- For WB-low a decision-level agreement measure exists (Finding 8) and should be reported.

### 4. [MODERATE] F_G is close to a binary indicator on this task, so F_G widths are lower bounds on transition duration.

**Evidence** (`r10.py`, `r11.py`).
- Among decisive flips there are only **2 distinct Bayes posterior types** in every neural condition, so essentially one decision boundary is crossed.
- Along the smooth interpolation P_w, F_G jumps from 0.00 to 1.00 between w = 0.25 and 0.26 (C) and between 0.48 and 0.49 (WB-high). Meanwhile G rises linearly with w.
- In the 16 confirmation acquisitions, the G 0.1 → 0.9 width is 175–450 steps (C_s202: 1900). The F_G width is 75–250 steps for the 15 non-exceptional ones (C_s202: 1850; C_s203: 400 against a G width of 350).
- At the last point with F_G ≤ 0.1, G is already 0.07–0.16 in the WB-low runs (G_conf up to 0.23).

**Assessment.**
- "A transition is called abrupt only when F_G ... changes within a few diagnostic intervals" (LEARNING_SIGNAL §5) uses a measure that is abrupt by construction for any monotone path.
- The empirical abruptness claim survives qualitatively, because G also completes within 175–450 steps. The quoted widths understate durations by roughly 1.2–2.3×.

**Correction.** Report the G (or G_conf) 0.1 → 0.9 widths beside the F_G widths in EXTENDED_RELATIONAL_LEARNING §5 and MAIN Q20. State that F_G is a single-boundary indicator here.

### 5. [MODERATE] "F_G = 0 for every rule that ignores the gate" holds only for rules that do not depend on the gate sources at all. t_first is not specific.

**Evidence** (`r4.py`).
- Take the gate-blind order-a optimum and add a random table over (x_{a−1}, y), standard deviation 1–4 over 20 draws. This gives F_G with a mean of 0.17–0.24 and a maximum of 0.29–0.34 (C), and a mean of 0.03–0.21 and a maximum up to 0.34 (WB-high), although such rules carry zero gate information.
- The flip edit changes only x_{a−1}, so any idiosyncratic sensitivity to that one source produces moves that hit the new Bayes argmax at roughly chance.
- t_A is protected: it also needs KL ≤ 0.75·KL_oa, which such rules cannot reach. The stage floor of F_G ≤ 0.1 is conservative in this direction.
- t_first (F_G ≥ 0.2) is not protected.

**Correction.**
- Restate the V6 property as "F_G = 0 for rules independent of the gate sources".
- Add a null reference: F_G under a reverse-flip or sham edit that changes x_{a−1} while keeping g fixed (possible when a ≥ 2, by also changing another gate source). Use it as the chance level for t_first.

### 6. [MODERATE] At arity 2, "gate value" and "agreement benefit" are computed against the order-2 optimum, not the gate-free mixture. The 0.66% match depends on that choice.

**Evidence** (`r1.py`).
- At a = 2 the order-2 class cannot represent the three-way evidence posterior, so the order-2 optimum is not "the best rule that cannot use the gate" (contrary to the docstrings of `xvidiag.py` and `xvi_math.py`).
- With the exact gate-free mixture as the reference, the numbers become:

| | low | high |
|---|---|---|
| gate value | 0.07972 | 0.13846 (ratio 1.737) |
| agreement benefit, order-1 − mixture | 0.07569 | 0.07704 |

- The agreement benefits are now **1.75%** apart, against 0.66%. The pre-set 2% tolerance passes only narrowly. HIGHER_ORDER §2 acknowledges the 0.1% gate-value mismatch but not its effect on the benefit match.
- The "agreement benefit" (order 1 → order 2) also includes every pairwise evidence interaction, (P1,P0) and (P2,P0) as well as (P1,P2). It is not agreement alone.

**Correction.**
- Report both definitions in DECISION_VALUE §3 and V4.
- Rename gain_gate at a = 2 to "value over the best order-a rule".
- Describe gain_intermediate as "pairwise-evidence benefit".

### 7. [MODERATE] The A_conf reference is not 1 at arity 2.

**Evidence.**
- `src/xvidiag.py` l.17 claims "a tempered order-a optimum has A_conf = 1 at every temperature".
- V6 and `score_XVI.json` refs give A_conf(order-a) = 1.0056 (low) and **1.1161** (high). Only at a = 3 is it 1.0000, where the order-a optimum equals the mixture used as regressor dG. At a = 2 even the mixture gives 1.019 and 1.096.
- Because A_conf = A_p·sB_a/sB, a tempered order-a optimum gives A_conf = A_p(order-a), not 1. V6 checks only constancy, not the value.
- The scorer uses A_ref, so the stage threshold is internally consistent.
- Raw A_conf values in WB-high are about 11% inflated relative to WB-low and to arity 3. Any cross-condition comparison of raw A_conf is biased by that.

**Correction.** Fix the docstring, and divide by A_ref in every cross-condition A_conf comparison.

### 8. [MODERATE] The rejection of a decision-level agreement measure was generalised beyond the conditions where it holds.

**Evidence** (`r3.py`).
- FROZEN_PROTOCOL §1 says "No pair-single swap changes the gate-free decision without also changing the order-1 decision", and the `xvidiag.py` docstring says the same.
- This is true in C, D and WB-high: 162 of 162.
- It is false in WB-low: **54 of the 162** inputs whose order-2 argmax changes under the swap leave the order-1 argmax unchanged. The same mechanism produces the 18.9% decision divergence in Finding 1.

**Correction.**
- Restrict the claim to b0 = b1 conditions.
- In WB-low, a decision-level agreement-following index (over those 54 input types) was available. It would give the only decision-level stage test in the package and could still be computed exploratorily from the saved snapshots.

### 9. [MODERATE] The frozen record is inconsistent on P2, and the FAST_UNLOCK interpretation does not follow from the exposure account.

**Evidence.**
- The hashed `scripts/score_XVI.py` l.24 states "Frozen expectation: CONT_BROAD". The hashed `FROZEN_PROTOCOL_XVI.md` l.118, EXTENDED §1 and MAIN l.40/148 state CONT_SELECTIVE, and report it as falsified.
- FAST_UNLOCK was frozen as "would point to a restart effect rather than accumulating exposure" (protocol l.124). It fired (median 875). Yet MAIN l.65 concludes "Nothing indicates a restart-specific effect".
- The frozen matched-exposure call was CONT_LOWER (fresh 4/4 against continuations 4/6 at E* = 5.90). MAIN l.65 instead reports a post-hoc comparison at exposure 3.0 and calls the two routes "comparable".
- The descriptor's threshold is not derived. Under a pure-exposure account calibrated on the fresh cohort (E_A = 1.0, 1.5, 2.9, 3.4), models that survived to E = 2.0 would acquire at about 2.9–3.4. That is about 1000–1500 post-restart steps, which straddles the 1000-step threshold. So FAST_UNLOCK cannot discriminate restart from exposure at n = 4.

**Correction.**
- Note the docstring/protocol conflict in the adjudication.
- Report the frozen CONT_LOWER call in MAIN.
- State that FAST_UNLOCK fired but is non-diagnostic, rather than reinterpreting it silently.

### 10. [MINOR] The proof of Proposition 2's upper bound is mis-stated, and its lower bound (with Proposition 3) can be upgraded from [num]/[argument] to [proof]. I also verified it at the decision level.

**Upper bound** (HIGHER_ORDER §2). "The log of g·L_pair + (1 − g)·L_single ... is a sum of tables" is false as written: a log of a sum is not a sum of tables. The correct step uses the fact that g ∈ {0, 1}:
- log(g·L_p + (1 − g)·L_s) = g·log L_p + (1 − g)·log L_s;
- g·ℓ(P1, y) is a table over (all a gate sources, P1, y);
- the noise-pair factor in L_s is y-independent and cancels.

**Lower bound and Proposition 3, a complete proof.**
- Let H = {t ∈ Z₃^a : Σt = 0} act by translating the gate sources. The joint law is H-invariant, the order-≤a class is closed under H, and the expected NLL is strictly convex in the posterior (p(x) > 0 everywhere). So the optimum is H-invariant.
- Averaging any table over H removes its dependence on a proper gate subset, since H projects onto all of Z₃^|A|. A table over all gate sources reduces to u(Σx mod 3, y).
- Label-permutation symmetry, which acts jointly on y and the evidence values, then makes u constant in y.
- Hence the order-a optimum is gate-blind: G = 0, F_G = 0, and KL ≥ the gate value > 0. For a ≥ 3 it equals the mixture.
- Proposition 3's sentence "additive in the logit, so it cannot select" skips the label-bias case that this symmetry step closes.

**Decision level** (`lp.py`). An LP over all order-k logit tables, requiring a margin of 1 for the Bayes argmax on every input with a unique Bayes argmax, gives:
- infeasible for k = a in `2:0.9:0.9:0.05`, `2:0.9:0.55:0.05`, `3:0.9:0.9:1/3` and `4:0.9:0.9:1/3:7` (k = 4);
- feasible for k = 5 at arity 4, with Bayes accuracy 0.9000.

So no order-≤a rule reproduces even the Bayes *decisions*. "Minimal decision order a + 1" therefore also holds in the literal argmax sense.

**Correction.** Replace the upper-bound sentence, add the symmetry proof, and change the labels to [proof].

### 11. [MINOR] The "structural" difficulty confound is a one-line proof, not a search result.

P(agree | g = 1) = b1² + (1 − b1)²/2 = 0.815 has the roots b1 = 0.9 and −0.233. An identical agreement law therefore pins b1, and ν, uniquely, which leaves b0 as the only free parameter. The Bayes NLL is strictly monotone in b0, so the difficulty must change. The 240-point grid is 20 one-dimensional families.

**Correction.** Label the claim [proof] (DECISION_VALUE §3 l.37).

### 12. [MINOR] The "schedule-matched continuation is provably inert" proof is correct but answers only the literal reading.

**Verified.**
- In optax 0.2.8, `warmup_cosine_decay_schedule(0, 1e-3, 200, 4000)` returns exactly 0.0 at counts 4000, 4001, 6000 and 8000 (1.8e-10 at 3999).
- `optax.adamw` scales both the Adam step and the weight decay by the learning rate.
- XVI's continuation starts at count 4000. So a continued XV schedule changes no parameter, whatever happens to the Adam moments.

**Limitation.** Mandate A2 asks for a no-restart continuation to separate *restart* from *continued effective optimisation*. That separation is still missing. The restart is also confounded with the moment reset, which is disclosed. MAIN l.62, "Continuing the old schedule cannot work", is a tautology: it is identical to stopping.

**Correction.** Phrase it as "the literal schedule continuation is identical to stopping [proof]; a non-degenerate no-restart control (for example, continuing at a fixed small rate with the moments kept) was not run".

### 13. [MINOR] Several verification checks are identities, labelled [num].

- V5 "gate value matched to arity 3 within 0.5%": by Proposition 3 both equal NLL(mixture) − NLL(Bayes), which does not depend on a. The ~1e-15 agreement is an identity.
- V6 "F_G(Bayes) = 1 at T = 0.5, 1, 2" and "F_G(order-a) = 0": these follow because tempering preserves the argmax and because the order-a optimum is gate-source independent (Finding 10).

These should be labelled [proof], with the numerics as sanity checks. This does not change any result.

### 14. [MINOR] The scaling statements are appropriately limited, but the "2–9×" figure mixes schedules.

- HIGHER_ORDER §5 / MAIN Q17 compare XV arity-2 exposures (0.37–0.52, cosine schedule) with XVI arity-3 exposures (1.0–3.4, constant rate). The arithmetic is right: 1.0/0.52 = 1.9 and 3.4/0.37 = 9.2.
- Cumulative learning rate is not schedule-invariant, though, and the label [expl] is the right one.
- The claim that no growth law is identifiable from two neural arities is correct. One small overstatement: the arity-3 against arity-4 construction is called "cleaner" than arity 2 against arity 3, yet the only difference is the 0.1% order-2 mismatch. The real gain is that the order-a optimum equals the mixture for both arities.

---

## Answer to the mandated question: has the value-versus-cue confound been resolved?

**Partly.**
- Resolved exactly: the competing cue's law. In fact the whole input distribution P(x) is identical between the two conditions (max |Δp(x)| = 2.7·10⁻²⁰), and seed-paired streams differ only at P0 on the 17.5% of questions where g = 0 and the b0 draw falls between 0.55 and 0.9 (`r7.py`).
- Resolved approximately: the competitor's absolute NLL benefit, 0.66% apart (1.75% with the exact gate-free mixture as reference).

**Residual confounds:**
1. The competitor's decision-level benefit: 0.037 against 0.000 accuracy, and 18.9% against 0% decision divergence.
2. The competitor's strength relative to the gate: 0.94 against 0.54.
3. The gate's accuracy gain ratio of 1.45, below the 1.5 tolerance.
4. Overall difficulty, unavoidable within the family (proof in Finding 11).
5. The locus of the value change, which is concentrated on disagreeing-pair inputs.
6. No gradient-level matching.
7. n = 4 pairs.

Items 1 and 2 are not disclosed. Together they mean the frozen P3 alternatives did not map onto the benefit and competition accounts (Finding 2). The INCONCLUSIVE result therefore cannot favour the competition account.

## Claims that survive

- **S1.** The two-reliability generator is correct.
  - b0 leaves the agreement law, and in fact the entire input law, exactly unchanged.
  - b1 = b0 reproduces XV bit for bit (checked independently for a ∈ {2, 3}, ν ∈ {0.05, 1/3, 0.45} and b ∈ {0.6, 0.9}).
  - Random-number consumption is independent of (b1, b0, ν), including the final RNG state.
  - `joint`, `bayes` and `gatefree` are correct (re-derived by hand; V2/V3 agree).
- **S2.** Proposition 1 (no proper-subset shortcut): the proof is valid. MI ≤ 3·10⁻¹⁵.
- **S3.** The minimal order is a + 1, for both the posterior and the decisions (arity 4 = 5). The KL table and the G = 0 table for orders ≤ 4 reproduce. The proof needs the repairs in Finding 10.
- **S4.** Proposition 3: the order-a optimum equals the gate-free mixture for a ≥ 3. So arity 3 and arity 4 (S = 7) have identical gate value 0.277018, intermediate gain 0.024726, accuracy gain 0.108611 and Bayes NLL.
- **S5.** The selected value pair: gate ratio 1.743 (1.737 against the mixture); identical agreement law (0.815 / 0.05; MI 0.3453; overlap 0.8825); 15 tolerance pairs; Bayes accuracy difference 0.075–0.25 and NLL difference 0.20–0.40 across them. All reproduce.
- **S6.** F_G is invariant to every argmax-preserving confidence change and is exactly 0 for gate-source-independent rules. A_conf is invariant to global temperature for *any* model, not only the reference: A_p and sB both scale by 1/T.
- **S7.** t_A is robust: P3 stays INCONCLUSIVE under every threshold variant I tried, and P5 = STAGE_ROBUST survives an added graded-gate floor.
- **S8.** The literal schedule-matched continuation is inert (proof verified against optax 0.2.8).
- **S9.** The scaling non-identifiability statements are correct.
