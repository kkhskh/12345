# R5: Baseline reviewer

**Package:** `step1_mechanism_identification_continuation` (commit d1aa97d)
**Scope:** whether each claimed advantage or contribution reduces to standard nonlinear computation or to known interpretability methods; whether the baselines are fair; whether the novelty claims are honest.
**Reviewer scripts** (all new, in `reviews/R5_scripts/`, no package file modified):
- `baseline_probe.py`: extra baselines with the package's data, optimizer, steps and validation set; seeds 0–2.
- `probe_vs_readout.py`: compares the agreement probe with the readout direction.
- `x4_on_twin.py`: runs the X4 criterion on the relation-free twin of C′.

Severity scale: **H** = a headline claim is wrong or does not follow; **M** = a claim is misattributed or confounded, but the conclusion can be repaired; **L** = wording or bookkeeping.

---

## Findings

### 1. [H] X4 does not separate "uses the relation" from "uses a per-source nonlinearity". The relation-free twin passes it.
- **Location.** MAIN positive #2; Q18 #2; Q15; `MATCHED_RELATION_INTERVENTIONS.md` §4 ("strongest on-manifold relation result").
- **Claim.** X4 is "an on-manifold identification criterion that separates 'uses the relation' from 'uses the aggregate'".
- **Evidence.** I applied the X4 procedure to the T-RF twin (`src/relation_free_equivalent.py`), where no unit reads two sources: ablate the block-1 MLP, then compare `m_blind` with the full margin (`x4_on_twin.py`).
  - The twin's `m_blind` is the constant −0.1973 on every input (200/200 random inputs).
  - X4's own pairs, e.g. (1, 1) vs (3, −0.238), therefore have equal `m_blind`, and the twin gives them opposite decisions (+0.211 vs −0.243).
  - The twin passes X4 more widely than C′ does. It also "matches" strong agreement (2, 2) against (3, −2.5), which T-R says C′ cannot match.
- **Diagnosis.** X4 is mediation by ablation on matched pairs: a natural-direct-effect or interchange design (Vig et al. 2020; Geiger et al.).
  - It shows that the ablated component is necessary on the matched pairs. It does not show what that component computes.
  - The "relation" label comes entirely from arrow 3, the closed-form characterisation of `q_R`.
  - Q15's rebuttal ("the twin fails step 1, because it has no relation feature to ablate") concedes this: the discriminating step is the declaration, not the intervention.
- **Correction.** Restate X4 as "decision-level necessity of a declared component on matched inputs, reachable within T-R". State that it discriminates relation use only *given* an independent proof that the ablated component computes `sign(ab)`. Delete "separates relation from aggregate" as a property of the criterion itself.

### 2. [H] T-CO is the intermediate value theorem, and its stated consequence is false.
- **Location.** `CONDITIONAL_CAUSAL_THEOREMS.md` T-CO; `CPRIME_…` §5 and status table; MAIN arrow 6 and negative #4; Q2, Q14, Q19 #2.
- **What the proposition says.** Part 1 is correct: if `m(a0, 0) ≠ 0`, T fails near `(a0, 0)`. But that is only the statement that a continuous margin must vanish on the target's decision boundary, which is true of every discontinuous label set (e.g. T2 on `a + b = 0`). It says nothing specific to relations.
- **What does not follow.** "Every relation-gated sign rule needs a dead band" (positive width) and "the historical C′1 had an impossible form".
  - `m = min(a, b)` is continuous and satisfies the historical form exactly: `sign a` when `ab > 0`, `m < 0` when `ab < 0`. I found 0 violations in 10⁶ uniform points.
  - A relation-free continuous margin, `clamp(ka) + clamp(kb) − 0.3`, has violation rate 2.9e−3, 3.1e−4 and 3e−5 at k = 40, 400 and 4000. Bands shrink as 1/k.
  - C′'s floor (≈ 0.12) and conflict band (≈ 0.6%) are C′-specific. The package's own E7 explains why: the relation feature is second order while the aggregate is first order. They are not forced by continuity.
  - The historical C′1 is stated on the open cells `ab ≠ 0` (CPRIME line 7), where the obstruction does not apply.
- **Correction.**
  - Rename T-CO as a remark: "a continuous margin must vanish on the positive axes".
  - Delete "every relation-gated sign rule needs a dead band" and "its form was impossible".
  - Attribute the C′ bands to E7/E8 (C′'s design), not to T-CO.
  - Remove T-CO from the "new negative results" (Q19 #2) and from MAIN negative #4.

### 3. [H] For T1 the "relation feature" is the label, so the T1 vs T3 interchange contrast is mostly a readout-alignment fact.
- **Location.** MAIN positive #3; Q8, Q9, Q18 #3; LEARNED §1.1 and §4.
- **Evidence** (`probe_vs_readout.py`, all 30 checkpoints, final residual site):
  - In T1 (where y = agreement), `|cos(agreement probe, LN-centred readout direction)|` is 0.90–0.99 in all 15 models. Interchanging along the **readout** direction transfers 1.000 of the differing decisions, the same as the probe.
  - In T3 the cosine is 0.008–0.053. The probe transfers 0.000, but the readout direction again transfers 1.000.
  - So the headline "agreement probe transfers 1.0 in T1 and 0.0 in T3" restates that in T1 the agreement variable *is* the output, and in T3 it is not.
  - §3's checklist row "direct candidate-margin information: the readout direction is used only for step 6" never ran this control.
  - Also, XNOR = `sign(ab)` is fit by a 12-parameter bilinear model at 0.995 (`B-bil-pgated`, T1). "Small Transformers learn a cross-source interaction for XOR" is Minsky–Papert / Rumelhart–Hinton–Williams (1986).
- **What survives.** The ablation evidence (the unit population halves Δagree, and accuracy drops to ≈ 0.75 in 7/10 models; checked against the LEARNED §3 table) and the mid-site results. But `resid_mid0` probe transfer is mixed across seeds: 0.0, 0.5 or 1.0 (`learned_posthoc_*.txt`).
- **Correction.**
  - Report the probe–readout cosine and a readout-direction control. Base the T1 "used" claim on mid-network sites and ablation, not on the final residual.
  - State plainly that for T1 relation use is entailed by the objective, so the positive claim is "standard XOR interaction, localised", not a distinctive finding.

### 4. [M] B-add is relation-free but over-specified: it gives p to both f and g. The fair baseline is weaker and still wins.
- **Location.** `src/learned_posthoc.py` (`badd_train`); REPARAM §2 reading 1; LEARNED §1.3.
- **The issue.** `f(a, p) + g(b, p)` lets each term compute source–prior agreement, e.g. `sign a == p`. That is a relation, if the prior token counts as a source. The docstring's "no term can read both sources" is true only for {A, B}. The T3 target is in fact `MAJ(sign a, sign b, p)`, a textbook threshold function, so p does not need to enter f or g.
- **Evidence** (seeds 0–2, same data, optimizer and steps):

| model | params | T3 | T5b |
|---|---|---|---|
| B-add (package) | 2,436 / 2,758 | 0.997 | 0.991 |
| B-add-strict `f(a) + g(b) + h·p` | 2,374 / 2,575 | 0.993 | 0.972 |
| linear on per-source support `[sign a, sign b, p]` / `[onehot argmax zA, onehot argmax zB, P]` | 8 / 30 | **1.000** | **1.000** |

  - The conclusion (T3 and T5b need no A–B relation) is strengthened: an 8- or 30-parameter relation-free linear rule beats every Transformer on T5b (0.984–0.999).
  - The B-add specification is not the clean test the text describes.
- **Correction.** Replace B-add with the strict form, plus the support-linear rule, as the "same information, no relation" baseline. Note that T-RF is exactly the support-linear rule.

### 5. [M] B-sum and B-lin "fail T3/T5b" because they are linear in raw evidence, not because they lack a relation.
- **Location.** REPARAM §3 row "net-evidence-only predictor"; §2 table.
- **Evidence.** Net *support*, `sign a + sign b + p`, solves T3 at 1.000 (Finding 4). B-sum's 0.859 measures the missing per-source squashing.
- **Correction.** Describe B-sum and B-lin as "raw-linear". Add a net-support baseline. Do not cite B-sum's failure as evidence that anything beyond aggregation is needed.

### 6. [M] B-bil vs B-add (T5b) is confounded, so it does not test T-B-IMP as written.
- **Location.** REPARAM §2 reading 3; `learned.py` B-bil.
- **Why the comparison is unfair.** B-bil has 57 parameters, no per-source nonlinearity, and enters p only linearly. B-add has about 2.6k parameters and nonlinear per-source MLPs. T5b's output needs `argmax zA`, which no model linear in zA can produce, whatever relation it has.
- **Evidence:**
  - B-bil with a p-gated bilinear term: 0.888.
  - B-add plus a bilinear term: 0.990, identical to B-add, so the bilinear relation neither helps nor hurts.
  - Linear plus the **oracle** argmax-agreement indicator and its products: 0.9998.
  - So the correct contrast for "the bilinear relation is the wrong relation" is B-bil (0.855/0.888) vs the oracle-gated linear model (0.9998). That contrast supports the claim; B-bil vs B-add does not isolate it.
- **Correction.** Cite the oracle contrast. Drop "as T-B-IMP predicts" from the B-add comparison. Note that T-B-IMP concerns the *agreement indicator*, while T5b's *decision* is additively representable (T-RF), so T-B-IMP is not the operative obstacle in T5b.

### 7. [M] The novelty labels contradict MAIN's own disclaimer.
- **Location.** `CONDITIONAL_CAUSAL_THEOREMS.md` heading "T-RF … (new, exploratory)"; Q19 heading "strongest **new** negative results"; Q14 bullet "T-CO shows which form a gated rule must take".
- **The reductions:**
  - **T-RF:** a weighted plurality / majority vote is a linear threshold function, and XNOR is not (Minsky & Papert 1969). The binary case is MAJ3; the K-ary case is weighted plurality voting.
  - **T-B-IMP:** fixing z_B makes the form affine in z_A; a half-space cannot equal a polyhedral cone whose complement is non-convex. This is elementary convexity, and it is limited to a single score, zero LN bias and linear encoding (TF2-AO reaches 0.991 on T5b).
  - **T-CO:** the intermediate value theorem; see Finding 2.
  - **Decodability without use** (Q19 #4): known from control tasks (Hewitt & Liang 2019), amnesic probing (Elazar et al. 2021) and Ravichander et al. (2021). In T2 it is the identity `ab = ((a+b)² − a² − b²)/2`.
  - **Unit criterion not sufficient for use** (Q19 #5): the standard correlational-vs-causal caveat.
- MAIN's "does not add beyond known theory" section is honest, but it is contradicted by these labels.
- **Correction.** Label T-RF, T-B-IMP and T-CO as "standard results instantiated for this setting". Retitle Q19 "negative results (known in kind)".

### 8. [M] "Gauge-robust" at the population level is close to tautological.
- **Location.** REPARAM §1; Q11; Q18 #3.
- **Why.**
  - The unit criterion (`I_agree`, `D_sum` thresholds) is permutation-equivariant by definition.
  - A split produces two units with proportional activations, so both are flagged and ablated together by construction.
  - Invariance of the set-level ablation effect under these gauges is therefore guaranteed, not found.
  - Residual rotation, the non-trivial gauge, is untested; the package says so.
- **Correction.** Say "invariant by construction under the applied gauges". Do not count it as evidence.

### 9. [L] "Matched-capacity" B-MLP is not capacity-matched, but the bias favours the Transformers.
- **Evidence.** Parameter counts:
  - B-MLP: 1,250 (T1) and 1,475 (T5b);
  - TF1: 2,338; TF2: 4,514; TF2-AO: 2,306;
  - B-bil: 57; B-add: about 2.4–2.6k.
- **Check.** A B-MLP with about 2.2–2.5k parameters reaches T4 0.994 and T5b 0.987, so the ties stand.
- **Correction.** Report parameter counts, and drop "matched-capacity" or match it.

### 10. [L] The claim that B-add's 0.92 on T4 is an optimization limit was asserted, not shown.
- **Location.** REPARAM §2 reading 1; LEARNED §2.
- **Evidence.** B-add trained for 12,000 steps (4× the steps) reaches 0.9857, 0.9752 and 0.9745 (mean 0.978).
- **A note on the LP check.** Grid LP feasibility alone is weak evidence, because a finite grid does not bound the needed jump. Analytically, `f(t) = t` for `t > 0` and `f(−t) = 10 − t`, with `g(t) = −f(−t)`, gives an exact bounded additive T4 rule with a jump at 0.
- **Correction.** Cite the longer-training result and the explicit construction.

### 11. [L] "Directionally predictable" is quoted for TF1 only, but Q18 #3 states it for all T1 models.
- **Evidence.** TF2 hits are 100/141 = 71% (dev) and 73/97 = 75% (held-out). TF2-AO has no MLP units to predict. The method is direct logit attribution through frozen `W_out·γ_f·ΔW_U`.
- **Correction.** Say "86–94% (TF1), 71–75% (TF2)", and name the method as standard DLA.

---

## Verified claims (checked, survive)

1. **Training matches.** All baselines and Transformers use the same STEPS (3000), BATCH (512), LR schedule, loss, generator and validation set (rng 1000, n = 20,000). The training streams differ only in rng seed and are i.i.d. from the same generator, with online sampling and no reuse (`learned.py` L137–217; `learned_posthoc.py` L29–66).
2. **Logs reproduce the reported tables.** The B-add accuracies in the REPARAM §2 and LEARNED §2 tables match `logs/learned_posthoc_{dev,heldout}.txt` (e.g. T1 0.473, 0.507, 0.466 → 0.482; T5b 0.991). The ablation accuracy count of "7 of 10 ≈ 0.75" matches the LEARNED §3 table.
3. **T-RF scores are correct.** I rechecked the four-case proof of `score_k = 1[k_A = k] + 1[k_B = k] + 1.5·1[p = k]`. The support-linear models in Finding 4 learn exactly this rule (1.000).
4. **XNOR is not source-additive.** The T-RF contrast proof is correct, and B-add on T1 is at chance (0.47–0.52) on all five seeds.
5. **T-B-IMP's proof is correct** for its stated class. The midpoint counterexample (0, −½, −½) checks out.
6. **The core negative conclusion is robust and, if anything, understated.** No positive result gives a predictive advantage over relation-free baselines. Relation-free rules with 8 or 30 parameters reach 1.000 on T3 and T5b, and a 12-parameter bilinear model reaches 0.995 on T1.
7. **MAIN's "What this does not add beyond known theory" section is accurate.** The package should make every other document consistent with it (Findings 2, 7).

## Bottom line on distinctiveness (Q18)

- **Claim 1** (C′1*, A1*, B-TM). These are correct closed-form decision analyses of specific hand-built circuits. They are useful, but standard calculus on softmax and LN rather than a new method.
- **Claim 2** (X4 + C2). This is standard ablation and interchange mediation. X4 does not discriminate relation use by itself (Finding 1). The reachability bound T-R is the one architecture-specific quantitative addition.
- **Claim 3** (learned T1). The objective entails relation use, so this reduces to XOR needing a multiplicative interaction (Finding 3). The gauge robustness is by construction (Finding 8).

What remains distinctive is the bookkeeping placement: when "consistency" has behavioural content (candidate-symmetric rules) and when it does not. That placement is itself a direct corollary of threshold-logic theory.
