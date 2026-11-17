# Review B: mechanisms (adversarial agent review, not human peer review)

**Scope.** T4 Stage I (I-A controls, I-B persistence), GATE mechanisms (§§1–3b), the cross-family comparison (GATE §5, CROSS_FAMILY_CONFIRMATION, research map), and the reading of X1.

**What I did.**
- I read the docs, freezes, harness, scorers and logs listed in the mandate.
- I ran three light checks of my own. Scripts and outputs are in `reviews/review_b_checks/`. They use reviewer-only streams 91_000_000 and 98_300_000+seed. No training was done and no existing file was modified.
  1. **`route_share.py`:** the route attention share under each Stage I condition, on the base checkpoints at steps 25, 50 and 100 (inside the intervention window).
  2. **`das_check2.py`:** GATE DAS with k = 1 shared across tokens (3 restarts), a separate 1-d DAS direction per evidence token (2 restarts), and shared k = 2, for s0–s2 at the package's sites.
  3. **`toggle_check.py`:** what the decoy toggle actually edits, and how d and g depend on each other.

**Severity labels.**
- **major:** changes a headline claim;
- **moderate:** changes a qualifier or a secondary claim;
- **minor:** presentation, or a missing control.

---

## A. T4 Stage I

### A1 (major). Both controls *raise* the route's attention share during the window, not only the sham

The document says the shams reinforce the route ("Explanations that remain open"). It presents the location-matched noise as a disturbance that does not block. In fact the noise also *raises* the route's attention share inside the steps 0–100 window. The mechanism is the log-normal effect: E[e^{σZ}] = e^{σ²/2}, so the noised edges gain weight on average.

`review_b_checks/route_share.json` (own-question readout mass in the intervened layer):

| base, step | none | real | off-route sham | location-matched noise (3 draws) |
|---|---|---|---|---|
| E L0, 25 | 0.134 | 0 | **0.326** | **0.217–0.219** |
| E L0, 100 | 0.237 | 0 | **0.453** | **0.305–0.307** |
| M L2, 25 | 0.142 | 0 | **0.176** | **0.231–0.233** |
| M L2, 100 | 0.170 | 0 | **0.205** | **0.266–0.268** |

What the noise does instead:
- **E:** total source mass rises from 0.60 to 0.82, while the own share *within* sources falls only slightly (0.39 → 0.36).
- **M:** the noise raises own mass by about 60%, more than the sham does.
- The matched "attention TV" is blind to direction. Most of the noise's TV is mass moving *towards* the route edges, plus reshuffling within the sources.

**Consequence.** Every comparator in R-S and R-N moves the route share in the same direction, upwards: away from the real block and towards the base organisation. So a no-flip result is expected for both. These controls exclude:
- "any masking of these rows";
- "any logit perturbation of these edges".

They do **not** exclude "any reduction of route attention share or access early on". The sentence "comparable interventions that do not block it do not [change the organisation]" overstates this.

**Corrections.**
- State that both controls are pro-route in the window.
- Relabel R-N as "disruption that increases route share".
- Note that an off-route mask can never be share-neutral, because softmax renormalisation always inflates the edges that remain.
- The missing control is an **attention diversion that preserves access**, for example:
  - a finite positive bias on off-route edges, or a finite negative bias (−b, not −1e9) on route edges;
  - calibrated so that it removes the same share as the real block, while every route edge remains reachable.
- XIII-E's partial block (`ro0p`) is a share-matched block, but it is not an access-preserving one.

### A2 (moderate). The sham-validity check is made at the wrong time, and is one-sided by construction

`i_sham_validity.py` evaluates the sham on the **step-1211 trained** baselines. The sham acts during steps 0–100. The rule also only forbids *decreases* of more than 0.01 in own mass. A renormalising mask can only increase the remaining mass, so it passes automatically.

In the window the sham raises own L0 mass by 2.4× on E (0.134 → 0.326), and its max-head value from 0.19 to 0.46. "Valid on all 6" therefore means "not disruptive to trained models". It does not mean "neutral during training".

**Correction.** Report the in-window route-share change for each control. Restate validity as "non-disruptive at step 1211".

### A3 (moderate; in the authors' favour, but not stated). On E the real-versus-sham contrast is sharper than claimed, and the minimal block is untested

- **What each condition blocks.** ro0 blocks the own-question *and* other-question sources. The sham `oq0` blocks exactly the other-question sources.
- **What the contrast isolates.** Real minus sham therefore isolates the **own-question edges, given that the others are removed**.
- **Mass removed is comparable.** The sham removes 0.44–0.46 of the mass against 0.60–0.68 for the real block (`i_calib.json`). Removing the *larger* other-question share does not flip E; removing the own-question share as well does. This is a near mass-matched contrast and worth saying.
- **The minimal real block was never run.** The own-only block (`own0`) is the "route" block proper. ro0 also removes all source access in layer 0. The claim "blocking the pathway itself" should therefore name ro0, not "the route".

### A4 (minor). The R-N result on E is partly a weaker-disruption result

- **Loss disruption.** The E noise disrupts the loss about 1/3 as much as the block (0.18 against 0.55); the document reports this correctly.
- **E-21 is not null.** Its noise run shifts δ = 0.17 (mix 0.34 against 0.16 for the baseline).
- **The M argument is sound.** The real M block barely changes the loss.
- **Correction.** Report R-N as decisive only on M.

### A5 (moderate). On M, the outcome measure is the attention statistic of the intervened edges

- **The coupling.** `mix_mass` is defined as the maximum, over layer-2 heads, of the readout attention mass on the own-question sources (`metrics.py`). These are exactly the edges that l2q zeroes and that the M sham and noise inflate. So on M, δ is partly a direct readout of what training did to those logits.
- **The flips themselves are corroborated.** lik_RO1 is a functional patch measure, and it agrees: dlik is 0.65–0.68 for the M real block, and lik is 0.72–0.73, against about 0.05 for the base. So the flips are real.
- **What it does weaken.** For the controls, "no change in mix_mass" partly measures the same quantity the control inflated.
- **Correction.** Use lik_RO1 (or route_Q/route_R) as the primary δ on M, with mix_mass secondary.

### A6 (major). "Late 100-step blocks on E: 2 of 3 runs have no measurable effect at any time" is false

From `unit_metrics_I.jsonl`, the late ro0 runs against the step-matched baseline (`stop1146_br546_XIVbref`):

| E pair | lik_RO1 at release (r = 0), run / base | KL at r = 0, run / base | lik at r = 50 | lik at r = 200 |
|---|---|---|---|---|
| D = 21 | **0.09 / 0.69** | **0.46 / 0.07** | 0.55 / 0.67 | 0.67 / 0.72 |
| D = 801 | **0.15 / 0.71** | **0.56 / 0.09** | 0.57 / 0.68 | 0.64 / 0.66 |

`score_I.json` itself records late dlik(r = 0) of 0.60 and 0.56.

**What actually happens.** The late E block abolishes the early readout immediately, and it recovers within about 200 updates. The frozen δ (on mix_mass) misses this, because blocking ro0 removes the early route without creating a mixture head. These pairs are excluded from R-P only because δ is measured on an axis the block does not move.

**Consequences.**
- With dlik, PI_late ≈ 0.02–0.03 on these two pairs. That would *strengthen* "late changes are transient".
- But it falsifies the sentence "no measurable effect at any time". It also falsifies the summary "mostly ineffective on E".

**Correction.** Rewrite I-B item 3:

> Late E blocks cause a large immediate loss of the early readout (Δlik 0.56–0.60; KL ≈ 6×). The early readout recovers within 200 updates in 2 of 3 pairs. In E-802 the late block converts to HYBRID.

Report both axes in the R-P table.

### A7 (major). PI is not the same quantity in the two arms; "greater early persistence" is a loaded name

**What PI measures in each arm.**
- **Early arm.** The baseline at release (step 100) has no organisation yet (mix 0.17–0.19, lik 0.11–0.14), and the early δ rises monotonically to r = 500. So max δ(r ≤ 500) is just δ(500), and PI_early = δ(1900)/δ(500) ≈ 1. This holds for *any* intervention whose effect develops after release, regardless of persistence.
- **Late arm.** PI_late measures the decay of an immediate perturbation of an already-formed attractor.

The two arms therefore measure attractor *selection* (early) and *elasticity* (late), not two degrees of one property. The authors note half of this ("not an immediate change that persists"), but still state the verdict as "greater early persistence".

**Correction.** Primary statement: "a 100-step block during formation selects the final organisation (6/6). The same block after formation is reversed (5/6 on δ or Δlik), with 1/6 converted."

A cleaner frozen-style statistic would be the change in final class: early 6/6 against late 1/6. Note that the units are not independent.

### A8 (moderate). In practice R-P rests on M alone

- **Composition of the four valid pairs.** Three are M pairs, which share one L/W initialisation; the fourth is E-802, where PI_early − PI_late = +0.04, a tie.
- **Effect on "4/4".** The count includes that tie.
- **Effect on the CI.** A bootstrap of n = 4 values (≈1.07, 1.08, 1.05, 0.04) is not a meaningful interval.
- **Correction.** State "R-P is confirmed on M. On E it is uninformative under the frozen δ, and on Δlik it is consistent (see A6)."

### A9 (moderate). Learning-rate and optimiser confounds are acknowledged, but asymmetric in their bite

**Learning rate.**
- **Block mass.** The late block has 4× the Σlr of the early block (0.097 against 0.025). Recovery after release is 1.98 for early against 1.46 for late.
- **M.** The reversal finishes within about 200 updates (Σlr ≈ 0.2). Post-release budget cannot explain it, so the doc's argument holds there.
- **E-802.** The late conversion is gradual (δ 0.06 → 0.75 over 1900 updates). Its baseline is nearer the mixture side at the branch: mix 0.25–0.26 at steps 600–646, against 0.16–0.18 for E-21 and E-801. A **marginal-basin** reading of E-802 fits better than "a late block can leave a lasting change".

**Optimiser state.** Adam moments are another uncontrolled difference. At step 546 they are adapted to the formed solution.

### A10 (minor). Recovery checkpoints: verified correct

I checked `score_I.py` against `I_design.json` and the unit list.
- Early runs are compared with the baseline at 100 + r, and late runs with the bitwise branch replay at 646 + r. The tail branches for r = 1111 and r = 1900 come from the correct sources.
- Competence at step 4000 is applied.

Residual limitations:
- Sham/noise persistence to step 4000 was checked only for D = 801.
- All interventions are synthetic training-time masks. No natural manipulation was tested, for example a change to the data distribution that removes the route's utility. "Changes the learned organisation" is therefore established for synthetic access removal only.

---

## B. GATE mechanisms

### B1 (major). "Not carried by one linear direction" / "≥ 2 dims" conflates shared-subspace dimension with per-token dimension

The package's DAS uses **one subspace shared by the 4 evidence tokens**. I added a separate trained 1-d direction for each token: 4 directions, each 1-d. Results are on the held-out half, with Bayes-decisive filtering identical to the package (`review_b_checks/das_check_s*.json`):

| | shared k = 1 (3 restarts) | per-token 1-d (2 restarts) | shared k = 2 |
|---|---|---|---|
| s0, resid_pre1 | 0.18, 0.18, 0.21 | 0.64, 0.64 | 0.74 |
| s1, resid_pre1 | 0.58, 0.58, 0.65 | **1.00, 0.99** | 1.00 |
| s2, resid_pre2 | 0.61, 0.61, 0.61 | 0.59, 0.60 | 1.00 |

**What the restarts and the per-token runs show.**
- **Shared k = 1 is stable across restarts.** The package's k = 1 numbers are not an optimisation artefact.
- **In s1, the gate is carried by a single direction per evidence token.** "≥ 2 dims" in s1 is an artefact of forcing the 4 tokens to share. This also tests the doc's own [spec] "one direction per role" reading, and supports it for s1.
- **The class-mean 1-d failure in s1 (0.21) is a direction failure, not a dimension failure.** A 1-d causal direction exists in s1, so the failure shows that the encoding direction differs from the causal one.
- **"Not one direction" survives only for s0 and s2,** and even there per-token 1-d reaches 0.60–0.64.

**Not run by me.** Null-donor and random controls for the per-token directions.

**Correction.** Replace with:

> The gate is mediated by a low-dimensional code at E. Its dimensionality per token is 1 in s1, and above 1 in s0 and s2 at the tested sites. A shared subspace needs 2 (s1, s2) or more than 2 (s0).

Also amend the research map ("needing a ≥ 2-dimensional subspace").

### B2 (major). The T4-versus-GATE dimensionality row ("different") does not follow

- **T4.** Ledger C4: "1–2 stable directions suffice, the number is model-specific".
- **GATE.** s1 and s2 need k = 2 shared, and s1 needs 1 per token (B1).
- **Overlap.** The two ranges overlap. The procedures also differ: DAS site, sharing across tokens, and the transfer metric.
- **Correction.** Change the row to "not distinguishable with current measurements".

### B3 (moderate). Probe accuracies are reported without the chance level; several "presence" values are at or below chance

- **Chance level.** P(g) = 0.50 on held-out x₀ = 2 (`toggle_check.py`).
- **Why some values fall below chance.** On x₀ = 2, an open gate means x₁ = 1. In training, x₁ = 1 was never an open value (open x₁ was 0 at x₀ = 0 and 2 at x₀ = 1). So a probe that learned an x₁ lookup is *inverted* on the test set. This explains the values of 0.01–0.46 (for example s3/s4 resid_pre1@G at 0.04/0.01).
- **What follows for the cited presence values.** They are not "low presence". They are evidence of a lookup code. Also, s2's "decodable at G after layer 1 (0.60–0.92)" has its lower end, resid_pre2@G = 0.60, near chance.
- **Decoy probe.** It sits at its own chance level: P(d) = 0.25 at x₀ = 2, so chance is 0.75, against the observed 0.77.
- **Correction.** Report balanced accuracy or AUC with CIs, and state the lookup-inversion baseline.

### B4 (major). "Decodable ≠ used in every model" fails for s2; the s2 instance uses an architecturally null site

**The architectural fact.** `fwd3.forward` reads the answer from the readout tokens only (`rd = zf[:, 6Q:7Q]`). So patching G or E at **resid_post2**, after the last layer, *cannot* transfer anything. Every post2@G and post2@E entry is 0.00, in all five models.

**The s2 pairing does not hold up.**
- resid_post2@G (0.92 against 0) is an architectural zero.
- resid_pre2@G (0.60 against 0) is at chance (B3).

**The valid instances:**
- s0: resid_pre2@E, 0.97 against 0.03;
- s1: resid_pre2@R, 1.00 against 0.08.

**"Used" means single-site sufficiency.** Patching measures whether replacing one site alone is enough to carry the flip, not whether the site is read. s0's E at layer 2 may well be read, and simply be overridden by an R that already carries the decision.

**Correction.**
- Say "decodable ≠ sufficient alone, in s0 and s1".
- Drop s2 from the list.
- Mark post2 G/E as structural nulls.
- Add necessity tests (resample-ablation of the site).

### B5 (moderate). The decoy toggle edits less than its label says, and the decoy "ideal 0" is not guaranteed

**Code check.** `toggle()` is correct: the `alt` formula keeps the gate closed (verified). The variable `other` is unused.

**Coverage.** The toggle edits only closed gates, which is 50% of episodes. It changes d = [x₀ = x₁] only when x₀ ≠ 0, so d changes in **33%** of episodes (`toggle_check.py`).

**Dilution.** `decoy_toggle_mean_abs` averages over all n episodes. Divided by the coverage (n_changed/n ≈ 1/3), the response is ≈ 0.09 for s0, about 4% of the gate flip. That is still small, but the table label "g kept, [x₀ = x₁] changed" overstates the edit.

**Dependence of d and g.** They are marginally independent (P(g | d) = 0.49, against P(g | ¬d) = 0.50), but strongly dependent given x₀:
- at x₀ = 0, d = g exactly;
- at x₀ = 1 and x₀ = 2, corr = −0.58.

Hence the class-mean decoy direction need not be orthogonal to the g code, and the "ideal 0" for the decoy-direction interchange is not a theorem. Its 0.09 transfer, identical in s0–s2, may simply be overlap with the g code.

**Correction.** Report cos(U_g, U_d) and the edit coverage.

### B6 (moderate). The DAS controls are weaker than presented

- **Random subspaces are not variance-matched.** They carry 0.06–0.35 of the DAS subspace's variance, so their ≤ 0.11 is a weak contrast.
- **"Same subspace at G or R: 0.00" is nearly vacuous.** The subspace was trained on E-token geometry. A real site-specificity control trains DAS *at* G or R. For s0, R at resid_pre2 would almost surely succeed (full patch 0.97). That is fine, but it shows the control does not test site specificity.
- **The null-donor claim was not computed.** "Every change is at Bayes ties" was never computed for the interchange (`shift_ties.json` is behavioural only). For s0 the DAS null-donor rate is 0.065–0.088. That exceeds the s0 behavioural shift decision change of 0.051, so "equals the tie rate" is unverified for s0.

### B7 (minor). Variance-matched random 1-d directions match the wrong quantity

- **What is matched.** The marginal variance of the base activations.
- **Code check.** `vm_dir` is mathematically correct: a is orthogonal to the extreme eigenvectors, so it is C-orthogonal to b.
- **What should be matched.** The intervention magnitude is E[((h_s − h)·u)²], the squared donor–base difference along u. The g direction maximises it by construction. Matching it would make the random control fair.

### B8 (minor). The ideal references are asserted, not computed

No synthetic "ideal mechanism" model was built and measured. "The ideal references discriminate here" should be "against asserted ideal values".

B1 also shows that the 1-d class-mean ideal of 1 presumes the encoding direction equals the causal direction. That presumption is false in s1.

### B9 (minor). Patching and behaviour code: checked, no bugs found

I checked the following and found them correct:
- `edit_q` (shapes and per-episode question);
- `gate_flip` (on-support in both directions);
- `gate_keep_shift`;
- the interchange formula h + ((h_s − h)·u)u;
- the train/test split in `g_mech` (classdir on `half`, scoring on `~half & dec`);
- in `g_das`: training on `tr & dec`, testing on `te & dec`, QR orthonormalisation and the mask construction.

The §1 behavioural claims (gate-following, decision-level invariance to the shift except at Bayes ties) are well supported.

---

## C. Cross-family comparison (GATE §5, CROSS_FAMILY_CONFIRMATION §4, research map)

### C1 (major). Every "same" row is task mathematics or a methodological truism; none is a mechanism similarity

- **Task mathematics.** "Trust invisible to single sources" and "source-wise ceiling far below Bayes" are properties *designed into* both tasks. They hold for any learner.
- **A truism.** "Decodable ≠ used" is true of probes in any network. In GATE it also holds only for s0 and s1 (B4).
- **An architecture-forced dichotomy.** The "partly same" storage row: when the answer is read at R, trust must be combined at R, either from source-side storage or by early pooling. That has little cross-family content.
- **No causal cross-family comparison exists.** There is no GATE training-time block (§6), no T4 counterpart of X1, and no matched intervention on both families.

**Correction.** Say explicitly that the cross-family similarities established are mathematical and methodological, and that no mechanism-level similarity was tested causally.

### C2 (major). Comparisons that fail or are mis-labelled

- **Dimensionality:** not distinguishable (B2).
- **Combination operator.** It is not identifiable in GATE, so "different" is a category error. Use "not comparable".
- **Learnability from scratch:** "different for the Transformer".
  - The models differ in architecture and recipe: T4's M Transformer against GATE's 3-layer, 4-head, d = 64 from-scratch recipe.
  - No matched-architecture comparison was made, so a task-caused difference is not established.
- **"s0 is early-readout-like".** This is an analogy of *where the combination happens*. T4's early organisation reads candidate evidence at the readout in layer 0, whereas s0 moves *gated* evidence there in layer 1. The analogy is loose, and should not count as a "same".

---

## D. X1 interpretation

### D1 (major for the wording; the verdicts stand). s3 does not "stay at the order-1 ceiling"; it is a partial order-2 acquisition overtaken by the gate

**The frozen verdicts are scored correctly.**
- X1a and X1a′: CONFIRMED.
- X1b and X1b′: FALSIFIED.

**The prose is not supported for s3.** It says the models "first reach the exact order-1 ceiling and stay there" and "leap from order 1 to order 3". From `score_X.json` and `score_X_amend.json`:
- **NLL.** s3 goes **below the exact order-1 optimum** (0.4622) while G ≤ 0.011: 0.461 at step 300, 0.453 at 325–350, 0.449 at 425. KL_bayes at 425 is 0.268, against the order-1 optimum's 0.280.
- **Share of the order-2 gain.** About 40% of the order-2 NLL gain (0.013 of 0.031 nats) is realised before the gate. A_p reaches 0.27, about 28% of the order-2 reference of 0.96.
- **Which rule the model is closest to.** KL to the gate-free (order-2) rule is *below* KL to the source-wise rule from step 175 in s3 and step 125 in s4. Note that `rule("sourcewise")` is a heuristic order-1 rule, not the exact order-1 optimum.
- **s4 is closer to a clean leap.** A_p ≤ 0.09, and NLL is 0.464–0.471 at steps 200–250.

**Supported reading:**

> Both models first reach order-1 *accuracy*. Pairwise-agreement use then accumulates slowly: substantially in s3, barely in s4. Both acquire the gate abruptly before agreement use reaches the order-2 level.

**Not supported:** "stays at the order-1 ceiling" and "skips order 2", for s3.

**Correction.** Rewrite §3 "Supported" and "Falsified" accordingly, and report NLL relative to the exact order-1 and order-2 optima.

### D2 (moderate). The order-1 phase criterion cannot separate order 1 from partial order 2

- **Accuracy does not discriminate.** It is 0.801 for both order-1 and order-2 optima.
- **The NLL clause does not either.** NLL ≤ 0.50 sits above both optima.
- **A_p ≤ 0.3 admits about a third of order-2 use.** So the 14-snapshot "order-1 phase" in s3 includes snapshots that beat every order-1 rule.
- **Future freezes.** Use KL to the *exact* order-1 and order-2 optimum rules, or NLL relative to 0.462 and 0.431.

### D3 (minor). Alternative readings of the "leap"

- **A plateau-and-escape dynamic** for a component with no lower-order projection (Proposition 3), as in leap or staircase accounts. It predicts a long plateau and then a rapid rise, which is what is seen. It is compatible with the doc's [spec] magnitude explanation, and not distinguished from it.
- **The negative A_p after the leap (down to −0.18 in s4)** may be a collinearity artefact of the joint regression, because the gate-free and Bayes swap responses are correlated. It should not be read as "anti-agreement".
- **Sample size.** n = 2 frozen models plus one development seed.

---

## Claims that survive this review

1. **T4, R-S and R-N, as frozen verdicts.** A 100-step early block (ro0 on E, l2q on M) changes the final organisation in 6/6 pairs, while an off-route mask and a TV-matched logit noise do not (0/6). This excludes "any masking of these rows" and "any logit perturbation of these edges" as causes.
   - It does **not** exclude "any early reduction of route share or access" (A1).
   - On E, removing other-question access alone (about 0.45 of the mass) does not flip, and removing all source access does (A3).
2. **T4, I-B, restated as attractor selection against elasticity.**
   - A block during formation (steps 0–100) determines the final organisation (6/6, persistent to step 4000).
   - The same 100-step block after formation (steps 546–646) produces large immediate disruption on M (δ) and on E (Δlik). This reverses within about 200 updates in 5 of 6 pairs; in the sixth, E-802, the run converts to hybrid, from a baseline nearer the boundary (A6–A9).
   - The R-P "persistence index" verdict holds formally, but rests on M.
3. **GATE behaviour (§1).** These hold in all 5 models:
   - Bayes-level accuracy;
   - gate-following (G ≈ 0.99);
   - rejection of the gate-free and order-1 rules;
   - decision-level invariance to relation-preserving edits, except at exact Bayes ties.
4. **GATE localisation.**
   - Patching at E transfers the gate flip, and E is the mediating token set in every model.
   - The mechanism layouts differ across models (s0, s1, s2/s3/s4).
   - "Decodable ≠ sufficient alone" holds for s0 and s1 (not s2; B4).
5. **GATE low-dimensional mediation.**
   - A 2-d subspace shared across tokens is sufficient in s1 and s2. s0 needs 4–8.
   - Random, null-donor and decoy controls are low, with the caveats in B5–B7.
   - "Not one direction" survives for s0 and s2 only. In s1 one direction per token suffices (B1).
6. **X1.**
   - X1a and X1a′ CONFIRMED, X1b and X1b′ FALSIFIED, as frozen.
   - Order-1 accuracy comes first.
   - The gate is acquired abruptly, before agreement use reaches the order-2 level.
   - "Stays at the order-1 ceiling" and "leaps from order 1" hold for s4, not for s3 (D1).
7. **Cross-family.** Only the mathematical properties of the tasks (relational invisibility to single sources, and source-wise ceilings far below Bayes), plus the methodological point that probes do not establish use, are shared.
   - No mechanism-level cross-family similarity is established causally.
   - The dimensionality "difference" is withdrawn (B2, C1, C2).
