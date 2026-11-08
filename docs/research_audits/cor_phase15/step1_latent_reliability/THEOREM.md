# Theorem obligation (mandate §11): three levels, kept separate

This document makes three different kinds of claim:

- **Level 1 (Bayes)** is about the generative model and its exact posterior.
- **Level 2 (engineered)** is about a hand-specified attention computation.
- **Level 3 (learned)** is about the trained Transformers. It is empirical: it holds on the measured models and data, with the measured tolerances, not as a proof about Transformers in general.

None of the three is evidence for the others. Level 1 says nothing about what a network learns. Level 2 shows that a computation is *possible*. Level 3 shows that the trained models are *close to* the Level 2 computation in the sense measured here.

## Level 1: Bayes theorems (proved; numerically checked)

**Theorem S (sufficiency of agreement counts)** — `MATH_VALIDATION.md` §5.
- **Assumptions.** The roles are reliable, noise or colluding; errors are uniform over the wrong candidates; Y priors are uniform.
- **Statement.** P(R | X) depends on X only through the per-group within-pair agreement counts n_g. The leave-one-out posterior π_{−q} depends on X only through the leave-one-out counts n^{−q}.
- **Check.** The maximum deviation within a count cell is 6.4e−15 (T1) and 8.8e−15 (T4).

**Theorem M (mixture / reuse identity).**
- **Statement.** P(Y_q | X) = Σ_R P(R | X) · P(Y_q | X_q, R).
- **Consequence.** One episode-level vector suffices for every question (2 dimensions in T1, 5 in T4).
- **Check.** Error 1.9e−15.

**Corollary T1 (log-linear group weights).**
- **Statement.** P(R = g | X) = softmax_g(λ · n_g) with λ = log(LR_agree / LR_disagree) = 1.357 (α = 0.8, K = 3).
- **Consequence.** In T1 the Bayes answer is a mixture of the groups' per-question evidence, with weights log-linear in the group's agreement count.

**Corollary T4 (non-monotone).**
- **Statement.** P(R = (g,h) | X) contains the factor 1[n_h = Q]. A group that agrees on every question is as likely to be the colluder as the reliable group. P(g reliable) is therefore non-monotone in n_g; for example 0.98 at n_g = 5 and 0.50 at n_g = 6, with the other counts (6, 2).

**Scope (MC3).** Fixed known groups; errors uniform over the wrong candidates; uniform priors on Y and R; noise uniform. **Correction (MC2).** The T4 non-monotonicity is *conditional on the other counts*: e.g. with counts (n_0, 6, 2), P(G0 reliable) = 0.06 → 0.98 for n_0 = 0 → 5 and 0.50 at n_0 = 6. It is false that a perfectly agreeing group is always as likely to be the colluder (with counts (6,2,2), G0 must be the colluder).

These are statements about the *task*. They show that a consistency statistic, within-group agreement, is the Bayes-sufficient summary. They do not show that any network computes it.

## Level 2: an engineered tag-and-route computation (constructive; stated, with the derivation)

**Model class.** A bidirectional attention network over tokens (q, s) and readout tokens r_q.

**Construction.**

1. **Tag (stage 1).** Write into every source token (q, s) a scalar tag τ_{q,s} = a · n_{g(s)} + b along a fixed direction u. Here n_g is the group's agreement count, either leave-one-out or episode-level.
   - This is computable by one attention layer. Each source token attends uniformly to the partner source of its own group in every question.
   - An MLP then reads the per-question equality indicators. Equality of two one-hot vectors is their inner product, a bilinear form that one GELU MLP can approximate.
   - The result is summed over questions.
2. **Route (stage 2).** In the last layer, a head at r_q has query–key score β · τ_{q,s} + c_s on the source tokens of question q.
   - Its value/output map writes c_v · onehot(x_{q,s}) into the answer logits.
   - Its attention weight on group g is then w_g ∝ exp(β a n_g) · (within-group terms).
3. **Prediction (derivation).**
   - The answer logit is ℓ_y = c_v Σ_s w_s 1[x_{q,s} = y] + (terms independent of X_{−q}).
   - If βa = λ **and the tag uses episode-level counts**, the routing weights equal the Bayes group posterior (Corollary T1). With leave-one-out counts this is only approximate (reviewer: 93.6% vs 97.2% argmax agreement with Bayes; MC4).
   - ℓ is then a P(R | X)-weighted *logit-linear vote*. It is **not** the Bayes probability mixture (Theorem M); it agrees with Bayes at the argmax level except near ties (5.4% exact vote ties), and it does not by itself explain calibration (MC5).
   - For a reliable pair that agrees, the vote and the evidence have the same argmax. When the pair disagrees, both are split. So the routed vote has the same argmax as Bayes except near ties.

**Intervention prediction.**
- An intervention that sets the tags of question q's source tokens to their values under an episode E′ (which shares X_q) changes the routing weights to w_g(E′).
- It therefore changes the candidate margin ℓ_y − ℓ_{y′} by c_v Σ_s [w_s(E′) − w_s(E)] (1[x_s = y] − 1[x_s = y′]).
- It changes nothing if X_{−q} is changed in a way that preserves the counts (reviewer test: model argmax changes 3.2–4.0% vs 26–27% for cond; MC7). The package's `target` pairs change X_q only and are a value-content control, not this test.

**Downstream assumptions needed.**
- (i) No other path from X_{−q} to ℓ at r_q, or bypass paths that are negligible.
- (ii) The last-layer value path carries the reported value, not the tag.
- (iii) The key depends on the tag linearly.
- (iv) Pre-LN: the tag passes LayerNorm before the key projection, so "linear" holds only approximately; other heads' and the MLP's contributions to the readout are ignored (MC8).

**Bypass handling.** Assumption (i) is falsifiable. Freeze the route heads' attention pattern while patching the tag: the effect should vanish.

## Level 3: learned models (empirical; with measured tolerances)

Numbers in the table are for **T1/M seed 0** (confirmation split; post hoc stream for the equation checks). **Per seed (MC9, B1):** seeds 0 and 1 match the table closely (seed 1: tag cond IIA 0.83, target change 0.104 — fails K4′ by 0.004); **seed 2 does not localise the variable on the keys**: key-only IIA 0.33, readout-query-only 0.47, joint 0.78 (KL 0.97). The cross-seed claim is the routing (Type B, 3/3), not the tag location. The tag is a group-indexed contrast of all groups' counts, with a seed-dependent encoding (B2). Measured βa (log-attention per count, unit-free; MC6): roughly 0.35–1.29 across seeds and heads, all below λ = 1.357 — the learned routing is less sharp than Bayes; accuracy is insensitive to this.

| Level 2 element | measured in the learned model (T1/M seed 0) |
|---|---|
| tag direction u at the source tokens, at the input to the last layer | output-free contrast direction U_D2 (1-D) at resid_pre2 |
| τ affine in the agreement count | tag mean by own-group leave-one-out count 0…5: −9.5, −7.6, −5.6, −3.3, −1.1, +0.6 (roughly linear). Corr with the Bayes reliability logit 0.89. Linear in all raw reports (non-relational): R² 0.04 |
| log attention linear in τ (route heads) | heads 0 and 1: slope β = 0.43 and 0.48; R² 0.84 and 0.69 |
| ℓ ≈ c_v · routed vote | R² 0.945 (c_v = 4.1). Routed vote vs Bayes log posterior: R² 0.94. Model vs Bayes: R² 0.996 |
| intervention prediction (tag patch from cond E′) | IIA 0.83, kl_recovered 0.99. Matched-histogram (perm) IIA 0.84 |
| no effect under count-preserving `target` changes | change rate 0.08 (random-direction baseline 0.05) |
| bypass assumption (i) | route heads' pattern frozen during the tag patch: IIA 0.83 → 0.07. Only those two heads matter (freezing all heads gives the same 0.07) |
| identification independent of outputs | U_D2 comes from input contrast pairs; its content is the Bayes tag (R² 0.79), not own values (0.002) or histograms (0.0001) |
| reuse across questions | U_D2 fitted on questions 0–2 transfers to questions 3–5 (IIA 0.83) |

**What the learned result is.** In the T1 M models the answer is produced by an *operation-level* mechanism:

- an explicit 1-dimensional, output-free-identifiable, count-linear reliability tag on source tokens;
- consumed by reliability-gated attention routing.

The frozen "explicit mediator at the readout token" hypothesis (Type A) is **false** for these models. Reliability is linearly decodable there (R² 0.65–0.81) but causally inert (IIA ≈ 0.01). This is a decodable-but-unused representation.

**What is not shown.**

1. That the model's βa equals λ. It does not (estimates 0.35–1.29 < 1.357).
2. That the mechanism is unique. The protocol cannot exclude a functionally equivalent network without it; see `reviews/R_arch_counterexample.md` (B1: seed 2 is a functionally equivalent network without the key-side localisation).
3. Anything spontaneous for T4: the M models did not learn T4 from scratch. **Post hoc**, T4 models fine-tuned from T1 implement the non-monotone version (a 2-D tag whose second coordinate rises with the count 0→4 and drops at 5; route-head log-attention R² 0.75 from the tag vs 0.01 from the count); see `MAIN_ANSWER.md`.

## Comparison with known theory

- **Truth discovery / crowdsourcing.** Estimating source reliability from inter-source agreement and then taking a reliability-weighted vote is the structure of Dawid–Skene (1979) EM and of "truth discovery" methods. The learned tag is analogous to an E-step reliability estimate, and the routing to the weighted vote. Here it is computed in one forward pass, and the tag is linear in the agreement count, which in T1 is exactly the Bayes log-odds form.
- **Attention as soft selection / gating.** The route heads implement content-based gating (mixture-of-experts style) keyed on a computed feature. That is a standard attention capability.
- **In-context Bayesian inference.** That trained sequence models approximate Bayes posteriors over latent task variables is a known observation (for example Xie et al. 2022). What this package adds is a localized, causally verified two-stage implementation for one latent: reliability.

**Novelty.** The consistency interpretation adds a *precise statement of which consistency statistic is computed*: the within-group agreement count, the Bayes-sufficient statistic, linear in log-odds. It also shows *where* it is consumed: as attention keys, not as a readout variable. It does **not** add a new decision theory beyond Bayes mixture / weighted voting.
