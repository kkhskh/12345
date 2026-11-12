# Full layer-2 causal reconstruction (development)

> **Post-review [R2, R3].** Corrected: §3 (where the response is generated), the meaning of D, the baseline dependence of the
> route split, the natural-check tolerance, the scope of the token attribution, and the injection site. Pre-review text:
> commit 76a9d60.

**Status:** exploratory, on the 16 Investigation VI parents (development data). Only the input-route prediction (P7) was frozen
for the new lineages; see INDEPENDENT_CONFIRMATION.md.

**Code and outputs.** Code: `scripts/layer2_reconstruction.py`. Outputs: `logs/recon_dev/*.json` and `.npz`. Pairs come from
`logs/eval_dev.npz`: 469 F45-R1, 98 R2, 793 R3, 469 F34-R1 and 768 F54o.

## 1. The forward equation (actual code)

The parameters come from `nets.py` and are read-only. The architecture is pre-LN with d = 64, 4 heads (head dimension 16) and a
GELU MLP of width 256. For the readout token t = 36 + q, layer 2 computes:

1. z = resid_pre2 (43 × 64). It is the output of the frozen embeddings and layers 0–1, the same for every condition of a parent.
2. q_h = LN₁(z_t) W_Q^h; k_h(s) = LN₁(z_s) W_K^h; v_h(s) = LN₁(z_s) W_V^h, with one LN₁ shared by all three.
3. A_h(t, s) = softmax_s(q_h · k_h(s) / 4); o_t = Σ_h Σ_s A_h(t, s) v_h(s) W_O^h.
4. x = z_t + o_t; u = W₂ GELU(W₁ LN₂(x) + c₁) + c₂.
5. logits = W_A^T LN_f(x + u).

Transfer trains layer 2, LN_f (g_f, b_f) and W_A; everything else is frozen. The inherited W_A of a P-count parent is its random
initialisation: its relative difference from `train_tasks.init(seed)` is 1.4e-4, because P-count pretraining uses a separate
head (WX). The T1 parent's W_A moved about 15 % during T1 pretraining. During transfer, W_A changes by about 15 % in both arms.

## 2. Exact input-route factorial (5 switches, 32 exact forwards per pair)

**Switches.** Each switch chooses the before-edit or after-edit copy of one input:

- D: residual z_t;
- Qy: z_t as used for the query;
- Kt: z_t as used for its own key;
- Ks: all other keys;
- V: all values.

The natural forwards reproduce the model to ≤ 1.9e-5 (`natural_check_max_abs`; review R2). Möbius terms sum to the total within 4e-7. Möbius coefficients give an exact additive
decomposition of every response into main effects and interactions.

**Readout-token route vs other-token route.**
- "Readout-token only" is the sum over subsets of {D, Qy, Kt}. It measures switching the readout token's copy *from the before
  state*.
- "Other tokens only" is the sum over subsets of {Ks, V}. Note that "other tokens" includes the other readout tokens and the
  episode token, not only source tokens.
- The interaction is the remainder.
- Values are means over parents.

**Baseline dependence (review R2-4).** Removing the readout-token route *from the after state* instead gives a P-count mean of
−0.37 rather than −0.74. P84 is the extreme case: −2.18 vs −0.14. The sign is negative in 8/8 parents under both baselines. The
magnitudes below are baseline-specific; only the sign is robust.

| pairs | arm | total Δm | readout-token only | other tokens only | interaction | parents with readout-token-only < 0 |
|---|---|---|---|---|---|---|
| F45-R1 | T1 | +0.203 | +0.095 | +0.099 | +0.008 | 0/8 |
| F45-R1 | P | **−0.489** | **−0.737** | −0.119 | **+0.368** | **8/8** |
| F45-R2 | T1 | −1.852 | −0.268 | **−1.594** | +0.010 | 8/8 |
| F45-R2 | P | −1.771 | **−1.556** | −0.358 | +0.144 | 8/8 |
| F45-R3 | T1 | −1.728 | −0.166 | **−1.561** | −0.001 | 8/8 |
| F45-R3 | P | −1.950 | **−1.606** | −0.414 | +0.069 | 8/8 |
| F34-R1 | T1 | +0.398 | +0.138 | +0.267 | −0.008 | 0/8 |
| F34-R1 | P | +0.367 | −0.090 | +0.102 | +0.355 | 2/8 |
| F54o | T1 | −0.208 | −0.100 | −0.116 | +0.008 | 8/8 |
| F54o | P | +0.501 | +0.370 | −0.235 | +0.366 | 0/8 |

**Main effects by regime** (means over parents):

D is **not** a pure residual term. It is the readout token's own copy entering both the residual sum and the MLP input. In P80
only −0.07 of D's −0.19 is the pure residual path; the rest is the MLP reading z_t (review R2-3).

| pairs | arm | D | Qy | V | Ks |
|---|---|---|---|---|---|
| F45-R1 | T1 | −0.018 | +0.096 | +0.064 | +0.062 |
| F45-R1 | P | −0.278 (8/8 < 0) | −0.569 | −0.092 | −0.036 |
| F45-R2 | T1 | −0.067 | −0.177 | −0.195 | **−1.499** |
| F45-R2 | P | −0.344 | **−1.391** | −0.240 | −0.198 |
| F45-R3 | T1 | −0.030 | −0.106 | −0.198 | **−1.462** |
| F45-R3 | P | −0.304 | **−1.490** | −0.275 | −0.234 |

**Two different implementations of the same legitimate discount (R2/R3).**

- **T1 parents** implement it almost entirely through the **keys of the other tokens** (Ks ≈ −1.5). The edit changes the
  source tokens' frozen states, and the readout token's attention re-targets accordingly. The decomposition is additive
  (interaction ≈ 0). The readout token's own changed state contributes little.
- **P-count parents** implement it mostly through the **readout token's own frozen state**: the query Qy (≈ −1.4) plus a direct
  residual term D (≈ −0.3). This is the token that P-count pretraining trained to encode every group's count (`train_tasks.py`:
  per-question, per-group count head on the readout tokens). The other tokens contribute little, and a sizeable interaction term
  appears.

**The R1 error lives in the P-count route.**

- The readout-token route gives **−0.74 in R1**, negative in 8/8 parents. Its direct component D (−0.28) is nearly the same in
  R1, R2 and R3 (−0.28 / −0.34 / −0.30), i.e. **regime-insensitive**.
- The query component is less negative in R1 than in R2/R3, but still negative.
- The interaction term (+0.37) is the only regime-sensitive correction, and it is too small.

In the terms of Theorem 1: the P-count route behaves like a separable "a group became perfect → discount" signal, carried by a
state that does not encode the cross-group relation. Probes (FROZEN_REPRESENTATION_SUFFICIENCY.md §3) show that relation is
poorly accessible in P-count states.

## 3. Attention and the MLP

**MLP frozen at its before value** (direct path plus attention only):

| arm | total | MLP frozen | so the MLP adds |
|---|---|---|---|
| T1 | +0.203 | +0.058 | +0.144 |
| P | −0.489 | −0.215 | −0.274 |

**Correction (review R2-1).** The pre-review conclusion, "most of the R1 response is produced after attention", was **wrong**.

Switching only the attention inputs while the MLP responds freely gives most of the total:

| parent | attention-only (MLP free) | total |
|---|---|---|
| P80 | −0.51 | −0.66 |
| P81 | −0.28 | −0.48 |

The pure residual path gives only about −0.07. The P-count response **originates in attention, mainly the query**, and is
**amplified 3–5× by the MLP**.

The −0.079 attention projection below counts only attention's *direct* path into LN_f, which bypasses the MLP. "Exact =
linearised" is an algebraic identity, not a check.

**Projection of the attention change onto the final-LN-linearised margin direction (direct path only):**

| arm | R1 attention projection |
|---|---|
| T1 | +0.062 |
| P | −0.079 |

Attention re-weighting by token set, in the mean projected contribution in R1 (direct path only; an MLP-inclusive Jacobian gives
terms 4–5× larger, review R2-2):

| token set | T1 | P |
|---|---|---|
| q-tokens of the edited group g | −0.322 | −0.188 |
| q-tokens of the other perfect group h* | +0.390 | +0.160 |
| q-tokens of the third group f | +0.013 | +0.006 |

Attention mass on h*'s q-tokens before the edit is 0.396 in T1 and 0.129 in P.

So T1 readout tokens already attend strongly to the sole perfect group, which is the certain colluder, before the edit. After
the edit they re-distribute towards g. P-count models attend to h* less on average, though the difference is weak: P81's
MLP-inclusive h* term is +0.80. The third group's tokens play almost no role in either arm. Theorem 2's "third-group inflation" therefore has **no attention-level counterpart**. It is a behavioural description
only, and is not a mechanism.

**Per-head switches (R1).** In every P-count parent (8/8) the single largest one-at-a-time effect is a **query** switch of one
head, and it is negative: −0.12 to −0.93. In T1 the largest effects are mixed (Q in 5/8, plus V and K) and positive in 7/8. Heads
are not aligned across parents, so "the" head is not a shared object.

## 4. Redundancy, cancellation and non-additivity

- **Injection site (review R2-6).** The repair's injected features are added at tokens 36–41 *before* LN₁. They therefore
  enter the queries, keys and values of all six readout tokens, and also the residual stream, the MLP and the readout. `_rtemb`
  likewise replaces all six readout tokens, not only t. The per-head K switch flips all of that head's keys.
- **P-count R1 = a cooperating negative readout-token route (D, Qy, the MLP) partly cancelled by a positive interaction.** It is
  not one dominant operation. Removing the readout-token route (the `_rtemb` training intervention: the readout token's layer-2
  input replaced by its embedding-level state) does **not** remove the error: P80 −0.60, P81 −0.75, against −0.90 / −0.72
  originally. The downstream learner then builds a similar discount from the source tokens. The readout-token route is
  sufficient to produce the error in the trained models, but it is **not necessary** for the error to arise.
- **T1 R1** is additive and small, made of cooperating positive contributions.
- **Hybrid-state caveat.** The interior Möbius terms mix copies of the input across pathways. They are exact functions of
  well-defined inputs, but those inputs are not naturally reachable states. The corner identities (natural before and after)
  and the training interventions (_rtemb, injection) are the non-hybrid evidence.

## 5. What causes the negative R1 update (strongest supported statement)

**In the trained P-count models.** The 4→5 edit changes the readout token's frozen count-bearing state. Layer 2 reads that change
mainly through one head's query, amplified by the MLP; a small pure residual term is also involved. It reads it as a largely
regime-insensitive discount. The sign of this route is robust; its magnitude depends on the baseline. The regime-sensitive
information, which sits in the source tokens in a form T1 states expose and P-count states do not (probes), corrects it only
partly.

**Why that solution is learned.** The cross-group statistic needed for the correct R1 update (Theorem 1) is not readily
accessible, and a separable discount fits 97 % of decisions well. This is shown by the injection results and the oracle readers.

**Limits.**
- The route analysis identifies *where* the trained solution reads the edit. It does not show that blocking that route repairs
  the decision; `_rtemb` shows it does not.
- The operation-level account (query + MLP) is a description of 16 trained solutions. It is not a unique mechanism: heads
  differ across parents.
