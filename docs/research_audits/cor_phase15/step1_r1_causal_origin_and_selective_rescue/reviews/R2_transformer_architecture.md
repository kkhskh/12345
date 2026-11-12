# Reviewer 2: Transformer architecture (pre-review commit 76a9d60)

Checks were run single-threaded; the scripts are in /tmp/r2chk, outside the package.

## Verified (defended)

- **block = forward.** NL.block∘frozen_z + LN_f + W_A equals `nets.forward()["answer"]` exactly on xP80 and xT70 at step 4000 (max diff 0.0). The structure matches nets.py: pre-LN, 1/√16 scaling, no QKV bias, GELU-tanh, readout on tokens 36–41.
- **The factorial is correct.**
  - LN₁ is applied to each copy separately, which is valid per token.
  - Q comes from Qy; the key at t from Kt and the other keys from Ks; all values from V.
  - D feeds the residual sum and therefore LN₂/MLP.
  - The corners are the natural forwards (logged max 1.9e-5).
- **The Möbius algebra is correct.** The coefficients sum to the total within 4e-7. Every §2 table reproduces from logs/recon_dev.
- **Per-head claims hold.**
  - P-count: the largest switch is a negative Q switch in 8/8 parents (−0.12 to −0.93).
  - T1: the largest switch is Q in 5/8 parents, and positive in 7/8.
- **Freezing and readout facts hold.**
  - P80's W_A differs from `init(80)` by a relative 1.44e-4: it received weight decay only.
  - P80's g_f has mean 1.125, so it was trained.
  - T70's W_A moved 15.3 % in pretraining and moves 15.7–17.4 % in transfer.
  - The frozen parameters are bit-identical between pre and step 4000.
- **The trainer is exactly equivalent.** I ran the actual downstream.py C1 path (block/frozen_z, make_loss, targets, subset AdamW, 4000-step schedule) against cohort6_train.py's masked loop. Over 20 steps the parameter difference is 0.0 and the loss is identical. Comparing re-run conditions with the archived C1 is legitimate.
- **The dynamics tables reproduce** from the log.

## Findings

**1. Major. FULL_LAYER2 §3: "Most of the R1 response is therefore produced after attention".**
- In exact replays I switched every attention input to its after-value, kept D at before and let the MLP respond. This gives −0.507 of −0.662 in P80 and −0.277 of −0.483 in P81.
- The pure residual (D after, MLP output held at before) gives only −0.07 / −0.09.
- The −0.079 "attention projection" counts only attention→LN_f. It omits attention→LN₂→MLP. With the MLP frozen, attention alone gives −0.094.
- So the response is attention-initiated (query) and MLP-amplified about 3–5×, which is consistent with Qy = −0.57 and with §5.
- *Correction:* rewrite the sentence and report the attention-only replays with the MLP live and frozen.

**2. Moderate. Token-set attribution.**
- J differentiates only LN_f∘W_A, so the attribution is direct-path only (about 15 % of the attention-mediated effect).
- "attn_exact = attn_lin" is the algebraic identity ΔA·v_b + A_a·Δv = Δ(Av), not a validation.
- With an MLP-inclusive midpoint Jacobian the re-weighting terms are about 4–5× larger:
  - P80: g −0.79, h* +0.46, f +0.03.
  - P81: g −0.78, h* +0.80, f 0.00.
  - T70: g −0.93, h* +1.07, f +0.03.
- "Third group ≈ 0" survives. "P-count attends to h* much less" is weakened as an explanation: P81's h* term is as large as T1's.
- *Correction:* label the table as direct-path only and add the MLP-inclusive version.

**3. Moderate. D is mislabelled "direct residual".**
- In P80, D = −0.194, of which the pure residual is −0.070. The rest is the MLP reading z_t.
- "Setting the MLP's input to its after value… (the D-only corner)" is wrong: the MLP input there is LN₂(z_t^a + o_b).
- *Correction:* call D "z_t residual copy (direct + MLP response)".

**4. Moderate. The route magnitude depends on the baseline.**
- "−0.737" is the effect of {D, Qy, Kt} added to the before-state.
- Removing those switches from the after-state (total − other-only) gives a mean of −0.37. Per parent: xP84 −2.18 vs −0.14, xP81 −0.36 vs −0.17.
- The sign is negative in 8/8 parents under both baselines and under Shapley, so P7 is robust. The magnitude is not, because of the +0.37 interaction.
- *Correction:* report both baselines, or the Shapley value (≈ −0.55).

**5. Moderate. The random-injection control is far weaker than described.**
- `inj_feats("rand")` computes ((S·a·b·c) mod 7) < 3 with S = Σ X·index. Every feature is a function of S mod 7.
- On 5000 dev episodes there are **7 distinct patterns**. That is ≤ 2.8 bits per episode, with no per-question variation.
- It matches the parameter count only. "9 pseudo-random binary episode features / capacity-matched" overstates it.
- P2 stands as a parameter-count control, not an informative-irrelevant-feature control.
- *Correction:* describe it accurately. A stronger control would be m from another question, or m permuted across episodes.

**6. Minor–moderate. The injection site is incompletely described.**
- `zfun` adds F·Wi to resid_pre2 at tokens 36–41, *before LN₁*. The injected vector:
  - is renormalised jointly with z_t;
  - enters the queries, keys and values of all six readout tokens, so each readout token can attend to every question's m;
  - flows through the residual into LN₂/MLP and LN_f/W_A, bypassing attention.
- The rescue therefore need not use any layer-2 attention operation, which is consistent with Outcome D "not reached".
- *Correction:* state all of these paths in SELECTIVE_RESCUE.md and FROZEN_CONFIRMATION_PROTOCOL.md.

**7. Minor. Route partition wording.**
- V includes v(t) in the reconstruction, but dynamics `route_split` puts v(t) on the readout side. This is immaterial: the V_t main effect is ≤ 0.005.
- "Other tokens" includes the five other readout tokens and the episode token. The T1 statement should say "other tokens' states", not "source tokens'".
- `_rtemb` replaces all six readout tokens, so "builds… from the source tokens" should read "source and episode tokens".

**8. Minor.** The per-head K switch toggles Kt and Ks together. This is undocumented.

**9. Minor.** check_sg_equivalence.py tests N.forward with stop_gradient, not the nnlib path, so the nnlib docstring overclaims. My direct test closes the gap; add it to the package.

**10. Minor.** "Natural check < 1e-5" should read "< 2e-5" (logged max 1.9e-5).

**11. Minor. C3 is a weak test of H-READ.**
- diag(g_f)·W_A, with W_A trainable, spans the same function class, and b_f only adds a constant logit offset. C3 ≈ C1 is therefore expected.
- *Correction:* scope "the final LN does not matter" to that re-initialisation.

**12. Minor.** "Deepens as KL falls" does not hold for P81: Δm_R1 peaks at step 1000 (−0.64) and then shrinks to −0.48.

## Hybrid inputs

Each switched copy is a natural state of its own token; the hybridity is across roles. The *sign* of the readout-token route is robust: 8/8 under both baselines, and confirmed on new lineages. Its *magnitude* is not (finding 4).

The defensible statement is: in trained P-count models, the edit-induced change of z_{36+q} acts mainly through the layer-2 query (attention re-weighting), and the MLP amplifies it into a regime-insensitive negative update. It is sufficient but not necessary (`_rtemb`). "Produced after attention" and "direct residual" are not justified.

## Verdict

The implementation is **correct and exact**: forward equivalence, natural corners, Möbius algebra, freezing semantics, W_A/LN facts and trainer equivalence all hold. No headline result is overturned.

Required revisions:
- Findings 1–3: one substantive mis-statement of where the R1 response is generated, plus the attribution and labelling fixes.
- Finding 4: report both baselines.
- Findings 5 and 6: describe the random control and the injection paths accurately.

**Accept after these corrections.**
