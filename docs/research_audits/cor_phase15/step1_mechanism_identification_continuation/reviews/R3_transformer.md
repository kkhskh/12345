# R3 — Transformer reviewer (adversarial)

Package: `step1_mechanism_identification_continuation` @ d1aa97d, with `../step1_causal_bridge_deep_investigation/src/cbmath.py`.
Scope: Q/K/V implementation, residual paths, MLP gating, multi-layer and attention-only claims.
Method: I read `src/learned.py`, `program_b.py`, `program_c.py`, `relation_free_equivalent.py`, `reparam.py` and `cbmath.py`, and the three target docs. I re-ran the key computations from stdin scripts that import the package modules and load `checkpoints/`. The only file written is this one.

No FATAL findings. The forward passes, masks, scaling and patch sites are implemented correctly. The problems are theorem statements that are stronger than the math, one interpretive claim about learned T1 models that the package's own data contradicts, and several construction or test details that the docs describe inaccurately.

---

## Findings

### 1. MAJOR — The final-LN clause of T-B-TM is false
**Location.** `DYNAMIC_ATTENTION_AND_DEEP_MECHANISMS.md` B3 (proof step 3); `CONDITIONAL_CAUSAL_THEOREMS.md` T-B-TM.

**Claim.** The final LN with zero bias "multiplies the logit difference by 1/σ > 0, so the sign is unchanged". The theorem's class is said to include this LN.

**Problem.** The theorem is about the *change* in margin between the full and the routing-ablated runs. The LN scale σ is itself a function of the routing, so σ_full ≠ σ_ablated. Then Δ_full/σ_full − Δ_abl/σ_abl need not have the sign of Δ_full − Δ_abl. The sign of each individual margin is preserved; the sign of the change is not.

**Evidence.** I re-ran the B3 check with `qk3(final_ln=True)` (bias 0), using the same 3,000 draws and the same predictor. The predicted sign is correct on only **2980/3000**. Examples are draw 441 (dm = −0.050) and draw 462 (dm = −0.017). The logged 3000/3000 used `final_ln=False`, so the LN clause was never tested.

**Correction.** Drop the final LN from the class. Alternatively, state the theorem for the pre-LN residual margin, and add the condition that the routing change does not reverse the ratio σ_full/σ_abl enough to flip the sign. Also check the downstream uses in FINAL §"B-TM", MAIN row 7 and T-C4 conclusion 2.

### 2. MAJOR — The corollary to T-B-IMP is overgeneralised
**Location.** DYNAMIC B1 ("So no single attention score can switch sign exactly on candidate-relative agreement when K ≥ 3"); `CONDITIONAL_CAUSAL_THEOREMS.md` T-B-IMP statement; FINAL_RESEARCH_SYNTHESIS l.152 and l.177; MAIN l.50.

**Problem.** The proof covers forms that are bilinear in the *raw, linearly encoded* evidence. A score is bilinear in whatever features sit in the residual stream. Any earlier per-position nonlinearity changes those features. An MLP that reads only one source, or a nonlinear value encoder, is enough.

**Counterexample (run).** Take the per-source ReLU feature `u_k = relu(z_k − max_{j≠k} z_j)`. It is exactly computable by a ReLU MLP because max(a, b) = b + relu(a − b). Embed it with an anchor, apply LN, and use one attention score `LN(x_A)ᵀ W_Q W_Kᵀ LN(x_B)` with `W_Q W_Kᵀ ∝ 2I − 11ᵀ`. Its sign equals argmax agreement on **20000/20000** T5b draws.

**Correction.** Restrict the statement: "no attention score whose query and key are linear in the source evidence (e.g. first layer, or pure copies)…". The K ≥ 3 limitation belongs to `qk3`'s *linear copy* design, not to attention.

### 3. MAJOR — The "redundant via attention" explanation for T1 learned models contradicts the package's own data
**Location.** LEARNED §3, T1 reading ("In the three models where ablation leaves accuracy intact, the relation is computed redundantly. Attention already makes sign(ab) decodable at 0.998 before the MLP"); FINAL l.58.

**Evidence** (from `logs/learned_analysis_{dev,heldout}.{json,txt}`):

| model | sign(ab) probe acc @ resid_mid0 | acc after unit ablation | probe interchange @ resid_mid0 |
|---|---|---|---|
| TF1 s0 | 0.998 | 0.746 | 0.503 |
| TF1 s1 | 0.999 | 0.999 | 1.000 |
| TF1 s2 | 0.999 | 0.749 | 0.503 |
| TF1 s3 | 0.998 | 0.749 | 0.513 |
| TF1 s4 | 0.998 | 0.750 | **1.000** |

- Decodability after attention is about 0.998 in *every* TF1 model, including the four where ablation drops accuracy to 0.75. It cannot explain why s1 is robust. This is the same "decodability ≠ use" point the doc makes itself in §1.4.
- The interchange column does not rescue the story either. s4 has full post-attention interchange transfer and still loses a quadrant pair under MLP-unit ablation, because the ablated units are also the downstream readers of that direction.
- In TF2-AO, the resid_mid0 probe is 0.991–0.998 but probe-direction interchange there transfers **0.0** in all 5 seeds. The decodable layer-0 feature is not the one used.

**Correction.** Report the redundancy explanation as unexplained or untested. To test it, ablate the units *and* patch or mean-ablate the head outputs at R, then report the joint versus separate effects.

### 4. MINOR — TF2-AO mechanism: the conclusion holds, the stated mechanism is imprecise
**Location.** LEARNED §1.5; REPARAM §2 item 4 ("softmax/LN normalization across positions").

LN is per-position, not "across positions". I ran an ablation using the same weights, which reproduces `Lm.forward` to 1e-4. Accuracy is on 20,000 T1 validation draws, seeds 0–4:

| intervention | accuracy |
|---|---|
| full model | 0.9998–1.0 |
| both layers' patterns frozen to their dataset mean | **0.25–0.50** |
| LN σ frozen to dataset mean in all layers | 0.63–0.77 |
| final-LN σ frozen | ≈ 1.0 |
| patterns and σ both frozen | ≤ 0.49 |

So **input-dependent softmax routing is necessary**, LN-scale coupling contributes, and the final LN is irrelevant. Suggested wording: "the cross-source interaction requires input-dependent attention patterns (softmax normalization over positions); per-position LN scales contribute but are insufficient alone."

### 5. MINOR — The residual-rotation gauge *can* be applied to the learned models
**Location.** REPARAM §1 "Gauges not applied"; `reparam.py` docstring.

**Problem.** The stated obstruction does not hold. With R orthogonal and R1 = 1, and with no zero gains (min |g| over the T1 checkpoints is 0.84), the following conjugates through every LN exactly, with no new parameters:
- set g′ = 1, b′ = (b ⊘ g)Rᵀ, W′ = R·diag(g)·W for Q, K, V, W1 and WU;
- multiply every residual writer (Wval, Eid, Wp, WO, bO, W2, c2) by Rᵀ on the right.

**Evidence.** The max logit change is 2.9e−6 (TF1), 2.9e−6 (TF2) and 3.8e−6 (TF2-AO) on T1 s0, against a logit scale of about 6.

**Impact.** Small. The residual-probe and interchange outputs are linear refits, so they should be invariant under rotation too. But the claim that residual results are intrinsic "because the residual stream was not transformed" is weaker than necessary. Apply the gauge and report the result.

### 6. MINOR — C2: the exact 0.000 share for q_B is a symmetry artefact of the (a, b) → (a, −b) intervention
**Location.** DYNAMIC C2 table and "Path-specific conclusion"; T-C4 assumption (i).

**Why the zero is forced.**
- The block-1 MLP pairing gives q(x, −y) = −q(x, y) exactly, so the patched q_B has the same magnitude.
- The (a, −b) residual at B has the same norm.
- The value route sees q_B only through σ_B2 (|q_B|²), so its effect is identically zero (I measured max |Δm| = 1.8e−15).

**Asymmetric counterfactuals.** With b → −0.3b and b → −3b (or −b/3), on 400 pairs:
- q_B share: median 3e−4, range −0.117 to +0.043, |Δm| up to 0.555;
- q_R share stays in 0.997–1.000, and decisions match on 100%.

So the conclusion survives, but "carries nothing measurable" overstates it.

**A second issue.** T-C4 (i) cites "values carry 1.8%" as its check that the relation affects m only through the scores. The 1.8% is the effect of the whole value interchange, which is dominated by the direct evidence change. It is not the relation-through-values path, which is the q_B route.

**Correction.** Report the asymmetric results, and cite q_B rather than V for assumption (i).

### 7. MINOR — The C3 "redundant" and "competing" relation features are not what the labels say
**Location.** `program_c.py variant()`; DYNAMIC C3 table ("function unchanged").

- **redundant.** The duplicate units write λ1·q onto `sp` at *every* position, which changes the LN scales at R and at the sources. The function is not unchanged: max |Δm| is 0.328 on a 41×41 grid in [−3, 3]² (median 1.7e−4). Decisions agree on the whole grid.
- **compete = c.** The query reads `6(e3 + sp)` with sp = −c·q, so the gate gain is just 6(1 − c), up to LN-scale effects. "Competing feature 1.5" is the same gate with gain −3. That is a reparameterised sign flip, not two mechanisms competing.
- **Single-copy ablation.** Ablating one redundant copy halves the query gain (6 × 0.5 remains), which is why 200/200 decisions survive. Ablating both makes R's query exactly 0, so the 167/200 is uniform routing by construction.

**Correction.** Label these honestly ("duplicated, LN-perturbed copy" and "gain rescaling"). Redundancy with genuinely independent paths, such as a second MLP with different weights, would be a stronger test.

### 8. MINOR — The C4 bypass check computes the values from the wrong state
**Location.** `program_c.py` l.203: `Z = layernorm(tr.mid[0], …)`.

The block-2 input is `tr.resid_in[1] = mid[0] + mlp_out[0]`. The code omits block 1's MLP output (e3), which changes σ at each position. I recomputed with the correct state on 900 resampled points. The condition outcomes did not change (β = 6: 108 points, 0 violations), so the result stands. The conclusion itself is trivial: |Σα v| ≤ max|v|.

**Correction.** Use `resid_in[1]`.

### 9. MINOR — Dead code in `ablate_dir`
**Location.** `program_c.py` l.176.

`arr[3] = arr[3] - np.outer(...)[0] if arr.ndim == 1 else arr[3]` is a no-op for the (T, d) array; I checked this. Line 177 does the real projection, and only at R. The result is correct, but the docs never say that the B, A and P copies are left intact. Since the query is read only at R, this happens to be the right site.

### 10. MINOR — The downstream MLP does not satisfy T-C4's assumption (iii)
**Location.** `program_c.py` L3 `down_mlp`; T-C4 conclusion 3 ("strictly increasing φ … γ ≤ 1 VERIFIED").

**Problem.** The unit reads `LN(h_R)·fU`, so its output depends on σ_R ∈ [0.95, 2.71] (measured), not on m alone. It is not a scalar map φ(m). For γ = 1, the local slope 1 − γ/σ_R reaches −0.05, so it is not strictly increasing everywhere.

**Evidence.** There are no sign flips for γ ∈ {0.5, 1}. For γ = 3 there are 1,672 flips on a 60×60 grid, measured against the pre-MLP margin. The `ms = np.linspace(...)` monotonicity check (l.215) is computed and never used, so the only "verification" is grid accuracy.

**Correction.** Either build a true scalar φ (no LN read, i.e. read the raw coordinate) or restate the condition in terms of σ_R.

### 11. MINOR — The qk3 description omits layer-1 leakage at B and has a wrong scale factor
**Location.** DYNAMIC "B-construction qk3".

- **Leakage at B.** In layer 1, B's query is 0, so B attends 0.5/0.5 to A and B. It writes 0.5·z̃_A onto fQ and fU *at B*. Measured layer-1 pattern rows: P(B) = [0.5, 0.5, 0].
- **Consequence.** B's LN2 scale, and with it its key and value, depends on z_A. For z_B = e0, V_B is 2.22 when z_A = 0 and 1.23 when z_A = (3, 0, 0). So z̃_B is not "B's LN-scaled evidence" alone.
- **What still holds.** The sign of s_RB (B2) is unaffected. The score identity s_RB = λ⟨LN2 fQ-coords at R, LN2 fB-coords at B⟩/√3 holds exactly; I measured −13.6797 against −13.6797.
- **Scale factor.** The doc's "z̃_A = (wQ/3)·LN-scale·z_A" is wrong. The measured factor is wQ/(σ_A,L1·σ_R,L2), with no /3: the ratio is 0.7325, which equals 1/σ_R2.

**Correction.** Document the leakage and fix the formula.

### 12. MINOR — Two checks are degenerate
- **B3 "3000/3000".** In qk3, w_A = w_R = 0 exactly (the VO norms at A and R are about 1e−16). So w̄_{−B} = 0, and the prediction sign(Δα_B)·sign(w_B) is algebraically the measured quantity. The check never exercises the w̄ term. Add a variant with a nonzero value at A or R.
- **B1 "K = 2 bilinear sign(tA tB) exact: 1.0".** `program_b.py` l.109 compares `(tA*tB>0) == (tA*tB>0)`, which is a tautology. Either remove it or fit a classifier.

### 13. MINOR — C2-manifold grid size is misreported
The log says "241x241 grid". The code uses `gg[::4]`, which is 61×61, and the "refine near the best coarse cell" comment has no code behind it. The conclusion is robust anyway: a distance of 5.49 against a resolution control of 0.35. Fix the text.

---

## Verified claims (reproduced or checked against the code)

- **`learned.py` architecture.**
  - The causal mask `tril` (query t sees s ≤ t) is correct.
  - Scores are divided by √DH, with DH = 8 constant.
  - The network is pre-norm, with LN using biased variance and eps = 1e−5.
  - Residual adds are `x + Σ_h head_out + bO` and `x + MLP`, and the final LN is applied at R only.
  - Hooks wrap the right tensors. TF2-AO has no MLP.
- **`learned.py` training.** Soft-target cross-entropy, Adam with cosine schedule; baselines use matched data and steps.
- **`cbmath.forward`.** The `triu(k=1)` mask, √dh scaling, per-head Q/K/V/scores/pattern hooks, `mid`, `mlp_out` and `resid_in` all match their docstrings.
- **B4 patch sites.** `replace_site(1, "Q0"/"K0"/"V0", …)` patches layer 2 (index 1), head 0, at all positions.
  - Q_R is exactly independent of z_B (max diff 0), so the Q-only margin of −0.617 equals the base margin.
  - K-only gives −3.081 and V-only gives +1.847, as documented.
- **Routing ablation ≡ qk_off.** Setting W_Q = 0 equals patching R's layer-2 pattern to uniform: both give 1.84658 without the rereader, and both give −1.35435 with it. Only s_RB changes, because s_RA = s_RR = 0 already.
- **B-level numbers.** B2 (sign(s_RB) = sign⟨z_A, z_B⟩ always; 8.4% and 43.5%), B3 without LN (3000/3000), B5 (all rows) and the kC sweep match the logs and docs. The sweep is exactly linear in kC.
- **`program_c` variants.**
  - The bypass, extra head (same Q/K as the gate, values −(e1 + e2), writing w·fU), downstream MLP and rereading head (uniform head reading e1 + e2 and writing w·fU) are built as described. Caveats are in findings 7 and 10.
  - The C3 table numbers match the log.
- **Relation-free twin (T-RF construction).**
  - Block-1 attention is off. Each of the 4 ReLU units reads only e1 or only e2 of its own position's LN state. Block 2 is a uniform head (W_Q = W_K = 0).
  - The margin is exactly source-additive: the interaction residual m(a, b) − m(a, 0) − m(0, b) + m(0, 0) is at most 4e−16.
  - `cprime.closed_form` equals the actual C′ forward pass to 3.5e−14.
  - On a 240×240 grid down to |a|, |b| = 0.005, the decisions differ from C′ at 182 cells. All have min(|a|, |b|) ≤ 0.081, inside C′'s stated dead bands (agreement 0.09, conflict 0.02).
  - The decision-level equivalence claimed for the region outside the dead bands is real. The docstring's "(φ_A + φ_B − τ)/4" holds only up to the per-position LN scales, as the docstring itself says.
- **Learned-model gauges.**
  - QK (√DH unaffected because DH is a constant), VO, head permutation (permuting WQ/WK/WV/WO jointly, with shared bO), unit permutation, and unit split (duplicating the column and halving both W2 rows) are all exactly function-preserving.
  - The logged max logit diffs are float32 noise (≤ 8.5e−5).
  - The set-level ablation invariance under splitting holds: both copies are flagged.
- **TF2-AO and TF1 accuracy and decodability.** TF2-AO solves T1 (0.9998–1.0) and needs input-dependent attention (finding 4). TF1's sign(ab) probe at resid_mid0 is 0.998–0.999 (but see finding 3 on what that does and does not show).
