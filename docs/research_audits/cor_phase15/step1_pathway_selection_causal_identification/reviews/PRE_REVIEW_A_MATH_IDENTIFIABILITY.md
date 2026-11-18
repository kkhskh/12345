# Pre-experiment review A: mathematics and causal identification (XVIII)

**Reviewer.** Agent reviewer A (mathematics and identifiability). This is an agent review, not human peer review.
**Object.** `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md` (the "design"), `scripts/xviii_train.py`, `scripts/unit_modes.py`, `logs/unit_modes.json`, `src/xviiim.py`, `logs/dev/measure_dev.jsonl` (10 records as read: 4 XIV + 6 XVIII dev), the mandate §2–§4, §9, §11–§12.
**Labels.** [proof] derivation; [code] read from the implementation (file:line); [num] my own numerical check (scripts in `/tmp/preA18/`, all single-threaded, each < 1 min); [dev] the package's development records; [argument] reasoning that is not a proof.

---

## 0. Verdict

**REJECT (redesign) of the primary comparison as specified in §5 and §8.**

The baseline and block runs on the fresh orders D = 811–813 do not depend on the design: every redesign needs them as anchors. They may start now. No component-arm confirmation should run until the changes in §5 of this review are made and unit-checked.

The reasons, in order of weight:

1. **§5's "core" (gsg × fst) does not separate the hypotheses as stated.**
   - The H2g column is mis-specified. fst does not supply "the routing gradient". It supplies a straight-through signal that is self-extinguishing and evaluated off-trajectory (§1.5, §2.2).
   - gsg is not "(R) removed". It keeps the complement half of the share-recruitment signal: a fraction n_T/(n_T + n_C) at the logit level, which is 0.84 on E (ro0) and 0.14 on M (§1.4).
   - fst also changes the complement routing gradient. On E the cosine between fst's and the block's complement-logit gradients is 0.50 at init and 0.17 at the baseline's step-100 state (§1.5).
2. **fst is close to guaranteed to reproduce the block** (Check A), and its dev result is already in on both bases. The only hypothesis it can refute is a nominal one ("T's logits receive a nonzero gradient").
3. **H1 against "route establishment"** is ill-posed as a contest between rivals. It is well-posed as a controlled-direct-effect question, and that needs one new mode: an exogenous share clamp (§3.3).
   - The dev data already contain a near-share-matched contrast on M that the design overlooked. gsg-M and fst-M end the window with the same own-question share, φ_share 0.85 against 0.74, yet their outcomes differ: φ_nec 0.27 against 1.00.
   - So the reading in §6 item 4 ("outcome tracks share growth") is contradicted on M once share is measured at release (step 100) rather than at step 150, which is after release (§3.2).
4. **The outcome scoring is not prospective enough** for hybrids (§4). φ is undefined, and it ranges 0.01–0.99 across diagnostics for the same run. The competence rule "fail-closed = not a flip" favours the hypotheses that predict "—". Hybrids drift between steps 813 and 1211.
5. **L1's dismissal of mandate A3 is valid only for the same-row complement.** The competing route of each base lies in another layer (E: layer-2 own-q; M: layer-0 own-q), where L1 does not apply (§1.1).

What the dev data already establish, conditional on confirmation: H3 in its immediate-loss form is contradicted on both bases, newly on E by vrs (§1.6). H2a in its instantaneous-mass form is contradicted by vrs. H2g in its nominal form is contradicted by fst.

---

## 1. The decomposition and the lemmas

### 1.1 §2 decomposition (F, R, V) and L1

**§2 equations: correct [proof].**
- With a_j = 0 exactly for j ∈ T, ∂L/∂s_j = a_j(u_j − ū) = 0, where u_j := ∂L/∂a_j = ⟨∂L/∂ctx, v_j⟩.
- ∂L/∂v_j vanishes for that head and row.

**Two clarifications that matter for the design.**
- **(F) has two parts.**
  - (F_c) T's content is absent.
  - (F_r) T's mass is redistributed onto C.
  - On E (ro0), C = 7 tokens (6 readouts and the episode token), so (F_r) is a large forward distribution shift: Δloss at the baseline's step 100 is 0.578 [num], against 0.548 in XIV's own calibration (`i_calib`).
  - vrs removes (F_c) without (F_r). That is why vrs's Δloss on E is only 0.126 [num, `dloss.py`].
- **(R) removal under the block does not freeze T's latent share.** W_Q and W_K are shared with other rows (in layer 0, every source row) and with the complement edges. The upstream layers keep learning.
  - [num, `rates3.py`] At matched parameters (the shared init), the block's raw-gradient instantaneous rate of change of the unmasked T share is 1.5–2.1× the plain rate on E and 0.92–0.98× on M.
  - So "R removed" is a statement about ∂L/∂s_T, not about the dynamics of T's share.

**L1: correct as stated [proof].** s + c·1_C = (s − c·1_T) + c·1, and softmax is shift-invariant. The equality holds as a function, so gradients and training are also identical.
- **The decision "no diversion arm" does not follow for mandate A3.** A3 asks for boosting "a carefully controlled competing route". On E, the competing (mixture) route is layer-2 readout → own-q sources. On M, the competing (early) route is layer-0 readout → own-q sources.
  - These edges are in another layer from T. A boost there leaves T's logits, T's share and T's content untouched in the forward pass, so L1 does not apply.
  - This arm makes a distinct prediction (§3.4, "xbo"). Either run it, or state that A3 is not addressed and why. Do not cite L1 for it.

### 1.2 L2 (graded suppression)

- **Correct as an equivalence along the offset family [proof].** With offset b on all of T, the within-T and within-C distributions are unchanged. At fixed parameters, o = A(b)·ō_T + (1 − A(b))·ō_C, and ∂L/∂s_j ∝ a_j. Every T-quantity is a function of the single scalar A(b).
- **Caveat that must be written into H1.** In a deterministic network, o_T is recoverable from o at every finite b, given ō_C and the scale. So Shannon "information through T" is not reduced by a finite offset at all. Only signal relative to interference and finite precision falls.
  - So H1 must be defined as *decodable or effective* information, e.g. the gain in Bayes-optimal readout from o_T, or SNR.
  - Otherwise H1 predicts "no effect at any finite b", which is a different prediction from "monotone". The mandate (§3, H1) explicitly asks for this definition.
- **L2 understates the arm's use.** Graded doses give additional values of the mediator (achieved share) under a third mechanism, and that is the calibration a single-index test needs (§3.2).
- Practical point [num]: on E, b = 4.6 already gives Δloss 0.541 at the baseline's step 100, against 0.578 for the block. b = 2.3 gives 0.343. On M the values are 0.017 and 0.021, against 0.021 for the block. Doses should be scheduled and reported by *achieved* share (L3), not by nominal b.

### 1.3 L3 and L4

- **L3: correct.** gof keeps the gradient on s_T, so the network can undo a finite offset.
- **L4: correct on the XIV data**, and it now extends to E [num]:
  - vrs flips E in dev with immediate Δloss 0.126 at the baseline's step 100;
  - that is below the non-flipping E location-matched noise (σ = 2.5, Δloss ≈ 0.18 [XIV]).
  - So the immediate-loss form of H3 is contradicted on E as well, for one order, pending confirmation.

### 1.4 L5: gsg [code: `xviii_train.py:91`]

**The implementation is correct.**
- `s = XM*sg(s) + (1−XM)*s` gives an identical forward: max |Δlogits| = 0.0 [`unit_modes.json`].
- ∂L/∂s_T = 0 (measured 0.0, against −0.0068 and −0.0056 in plain runs).

**What gsg does not remove [proof].**
- Write A for T's share in a row. Softmax rows satisfy Σ_j ∂L/∂s_j = 0, and my check of the row-sum residual gives max |row-sum| = 0.0 [num].
- So the T-against-C competition signal is carried *jointly* by T's and C's logits. gsg removes only T's part.
- Under free-logit gradient flow, Δs = −η g, with dA/ds_j = a_j(1[j ∈ T] − A):
  - T part = −η (1 − A)² Δ Σ_{T} a_j²;
  - C part = −η A² Δ Σ_{C} a_j², where Δ = ū_T − ū_C.
- With uniform attention within groups (Σ_T a_j² = A²/n_T, Σ_C a_j² = (1 − A)²/n_C):

  **fraction of the logit-level recruitment of A retained by gsg = n_T / (n_T + n_C).**
  - E (ro0): 36/43 = 0.84. M (own2): 6/43 = 0.14. In the 2-token case it is exactly ½.
- Numerical decomposition on the real model [num, `logitsplit.py`, 3 batches of 256]:
  - E: 0.82–0.92 retained at init; 0.72–0.74 at the baseline's step 100 (2 of 3 batches; the third has a near-zero total).
  - M: 0.26–0.48 at init; 0.03–0.15 at the baseline's step 100.
- **On E with T = ro0, gsg removes 100% of the logit gradient on the *within-T* log-odds** (own-q against other-q sources), because C's logits shift all of T equally [proof]. The E route is own-q (Finding A1-1; own0 necessity). So on E, gsg is mainly a *selectivity* intervention, not a share intervention. That confounds the cross-base comparison.

**The §6 item 5 statement "gsg: identical to the plain run by construction" is false after step 0 [proof + num].**
- What is guaranteed is only the *same-θ* immediate Δloss = 0.
- θ_gsg ≠ θ_base from step 1. At step 100, ‖θ_gsg − θ_base‖ = 0.56 (E) and 0.71 (M) of ‖θ_base − θ_block‖ [num, `dist.py`].
- So the window losses differ. The statement must be corrected.

### 1.5 L5: fst [code: `xviii_train.py:93–95`]

**What it is [proof].**
- a = a_f + sg(a_b − a_f) has the value a_b, and ∂a/∂s = J(a_f) = diag(a_f) − a_f a_fᵀ. So ∂L/∂s = J(a_f)ᵀ u, with u = ∂L/∂a evaluated at the blocked forward state.
- Since ctx = Σ a v and a has the value a_b, ∂L/∂v_T = a_b,T(…) = 0. So (V) is absent, as stated.
- **T logits:** ∂L/∂s_j = a_f,j (u_j − ū^{a_f}). This is the first-order loss change from moving infinitesimal mass onto j, at the deprived state: "would admitting T help now?" It is well defined, but:
  1. **The update field is not the gradient of any scalar loss** (straight-through fields are non-conservative). So fst is a dynamical intervention, not "the block with a different objective".
  2. **The signal self-extinguishes along fst's own trajectory** [num].
     - Summed T-logit gradient at init on E: −0.050 to −0.061 under fst, against −0.005 to −0.010 under plain (6–10× stronger push towards T).
     - At fst's own step-50 state: +0.001 to −0.005, which has collapsed. Plain's gradient at that state even pushes *away* from T (+0.008 to +0.010).
     - The parameter-level latent-share rate under fst at init is 3.6–4.8× plain's, and at fst@50 it is ≈ 0 (−0.006 to +0.009).
     - The warm-up places 75% of the window's cumulative learning rate in steps 50–99 [proof: lr_i = 1e-3·i/200]. So fst's push is strongest exactly when it is least effective.
     - fst's latent own-q share: 0.134 at step 25, 0.136 at 50, 0.155 at 100 (baseline 0.234) [dev].
- **Complement logits** — fst ≠ block + R_T [proof]. With A = a_f(T) and a_f,k = (1 − A) a_b,k for k ∈ C:

  ∂L/∂s_k|_fst = (1 − A)·[ ∂L/∂s_k|_block − A·a_b,k·(ū_T^{a_f} − ū_C^{a_b}) ].

  [num, `fstC_vrs.py`]
  - E: ‖fst_C‖/‖block_C‖ = 0.56, cosine 0.50 at init; 1.63 and cosine 0.17 at the baseline's step 100.
  - M: 0.87 and cosine 0.99, matching 1 − A ≈ 0.86.
  - So on E, the "R present" cell differs from the block in **two** ways: T's routing gradient, and a large re-weighting of C's routing gradient.
- Whole-gradient difference from the block: 7.8% (E) and 1.4% (M) [`unit_modes.json`]. The trajectory stays nearest the block: ‖θ_fst − θ_block‖ = 0.25 (E) and 0.45 (M) of the base–block distance at step 100 [num].
- **Fix, if fst is kept.** Use a custom VJP ("fst2") with:
  - forward a_b;
  - backward ∂L/∂s_C := J(a_b)ᵀu restricted to C (exactly the block's);
  - ∂L/∂s_T := a_f,T ⊙ (u_T − ū^{a_f}).

  Only then is the fst − block contrast "T's routing gradient only". **My recommendation is to drop fst from confirmation** (§5).

### 1.6 L6: vrs [code: `xviii_train.py:96–97`]

**The implementation is correct.** `ctx = Σ a(1−XM) v + Σ a XM roll(v, 1, axis=0)`. The roll is on the batch axis, on T edges only, and only during training (the evaluation path does not use `block_m`).

**Is the roll leak-free? Yes [code + proof].**
- `sample_fast` (`step1_latent_reliability/src/bayes.py:147–167`) draws R, Y, X and the noise independently per episode.
- So episode n − 1 is independent of episode n, and o_T^{vrs}(n) carries no information about Y_n.
- No cross-row leak:
  - On M, row q reads rolled values at its own sources, while row q′ ≠ q reads the real values at the same key positions. Layer 2 is the last layer, and logits are read per row (`xviii_train.py:113`), so nothing flows from row q′ to row q.
  - On E, readout rows can still read episode n's sources in layer 1, exactly as under the block.

**What vrs changes besides content [num, `fstC_vrs.py`].**
- Magnitude is preserved:
  - E‖o_T‖² within 0.05% (E init) and 6.5% (E, baseline step 100); M within 2.4% and 1.5%.
  - Variance within 2–6%.
  - Mean within 4–23%. The 23% (E, step 100) comes from breaking the key–value coupling once attention is selective: a_j is set by episode n's key, v_j by episode n − 1.
- corr(o_T^{vrs}(n), o_T^{plain}(n − 1)) = 0.998 (E init), 0.92 (E step 100), 0.84–0.89 (M). So vrs is essentially a *transplant* of the previous episode's T-contribution: content destroyed, magnitude kept.

**Gradients under vrs [proof].**
- (V) is **present but content-free in expectation**: ∂L_n/∂W_V ∝ Σ a^{(n)} δ_n z_{n−1}ᵀ, and E[δ_n z_{n−1}ᵀ] = E[δ_n] E[z]ᵀ. Only the mean or bias component is learned, plus gradient noise.
- (R) is present and carries "T is uninformative", which pushes T's share down. Dev: T's share stays at its initial level.
- The design's (V) row should list vrs as "V present, content-learning part zero in expectation".

**The L6 conclusion is correct only for instantaneous mass.** vrs separates H1 from H2a in its instantaneous-mass (or magnitude) form. Dev vrs flips with magnitude preserved, which contradicts that form. vrs does not separate H1 from the share-trajectory hypothesis, because vrs's share stays low: φ_share 0.88–0.95 at release.

---

## 2. Check A: outcomes fixed by construction

| claim | status |
|---|---|
| gsg forward and immediate Δloss = 0 at every step | **Guaranteed** at the same θ [proof]. Not evidence. The window-trajectory version is false after step 0 (§1.4). |
| fst forward = block forward | **Guaranteed** (1.4e-6) [proof]. |
| "fst flips" | **Near-guaranteed, not proven** [argument + num]. The gradient field is within 7.8% / 1.4% of the block's. Its one lever, T's routing gradient, collapses by step 50 and acts in the low-LR part of warm-up. The trajectory stays nearest the block. The dev flip on both bases already exists. Confirming it on 6 pairs buys little. |
| fst "—" under H2g | **Holds only for the nominal H2g** ("T's logits receive any gradient"). Under an integrated form (T is actually recruited during the window), fst also starves T (latent share +0.02, against +0.10 in the baseline), so H2g predicts *flip*. |
| vrs flips | **Not guaranteed, but predicted by every hypothesis except H2a in its instantaneous form**, because vrs's share stays low. Non-discriminating between H1 and share-trajectory (the design admits this). |
| graded b: monotone dose–response | **Predicted by all hypotheses** (L2). Descriptive (mandate P1) only. |
| Share clamp held low with content rolled (proposed cell) | **Predicted F by every hypothesis.** Do not run it (§3.3). |
| Share clamp at the baseline's level with real content | **Near-guaranteed ≈ baseline** if β stays ≈ 0. Dev-only validity check (log β), not a confirmation arm. |

On "is fst's backward a meaningful counterfactual?". It is meaningful as the marginal value of T at the deprived state. It is not the recruitment signal of any trajectory the network could follow with T in the forward pass. This is because the downstream readers that make u_T large in the plain run never develop under fst. The recruitment loop (use, then larger u_T, then larger share, then more use) is cut at "use". So (R) is endogenous to (F), and component knock-outs cannot assign the effect to "F" or "R" as separable causes [argument].

---

## 3. Prediction table, equivalences, and the H1 / mediator question

### 3.1 The hypotheses, restated so that each has a measurable index

- **H1d (forward content, direct).** The outcome depends on whether T's *decodable* content enters the forward computation during the window, holding T's release share fixed.
- **Hm (share trajectory).** The outcome depends on T's achieved share up to release. Index: the readout's own-q mean share at the target layer at step 100, measured unmasked. This holds whatever T carries and however the share is produced.
  - **This merges H2a-growth and H2g-integrated, which are observationally equivalent under every arm I can construct** [argument]. Any intervention that fixes the share trajectory fixes both, and they differ only in a mechanism story. Mark them as one hypothesis.
- **H2a-inst.** The outcome depends on T's instantaneous forward mass or magnitude ‖a_T o_T‖.
- **H2g-s (nominal).** The outcome depends on whether T's logits receive a nonzero gradient.
- **H2c (cross-route competition).** The outcome depends on the competing route's early share or contribution, with T untouched.
- **H3.** Immediate Δloss or gradient disturbance.
- **H4.** Content absence and low share are *both* required.

### 3.2 Dev data, scored at release (fixes §6 item 4)

§6 item 4 uses E own0_mean at step 100 but M own2_maxhead at **step 150**. Step 150 is after release, so that "mediator" is partly an outcome of the post-release dynamics. At release (step 100), with φ_share = (A_base − A_arm)/(A_base − A_block) and φ_nec as defined in §4.1 [num]:

| run | φ_share@100 (mean / max head) | φ_nec (log) | class |
|---|---|---|---|
| E gsg | 0.47 / 0.12 | 0.66 | HYBRID |
| E fst | 0.80 / 1.02 | 0.97 | MIX |
| E vrs | 0.95 / 1.05 | 1.03 | MIX |
| **M gsg** | **0.85 / 0.86** | **0.27** | HYBRID |
| **M fst** | **0.74 / 0.60** | **1.00** | EARLY |
| M vrs | 0.88 / 1.00 | 1.00 | EARLY |

**On M, gsg suppresses the share at release at least as much as fst, yet ends at φ 0.27 against 1.00.**
- Hm predicts that these coincide. They do not (one order).
- The difference between the two cells is forward content, together with its value learning (present in gsg, absent in fst).
- So the dev data already favour **"content matters beyond share at release"** on M. gsg's partial effect, with forward content fully intact, also shows that **content is not the whole story**: H1d predicts N for gsg.
- The current best description is therefore "both contribute". The decisive missing cell is *high share with destroyed content*.

### 3.3 The redesign: exogenous share clamp × content (controlled direct effects)

**Is "H1 against route establishment" well-posed?**
- As competing hypotheses, no. In the natural regime, missing content implies no recruitment, so content and share are cause and mediator on one path.
- As a mediation question, yes. With potential outcomes Y(c, m), content c ∈ {real, rolled} and share trajectory m ∈ {m_base, m_low}:
  - CDE_content(m) = Y(rolled, m) − Y(real, m);
  - CDE_share(c) = Y(c, m_low) − Y(c, m_base).
- Identification needs an intervention that sets m with no other effect (consistency, no side effect of the setting mechanism). That is testable in part: see the validity check below.

**Mode `shc` (share clamp by adaptive offset).** At the target layer, for head h at step i < 100:

- s_{n,h,q,j} ← s_{n,h,q,j} + sg(β_{h,i})·1[j ∈ T_q]
- β_{h,i} solves Ā_h(β) := mean_{n,q} [ Σ_{j∈T_q} e^{s+β} / (Σ_{j∈T_q} e^{s+β} + Σ_{k∈C_q} e^{s}) ] = A*_h(i).
- Ā is strictly increasing in β, with dĀ/dβ = mean A(1 − A) > 0, so the root is unique. Use 8 Newton steps inside the jit, clipped to |β| ≤ 20.
- Content is either real (standard ctx) or rolled (the vrs formula on T).
- Properties [proof]:
  - the forward differs from plain only in the aggregate T-against-C odds (per head; per-episode variation kept, within-T and within-C distributions kept);
  - ∂L/∂s_T flows, so H2g-s's quantity is present;
  - its effect on the head-mean share is undone each step by β.
- **Schedules.**
  - m_base: A*_h(i) = the paired baseline's per-head own-q mean share at step i. Record it by a logging replay of the deterministic baseline; the harness replays bitwise (`replay_check.json`).
  - m_low: A*_h(i) = A_h(0).
- **E edge set.** Use T = own0 (6 per row) for all component arms. The ro0 set confounds gsg on E (§1.4) and contains 30 non-route edges (A1-1). [num, preliminary] The own0-only block run at step 813 has nec_E 0.010, nec_M 0.777, KL 0.015: it flips E. Confirm it at step 1211 with the frozen scorer before the freeze.

**Cells and predictions** (bands from §4: F ≥ 0.8, P in (0.2, 0.8), N ≤ 0.2):

| cell | H1d | Hm | H4 | H1d + Hm additive | H2g-s | H3 |
|---|---|---|---|---|---|---|
| **shc-hi-rolled** (share = baseline trajectory, content destroyed) | **F** | **N** | N | P | N | measure Δloss; flagged on E if > 0.18 |
| **shc-lo-real** (share held at init, content real) | **N** | **F** | N | P | N | ≈ 0 |
| shc-hi-real (validity, dev only) | N | N | N | N | N | ≈ 0 |
| shc-lo-rolled | F | F | F | F | F | — | **not run (Check A)** |

Every pair among {H1d, Hm, H4, additive} differs in at least one of the two confirmation cells.
- H4 and "additive" differ only by band (N against P) in shc-hi-rolled. They are therefore weakly separated, and must be reported as such if P is borderline.
- Validity rule (prospective): if shc-hi-real is not N in dev, or |β| exceeds 1 for more than 10% of window steps in that cell, the clamp is not inert. The shc arms are then reported as uninterpretable, not reinterpreted.
- On M, gsg is an approximate *achieved* version of shc-lo-real (share stayed at 0.143 against 0.142 at init). The prediction for shc-lo-real on M, if the dev pattern holds, is P.

### 3.4 Cross-route boost (mandate A3; optional but must be addressed)

- **xbo:** s_{T′} += c for steps < 100, where T′ is the competitor route (E: layer-2 own-q; M: layer-0 own-q). c is calibrated on dev, then frozen. A suggested target is that the competitor's own-q share at step 100 equals the paired block run's competitor share at step 300 (E block own2_maxhead 0.66).
- T's forward is untouched.
- Predictions: H2c → F. H1d → N. Hm → whatever T's achieved release share is (measured; prospectively, "F only if φ_share ≥ 0.8").
- This is the only arm that tests "relative contribution reduced by an alternative mechanism" without touching T.

### 3.5 Pairs that remain observationally equivalent (to be stated explicitly in the report)

1. **H2a-growth ≡ H2g-integrated.** Merged into Hm. No arm separates them (§3.1).
2. **H1d against "value-gradient/reader development through T".** Content in the forward pass and (V) co-vary in every arm (block, fst: both absent; gsg: both present; vrs: content absent and V content-free).
   - If the separation is wanted, add `vsg`: ctx = Σ a(1 − XM) v + Σ a XM sg(v). The forward is intact and (V) is removed in that head only.
   - Otherwise, report H1d as "forward content, including the value learning it enables".
3. **H4 against additive** is separated only by band (above).
4. **H5** is not addressed by any within-parent arm.

---

## 4. The outcome measure

### 4.1 Define φ prospectively

The design's "φ ≈ 0.72 on E and 0.27 on M" is undefined. Across diagnostics, gsg's fraction-of-block-effect ranges 0.22–0.99 on E and 0.01–0.73 on M [num]:
- E: lik 0.66, mix_mass 0.99, ρ 0.22, nec_E 0.88, nec_M 0.51;
- M: lik 0.27, mix_mass 0.01, ρ 0.73, nec_E 0.10, nec_M 0.36.

**Required definition.**
- Primary: φ_nec = ½ Σ_{r∈{E,M}} [ℓ(nec_r^{arm}) − ℓ(nec_r^{base})] / [ℓ(nec_r^{block}) − ℓ(nec_r^{base})], with ℓ(x) = log(max(x, 0) + 0.01).
  - The anchors are the **paired** baseline and block of the same base × order.
  - Dev values: gsg 0.66 (E) and 0.27 (M); fst 0.97 and 1.00; vrs 1.03 and 1.00.
- Report ρ-φ and the structural φ as secondaries.
- ρ must not drive the class of hybrids. ρ ≤ 0.60 labels both gsg hybrids EARLY (ρ 0.57, 0.59), even though nec_M = 0.385, so the mixture route is about half as necessary as in the mixture baseline. ρ's bands were calibrated on pure classes only.

### 4.2 Bands and hybrids (prospective scoring)

- **Bands:** F if φ_nec ≥ 0.8 and the three-diagnostic unanimity class is the opposite organisation; N if φ_nec ≤ 0.2 and the class equals the baseline's; otherwise P.
- Every hypothesis states its band per arm before the freeze, as in §3.3.
- A P result **contradicts** every hypothesis that predicted F or N for that arm. It is never counted as "weak support for flip".
- The minimum meaningful difference between two arms is a band change, together with |Δφ_nec| ≥ 0.4 in at least 4 of 6 pairs.

### 4.3 Measurement time and drift

Hybrids are not stationary. gsg-E moves from nec_E 0.243, nec_M 0.242 at step 813 to 0.157 and 0.385 at step 1211 [num, `own0.py`, same measurement stream].
- Freeze 1211 as primary.
- Continue every P-band run to step 2546, on the unchanged LR schedule, and report it as a stability readout.
- A P that moves to a band edge is labelled "transient hybrid".

### 4.4 Competence and missing data

- KL ≤ 0.06 is reasonable: all dev and XIV runs are at 0.010–0.036.
- But "not competent = not a flip (fail-closed)" is **not neutral**: it scores in favour of every hypothesis that predicts N.
- Replace it with "not competent = missing". An arm is inconclusive if more than 1 of its 6 pairs is missing.
- A pair is non-evaluable, reported in the denominator, if its baseline is HYBRID or its block is not F. φ's anchors are then undefined.

### 4.5 Mediator index

- Freeze one mediator: own-q mean share at the target layer at step 100 (unmasked), with the max head reported.
- Do not use post-release times (§3.2).
- Pre-register the single-index test: Hm predicts the same φ_nec for arms with equal φ_share (within 0.15) on the same base. The dev pair gsg-M / fst-M already fails this test.

---

## 5. Arms: keep, drop or redesign

| arm | decision | reason |
|---|---|---|
| baseline, block (D = 811–813) | **keep; may start now** | anchors under every design |
| vrs | **keep** | the dev result contradicts H2a-inst and immediate-loss H3 on E; confirm it (cell "endogenous-low, rolled") |
| gsg | **keep, re-labelled** | not "R removed": it keeps n_T/(n_T + n_C) of the logit-level recruitment. On M it is the achieved "low share, content intact" cell, and its contrast with fst-M is currently the strongest evidence. On E, rerun it with T = own0 |
| **shc-hi-rolled** | **add (decisive)** | the only cell separating H1d from Hm and H4 |
| **shc-lo-real** | **add** | the clean exogenous version of gsg; required on E |
| shc-hi-real | dev only | clamp validity (§3.3 rule) |
| fst | **drop from confirmation** | near-guaranteed (Check A); it refutes only H2g-s, already done in dev; on E it also alters the C-gradient. If kept, use fst2 (custom VJP, §1.5) |
| graded b = 2.3, 4.6 | **secondary** (mandate P1) | reported by achieved share; non-discriminating (L2). On E, b = 4.6 ≈ block in Δloss |
| xbo | **add, or state that A3 is not addressed** | L1 does not cover the cross-layer competitor |
| vsg | optional | only if the reviewers want V separated from forward content |

**Compute** [estimate from the design's ~10 min per run].
- Confirmation: 6 arms (baseline, block, vrs, gsg, shc-hi-rolled, shc-lo-real) × 6 pairs = 36 runs, about 3 h on 2 lanes.
- Dev for shc: 3 cells × 2 bases = 6 runs, plus baseline logging replays (100 steps each).
- Graded and xbo as budget allows.
- Implementation must come with unit checks for shc:
  - forward share equals A* within 1e-3 per head;
  - ∂L/∂s_T is nonzero;
  - the roll is on the batch axis only;
  - β stays finite.

---

## 6. Smaller points

- **(R) in §2.** The statement holds per row and head. Under the block, the latent T share still moves through shared W_Q/W_K and upstream layers: the instantaneous rate is nonzero, 1.5–2.1× plain on E at init [num]. Say so, since gsg's "limitation" applies equally to the block.
- **Instantaneous raw-gradient share rates at matched θ do not predict trajectories.** On E at init, the block's and fst's rates exceed plain's, yet neither grows the share [num, `rates3.py`]. Any hypothesis must therefore be scored on *achieved* quantities, never on component presence.
- **The two gsg hybrids land on nearly identical numbers** (nec_M 0.3854 against 0.3844; mix_mass 0.9447 against 0.9452) from different bases and edge sets. That is plausible as a common attractor, but worth one sanity check of the run environments before it is described as "the same hybrid state".
- **Unit checks missing.**
  - gsg's complement-logit gradient should equal plain's exactly. This is trivial, but it is the property that limits gsg.
  - vrs should change nothing outside T: compare ctx on C-only rows.

## 7. Scratch scripts (not package files)

`/tmp/preA18/`:
- `rates.py`, `rates3.py`: instantaneous share-recruitment rates per mode at matched θ, 3 batches;
- `logitsplit.py`: T/C split of the logit-level recruitment, and fst's T-logit gradient;
- `fstC_vrs.py`: fst against block on C logits, and vrs value statistics;
- `dist.py`: parameter distances between arms;
- `dloss.py`: immediate Δloss of block, vrs and gof at the baseline's step 100;
- `own0.py`: preliminary necessity of the own0-only E block at step 813, and gsg-E drift.

All of these load the harness exactly as `unit_modes.py` does (the source executed up to the training loop), and none trains.

---

## Addendum: re-review of design v2

**Object.**
- `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md` §9–§10;
- `scripts/xviii_train.py` (md5 e90ce467…; modes sha, gsv, vrg; FX_SLOG);
- `scripts/unit_modes2.py`, `scripts/unit_modes3.py` and their logs;
- `scripts/xviii_design.py`, `logs/XVIII_design.json` (48 runs);
- `scripts/score_XVIII.py`, `scripts/preflight_XVIII.py` and `logs/preflight_XVIII.json`;
- `src/xviiim.py` (measure_v2, rho_fit_lapse, cls_behav2, nec_rq1);
- `logs/dev/rhoL_xiv.json`, `logs/dev/gsg_continuation.jsonl`, `logs/dev/measure_dev2.jsonl`.

New scratch scripts are in `/tmp/preA18/`: `newton.py`, `newton2.py`, `asym.py`, `logitsplit_gsv.py`, `dloss_own.py`, `um2.py`. No training was run, and no package file was changed except this one.

### Verdict: **APPROVE WITH REQUIRED CHANGES**

The v2 design implements the redesign faithfully. Every hypothesis in the frozen table now has at least one discriminating arm, and no decisive arm is guaranteed by construction. Five changes are required before the freeze:
- **A1, A2 and A3 are blocking.** A1 is a solver defect that would silently waste the decisive arm. A2 is a scorer asymmetry. A3 is an undeclared split of Hm.
- **A4 and A5 are required corrections** to the H3 rule and to the declarations.

The baseline and block runs may continue.

### (a) Does sha match the §3.3 specification?

**What matches [code: `xviii_train.py:100–108, 120–121`].**
- `sha_ell` gives l = logsumexp_T(s) − logsumexp_C(s) per (episode, head, readout row). With offset β on T, the row share is exactly σ(l + β). So `sha_beta` solves mean_{n,q} σ(l + β_h) = A*_h(i) on the stop-gradiented logits of the current batch at the current parameters, as specified.
- The offset is stop-gradiented, and ∂L/∂s_T flows: unit 3 gives −0.0071, which is nonzero.
- The schedule is the paired baseline's per-head share before the update at step i, on the same batch (identical data stream). The lo schedule is the baseline's step-0 row, so β(0) ≈ 0.
- Rolled content uses the vrg substitution.
- The logged diagnostics (`_shad`) recompute β and the achieved share at the pre-update parameters, so the scorer's clamp-error check is meaningful.
- I verified the scorer's window-log indices (β at w[8:8+nh], shares at w[8+nh:8+2nh]) against the logging code.

**A1 (blocking): Newton from β = 0 diverges in exactly the regime sha-hi-rolled will enter [num, `newton.py`].**
- Under rolled content, the routing gradient lowers T's *latent* share while the hi schedule rises. The E per-head targets go 0.086 → 0.294 in dev (`slog`).
- On synthetic per-row log-odds, the harness's 8 Newton steps (starting from β = 0, with clipping) fail in 24 of 112 cases (|achieved − target| > 1e-3). They oscillate onto the ±20 clip.
- The failures include current share 0.03 against target 0.234 (any spread), and current 0.01 against target 0.134.
- At β = −20 the forward pass *is the block*. The scorer would catch this (clamp_max_err > 1e-3 gives UNINTERPRETABLE), but the decisive arm would be lost.
- The unit check (`unit_modes3`) only tests the initial parameters, where targets are near the current share (max error 6e-8). It does not exercise this regime.
- **Fix:** warm-start at β₀ = logit(A*) − logit(mean_{n,q} σ(l)), then run the same 8 Newton steps. On 128 cases, including current shares down to 0.003 and targets up to 0.9, the maximum error was 1.7e-6 and there were no failures [num, `newton2.py`]. Safeguarded bisection also has no failures.
- Add a unit check with a latent share of 0.01–0.03 and targets of 0.13–0.4.

### (b) Is the scorer logic correct and symmetric?

**What is correct [code].**
- φ_nec matches §10.4.
- Bands require both the φ_nec threshold and the cls2 direction, at the same step and against the paired anchors.
- Competence uses KL ≤ 0.045 and acc ≥ the paired baseline's acc − 0.03. A non-competent run is MISSING.
- Evaluability requires a competent, unanimous baseline and a competent block with the opposite class.
- The arm is pooled at ≥ 5 of the definite evaluable pairs, otherwise MIXED or INCONCLUSIVE.
- int→str step keys round-trip correctly through `measure18.py`.
- Minor: a NaN φ (degenerate anchors) falls through to "P". It should be MISSING.

**A2 (blocking): the hypothesis-verdict rule is asymmetric between "med" predictions and fixed predictions** [num, `asym.py`, using the package's own preflight record builder].
- A fixed prediction is *contradicted* only when ≥ 5 pairs share a definite band that differs from it. Split outcomes give MIXED, hence INCONCLUSIVE.
- A med prediction (Hm on gsg and gsv) is *contradicted* by **any 2 mismatched pairs**. It is *consistent* with no minimum number of evaluable pairs; 2 matching pairs suffice.
- Demonstration with identical perturbations (2 of the 6 gsg pairs turned to P, everything else exactly as predicted):
  - when H1d is the truth, H1d's verdict is **INCONCLUSIVE**;
  - when Hm is the truth, Hm's verdict is **CONTRADICTED**.
- **Fix:** one rule for every hypothesis and arm.
  - Form each pair's predicted band (fixed, or med from that pair's mediator).
  - CONSISTENT if at least 5 definite evaluable pairs match.
  - CONTRADICTED if at least 2 definite pairs mismatch.
  - INCONCLUSIVE otherwise.
  - These conditions are disjoint for n ≤ 6.
- Re-run the preflight with heterogeneous scenarios: one noisy pair per arm, and two noisy pairs on one arm, for each hypothesis.

**Also required: per-base hypothesis verdicts.**
- Use the same rule with "3 of 3 match, any mismatch contradicts", reported next to the pooled verdict.
- Probe: H1d exactly true on E and Hm exactly true on M gives *every* pooled verdict INCONCLUSIVE except the H2g ones [num].
- Dev already shows strongly base-dependent magnitudes (gsg φ_nec 0.66 on E against 0.27 on M), so a base-specific truth is a live possibility, and pooling would hide it.

### (c) Is the mediator ("med") rule implemented as intended?

**For gsg and gsv, yes.** φ_share uses the unmasked own-question mean share at step 100, anchored on the paired baseline and block, and is mapped to bands at 0.2 and 0.8. Two caveats must be stated:
- **The mapping is an identity assumption** (mediator band = outcome band). The single-index idea only needs a *monotone* mapping. Either state that "Hm with identity mapping" is the version tested, or calibrate the mapping on gof23 plus the anchors per pair.
- **The denominator is small on M.** A_base − A_block at step 100 is about 0.024–0.03 (dev `slog`: 0.143 → 0.167). φ_share on M is therefore noisy to about ±0.1, even with the common episode set used in `measure18.py`. Report a bootstrap interval over episodes.

**A3 (blocking): the clamp separates two different mediators, and the table merges them.**
- sha holds T's *forward* share during the window. The routing gradient keeps acting on the latent logits, and β removes its effect only in the forward pass.
- At release β disappears, and the network carries its *latent* share: σ(l), not σ(l + β).
- So under sha the two candidate mediators come apart:
  - **Hm-window:** the forward share trajectory in steps 0–99. This is what sha sets. The fixed predictions in the table (sha-hi-rolled N, sha-lo-real F) belong to this hypothesis.
  - **Hm-release:** the latent share state at step 100, which is what the network continues from. Its prediction for every arm, *including vrg and both sha arms*, is med, computed from the measured latent φ_share.
- Expected consequences. These are not guaranteed, but the sign of the routing gradient makes them likely:
  - In sha-hi-rolled, the latent share falls (content useless), so Hm-release → F. That is **the same as H1d**.
  - In sha-lo-real, the latent share may grow as in the baseline, so Hm-release → N. That is **the same as H1d**.
  - **So the sha arms separate H1d from Hm-window but, in that case, not from Hm-release.** H1d against Hm-release is then separated only by gsg and gsv: content present, latent share low at release. The dev result on M (φ_share 0.85, observed P) already disfavours Hm-release with identity mapping.
- **Required:**
  1. Split Hm into Hm-window (fixed sha predictions as now; vrg → med on the window-mean *forward* share) and Hm-release (med everywhere).
  2. Report each sha run's latent φ_share at step 100 and β at step 99.
  3. Declare in §10.3: "if the latent φ_share of sha-hi-rolled is ≥ 0.8, sha-hi-rolled does not separate H1d from Hm-release". State the analogous condition for sha-lo-real (latent φ_share ≤ 0.2).
- The dev sha runs arriving in ~90 minutes will show which case applies. The rule must be frozen before they are inspected, not adjusted after.

**A3b.** Hm's fixed prediction F for vrg assumes vrg's share stays low. vrg's share is endogenous, so Hm should predict med for vrg, as it does for gsg and gsv.

### (d) Remaining guaranteed outcomes, and pairs no arm separates

**No decisive arm is guaranteed by construction.**
- block = F holds by the evaluability definition. That is intended: the block is the anchor.
- gof23 is predicted F by all hypotheses and was F on both bases in dev (`measure_dev2`). It is correctly secondary (P1 only).
- vrg is predicted F by every hypothesis except the two H2g variants, so it is decisive only against H2g. That is acceptable, but it should not be described as a content test between H1d and Hm.

**A4: the H3 rule on E compares two different metrics** [num].
- The 0.061 threshold is built from XIV's *same-θ immediate* Δloss (sham 0.031 at step 100). The rule then applies it to a *trajectory* Δloss: the arm's own window loss minus the baseline's, both on their own parameters.
- The trajectory metric includes the consequences of different learning:
  - gsg, whose immediate disturbance is 0 by construction, has a trajectory Δloss of +0.031 (E own0) and +0.034 (E ro0);
  - dev vrs-ro0 has +0.062 and would fail the threshold;
  - its immediate Δloss is 0.126 (ro0), and vrg-own0's is 0.129 at the baseline's step 100 (`dloss_own.py`). Both are below the non-flipping E noise control (≈ 0.18, same metric).
- **Fix:** score H3 on E with the L4 metric. Apply the arm's intervention to the paired baseline's step-25/50/100 checkpoints; this is deterministic and needs no training. Use as threshold the largest same-metric Δloss of a XIV E control that did not flip (noise σ = 2.5: ≈ 0.18 at step 100). Apply the same check to sha-hi-rolled on E. At the baseline's own parameters β ≈ 0, so it equals vrg's 0.129, but report it.

**A5: declarations still missing from §10.3.**
1. **gsv does not freeze T's share** [proof + num, `logitsplit_gsv.py`].
   - With s_T stop-gradiented and the T term of ctx stop-gradiented, the complement logits get ∂L/∂s_k = a_k(u_k − Σ_C a_j u_j). Their aggregate equals plain's with ū_T set to 0.
   - This signal depends on the gauge: it moves with any common shift of the value vectors, which plain training cancels.
   - The resulting logit-level recruitment of T's share under gsv is 0.6–1.0× plain's on E-own0 and ≈ 0.5× on M at init. At M's baseline step 100 it is ≈ 0.
   - H2g-RV's prediction (gsv = F) holds only for "learning signal through T's edges", not for "T's share is not learned". State this.
2. **Under uniform attention within groups, gsg's complement keeps the fraction n_T/(n_T + n_C).** For own0 the measured fraction is 0.5–0.7 (E init and baseline step 100), not 0.14. Correct the gsg row of the §10.2 table.
3. **Hm (med = P) and H1d + Hm additive both predict P for gsg and gsv,** so those arms cannot separate them; separation rests on the sha arms. **H4 and additive** are separated by band only (N against P) on the sha arms. This is already noted in A7; carry it into §10.3.
4. **Horizon.** The gsg hybrids are competent through step 4000 (KL ≤ 0.017). By steps 3000–4000, however, they are MIXTURE on two of three diagnostics on *both* bases (causal: nec_E 0.10–0.14 ≤ 0.15 and nec_M 0.45–0.48 ≥ 0.30; behavioural: ρ_L 0.76–0.86, lapse ≤ 0.017). They stay HYBRID only through structural lik_RO1 0.19 > 0.15.
   - So the drift runs towards F on E and towards N on M.
   - A P verdict at 1211 is therefore horizon-dependent, in opposite directions on the two bases. Declare this alongside the 2546 stability readout.

**Provenance (required, trivial).** `logs/unit_modes2.json` (17:36) predates the current harness (17:44). It still contains a check of the removed `shc` mode, including `rows_sum_to_1: NaN`. I re-ran it against the current harness (`/tmp/preA18/um2.py`):
- gsv and vrg reproduce exactly: gsv logits vs plain 1.2e-6 / 6e-7; vrg logits vs vrs 0.0; gradient deltas identical.
- `shc` is now a no-op.

Regenerate the log with `shc` removed, so that every unit log matches the frozen harness hash.

### Summary of required changes before the freeze

| id | type | change |
|---|---|---|
| A1 | code (blocking) | warm-start β₀ = logit(A*) − logit(mean σ(l)) in `sha_beta`, plus a unit check in the far-target regime |
| A2 | scorer (blocking) | one symmetric per-pair verdict rule (≥ 5 match → CONSISTENT; ≥ 2 mismatch → CONTRADICTED); per-base verdicts; NaN φ → MISSING; preflight with heterogeneous scenarios |
| A3 | pre-registration (blocking) | split Hm into Hm-window and Hm-release; vrg → med; report latent φ_share and β(99) for sha; declare the non-separation conditions |
| A4 | rule | H3 on E on the same-θ immediate Δloss metric, threshold from same-metric non-flipping XIV controls; also for sha-hi-rolled |
| A5 | declarations | gsv share recruitment via the complement; gsg fraction text; Hm(P) vs additive; horizon drift; regenerate `unit_modes2.json` |

With A1–A3 implemented and the preflight re-run, the design meets mandate §2 Check B. Every frozen hypothesis has a discriminating arm, and the remaining equivalences are declared.
