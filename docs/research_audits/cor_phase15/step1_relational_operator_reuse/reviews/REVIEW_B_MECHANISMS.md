# Review B: neural mechanisms and causal inference (Investigation XVII)

**Nature of this review.** This is an agent review (Reviewer B, mandate §15), not human peer review. The mandate asks this reviewer to reject undercontrolled patching claims aggressively.

**Scope.** I read `src/xviislot.py`, `scripts/xvii_confirm_analysis.py`, `scripts/xvii_slot_train.py`, `scripts/xvii_learning_arm.py`, `scripts/score_XVII.py`, all `logs/analysis/*.json`, `learning/*/slot_traj.json`, `logs/score_XVII.json`, `logs/xvii_battery_summary.txt`, the `logs/expl/` development files, and the seven reports named in the brief.

**Independent checks.** I recomputed every model-level number in the battery summary and the score from the raw analysis JSONs, and they match. I then ran new forward-pass checks on the fresh checkpoints (CPU only), plus two 200-step training controls. Scratch scripts and outputs are in `/tmp/revB17/`:
- `checks.py` → `checks.json`
- `kl.py` → `kl.json`
- `l2id.py` → `l2id.json`
- `l2b.py` → `l2b.json`
- the P5 control → `p5ctrl.json`
- the training controls `A3_s804_targetA1/` and `A2_s705_targetA4/`

I modified no package file other than this review.

**Severity scale.**
- **critical:** a headline claim does not follow.
- **major:** a frozen verdict's interpretation is wrong or unsupported.
- **moderate:** the claim is overstated or a control is missing.
- **minor:** wording or bookkeeping.

---

## Findings

### B1 (critical). Under Theorem P, "an unacquired slot can run the acquired slot's computation" is close to a tautology. P2 does not discriminate H2 from H3, and it does not show reusable relational computation.

**Evidence.**
- *What a slot is.* The package proves (Theorem P) that a slot is nothing but its code E_q[q]. Contents are i.i.d. across slots. The full-window relabel [0,3) is exactly "slot i's computation applied to slot j's content". So the claim that an unacquired content can be gated under the acquired code holds by construction (P1, 0.87–0.99).
- *What P2 adds.* The [0,2) relabel differs from [0,3) only in **one** respect: the raw E_q component at the input of layer 2 is set back. The derived slot identity is not set back:
  - At the layer-2 input I projected the residual off span(E_q). Readout-token slot identity is then still decodable by nearest centroid with accuracy 0.92–1.00 (4 models).
  - After the [0,2) relabel and its undo, the recipient's layer-2 input decodes as the **donor** slot in 96–100% of episodes, and its final-layer output still does in 90–100% (`checks.json`, `L2in_/L3in_relabel_recipient_decoded_as_i_frac`).
  - The raw code is only 32–43% of the norm of the layer-2 input.
  - So "the code in layers 0–1 only" is really "the donor's identity everywhere except the raw code vector in layer 2".
- *What makes a code "engaging".* Giving the recipient the donor's code **without exchange** (both slots then carry code i) does not transfer, even over the full window: 0.08–0.19 in [0,2) and 0.15–0.22 in [0,3) (`checks.json`, `dup_code_*`). The code works only as a **unique binding label**. "Engaging code" therefore cannot be separated from "being slot i". The package's own interpolation result in development makes the same point.
- *H3 is excluded by construction.* COMPETING_NEURAL_MECHANISMS.md §1, consequence 3 concedes that H3 "can only mean code-conditioned behaviour of shared weights". That is the same object as H2. The "strong H3" that P2 rejects ("borrowing another slot's code does not help") is already ruled out by Theorem P plus P1.

**Correction.**
- Delete the claims that P2 is evidence for "a common relational computation that every slot can use" (Synthesis Q12, Q14, Q18, Q23 "strong non-transferable per-slot implementations") and that it is a rejection of H3.
- Restate P2 descriptively: at partial checkpoints, the network gates for whichever content carries an acquired slot label in layers 0–1. Re-setting only the raw code vector at layer 2 leaves most of this intact. The unacquired slots' deficit lies in how the shared weights process their identity, not in their content. Their content is exchangeable by construction, so it could never have been the cause.

### B2 (major). P3, and half of P5's windows, are the same measurement as P2, rescored. They are not independent confirmations.

**Evidence.**
- *The identity.* Because the layers are permutation-equivariant, a relabel over [lo, L) on input X equals π ∘ (relabel over [0, lo)) on πX. I verified this to 4.6·10⁻⁶ – 9.2·10⁻⁶ in max |logit difference| in 4 fresh models (`checks.json`, `complement_maxdiff`).
- *P3 is P2.* The P3 statistic (donor under [2,3)) is the P2 recipient statistic on content-swapped episodes. The raw data show it:

  | model | donor under [2,3) | recipient under [0,2) |
  |---|---|---|
  | s802 | 0.60 | 0.65 |
  | s803 | 0.90 | 0.90 |
  | s804 | 0.79 | 0.78 |
  | s704 | 0.70 | 0.68 |
  | s705 | 0.82 | 0.83 |
  | s707 | 0.79 | 0.79 |

- *P5's windows are complement pairs.* [0,1) and [1,3) are complements, and so are [0,2) and [2,3). The "min over {[0,1), [1,3)}" in P5 is therefore redundant.
- *P1 is not an empirical test.* It is Theorem P checked numerically.
- *The real count.* "Every frozen mechanism prediction held, 6/6" counts P1 (a tautology) and P3 (a duplicate of P2). The empirically independent mechanism tests are P2, P4, P5 (mixing) and PL.

**Correction.**
- Report P3 as a restatement of P2, and P1 as a numerical check.
- Recount the frozen record as 4 independent mechanism tests plus P0.

### B3 (major). The "layer locus" (P3, Synthesis Q11 "causal site layers 0–1 for every slot", Q13 "necessary and sufficient") is not identified by code-window interventions, and it is contradicted in one fresh model.

**Evidence.**
- *Why the code windows cannot localise.* The [2,3) and [0,2) windows move only the raw E_q vector. Slot identity at layer 2 is mostly carried by derived features (B1): non-Eq slot-centroid differences are 66–92% of the layer-2 residual norm. So "layer 2 tolerates the other code" may only mean that layer 2 reads identity from derived features.
- *Causal check 1: restoring the derived identity.* After the [0,2) relabel, I also moved the recipient's derived identity at the layer-2 input back from i to j (a non-Eq centroid shift). Transfer fell:

  | model | transfer before | transfer after |
  |---|---|---|
  | s804 | 0.79 | 0.43 |
  | s803 | 0.86 | 0.62 |
  | s705 | 0.83 | 0.28 |
  | s704 | 0.72 | 0.21 |

- *Causal check 2: a layer-2-only identity swap.* I swapped raw code plus derived identity at the layer-2 input only, with no change in layers 0–1. In **A2_s704** the unacquired slot gained gate use:
  - 0.65–0.70 over 3 pairs;
  - against 0.06–0.08 for equal-norm random shifts;
  - against a baseline of 0.12 (`l2b.json`).

  That passes the P2 threshold with layers 0–1 untouched. In A3_s804, pair 1→3 reached 0.35 (random 0.04).
- *Caveat.* Centroid shifts are crude and off-manifold. Large random shifts also hurt donors (0.31–0.75), so the donor-side numbers are confounded. The recipient-side gain in A2_s704 is not: random shifts do not create Bayes-correct gate following.

**Correction.**
- Withdraw "the causal site is layers 0–1 for every slot" and "late layer largely code-agnostic".
- State instead that the raw code vector matters mostly in layers 0–1, and that the layer at which an unacquired slot is deficient varies across models (A2_s704 is deficient at layer 2).
- A residual-subspace identity intervention per layer is needed to localise.

### B4 (major). The routing transplant is not content-matched, so P4's "routing insufficient" is overstated, and routing is in fact necessary.

**Evidence.**
- *The implementation is correct as described.* In `_route`, the donor rows are column-permuted so that the donor's weights on its own tokens land on the recipient's tokens. Advanced-index shapes (N, |heads|, |rows|, T) are consistent.
- *But it imports the donor's content-dependent attention.* Layer-0 and layer-1 attention depends on token values: developmental layer 1 "routes evidence to gate tokens", which is a content-conditional pattern. That pattern is pasted onto different content.
- *A matched test.* I took the recipient's **own** attention rows from the [0,2)-relabel run (same input, acquired label) and inserted them into the unperturbed run, in layers 0–1, all heads:

  | model | matched transplant | package transplant |
  |---|---|---|
  | s804 | 0.32 | 0.07 |
  | s803 | 0.47 | 0.05 |
  | s704 | 0.11 | 0.06 |
  | s705 | 0.27 | 0.10 |

  Under P4's own threshold (< 0.3), 2 of these 4 models would fail ROUTING_INSUFFICIENT.
- *Routing is necessary.* Running the [0,2) relabel while forcing the recipient's attention to its baseline pattern cuts transfer to 0.02 (s804), 0.03 (s803), 0.32 (s704) and 0.15 (s705) (`checks.json`).

**Correction.**
- Report routing as **necessary** for transfer, and **partly sufficient** when the transplant is content-matched (0.11–0.47).
- Downgrade "attention routing alone does not carry it" (Synthesis, Stage 2; Q10; Q13) to: "transplanting the donor's content-specific attention does not carry it; matched routing carries part of it."

### B5 (major). PL's "interference" is generic single-slot forgetting, not evidence about shared computation or slot-specific maintenance.

**Evidence.**
- *The missing control.* The decisive control is to train on an **acquired** slot's loss only. The package never ran it. I ran it with the package script (`xvii_slot_train.py`, MODE all, same optimiser and data seed, 200 steps):
  - A3_s804, loss on acquired slot 1: the other acquired slot 4 fell from 0.87 to 0.34 at step 100 and 0.23 at step 200. That is the same as in the frozen targeted arm (0.28, then 0.34).
  - A2_s705, loss on acquired slot 4: slot 5 fell from 0.96 to 0.60 and then 0.61. The frozen targeted arm gave 0.33 and then 0.41.
- *The unacquired slots stay unacquired in this arm too.* In A3_s804 they reach 0.01–0.20, although under full training they acquire within 40–80 steps.
- *The cause is optimiser drift, not the targeted slot.* The arms restart with a fresh AdamW at 1e-3. Adam's normalised steps move the shared weights substantially even when the trained slot is already solved: shared-weight drift was 2.8–4.2 in 200 steps. Slots with zero loss get no restoring gradient. This is ordinary catastrophic forgetting.
- *Why U slots stay low.* A slot with zero loss gets no gradient on its own identity pathway, so its failure to acquire is expected under every hypothesis considered.
- *No acceleration is shown.* "No acceleration" was at the ceiling (acknowledged). Yet CROSS_SLOT_LEARNING_INTERVENTIONS.md Q17 says "yes, targeted pressure accelerates one slot". Nothing shows acceleration: the slot was 40–220 steps from natural acquisition anyway.

**Correction.**
- Report PL interference as non-diagnostic: any single-slot objective causes it.
- Remove "acquisition and maintenance of access are slot-specific" as evidence for H4 (Q16–Q18).
- Correct Q17 to "not testable here (ceiling)".

### B6 (moderate). P5's "code keying persists" rests on off-manifold hybrid states, and it rejects only a strawman H1.

**Evidence.**
- *A random-code control.* At the late checkpoints (A3_s804 and A3_s803), replacing a slot's code in layer 0 only with a random vector of matched norm drops F_G to 0.00–0.03 (3 draws each). A half-strength exchange gives 0.08–0.23. The full exchange the package used gives 0.32–0.40 (`p5ctrl.json`).
- *What that means.* Layer 1 tolerates another **trained** code in layer 0 much better than any other perturbation. That is partial cross-code generality, not keying.
- *Why H1 is a strawman.* With random, near-orthogonal codes entering nonlinear layers, exact code-invariance ("no code-specific processing") is not a live hypothesis. The [0,1) mixing state never occurs in training.

**Correction.**
- Report P5 descriptively: layers 0 and 1 must see a consistent trained label for full gate use; mixed labels keep 0.26–0.62.
- Do not present P5 as ruling out "uniform reuse".

### B7 (moderate). Imminence: every frozen selected checkpoint lies 20–220 steps before the unacquired slots acquire naturally, and the imminence rebuttal is a non-sequitur.

**Evidence.**
- *Gap from the selected checkpoint to natural acquisition of U* (from the `slot_acquisition_step` fields):

  | model | gap (steps) |
  |---|---|
  | s704 | 20 |
  | s705 | 40–220 |
  | s707 | 40–60 |
  | s802 | 40–60 |
  | s803 | 80 |
  | s804 | 40–80 |

- *The rebuttal does not address the confound.* FROZEN_CONFIRMATION §7.3 argues that the U→U control "shows the transfer is not mere imminence". But the U→U control tests whether any exchange helps. It says nothing about whether the recipient's downstream processing was already nearly ready.
- *Imminence does matter.* B3 shows it matters in at least one model (A2_s704: a layer-2-only swap suffices).
- *The only non-imminent case is development.* That case is C_s202 at step 1000, about 1000 steps before acquisition, with transfer 0.96. It is a development model, on a learning-rate-restart history.

**Correction.**
- Say that the frozen confirmation tests transfer only at checkpoints close to acquisition. Far-from-acquisition transfer rests on one development model.

### B8 (moderate). The P2 control for |U| = 1 is the baseline. My new controls mitigate this.

**Evidence.**
- A2_s704 and A3_s803 have no perturbation control in the frozen battery.
- I added a one-sided norm-matched random code in [0,2):
  - A3_s803: 0.01–0.05;
  - A2_s704: 0.01–0.07;
  - also A3_s804 0.00–0.01 and A2_s705 0.01–0.04 (5 draws each).
- The package's own window controls ([0,1), [1,2), [2,3) recipient) are also perturbations of the same size, and they give low values.

**Correction.**
- Add these controls to the record.
- In Synthesis Q8, do not describe the baselines as "controls of 0.01–0.12". The perturbation controls are 0.00–0.07.

### B9 (minor). The transfer is not a confidence or sharpening artifact.

**Evidence.**
- *KL and accuracy* in A3_s804, pair (1, 0):
  - the recipient's KL to Bayes falls from 0.272 (the gate-free floor) to 0.124 under [0,2) and 0.083 under [0,3);
  - its accuracy rises from 0.78 to 0.84;
  - its agreement index A_p moves from 0.68 to 0.43, toward the Bayes rule.
- *A2_s705, pair (4, 0):* KL falls from 0.113 to 0.061, and A_p from 0.64 to 0.22.
- *Other slots are unchanged* (`kl.json`).
- *F_G is argmax-based* and counts only decisive Bayes flips.
- *Caveat.* [0,2) recovers about 78% of the full-swap KL gain, so "transfer" is partial.

**Correction.** None beyond quoting KL alongside F_G.

### B10 (minor). Power, dependence and selection in F_G.

**Evidence.**
- *Power.* With 600 episodes there are 155–245 decisive flips per slot, so the binomial SE is at most 0.04. That is adequate for the 0.05-against-0.8 contrasts.
- *Dependence.* Every analysis in a condition reuses one diagnostic set (common random numbers). A and U are selected on the same episodes as the battery.
- *Inflated pair counts.* The U→U control depends only on (j, k), yet it is recorded once per donor. In A2_s707 the values [0.07, 0.01, 0.04] repeat three times.
- *A2_s704 is not a clean partial state.* Two "neither" slots sit at 0.72, and U's F_G is 0.12.
- *P0 is resolution-limited.* Spreads of 20–40 steps at a 20-step snapshot spacing cannot separate synchronous acquisition from fast staggering. With 600 episodes, threshold crossings near 0.8 and 0.2 are noisy at the 1-snapshot level.

**Correction.**
- Report pair counts net of duplicated controls.
- Describe P0 at arity 2 as "unresolved at 20-step resolution".

### B11 (minor). Wording about what moves.

**Evidence.**
- "No activation … moves between slots" is literally false. Layer-0 cross-slot attention mass is 0.18–0.80 (`dev_attn_C_s202_1000.json`), so every slot's state mixes other slots' content in every run.
- Each slot's Bayes posterior depends only on its own row (`xvitask.bayes`), so no **task evidence** about the recipient's answer moves. The claim about evidence transfer survives.

**Correction.** Say "no slot's input content is copied to another slot, and no task-relevant evidence moves".

### B12 (moderate). The H4 "mixture" conclusion and the identifiability statements.

**Evidence.**
- *The "reusable computation" half* is B1: architectural.
- *The "slot-specific learning and maintenance" half* is B5: generic optimiser forgetting plus zero gradient.
- *The labels themselves are not identified.* Q15 correctly states that H2 and "an adoptable code-conditioned implementation" are observationally equivalent. By the package's own Theorem P consequence 3, H3 (non-strong) is the same object as well. No intervention in XVII can distinguish H2, H3 and H4 as defined.

**Correction.**
- Replace "H4 mixture" with the descriptive statement in "Claims that survive" below.
- Extend Q15: H2, weak H3 and H4 are not identified. Only "is the unacquired slots' deficit pre-layer-2 in raw-code terms" is tested, and B3 shows even that is model-dependent.

---

## Claims that survive

1. **Theorem P** (exact equivariance; slot = code) and its numerical check [proof + num].
2. **Partial-slot states are real.** Some slots run the Bayes rule while others sit at the gate-free floor in the same network:
   - stark and long in C_s202 [dev];
   - short (80–160 steps) in all 3 acquiring fresh arity-3 models;
   - unresolved at arity 2 [frozen P0 MIXED].
3. **Relabelling the unacquired slot as an acquired slot in layers 0–1 produces Bayes-directed gate use in 6/6 fresh models (0.65–0.90).**
   - It is not a generic-perturbation effect: U→U exchange gives 0.01–0.06, a random code 0.00–0.07, and other windows are low.
   - It is not a confidence effect (B9).
   - The deficit of unacquired slots therefore lies in how the shared weights process their identity. The scope limits of B1, B3 and B7 apply.
4. **Giving an acquired slot an unacquired label in layers 0–1 abolishes its gate use** (0.02–0.17).
5. **Routing.** Transplanting the donor's content-specific attention does not restore gate use. Matched routing is necessary for the transfer and partly sufficient (B4, this review).
6. **After acquisition, inconsistent labels between layers 0 and 1 degrade gate use** (0.26–0.62), much less than a random code does (B6).
7. **Learning.** A single-slot objective, on an unacquired or an acquired slot, makes other slots lose gate use, and unacquired slots without loss do not acquire it. This is descriptive and non-diagnostic (B5).
8. **No input evidence or answer for the recipient is transferred** by the relabel (B11).

## Recommended verdict adjustments (for ADJUDICATION.md)

| | as filed | recommended |
|---|---|---|
| P1 | FOLLOWS_CODE | keep, as a numerical check of Theorem P, not a mechanism confirmation |
| P2 | TRANSFER | keep the observation; drop the "reusable relational computation" and "H3 rejected" interpretation |
| P3 | LATE_LESS_CODE_DEPENDENT | not independent of P2; the layer-locus reading is withdrawn (B3) |
| P4 | ROUTING_INSUFFICIENT | qualify: the transplant was content-mismatched; matched routing is necessary and partly sufficient |
| P5 | CODE_KEYING_PERSISTS | report descriptively; it does not exclude reuse (B6) |
| PL | INTERFERENCE_NO_ACCELERATION | interference is non-diagnostic (acquired-slot control); no acceleration is untestable (ceiling) |
| Synthesis | "H4 mixture" | replace with the descriptive claims above; H2, H3 and H4 are not identified |
