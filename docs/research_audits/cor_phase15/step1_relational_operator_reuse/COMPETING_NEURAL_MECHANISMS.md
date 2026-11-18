# Competing neural mechanisms: formal statements and discriminating predictions

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). **W3:** P2 does not discriminate H2 from the mandate's H3. Under Theorem P a slot is its code, so the H3 used here is a straw version. **W5, W6:** P1 is a numerical check of Theorem P; P3 re-measures P2 (complement windows, verified to ≈1e-5); P0, P3 and PL replicate development regularities and are not consequences of H1–H4. **W9:** P5 does not exclude H1-with-binding. **W7:** P4's transplant was content-mismatched. The "H4 mixture" conclusion is withdrawn: H2, H3 and H4 are not identified by any XVII intervention. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

**When this was written.** The hypotheses and predictions below were fixed in the XVII freeze, before any fresh confirmation model was trained. They were derived from the architecture (§1) and from **development** analyses of existing XVI models (C_s202, WB_low_s503, WA_const_s402), recorded in §4 and `logs/expl/`.

## 1. What the architecture allows: Theorem P [proof + num]

The GATE Transformer (`step1_latent_reliability/src/nets.py`):
- has bidirectional attention, **no positional encoding and no attention mask**;
- builds question q's tokens as E_val[x] + E_src[s] + E_q[q] (six source tokens) and E_read + E_q[q] (its readout token);
- shares every other parameter across questions.

**Theorem P.** For any permutation π of the six question slots,

  f_{θ_π}(X_π) = f_θ(X)_π,

where X_π permutes the questions' contents and θ_π permutes the rows of E_q.

*Proof.* Self-attention without positional information is equivariant under any permutation of its tokens. The embedding of the permuted input with permuted E_q equals the permuted embedding, and the readout reads token 36 + q. ∎

Checked numerically on three permutations: max |difference| ≤ 1.2·10⁻⁶ (`src/xviislot.theorem_P_check`).

**Consequences** [proof]:
1. **"Physical position" has no meaning in this model.** Moving a question's tokens to other sequence positions changes nothing. What distinguishes slots is only the slot code E_q[q], a 64-dimensional vector, 6 of them in all.
2. **Exchanging two slots' codes is exactly exchanging their contents.** Slot contents are exchangeable under the task, so a slot's gate use must follow its code. Mandate Experiment A ("embedding against content against position") is therefore settled by proof. Its only empirical content is a numerical check (P1).
3. **Every slot difference is a difference in how the shared weights respond to different codes.** "Separate parameter sets" cannot exist. A position-conditioned implementation (H3) can only mean code-conditioned behaviour of shared weights.

## 2. The hypotheses, made operational

**The intervention: identity relabel (lo, hi).** Inside layers lo … hi − 1, slots i and j exchange codes. Each keeps its own content, and the labels are restored afterwards. **No activation, evidence or answer moves between slots.** This is the source-matched control the XVI patching lacked, by construction. The full window equals a code swap (Theorem P).

**H1, shared and uniformly reusable computation.**
- Slot codes play no functional role in the relational computation, beyond slot identity for binding.
- Predicts: no stable partial-slot states (P0 SYNCHRONOUS). If partial states do occur, code exchange in a subset of layers is not specifically needed to transfer gate use. After acquisition, partial-window exchanges do not degrade gate use (P5 NOT_PERSISTENT).

**H2, shared computation with code-dependent access.**
- The shared weights implement a gate computation that engages for some codes and not others. An unacquired slot fails because its code does not engage that computation.
- Predicts:
  - giving the unacquired slot an acquired slot's code, only in the layers where engagement happens, restores gate use (P2 TRANSFER), with no content moved;
  - exchanging codes in the remaining layers matters less (P3);
  - code keying remains after acquisition (P5 PERSISTS).

**H3, distinct position-conditioned implementations.**
- Each slot's working computation is specific to that slot. An unacquired slot needs its own, not-yet-formed implementation, and borrowing another slot's code does not help.
- Predicts: P2 NO_TRANSFER.

**H4, mixtures and non-identifiability.** For example, transfer in some models but not others (MIXED), or transfer only with all layers.

**Routing against computation inside access (P4).**
- If the engaging difference were only attention routing, transplanting the acquired slot's attention pattern would restore gate use (ROUTING_SUFFICIENT).
- If engagement also involves code-dependent value and MLP processing, routing alone is insufficient (ROUTING_INSUFFICIENT), while the code relabel still transfers.

**Learning (PL).**
- Under H2, with the computation already present in the shared weights, training on one unacquired slot need not speed that slot up specifically. Shared-weight updates driven by one slot can also interfere with the others.
- The frozen test is the development pattern below: interference with the acquired slots, and no acceleration of the targeted slot.

**What cannot be distinguished** [proof/argument]:
- A transfer by code relabel shows that the unacquired slot can run the **same** shared-weight computation that the acquired slot runs, given that slot's code in the early layers.
- Whether one calls the code-dependent early processing "access" or "a code-conditioned implementation that any slot can adopt" is a matter of labelling. Both are the same mathematical fact.
- What is testable, and tested, is the strong H3 claim that the unacquired slot could not use the acquired slot's computation.

## 3. Development evidence used to set the predictions [dev]

| model, checkpoint | observation |
|---|---|
| C_s202 (arity 3, XVI restart), replay step 1000 | Per-slot F_G = [1.00, 0.00, 0.00, 0.99, 0.99, 0.00]: slots 0, 3, 4 acquired by step 350; 1, 2, 5 at exactly 0 through step 1000 |
| same: relabel windows, 9 pairs | recipient / donor mean F_G: [0,1) 0.52 / 0.01; [1,2) 0.00 / 0.66; [2,3) 0.00 / 0.96; **[0,2) 0.96 / 0.00**; [1,3) 0.01 / 0.55; [0,3) 1.00 / 0.00 |
| same: layer-1 attention | heads 0–1 route evidence to gate tokens with mass 0.73–0.99 in acquired slots, against 0.01–0.10 in unacquired slots |
| same: routing transplant (rows re-indexed onto the recipient's own tokens) | layer 1: 0.08–0.11; layers 0–1: 0.09–0.19; sham and unacquired donors: 0.00 |
| same: code interpolation toward an acquired code | recipient ≤ 0.17, while the donor falls to 0.13–0.25 (identity confusion). Toward an unacquired code or random: ≤ 0.04 |
| same: training only the unacquired slot's code (1200 steps) | code moved 2.0 (46% of its norm); F_G stays 0.00 |
| same: all parameters, loss on that slot only (1000 steps) | the slot stays at 0.00; the acquired slots fall to 0.56, 0.23, 0.09 |
| C_s202, step 2400 (all acquired) | consistent [0,2) exchange 0.96 / 0.92; mixed [0,1) 0.64 / 0.60; [1,3) 0.60 / 0.61 |
| WB_low_s503 (arity 2), replay step 1740 | slot 1 alone acquired. [0,2): 0.68 / 0.04; [0,1): 0.19 / 0.01; [2,3): 0.03 / 0.65; routing transplant ≤ 0.07 |
| WB_low_s503, step 2500 (all acquired) | every partial window degrades (0.2–0.7); consistent [0,2) gives 0.58–0.73 against mixed [0,1) 0.20–0.34 |
| XVI pooled scan (16 acquisitions) | only C_s202 has a long partial window (1750 steps). Others pass through intermediate pooled F_G within 25–200 steps; per-slot reconstruction of WB_low_s503 shows slots acquiring in sequence over about 300 steps |

**Development suggests H2.** The shared early-layer computation works for the acquired codes, is not routing alone, and is keyed by the code consistently across layers. These are development results from one arity-3 and one arity-2 model, not confirmation.

## 4. Frozen predictions and expectations (score_XVII.py)

| | test (model level) | expectation | falsifying outcome |
|---|---|---|---|
| P0 | partial checkpoint at 20-step resolution in acquiring models | STAGGERED (≥ 75%) | SYNCHRONOUS (≤ 25%) |
| P1 | full-window relabel: recipient ≥ 0.8, donor ≤ 0.2 | FOLLOWS_CODE (≥ 90%) | NOT_ALWAYS (would contradict Theorem P or the measure) |
| P2 | [0,2) relabel: recipient ≥ 0.5 and ≥ control + 0.4 | TRANSFER (≥ 75%) | NO_TRANSFER (≤ 25%): H3 |
| P3 | donor under [2,3) − donor under [0,2) ≥ 0.4 | LATE_LESS_CODE_DEPENDENT | OPPOSITE |
| P4 | routing transplant (layers 0–1) < 0.3 while relabel ≥ 0.5 | ROUTING_INSUFFICIENT | ROUTING_SUFFICIENT |
| P5 | late: some mixing window ≤ 0.8, and consistent [0,2) ≥ mixed [0,1) + 0.15 | CODE_KEYING_PERSISTS | NOT_PERSISTENT: H1 |
| PL | targeted training: acquired slots ≤ 0.5 against ≥ 0.9 under full training, and targeted slot not faster | INTERFERENCE_NO_ACCELERATION | NOT_SUPPORTED |

**Notes:**
- **Replication unit:** the fresh model. Pairs, slots and examples within a model are not independent replications.
- **Reporting:** pooled verdicts are primary. Per-arity verdicts and exact Clopper–Pearson intervals are reported.
- **Expected uncertainty:** P3 is the least certain expectation. At arity 2, development showed the late layer partly code-dependent (donor 0.65).
