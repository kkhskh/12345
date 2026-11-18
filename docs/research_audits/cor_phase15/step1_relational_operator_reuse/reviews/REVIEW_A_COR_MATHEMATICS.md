# Review A: COR mathematics and theoretical identity (Investigation XVII)

**Reviewer.** Agent Reviewer A (mandate §15). This is an agent review, not human peer review.

**Scope.** I reviewed:
- ORIGINAL_COR_OD_AND_POSITIONAL_REUSE.md;
- `scripts/xvii_od_check.py` and `logs/xvii_od_check.json`;
- COMPETING_NEURAL_MECHANISMS.md §1–4 and `scripts/score_XVII.py`;
- Theorem P, checked against `step1_latent_reliability/src/nets.py` and `src/xviislot.py`;
- MAIN_RESEARCH_SYNTHESIS.md answers 1–5, 12–15, 23, 25 and 26.

**Original sources read:**
- `docs/notes/Causal_Obstruction_Response_theory.pdf` and `..._Update.pdf` (via pdftotext);
- `docs/math_map_v0.md` and `docs/theory_mvp_theorem_set.md`;
- `cor_phase1/02, 04, 06, 10`;
- the overnight MATHEMATICAL_RECONSTRUCTION.md;
- COR_TEN_FAILURE_MATRIX.md §2 and `od_bridge.py`.

**Package files.** No package file was modified. Scratch work is in `/tmp/revA17/`:
- `od.py`: independent re-derivation of the Proposition S numbers;
- `lp.py`: exact LP representability check;
- `thp.py`: Theorem P and relabel identities;
- `late.py`: exploratory layer-2 slot-offset test.

**Compute.** CPU only. No training was run. Exact enumeration was done over all 729 inputs.

**Labels:** [proof], [num] (my independent computation), [expl] (my exploratory check, small n), [arg] (argument).

---

## Overall verdict

**Case D (O/D is silent on positional reuse and access) is correct, and I endorse it.** Specifically:
- Nothing in Definitions 1–4, Theorem 1, or Propositions 2–6 (original) / 1–7 (Update) constrains how a network implements or reuses a computation across token positions.
- Proposition 6 (original; Proposition 5 in the Update), the relative extension obstruction, supplies only vocabulary. The analyst would have to choose U, the stalks and ρ from the neural results.
- The non-identifiability propositions (3 in the original; 3 and 7 in the Update) are limitations, not predictions.

The supporting "limited negative property" (Proposition S) is **misstated**. Its headline numbers are **vacuous or wrong for the faithful Phase 1.5 form of D**. A corrected and stronger statement holds (F1, F2).

On the neural side, Theorem P is correct. However:
- P2 does not discriminate H2 from the mandate's H3.
- The window relabels at lo > 0 are not code exchanges, which undercuts the "late layer is code-independent / causal site is layers 0–1" reading (F7, F8).
- Several frozen predictions are development-derived regularities, not consequences of the hypotheses (F9).

---

## Findings

### F1 [major]: Proposition S is false as stated, and the "0 of 729" figure is vacuous or wrong

**Claim** (ORIGINAL §4; synthesis Q4):
- "for any weights … argmin_c D_c is an order-1 rule";
- "Every rule of order ≤ a ignores the gate … so argmin D never responds to a gate flip";
- table column: "inputs on which a gate flip changes argmin D = 0.000"; Q4: "0 of 729 inputs change under a gate flip, in every condition".

**Evidence:**
- The gate-flip column is computed only for D over the evidence sources (3, 4, 5) (`xvii_od_check.py`, `changed = … Pev …`). That D does not read the gate sources at all, so 0 is true by construction and says nothing.
- The faithful Phase 1.5 form (`od_bridge.py`) counts **all** sources of the question. The same table reports that form's accuracy ("all six sources"), but not its flip response. I enumerated every on-support flip exactly, under the same flip rule as `T.gate_flip` [num, `/tmp/revA17/od.py` → `od.json`]. Argmin D over all six sources:
  - changes on 53.5% (a = 2) and 50.6% (a = 3) of flips;
  - has gate-flip following F_G = 0.143–0.149 (a = 2) and 0.223–0.228 (a = 3).
  
  Bayes gives F_G = 1.000 in the same computation.
- "Every rule of order ≤ a ignores the gate" is false for arbitrary rules. The order-1 rule "evidence vote + 2·[x_{a−1} = y]" has F_G = 1/3 in all five conditions [num].
- What XVI V5/V6 actually established is narrower: the **expected-NLL-optimal** order-a rule is gate-source independent, so its F_G = 0. (XVI Review A, Finding 10, says the same.)
- I reproduced the reported accuracies exactly (0.7379 / 0.6076 / … / 0.6806 / 0.5575).

**Correction:**
- Restrict the gate-flip statement to the evidence-only D, and say that it is trivial.
- Replace "never responds to a gate flip" and "every rule of order ≤ a ignores the gate" with the representability statement in F2. Rules of order ≤ a can respond to flips; they cannot track the gate.
- Correct synthesis Q4 accordingly.

### F2 [moderate]: The count form of D is not "the simplest instance" of Definition 3. Corrected and stronger Proposition S′

**Claim** (ORIGINAL §2): source-level D = number of disagreeing sources is "one vertex per source, a scalar stalk, and identity compatibility".

**Evidence** [proof]:
- With scalar germs s_c(v), identity ρ and a connected graph, ker d_c is the constant sections. Then:
  - Definition 3 gives D_c = Σ_v (s_v − s̄)², a variance (a pairwise functional);
  - Definition 4 gives O_c = Σ_e w_e (s_u − s_v)², pairwise disagreement.
  
  Neither is the count.
- The count Σ_v [x_v ≠ c] is the squared distance to one **fixed** section, "every source reports c". It is not the distance to H⁰. It arises only with an added anchor vertex whose value is clamped, which is the relative setting of Proposition 6.
- So Proposition S as proved covers only the anchored count form. It does not cover Definitions 3–4 at source level.

**Corrected statement (Proposition S′)** [proof + num]:
- Take any source-level O_c or D_c with single-source germs, under any graph, linear maps ρ, weights W, metric M, and anchors. It is a quadratic form in the germs, so each is a sum of terms involving at most two sources. Hence −O_c and −D_c are order-≤2 scores.
- Exact LP feasibility over all 567 inputs with a unique Bayes argmax [num, `/tmp/revA17/lp.py`]:
  - No order-1 or order-2 scoring reproduces the GATE Bayes decisions, in any of the four conditions tested (a = 2 and a = 3).
  - Sanity checks: order 3 is feasible at a = 2 and infeasible at a = 3, as expected.
- Therefore no source-level O/D decision rule implements the GATE decision.

This strengthens Case C(i) and covers the actual definitions. Label it "mathematically justified restricted version" only in this corrected form.

### F3 [minor]: The "order-1 optimum" column is the NLL-optimal rule, not the accuracy optimum

**Evidence:**
- In 2:0.9:0.55:0.05, argmin D (evidence) scores 0.681, which is **above** the "order-1 optimum" of 0.668.
- A grid search over evidence weighted votes reaches 0.705, equal to the gate-free optimum's accuracy [num].
- In `xvii_od_check.py`, `acc = … P.argmax(1)` breaks ties by first index, whereas argmin D splits ties.

**Correction:** relabel the column "argmax of the NLL-optimal order-1 rule", or report the accuracy-optimal weighted vote.

### F4 [minor]: Source fidelity, numbering and omissions

**Evidence:**
- **Numbering.** The original PDF has Theorem 1 and Propositions 2–6. The Update has Theorem 1 and Propositions 1–7, where the relative extension obstruction is **Proposition 5**. The package cites "Update §5 … propositions 1–7" but uses the original's numbering ("Proposition 6").
- **Naming.** In the source, Definition 3 (distance to H⁰) and Definition 4 (Laplacian energy) are **both** called "obstruction". "Discrepancy" is later (Phase 1) nomenclature. The O/D mapping used is the conventional one and is acceptable, but this should be noted.
- **Omissions from the recovery table:**
  - Update Proposition 6 (local validity of the response approximation): XVII's interventions add full code-difference vectors, far outside any local region, so COR's response calculus would not apply even if sites were chosen. This reinforces silence.
  - Update Proposition 7 (superposition non-uniqueness): it bears directly on "shared direction" claims as a limitation (non-uniqueness), not as a prediction.
  - Most of the obstruction–response intervention calculus (eqs. 19–26, Algorithm 1) beyond "minimum-energy repair".

**Correction:** fix the numbering, add the two Update propositions to §2/§5, and state why neither changes Case D.

### F5 [moderate]: Case C(ii) and synthesis Q3 overstate what an equivariance axiom would give

**Evidence** [proof]:
- Theorem P is a symmetry of the **parametrised family**: θ ↦ θ_π permutes the rows of E_q. It is not a symmetry of a trained θ.
- For a fixed trained network, an analyst's slot-equivariant ρ and P do **not** make O and D slot-invariant. The acquired/unacquired asymmetry is exactly the failure of within-network symmetry.
- Even granting such an axiom, the consequence is invariance of O and D across (θ, θ_π). It is not a prediction that a network reuses a computation.

**Correction:**
- C(ii): "O and D computed for θ on X equal those for θ_π on X_π". This holds across networks and is silent about any one trained network.
- Q3: "no assumption set within or near O/D yields a reuse prediction. An equivariance axiom on ρ and P would be new mathematics and would still only give cross-network invariance."

### F6 [no defect]: Theorem P is correct. No hidden position dependence in the forward pass

**Evidence:**
- In `nets.forward` there is no positional encoding, no mask, per-token LayerNorm, and readout at tokens 36 + q.
- E_src[s] encodes the source role within a question and is identical across slots.
- The episode token (E_ep) attends symmetrically and does not affect the answer except through attention, which is equivariant.
- The proof is complete: the embedding of (X_π, θ_π) is a token permutation of the embedding of (X, θ).

**Independent check** [num, `/tmp/revA17/thp.py`, A2_s705 step 1400]:
- 5 random permutations: max |Δ| = 4.3·10⁻⁶ (float32 noise; the package reports 1.2·10⁻⁶ on other inputs).
- Full-window relabel = E_q row swap: 6.3·10⁻⁶.
- E_q row swap = content swap with outputs relabelled: 3.6·10⁻⁶.

**Notes:**
- Slot symmetry is broken only by the initial E_q rows and sampling noise during training.
- Consequence 3 ("H3 can only mean code-conditioned behaviour of shared weights") is correct. It has a corollary the package underuses; see F8.

### F7 [major for interpretation; the P3 verdict itself stands]: Relabel windows with lo > 0 are not code exchanges

**Claim:**
- COMPETING §2: "Inside layers lo … hi−1, slots i and j exchange codes";
- P3 reading "late layer less code-dependent";
- synthesis Q11 "causal site is shared: layers 0–1" and Q13 "the code in layers 0–1 is necessary and sufficient".

**Evidence:**
- `xviislot.forward` adds only the raw difference E_q[j] − E_q[i] at layer lo, and subtracts it after hi − 1.
- Code-derived features written by earlier layers stay in the residual. For example, values copy E_q through W_V W_O.
- Slot identity is linearly decodable with **accuracy 1.000** from the layer-2 input after projecting out span(E_q) [num; A2_s705 and A3_s804].
- A window starting at layer 0 changes the code everywhere downstream. A window starting at layer lo > 0 changes only the raw additive component. The windows are therefore not commensurable.

**Exploratory test** [expl, `/tmp/revA17/late.py`, 600 episodes, 2–3 pairs per model]. At the layer-2 input, I exchanged each slot's mean residual offset (the content-averaged per-token state, which includes code-derived slot identity) instead of only the raw code:

| model | pair | raw-code swap at layer 2: donor F_G | slot-offset swap at layer 2: donor F_G | acquired ↔ acquired control (offset swap) |
|---|---|---|---|---|
| A2_s705 | 4→0, 5→1 | 0.79, 0.93 | **0.21, 0.21** | 0.96 / 0.87 (4↔5) |
| A3_s804 | 1→0, 4→2 | 0.78, 0.61 | 0.55, 0.56 | 0.83 / 0.82 (1↔4) |

- The acquired↔acquired control shows the offset swap is not destructive in general.
- So the late layer's gate use **does** depend on slot-specific state carried from early layers: strongly in A2_s705, partly in A3_s804. It is insensitive mainly to the raw additive code.

**Caveats:**
- The offset includes content-averaged products of early computation, so it is not pure "code".
- n is small and the test is exploratory.

**Correction:**
- Reword P3 and Q11 as: "after early layers have processed the original code, re-adding the other slot's raw code at layer 2 matters little".
- Do not claim the late layer is code-independent, or that the causal site is confined to layers 0–1.
- P2's sufficiency claim ("the acquired code at the embedding, through layers 0–1, suffices") is unaffected.

### F8 [major]: P2 TRANSFER does not discriminate H2 from the mandate's H3; synthesis Q12–14, Q23 and Q25 contradict Q15

**Evidence** [arg + proof from Theorem P]:
- By Theorem P, the code is the **only** slot-specific ingredient. The mandate's H3 explicitly allows transfer after "replacing additional implementation-specific components". Giving the recipient the donor's code in layers 0–1 replaces exactly the component in which a code-conditioned implementation would live.
- So TRANSFER is predicted equally by:
  - H2: shared computation, code-keyed access;
  - H3′: code-conditioned early implementations, some formed and some not, feeding a late stage that is compatible across codes.
- The package's H3 ("borrowing another slot's code does not help") is a straw version. Under Theorem P, the full window must transfer. A [0,2) failure could arise only from late-layer code specificity, so P2 tests **where** code dependence sits, not H2 against H3.
- COMPETING §2 ("What cannot be distinguished") and synthesis Q15 concede this. But the synthesis also says:
  - Q12: "evidence for a common relational computation? Yes";
  - Q13: "evidence for position-dependent access? Yes";
  - Q14: H3 "rejected";
  - Q23: "strong non-transferable per-slot implementations ruled out";
  - Q25: "'the computation exists' and 'a position can use it' separate cleanly".
  
  These treat the indistinguishable descriptions as distinguished.

**Correction:**
- Q12/Q13: "P2 shows that an unacquired slot's content is processed correctly when the acquired slot's code drives layers 0–1. Whether to call this a shared computation with code-keyed access, or an adoptable code-conditioned implementation, is not identified [Q15]."
- Q14/Q23: "only the straw H3 (code transfer cannot help) is rejected".
- Q25: delete "separate cleanly".

### F9 [moderate]: Several frozen predictions do not follow from their hypotheses

**Evidence:**
- **P0 SYNCHRONOUS ↦ H1.** The mandate's H1 explicitly allows "temporary optimization or performance differences", so staggering does not bear on H1.
- **P1** is Theorem P. It has no hypothesis content; it is a numerical check, which the package correctly says.
- **P3** and **PL** are taken from development observations. COMPETING §2 says H2 "need not" speed the slot, which is not a prediction.
- **PL interference** (other slots degrade under single-slot loss) is expected under every shared-weight hypothesis, as ordinary interference. Its no-acceleration half was at ceiling (synthesis Q22 discloses this).
- **P5 NOT_PERSISTENT ↦ H1 is one-directional.** Under H1 the code still binds tokens to slots, and value paths copy E_q into the residual (F7). Raw-code mixing can therefore degrade gate use without any code-specific relational processing. My acquired↔acquired raw swaps, between two slots that both already use the gate, degrade it:
  - A2_s705: layer-1 raw swap 0.52 / 0.37; layer-2 raw swap 0.62 / 0.97;
  - A3_s804: layer-1 raw swap 0.21 / 0.78 [expl].
  
  So PERSISTS does not exclude H1-with-binding.
- `score_XVII.py` implements the stated rules faithfully (P1 cannot FAIL by construction, `lo = -1`; P4's fallback logic is correct). The issue is inferential, not computational.

**Correction:** in COMPETING §4 and the synthesis, mark P0, P1, P3 and PL as replication-of-development or architectural checks, not hypothesis tests. Mark P5 as "consistent with H2/H3′, does not exclude H1-with-binding".

### F10 [moderate]: The "shared relational operation" is not identified as an object

**Evidence:**
- Operationally, the "shared operation" is f_θ(·, code = c_A in layers 0–1). The "observational equivalence" claim (COMPETING §2; Q15) is correct, and in fact stronger than stated: in this architecture the two descriptions denote the **same mathematical object**, so they are equivalent by definition, not merely under these interventions.
- H2 could be given distinct, testable content, but none of these was tested:
  - code engagement factors through a low-dimensional subspace of code space;
  - the gate computation is present (decodable) but unused in unacquired slots;
  - a code-independent subcircuit is switched by a low-rank code signal.
- Q11 correctly states that no shared direction was tested.

**Correction:** state in Q15 that the equivalence is definitional in this architecture, and list the untested refinements that would give H2 separate content.

### F11 [minor]: Distinctness from Bayesian inference and conditional computation

**Evidence:**
- The package's own classification is correct: "ideal decision identical at every slot" is trivial exchangeability, and the mechanism is ordinary conditional computation (an additive, FiLM-like code).
- Nothing in XVII is a distinct theoretical construct beyond these. Q5 and NOVELTY say so.
- Only Q25's wording (F8) overreaches.

### F12 [minor]: Synthesis wording

- **Q4.** Answering "Yes" to "do counterexamples undermine the connection" presupposes a connection that Case D denies. Better: "there is no connection to undermine. Corrected Proposition S′ adds that source-level O/D cannot represent the GATE decision."
- **Q26.** Correct once Proposition S is replaced by S′.

---

## Surviving claims

1. **Case D:** the original O/D framework is silent on positional reuse, access and acquisition [proof by inspection of the definitions and propositions]. No XVII result confirms or refutes O/D.
2. Source-level O/D cannot represent the GATE decision, in the corrected form S′. Any single-source-germ O_c or D_c is an order-≤2 score, and no order-≤2 score reproduces the Bayes decisions at a = 2 or 3 [proof + exact LP, num]. The count-form D over the evidence sources is trivially gate-independent [proof].
3. The recovery of Definitions 1–4 (notation, the analyst-chosen graph, P, ρ, M, W), the P = 0 vacuity, O ≠ D, and the failed causal bridge is faithful, apart from the F4 corrections.
4. The relative extension obstruction is vocabulary only. Using it would be a new, circular, COR-inspired hypothesis [arg]. Correctly not pursued.
5. **Theorem P** and its consequences 1–3 [proof; independently verified, num]. "Physical position" is meaningless in this architecture.
6. The relabel moves no evidence or answer between slots, so P2 is source-matched by construction [proof].
7. P2's operational content: the acquired code entering at the embedding and active through layers 0–1 suffices for an unacquired slot to follow its own gate. Its interpretation as H2 over H3 does not survive (F8).
8. P4: attention-routing transplant alone is insufficient [frozen; not re-examined here].
9. The "access" versus "adoptable implementation" equivalence (Q15) survives, strengthened to definitional identity in this architecture.
10. The interpretation is ordinary conditional computation with Bayesian-exchangeable ideal decisions. No new operator was introduced silently. Proposition S is the only new mathematical object, and it was labelled, but it needs the F1/F2 correction.
