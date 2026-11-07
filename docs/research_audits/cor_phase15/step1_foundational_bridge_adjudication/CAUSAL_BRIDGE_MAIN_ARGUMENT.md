# The causal-bridge argument

Phase 1.5, Step 1. Proofs are in `proofs/PROOFS.md`. Checks are in `tests/test_bridge.py` (26 tests; log in `logs/`). Independent reviews are in `reviews/B, C, D, F` and their adjudication in `ADVERSARIAL_CROSS_EXAMINATION.md`. No pretrained model was run.

**Verdict:** `STEP1_NECESSARY_CONDITIONS_ONLY`. In the investigated classes, the consistency hypothesis reduces to ordinary normalization effects. The original `O`/`D` mechanism is unsupported and should be retired (§11).

## Summary

Steps 1A–1E tried to repair a residual between *assigned* states (`O`, `D`). This step kept the question and replaced the object. It looked for a relation among **architecturally canonical variables**: input tokens, component writes, attention weights, and the frozen readout. These involve no fitted projector, transport, metric, or graph, so the identification problems of Phase 1 do not arise.

With that object, the investigation obtained exact results:

- **where** in a residual Transformer one source's effect on a decision can depend on other sources;
- **through which operations** it does so;
- **what information** fixes its sign.

The answer is narrow, and it is negative for the consistency hypothesis.

1. Relations among writers into one additive cut are causally inert. This is an exact impossibility (SS).
2. Inside a softmax head, other sources enter a source's direct term **only through attention reallocation**: `(w'_j − w_j)(v'_j − M)` (CI.3*). The relation that then fixes the sign, `v'_j − M`, is the source's direct-logit attribution minus the head's. It is a readout-defined quantity, and every head satisfies the identity by algebra.
3. The final-normalization channel is a temperature together with a pull toward the bias ranking `κ` (NC). Its "alignment" reading is the norm of a sum, so by SS it is not a relation. In RMSNorm models it cannot flip any pair.
4. The metric-free test for a relation having *any* causal role (RB) is a generic nonremovable-interaction test. It cannot tell consistency apart from any other nonlinearity.
5. The decision-level step (assumption A4: the total effect has the sign of the direct term) is close to a restatement of the prediction. It is not an independent structural premise.

So **no noncircular, distinctive consistency–decision bridge survives**. What survives are necessary conditions and impossibility boundaries that any future consistency mechanism must satisfy. They are sharper and better identified than those of Steps 1B and 1C, but they are not a bridge.

---

## 1. Scientific question

Fix a frozen Transformer, a read position, and a decision scalar: the pair margin `m = L_c − L_{c'}` with a named competitor. A consistency–decision connection is a relation `C` among internal or input variables with three properties: it is identified without the decision response; it imposes a nontrivial, falsifiable restriction on `R = m(do I) − m` for a stated family `I`; and that restriction is neither "`m` moves as `∇m` or the readout says" nor true because `C` was built from `m`.

Three kinds of answer are kept separate: (E) an explicit mechanism, (P) a causal predictor, (N) a necessary condition.

## 2. Computational interpretation: what is consistent with what?

Candidate relations among canonical variables, and what each turned out to be:

| Relation | Is it a consistency relation? | What it turned out to be |
|---|---|---|
| (A) source–source agreement (task-semantic labels) | yes, by definition | causally relevant only through an operation that reads the sources separately (SS); by itself it does not determine the sign (FIB.1); invisible to RB (RB.2) |
| (B) source–consensus deviation `v'_j − M` in a head | nominally | DLA(source) − DLA(head) on the readout `ũ`. It enters only multiplied by a change in attention weight (CI.3*). Every head obeys this by weighted-mean algebra. It is not a dedicated operation that responds to agreement |
| (C) "writer alignment" `Σ c̃_k^T c̃_l` | no | it is `‖PΣc_k‖²`, the norm of the sum; screened by SS. It is the confidence and temperature channel |
| `O`, `D` (Phase 1) | nominally | surrogate error of fitted cochains (1A Theorems Z and App) |

**What is consistent with what?** The only nontrivial answer is (B): a source with the weighted consensus of one head. The head computes the consensus explicitly. The deviation matters only when the intervention changes the source's attention weight.

## 3. Identification

All variables used are deterministic functions of input and frozen weights. They need no fitted map, so manufactured agreement (Phase 1 Attack E; Sutter et al. 2025, Theorem 1) cannot occur. The scalars `w_j`, `v_j`, `M`, `a`, `σ`, `κ`, `m` and the *centered* writes are invariant under the architecture's function-preserving symmetries. The raw writes `c_k` are not: shifting `W_O` along `1` changes them and their cosines while leaving every logit fixed (review B, M3). RB's verdict is invariant to monotone transforms of one fixed decision scalar (RB.2). The scalar itself must be frozen, because with more than two candidates the probability is not monotone in the pair margin (review B, S1).

Remaining design choices that must be frozen (review C, SERIOUS-5): candidate pair and competitor; read position; evidence slots; token lists per level, including the neutral token, whose salience decides the sign in Construction D; contexts; layer and head set; number of levels; tie tolerance; the train and held-out split for RB.6; and the rule for combining tables across contexts. These are choices, not fitted parameters. Because about 86% of random 3×3 tables are relation-sensitive (review C, probe d), searching over them finds crossovers almost surely. Freezing them is therefore mandatory, not cosmetic.

**Identification succeeds. Relevance fails.** The identified relations are not consistency relations with a role of their own.

## 4. Formal assumptions

| ID | Assumption | Status |
|---|---|---|
| A1 | block outputs are sums of component writes | architecture fact (GPT-2, Pythia parallel blocks, LLaMA) |
| A2 | final LayerNorm with `ε > 0` and bias; linear unembedding | holds for GPT-2 and Pythia. Fails for RMSNorm models, where `κ ≡ 0` |
| A3 | evidence slots independently manipulable by substitution | design |
| A4 | sign(total effect on `m`) = sign(direct numerator term of the named heads) | **rejected as a bridge premise.** It asserts the response sign up to an identity (review C, FATAL-4). Even in one attention-only layer the `σ` channel can reverse it: in random models, 591 of 4000 substitutions (review B, S3). Downstream re-readers can also reverse it (D′; backup and negative heads, self-repair; review D) |

## 5. Mathematical results (all proved; see `THEOREM_REGISTER.md`)

- **SS (impossibility).** Same-cut relations are inert given the sum. Equal relation is compatible with different decisions.
- **CI and CI.3\* (exact characterization).** A head's direct numerator term changes by `Z'^{-1} Σ_j w'_j(v'_j − M)`. For a layer-0 substitution, this is `[w_j(v'_j−v_j) + (w'_j−w_j)(v'_j−M)]/Z'`. Dependence on other sources requires a change in attention weight.
- **CI-5 (exact decomposition, last layer).** `Δm = Δa/σ' + a(1/σ' − 1/σ)`. This is the LayerNorm decomposition of Rushing and Nanda (2024).
- **NC (norm channel).** `m = a/σ + κ`. Null edits move all margins over `C` through one `τ = 1/σ`. A pair has one crossing in `σ`, at most one flip along σ-monotone paths, a flip iff `aκ < 0` and `σ* > σ_min`, and the limit `κ` as the norm grows. It holds only within `C`, and is vacuous when `|V| ≥ d`. Under RMSNorm no pair ever flips.
- **RB.1–6 (necessary condition).** Relation-blindness ⇔ an LP is feasible. If `m` is relation-blind, no relation can set the direction of either source's effect. A crossover therefore implies relation-sensitivity. RB forces held-out orders (the necessary preference relation of Greco et al. 2008).
- **D, LN (existence).** With designed weights, the architecture can express "weaker agreeing evidence lowers the margin", both through softmax reallocation and through final-LN norm coupling. **D′ (counterexample).** One re-reading MLP neuron reverses the total effect. **NEG.** A per-source dedicated-heads design is relation-blind.
- **MIX, CORR, FIB.** These are as before.

## 6. Proofs

`proofs/PROOFS.md`. The corrections made after review are marked there.

## 7. Strongest counterexamples, with the arrow each one breaks

| Counterexample | Arrow broken |
|---|---|
| SS fiber (relation varies over `[−1, 1]`, logits fixed) | computation → relation, for same-cut relations |
| CI.3* with fixed weights; review C probe a (salience equalized removes the crossover) | relation → causal role: other sources enter only via reallocation, and the relation is DLA − DLA |
| `‖PΣc‖²` | relation → causal role, for "alignment": it is the norm of the sum |
| Random 3×3 tables: 86% relation-sensitive | causal role → *consistency* role: RB is generic |
| 591/4000 σ-channel sign reversals; D′; self-repair (Rushing and Nanda 2024) | causal role → decision consequence |
| Original §8 claim "sub-consensus support lowers the margin" (false for substitutions, review B S2) | decision consequence → distinctive prediction |
| RMSNorm `κ ≡ 0` | NC as a ranking mechanism outside LN-with-bias models |

## 8. Distinctiveness at equal information

| Baseline | Relation to the surviving results |
|---|---|
| head patching or the attention-weight gradient | yields exactly the CI quantity (review C SERIOUS-8; `∂M/∂w_j = (v_j − M)/Z`). Nothing is distinctive |
| additive DLA of the substituted value | a straw man. The fair DLA baseline is the head-output DLA change, which equals CI |
| entropy and token-frequency neurons, LN self-repair | NC and CI-5 are this mechanism (Stolfo et al. 2024; Gurnee et al. 2024; Rushing and Nanda 2024). Possibly new as minor corollaries: the reachable set, the flip criterion, the `κ` limit |
| additive conjoint measurement, robust ordinal regression, signed difference analysis | RB.1–6 are these (Luce and Tukey 1964; Scott 1964; Krantz et al. 1971; Greco et al. 2008; Dunn and James 2003) |
| causal abstraction (Geiger et al. 2025) | "a head is a weighted-mean aggregator" is an exact abstraction instance |

Possibly new, as applications only: using removable versus nonremovable interaction as an interpretability criterion (review F: no prior use found, moderate confidence), and the NC flip criterion. Neither is a consistency mechanism.

## 9. Scope of the negative

The verdict covers: same-cut writer relations; softmax value-consensus relations in any head (direct term); final-LN and RMSNorm norm-channel relations; and two-factor ordinal interaction of independently manipulable sources. It does **not** cover:

- relations computed by MLPs that are nonlinear in an agreement variable (e.g., a neuron that responds to the *product* of two evidence features);
- QK-composition, where one source changes another's key or query salience;
- multi-layer re-reading circuits identified in trained models.

Review C states the minimum needed to reopen the question: a relation computed nonlinearly in the agreement variable itself, with a direction prediction that survives with salience held fixed and a bridge premise that does not state the response. That is a different hypothesis. Rule C says it must not be generated here to rescue this one.

## 10. Scientific verdict

**`STEP1_NECESSARY_CONDITIONS_ONLY`.**

Arrow by arrow:

1. *Computation → relation.* Partly established. The architecture computes weighted consensus and norms. The only relation with nontrivial content is source versus consensus, and it is readout-defined.
2. *Relation → identified causal role.* Established only as a characterization. Other sources matter only via reallocation (CI.3*) or the norm (NC). Neither is a consistency-specific role.
3. *Causal role → decision consequence.* **Fails.** The σ channel and downstream paths can reverse the sign. A4 would restate the outcome.
4. *Decision consequence → distinctive falsifiable prediction.* **Fails.** The consensus-locus prediction is false as stated. Its corrected form equals head patching. RB.6 predicts from the generic additive null.

Why not the other verdicts:

- Not `CONDITIONAL_BRIDGE`: its condition (A4) is outcome-adjacent (Rule D).
- Not `CAUSAL_PREDICTOR_ONLY`: the only predictor reduces to head patching at equal information.
- Not `INCOMPLETE`: every mandated analysis was carried out. The verdict is scoped (§9).

`STEP1_ORIGINAL_CONSISTENCY_HYPOTHESIS_UNSUPPORTED`, scoped to §9, is an equally accurate secondary description. The primary label is chosen because the step's positive output is the set of boundaries in §5.

## 11. The original `O`/`D` formulation (mandate §28)

- **A.** Descriptively valid as quadratic forms: yes.
- **B.** Demonstrated causal interpretation: no.
- **C.** Relevant under justified restrictions: no. SS makes same-cut "local agreement" inert, and the identified relations are not coboundaries.
- **D.** Necessary for a successor: no.
- **E.** "Obstruction" justified: no.
- **F.** Reason to keep the graph or sheaf formulation: none.
- **G.** Meaningful without the name: the surviving content is ordinary normalization structure plus measurement theory. It needs neither.

**Recommendation** (it does not authorize any step): abandon the `O`/`D` decision mechanism. Do not open Step 2 on any successor drawn from this step.
