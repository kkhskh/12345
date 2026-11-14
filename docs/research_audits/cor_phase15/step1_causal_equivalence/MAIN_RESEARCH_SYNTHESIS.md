# Investigation X — main research synthesis

**Causal equivalence, heterogeneous relational mechanisms, and mathematical structure.**

**Evidence levels.**
- **[C]** confirmed under frozen predictions. Part I is at `aa12df6`, A1 at `827512f`, A2 at `0073e39`. The new parents
  are wP140–143; n = 3 complete parents, and wP143 is incomplete and excluded as pre-specified.
- **[F]** falsified.
- **[D]** development (Investigation IX models).
- **[P]** proved (elementary).
- **[N]** numerical.

## Answer to the central question

**Independently trained native repairs implement the same abstract relational computation through different concrete
pathways. This is H2, distributed equivalence. It is established at three levels, and refuted at one.**

1. **Same algorithm, behaviourally [C, F1 3/3].** On on-support edits that separate the candidate rules, every repaired
   model is closest to the Bayes eligibility rule. That rule uses perfectness *and* agreement at the target question.
   The repaired models sit at distance 0.11–0.27, against ≥ 0.65 for the alternatives. The original frozen-prefix
   models are identified as *different* algorithms: perfect-only (A1) or a count-threshold rule (A2).
2. **Same causal abstraction [C, F2 2/3, F3 3/3].**
   - Every repaired model has an interchangeable representation of the **per-group eligibility vector E**, at the same
     layer boundary: the downstream input, after the fine-tuned layers 0–1. Interchange accuracy is 0.85–0.97 against
     0.50 for random subspaces, and KL to the counterfactual is 0.015–0.16.
   - The pooled variable S interchanges only partially (0.47–0.55). By Theorem X1, this is the signature of a *per-group*
     implementation, not a pooled one.
3. **Same validity signature [C, F6 3/3].** In every repaired model, the E-subspace responds *selectively* to the input
   changes that define eligibility, through both routes (agreement at q, and counts). It does not respond to matched
   control edits. In every original model it does not.
4. **Same learning order [C, F7 3/3, plus the incomplete parent].** Every run first learns perfect-only eligibility (A1),
   briefly a threshold rule (A2), and then the Bayes rule (B). The switch comes between steps 546 and 1211 on the new
   parents, and 813–4000 in development. This order was invisible to Investigation IX's R1-based test, which was
   falsified, because A1 and B make identical R1 predictions.
5. **Different concrete pathways [C, F5 3/3; Investigation IX NP2].**
   - The abstraction is distributed over the readout and source tokens of the target question. Neither token type
     alone carries it.
   - The token subset that carries the R1 credit varies across parents (Investigation IX).
   - Head-level organization is inherited from the pretrained parent. Two fine-tuning seeds of one parent share head
     roles (cosine 0.78–0.87). Cross-parent head similarity after quotienting head permutations is only weakly above a
     permutation null.
6. **Refuted: a mechanistic marker [F, F4 1/3].** Interchange accuracy on discriminating counterfactuals did *not*
   separate repaired from original models. Two repaired models had weaker fits than their originals. Interchange
   accuracy without a selectivity check is not a reliable marker of the mechanism.

## Mathematics

- **Theorem X1 [P, N]: abstraction depends on the factorization, not on the function.** Three exact implementations of
  the same Bayes rule (pooled, per-group, pairwise) support different interventions under linear interchange: "set S",
  "set E", and neither.
  - A failed alignment therefore does not show a different algorithm, and a successful one does not show a unique
    intermediate.
  - Gradient search missed the exact S-alignment of the pooled implementation (error 0.26 vs 0). Search failure is weak
    evidence.
- **Proposition 1 [P]:** exact models are indistinguishable by every on-support input test. Behavioural discrimination
  works only through approximation differences, off-support inputs, or internal interventions.
- **Proposition 2 [P]:** relational edits that remove the only eligible group are off-support. They were detected and
  excluded.
- **Separation table [P]:** the on-support edit families D1/D2/R1–R3 jointly separate the five candidate eligibility
  rules (pairwise distances ≥ 1 on at least one family).
- **No Transformer lower bound is claimed** from the interaction-order results.

## Answers to the final questions

1. **Do the native mechanisms share a causally meaningful abstract computation?** Yes. They share the per-group
   eligibility vector E, interchangeable at the same layer, selective for its defining inputs, and part of the same
   Bayes rule [C: F1, F2, F3, F6].
2. **What is shared, and which interventions establish it?**
   - The *behavioural* Bayes eligibility rule (discriminating edits D1/D2/R2/R3).
   - A distributed E-subspace at the downstream input: DAS interchange with random controls, plus selectivity under
     defining vs control edits.
   - The A1 → A2 → B learning order (checkpoint behaviour).
3. **What supports different algorithms rather than failed identification?** For the *repaired* models, nothing: they
   are equivalent at every tested level above the token pathway. For the *original* models, the A1/A2 identification
   (F1) and the absence of a selective E-subspace (F6) show a genuinely different algorithm, not merely a failed
   alignment.
4. **Which pathways implement the operations?**
   - E is distributed over the 7 tokens of the target question at the downstream input [C].
   - The layer-0 MLP carries ≈ all of the edit's total effect [D].
   - Parent-specific layer-0 heads move agreement information. Their identity is inherited from pretraining [D].
   - Head-internal computation of "agreement × perfectness" is **not** identified.
5. **Why do parents develop different mechanisms and trajectories?**
   - *Mechanisms:* the head roles available to fine-tuning are fixed by pretraining. A same-parent seed replicate
     reuses them; different parents use different heads [D].
   - *Trajectory timing* varies (546–4000 steps); the *order* does not [C].
   - Why the timing varies is not established.
6. **What explains the frozen-state reader and incomplete-pretraining failures?**
   - *Incomplete pretraining:* the incomplete parent wP143 still reached Bayes-level behaviour under *full* fine-tuning
     (KL 0.002, absE_R1 0.18). With a frozen prefix it stayed information-limited (KL 0.096, absE R2∪R3 1.84). The
     missing count is recoverable when the prefix can learn.
   - *Longer pretraining:* not run.
   - *Frozen-state reader:* see the B3 section below.
7. **Invariant mathematical properties.**
   - The Bayes eligibility rule's behavioural signature.
   - Per-group (not pooled) interchange structure.
   - Selectivity to both input routes of eligibility.
   - The layer location of the abstraction.
8. **Prospectively predictable effects.** F1, F2, F3, F5, F6 and F7 were predicted and confirmed on new parents. F4 was
   not.
9. **Generalization tests.** No new generalization runs in Investigation X. The generalization-style test used here is
   the *discriminating on-support edit*. It separates algorithms that agree on ordinary episodes (A1 vs B on R1). Its
   off-support pitfall (D1 with S = 0) was found and excluded.
10. **New theorems, counterexamples and limitations.**
    - Theorem X1 and its search-failure caveat.
    - Propositions 1–2.
    - A **false-positive mode of interchange abstraction**: moderate IIA without selectivity in non-repaired models.
    - The F4 failure.
11. **COR connections.**
    - The order-1 vs order-2 distinction (Investigation IX) is reflected in the learned algorithm sequence: perfect-only
      is a per-group rule; B adds the q-agreement interaction.
    - No connection to GATE's mechanism was tested.
    - **No connection to the original O/D formalism.** E is a Bayesian sufficient-statistic component.
12. **Highest-value open questions.** See `OPEN_FOUNDATIONAL_MAP.md`:
    - the head-internal computation of E;
    - whether an S variable exists that the search missed;
    - cross-model mapped interchange;
    - why A1 is learned first;
    - a general validity criterion for interchange abstractions.

## B3 follow-ups (development, bounded; `logs/B3_followups.json`)

**Frozen-state reader (Investigation IX's falsified CP4).** The reader is an MLP over the 7 q-tokens of the frozen final
pretrained states, with no side information. Values are absE_R1 / E_R1 / KL.

| parent | labels, 8k (Inv. IX, conf9) | **Bayes targets, 32k (dev)** | labels, 32k (dev) |
|---|---|---|---|
| yP130 | 0.52 / −0.42 / 0.023 | **0.117 / +0.017 / 0.0010** | 0.75 / −0.01 / 0.056 |
| yP131 | 0.50 / −0.34 / 0.026 | **0.126 / +0.011 / 0.0013** | 0.80 / … |
| yP132 | 0.44 / +0.00 / 0.015 | **0.084 / +0.018 / 0.0004** | not run |
| yP134 | 0.51 / −0.44 / 0.017 | **0.133 / +0.012 / 0.0010** | not run |

- With exact-Bayes soft targets, the frozen-state reader reaches near-Bayes on all 4 parents. Frozen native states are
  therefore **sufficient and accessible** to a small reader.
- With labels, more steps make it *worse*. The reader trains on a finite pool of 50,000 episodes, so it overfits label
  noise.
- CP4's failure was an **objective / finite-sample effect, not an information limit**. Caveat: the 32k runs were scored
  on eval_dev and Investigation IX's on eval_conf9.

**Incomplete pretraining.** No longer-pretraining run was made. The pre-specified incomplete confirmation parent wP143
(count accuracy 0.765) gives the relevant contrast:
- with a frozen prefix (C1): KL 0.096 and absE R2∪R3 1.84, the information-limited pattern of Investigation IX;
- with full fine-tuning: KL 0.002, absE R1 0.18, and the same E abstraction (IIA 0.965), selectivity and learning order
  as the complete parents.

Fine-tuning the prefix recovers the missing count information.

**Seed replicates (same parent, new fine-tuning seed).** yP130/s1 and xP81/s1 are behaviourally Bayes-like (B-closest on
D1/D2/R2/R3, distances 0.09–0.15). They carry the E abstraction (IIA 0.946 / 0.964, KLcf 0.025 / 0.019).

## Limitations and disclosures

- n = 3 complete confirmation parents. The thresholds are counts, not significance tests.
- One seed per confirmation parent.
- **Reviews:** by the same agent in separate passes (`REVIEWS_AND_ADJUDICATION.md`), not independent.
- **Compute:** about 4.5 h of active compute on 2 CPU cores, with one environment restart. Every lost run was rerun.
- **Not done:**
  - head-internal path patching;
  - cross-model mapped interchange;
  - longer pretraining of an incomplete parent;
  - all generalization variations;
  - TRANS, adaptive sources and O/D work.
- **Scope.** Step 1 is not closed, and no COR thesis is selected.
