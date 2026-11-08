# Computational equivalence and invariance

> **Revised after reviews R1 and R2** (see `ADJUDICATION.md`):
> - **E3** is a post hoc *continuous* index, and it rests on the p_c = 0 shift;
> - **E_q** is the answer code, not a separate level;
> - **Shapley invariance:** withdrawn;
> - **I6:** downgraded to a conjecture.

**Question (mandate A).** Several successful implementations perform the same source-reliability-dependent decision. What do they have in common, and under which notion of equivalence are they the same?

**Implementations compared** (all T4-capable unless stated):

| id | implementation | where it comes from | key evidence |
|---|---|---|---|
| I-tag | key-localised tag, then last-layer routing | Transformer curriculum seeds 0, 1, 11, 13 | replication R2 key-only IIA 0.76–0.84; one or two dominant route heads |
| I-split | tag split between keys and readout query | Transformer seed 2 (also T1 seed 2) | joint key+query IIA 0.73; neither part alone ≥ 0.4 |
| I-redund | tag with three partially redundant route heads | Transformer seed 10 | Shapley 0.35 / 0.29 / 0.28 |
| I-reread | direct last-layer cross-question reading, weakly gated routing | Transformer seed 12 | 2 heads put 52–76% of readout attention on other questions; within-question group-mass R² ≤ 0.25 |
| I-pool | explicit pooled summary | DeepSets T4 seeds 0–2 | pool IIA 0.94 |
| I-mlp | per-group-histogram MLP under curriculum | not saved; behaviour only | T4 acc 0.697–0.729 |
| I-eng | engineered tag-and-route | `../step1_latent_reliability/THEOREM.md` Level 2 | |

## 1. A ladder of equivalence relations

Each rung is strictly finer than the one above. Every implementation is placed on the ladder by measurement.

| rung | relation | test used | I-tag | I-split | I-redund | I-reread | I-pool |
|---|---|---|---|---|---|---|---|
| E1 | same behaviour on the training distribution (≈ Bayes) | accuracy within 0.01 of Bayes; argmax agreement with Bayes 0.86–0.92 | ✓ | ✓ | ✓ | ✓ | ✓ |
| E2 | same **decision response to natural relation changes** (PP1 sign rule; G5 relation-only edits) | PP1 sign agreement 99.4–100%; boundary 96–100%; G5 change agreement 0.85–0.93 | ✓ | ✓ | ✓ | ✓ | ✓ |
| E3 | same **off-support extrapolation** | PP2: siding with the global plug-in versus the local reference under shift | mixed: seeds 1 and 11 local; 0 and 13 global-leaning | mixed | **local** | **global** | **global** |
| E4 | same **relation-carrying internal operation**: reliability-gated readout routing in the last layer (pattern patch IIA ≥ 0.8, target transfer ≤ 0.05) | `spec_s*.json`, replication | ✓ | ✓ | ✓ | ✗ (0.64; value-path reading) | n/a (no attention) |
| E5 | same **relation representation site** (key tag, ≤ 2-D) | replication R2 key-only; joint patches | ✓ | ✗ (split) | ✓ | ✗ | ✗ (pool) |
| E6 | same mechanism up to reparameterisation | not testable across architectures; within the Transformers, head splits (I5) and Q/K gauge (I6) already separate E5 from E6 | — | — | — | — | — |

**What the ladder shows.**

1. **A common class exists at E2.** Every successful model, across three architectures and five implementations, satisfies the same candidate-decision response law: the Bayes margin-change sign rule (Theorem D, Corollary D1), applied to natural edits, including the non-monotone boundary.
   - This is a genuine **implementation-independent causal characterisation**: interventions on inputs produce the same direction of change of candidate margins.
   - It was **predicted before testing** (PP1) and passed in 10/10 models.
2. **The class is not closed under distribution shift (E3)**, as shown by the p_c = 0 shift only; the ρ shifts are non-discriminating (review R1-e). Models that are E2-equivalent on the support disagree off the support.
   - The disagreement is organised. A *post hoc, continuous* index (review R1-g) runs from a *global* configuration-constrained posterior to a *local* factorised per-group posterior (`logs/pp2b_local_posthoc.json`).
   - DeepSets consistently implement the global form: side_global 0.67–0.89 under every shift.
   - Transformers vary by seed.
   - **The tempting generalisation "Bayes-level on the training distribution ⇒ the same computation" is false.** Rungs E1 and E2 do not fix E3.
3. **The internal common structure among the Transformers is an operation, not a representation (E4 ⊋ E5).**
   - Six of seven seeds route the readout's attention over the target question's sources by reliability. Seeds 0, 1, 2, 10, 11 and 13 have pattern patch IIA 0.84–0.86 and target transfer ≤ 0.05.
   - Where the reliability information sits upstream (keys, keys plus query, several heads) varies by seed.
   - Seed 12 breaks E4: it reads the other questions directly through values in the last layer.
4. **A shared downstream site exists, but it is the answer code** (review R2-d), so it adds nothing as an abstraction level.
   - In every Transformer the last-layer attention output at readout q carries nearly the whole cross-question effect (path IIA 0.87–0.95, 5-D subspace 0.85–0.94).
   - It also transfers *target* changes (0.88–0.97). So it is the **reliability-weighted candidate-evidence variable**, the merge point of relation and target, not a relation variable.
   - Seed 12's lower target transfer (0.34) shows that part of its X_q evidence enters through other paths.

## 2. A common causal abstraction, and where it stops

**High-level model H.**

X → V = n^{−q} (leave-one-out agreement counts) → W = reliability weights over groups → E_q = Σ_g W_g · evidence_g(X_q) → decision.

**Alignments that survive meaningful interventions.**

| level of H | I-tag / I-redund | I-split | I-reread | I-pool |
|---|---|---|---|---|
| X → decision (input interventions G1–G5) | ✓ | ✓ | ✓ | ✓ |
| W (routing weights) | readout attention pattern, last layer: ✓ (IIA ≥ 0.84) | ✓ (0.85) | partial (0.64) | — |
| V or W representation | key tag at the last-layer input: ✓ (0.76–0.84) | key + query jointly (0.73) | **no aligned site at the last-layer input** (q-token full residual 0.67) | pool: ✓ (0.94) |
| E_q | last-layer attention output at readout q: ✓ (0.93–0.95) | ✓ (0.93) | ✓ (0.87) | readout MLP input (not tested) |

**Obstacle (mandate A3).** No single *internal* alignment covers I-reread together with the others at the V/W level. I-reread builds part of W·evidence inside the last layer's value path, so there is no state before the last layer where V is fully present for question q.

The common abstraction is therefore exact only at the input and decision levels (E2), plus approximately at E_q (the merge). Its middle variables are realised at different sites, or are not realised at a single site at all.

**Counterexample.** Seed 12 is a learned network that is E2-equivalent to the tag-and-route seeds, yet admits no ≤ 5-dimensional alignment of V at the input to the last layer. It is the existence proof that a common causal abstraction with a localised relation variable does not survive across implementations.

## 3. Invariance under function-preserving transformations (mandate F1)

| transformation | function-preserving in this pre-LN Transformer? | effect on explanatory quantities |
|---|---|---|
| Q/K change of basis per head (W_Q A, W_K A^{−T}) | yes, exactly: the scores q·k are unchanged | patterns, outputs and Shapley shares are invariant. The *coordinates* of a query-side or key-side tag are not; the subspace transforms covariantly |
| V/O change of basis per head | yes, exactly | head outputs are invariant |
| head split (I5) | yes, exactly (checked: Δp ≤ 2.4e−7) | **head counts and top-k head criteria change** (R3 pass → fail). Shapley shares are **not** conserved (review R2-c) |
| residual-stream orthogonal rotation | **no** in general: pre-LN mean-centring and per-coordinate gains do not commute with arbitrary rotations. Only rotations fixing **1** combined with uniform gains preserve LN | residual "directions" are convention-dependent, but only up to the LN-compatible group, not up to arbitrary rotations |
| unit permutation inside the MLP | yes | — |
| query vs key placement of the reliability information (I6) | **conjecture** (pre-LN obstacle; review R2-e) | — |

**Most invariant internal object found.** The **readout attention pattern** over the target question's sources: a distribution over positions with no basis. It is invariant under the exact symmetries in the table, except head creation. Other function-preserving changes (moving attention mass between positions with equal value outputs) can alter it (review R2-j). Up to seed 12's partial exception, it is also where the relation enters the decision.

## 4. What remains unknown

- **Why seed 12 chose rereading** (and why seed 2 chose split keys and query): see `GENERALIZATION_AND_EMERGENCE.md`. No predictive account exists yet.
- **Whether I-mlp shares E2.** Its checkpoints were not saved in the latent-reliability package, so only its accuracy is known. This is a gap.
- **Whether E3 divergence can be predicted from internal structure.** Post hoc, seed 12 (the rereader) and DeepSets (the pooler) extrapolate globally, while key-tag seeds vary. Seven seeds give no reliable predictor. **Open; needs held-out seeds.**
