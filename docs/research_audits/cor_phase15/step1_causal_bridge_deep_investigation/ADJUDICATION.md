# Adversarial cross-examination and adjudication

## Independence, and its limits

There were four reviewers. Each ran as a separate subagent with a fresh context and saw only the package files and the mandate:

| Reviewer | Role |
|---|---|
| R1 | mechanism construction and attack |
| R2 | mathematics and counterexamples |
| R3 | identifiability |
| R4 | baselines and novelty, with web search |

Each was told to write only its own review file. R1 noted that the working tree changed while it reviewed (I was writing other documents), so R1 recorded the commit and file hashes it actually reviewed. Reviewers' scratch scripts lived in `/tmp` and are not part of the package. Their numbers are quoted from their reviews, and I reproduced the decisive ones myself.

**This adjudication is mine, written in the same context as the package. It is not independent peer review.**

The external independent audit (`sources/independent_audit_handoff/`) was produced in a different session by a different system, before this step. Its three findings are the mandate's corrections A, B and C.

## Confrontation: strongest positive case against strongest negative case

**Positive.**
- Three architecturally distinct Transformer circuits compute a relation between evidence sources and use it in the decision:
  - an MLP XNOR (A);
  - a query–key gate (B);
  - MLP-gated amplification (C), and query-composed rereading (C′).
- Each has a class theorem with proved boundaries.
- C′ makes depth necessary: no single attention layer can route on the relation (C′2).
- Interchange shows the agreement variable is a real mediator: necessary and sufficient in C and C′.

**Negative.**
- Every directional conclusion is supplied by sign hypotheses on the weights (R2). Mirrored circuits implement the opposite rule just as easily.
- Source polarity is identified only through the outcome pathway (R3).
- Candidate-relativity fails beyond a binary candidate pair (R1).
- Head-level claims fail with a downstream re-reader or a parallel writer (PB9; R1 S3).
- Each mechanism is a standard XNOR unit, a QK composition, or a gate (R4).

**Resolution.** The positive case establishes *existence* and *class theorems*. Nobody disputes either. The negative case defeats *identification* and *distinctiveness*. The mandate's §17 separates exactly these. Neither side's points are rejected.

## Finding by finding

| Finding | Accepted? | Action |
|---|---|---|
| R1 S1: A1 needs source separation; A1.3 false with two mixing heads | yes | (A0) added; A1.3 scoped; counterexample registered |
| R1 S2: candidate-relativity fails with 3 or more candidates | yes | class-formula example reproduced (`⟨x,y⟩ = −1.70`, B's weight 0.333 → 0.003); claim restricted to a binary pair with scalar evidence |
| R1 S3: B1 must exclude parallel writers | yes | (B4) strengthened |
| R1 S4: PB6 misscored | **yes** | registered rescoring (addendum); PB6 FAILED |
| R1 S5: PA11 cannot fail | **yes** | reclassified NON-DISCRIMINATING |
| R1 S6 / R2 S2: C matched at the wrong site | **yes** | rematched at the block-2 reader. Gap 4.318, and exactly 0 with block 1 ablated. The "LN coupling" explanation is withdrawn |
| R1 S7: C has no attention rereading | **yes; this was the main coverage gap** | built C′ with predictions registered first (`fdc87d8`). PR1, PR3, PR4, PR5 pass; PR2 fails and is kept |
| R1 S8: no B/C perturbation analysis | yes | run. B robust (unanimity 20/20 at 0.03, 18/20 at 0.1). C sign law fragile (0/20 seeds at 0.03), amplification robust (20/20). C with final LN and bias: 34 errors (R1: 36) |
| R1 M1: A with moving attention | accepted | single head holds (R1, 0/7200 errors); recorded in A1.3's scope |
| R1 M2: labeling, reachability | yes | dissection labels added; A4 repeated on reachable inputs with input substitution only |
| R1 M3 / R2 S3: B1.4 with `s_0 ≠ 0`; `v_A` sign | yes | stated in score-difference form, with the offset case and its reversed direction |
| R1 M4: A4 conditions | yes | added |
| R1 M5: PC2 prior chosen from margins | yes, already disclosed | kept labeled as a demonstration |
| R1 M6: A with 3 candidates | yes | consistent with A1.2 |
| R2 S1: C1.2 "strictly" false | yes | corrected |
| R2 S4 / R3 S1–S2: identifiable list overclaimed | **yes** | replaced by score differences, pattern mass over differing values, the reader even part grouped by direction, products and readout-normalized signs |
| R2 S5: A1 never links coordinates to sources | yes | (A0) |
| R2 S6: A2 fails with final LN | yes | A2 scoped to no final LN and `h(0) = 0` |
| R2 MINOR: SS* needs B not descendant | yes | added |
| R2 MINOR: ID1 sum of individual patch effects | yes | the total patch is invariant. The sum of individual patch effects is invariant here because the downstream map is linear in head outputs |
| R2 MINOR: B1 notation, hidden conditions, final-LN scale | yes | `k` absorbs the factor 2; no final LN in B4; `Δu^T h` before the gate must equal `v_A` |
| R2 MINOR: C2 needs source-independent routing | yes | in a model where keys depend on the sources, keys are a bypass; C2 holds for C's `W_Q = 0` block 1 |
| R2 MINOR: B1.5 only at site level; input-level 2×2 interaction is 0 | yes, and kept as a result | the gate is invisible to sign-symmetric 2×2 designs |
| R2 C3: sketch incomplete; non-representability distinguishes | yes | replaced by the piecewise-affine argument; the "I/O does not identify" headline withdrawn |
| R2 §5: should be rejected as an explanatory bridge | **yes** | §5 now rejects it; the results are labeled implemented-mechanism existence theorems |
| R3 S3: coordinates and conditions change under gauges; LN gain is a gauge | yes | ID2 extended; the earlier non-uniform-gain statement withdrawn |
| R3 S4: answer §9 for the actual quantities | yes | `IDENTIFIABILITY…` §3 rewritten per quantity |
| R3 S5: polarity not identified; enters via the outcome pathway | **yes** | stated in ID and in the bridge checks |
| R3 MINOR: G ill-typed; C3 compares across architectures; attention↔MLP move unproved | yes | site correspondences; C3 revised; conjecture labeled |
| R4: everything reduces to known mechanisms; ONLY_KNOWN | yes | `BASELINES_AND_NOVELTY.md`; verdict |
| R4: attention scores identifiable only modulo listed groups; ID1 non-generic | yes | done |

**Rejected findings:** none.

**Remaining disagreement:** R1 suggests the verdict could be `STEP1_INVESTIGATION_INCOMPLETE` if S7 were not fixed. With C′ built, every family the mandate names has been examined: MLP, moving Q/K, and multi-block rereading through both MLP and attention. Remaining coverage limits are listed in `FINAL_SCIENTIFIC_DECISION.md` §4. They do not include an unexamined central class.

## Adjudicated verdict

**`STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`.** R4 recommends this label. The existence conditions of `STEP1_RESTRICTED_IMPLEMENTED_MECHANISM_PROVED` are also met. The more specific label is chosen because distinctiveness was tested and found negative, not left unresolved (R4's reasoning).
