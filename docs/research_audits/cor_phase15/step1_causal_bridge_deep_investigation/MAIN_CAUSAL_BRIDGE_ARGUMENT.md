# Main causal-bridge argument

**Scope.** Phase 1.5, Step 1, deep investigation.

**Branch.** `cor-phase15-step1-deep-investigation`, created from `49c1ec6` (the previous package).

**Commits that fix the record:**

| What | Commit | When |
|---|---|---|
| Predictions | `ddf935d` | before any run |
| Constructions | `b85222b` | before any run |
| Addendum for the post-review mechanism C′ | `fdc87d8` | before its code |

**Reviews.** `reviews/R1–R4` were written in separate contexts. The adjudication (`ADJUDICATION.md`) was written in the same context as this package and is not independent. The external audit that motivated corrections A–C is in `sources/independent_audit_handoff/`.

## The question

> Can an independently defined, computation-grounded consistency relationship between internal Transformer representations have a noncircular causal connection to candidate-specific decisions?

## Answer

**As an implemented mechanism: yes.** Transformer blocks can compute a relation between two evidence sources and let it determine or modulate a candidate preference. This step builds four circuits:

- **A:** an MLP that computes sign agreement and turns it into the decision;
- **B:** a query–key gate whose attention score is the product of two sources' features;
- **C:** an MLP that amplifies net evidence when the sources agree;
- **C′:** a later attention head whose query reads an earlier agreement feature. It follows the sources when they agree and defers to a prior when they conflict.

Each has a class theorem, executed interventions, controls, ablations, and counterexamples at its boundaries.

**As a theory: no.** No noncircular, independently identified, distinctive consistency-to-decision bridge survives.

**Verdict: `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`**, adjudicated in `FINAL_SCIENTIFIC_DECISION.md`.

## 1. The strongest candidate and its exact causal chain

The strongest candidate is **C′**: conditional rereading through attention. It is the most transformer-specific of the four, and the only one where depth is provably necessary. B and C are the same idea in other operators.

| Arrow | Content | Status |
|---|---|---|
| **Computation → relation** | Block 1's GELU MLP writes `q ∝ h(Ŝ) − h(D̂)` onto `e_3` at the read token. Its sign is the sources' sign agreement (A1). | **PROVED** (A1, given A0); **VERIFIED** |
| **Relation → causal role** | Block 2's query reads `q̂`. Its score differences favor the sources when `q̂ > 0` and the prior token when `q̂ < 0`. Interchanging `q̂` flips the routing (source mass 0.000 → 1.000) and the decision (−2.631 → +1.879). Ablating it makes routing uniform. | **VERIFIED**; mediator necessary and sufficient |
| **Depth** | No single attention layer can route on the relation, because `log(α_A/α_P)` does not depend on B. | **PROVED** (C′2); 30 random heads, spread ≤ 3.6e−15 |
| **Causal role → candidate decision** | When the sources agree, the decision is their shared candidate. When they conflict, it is the prior candidate, whichever source is stronger (margin constant at −2.631). | **VERIFIED** (PR1, PR5). PR2 (monotone routing in `q̂` across cells) **FAILED** and is kept |
| **Decision → distinctive prediction** | Falsifiable in a frozen model (§5 of the final decision). | Predictions exist but **are not distinctive**: they are those of a Q-composition account |

## 2. Where the chain stops being an explanatory theory

1. **The direction comes from sign hypotheses on the weights.** Every directional conclusion is supplied by sign hypotheses: the unit output signs, the key signs at the sources and at P, `k > 0`, `Δu^T v > 0`. Flip a key sign, the gate sign, or `γ`, and the same architecture implements the opposite rule. For example, mirrored C amplifies disagreement (3.185 against 0.246). So nothing in the architecture implies that consistency should favor or stabilize decisions, which is the original thesis. Under mandate §8, check 5 fails, and the bridge is **rejected as an explanatory bridge**.
2. **The relation is not identified independently.**
   - Source polarity is fixed only through the source's direct path to the readout (R3).
   - "The circuit reads `x − y`" is not invariant: cancelling linear units can add such a reading to any reader.
   - Raw scores, patterns, preactivations, per-head and per-unit quantities, and summed writes all change under exact function-preserving maps.
   - What is identifiable: function-level effects of the coded inputs, score differences such as `x̂ŷ`, readout-normalized reader even parts, and `q̂` up to scale (`IDENTIFIABILITY_AND_REPARAMETERIZATION.md`).
3. **Candidate-relativity is narrow.**
   - A is candidate-symmetric: `m(a,b) = m(−a,−b)`, so sources agreeing on `c'` push toward `c`.
   - B, C and C′ are candidate-relative only for a binary pair with scalar evidence. With three or more candidates, B's relation is an inner product of features, not agreement about the preferred candidate (R1 S2).
4. **Head- and unit-level claims do not fix the decision when other components exist.**
   - A downstream re-reading MLP produces 26 cells where the gated head's change and the margin change have opposite signs (PB9).
   - One parallel head breaks B's unanimity in 348 of 450 cells (R1 S3).
   - A third block makes C decide against net evidence (`m = −8.02`, R1).
   - C's sign law fails under weight noise of 0.03 in every seed, though its amplification is robust.
5. **Every mechanism is a known one:**
   - A: an XNOR unit (polarization identity);
   - B: Q/K composition and content-based routing;
   - C: a multiplicative gate;
   - C′: Q-composition from an MLP feature.

   See `BASELINES_AND_NOVELTY.md`.

## 3. What the positive control (A) teaches

A reproduces the audit's circuit exactly (`±3.2142216080`), with three independent implementations. Its relation is sign agreement *relative to the reader's pairing*:

- In the residual, the two source writes are orthogonal. Their cosine and Euclidean distance are the same in every cell and explain nothing.
- The magnitude of the effect is product-like for weak evidence and min-like for strong.

A carries one lesson for the decision question: **binary data cannot tell a relation-computing circuit from a nonlinearity of net evidence.**

- A reader of net evidence alone reproduces A's binary margins exactly (A2, without final LN).
- Interactions and odd–odd contrasts cannot separate them (A3).
- Only a contrast matched at the reader's post-normalization input does (A4). It also works on reachable inputs with input substitution alone: 0.307 / 0 / −0.720, against a constant 0.307412.
- The preregistered match on the raw sum failed (PA9), because pre-norm LN reads the residual norm.

## 4. Each candidate and where it fails

| Candidate | Where it fails |
|---|---|
| Original O/D | causal role (Phase 1; Steps 1A–1E) |
| Softmax consensus, final-LN norm channel (previous package) | distinctiveness; the relation is defined by the readout |
| A: MLP XNOR | candidate relevance (candidate-symmetric); distinctiveness |
| B: QK gate | identification (polarity; per-head gauge); parallel writers and re-readers; three or more candidates; distinctiveness |
| C: gated amplification | direction is not architectural (mirror); sign law fragile; decision equals `sign Ŝ` without a prior; distinctiveness |
| C′: conditional rereading | direction set by key and value sign hypotheses (mirror); identification; distinctiveness (Q-composition) |

## 5. Bottom line

The architecture *permits* consistency-to-decision rules of either sign, with equal ease and through ordinary mechanisms. Their identifiable content is the content of those ordinary mechanisms. That is the scientific determination this step was asked to make.

The 25 questions, coverage, the falsification specification and the next decision are in `FINAL_SCIENTIFIC_DECISION.md`. The ten failures are adjudicated in `TEN_FAILURE_ADJUDICATION.md`.
