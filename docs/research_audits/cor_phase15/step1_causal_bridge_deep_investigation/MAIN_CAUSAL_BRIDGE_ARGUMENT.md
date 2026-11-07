# Main causal-bridge argument

Phase 1.5, Step 1, deep investigation. This step adds three things that were missing before:

- the mechanism classes the previous package left out (agreement-sensitive MLPs, moving Q/K attention, multi-block rereading);
- the three corrections (no-bypass screening, per-head identifiability, coverage);
- an evaluation of the positive control from the independent audit.

Base: branch `cor-phase15-step1-deep-investigation` from `49c1ec6`. Predictions committed in `ddf935d` and constructions in `b85222b`, both before the first run. Reviews: `reviews/R1–R4` (separate contexts) and `ADJUDICATION.md` (same context; see the disclosure there).

## The question

Can an independently defined, computation-grounded consistency relationship between internal Transformer representations have a noncircular causal connection to candidate-specific decisions?

## Short answer

**Yes, as an implemented mechanism. No, as a distinctive or identified theory.**

Transformer blocks can compute a relation between two evidence sources and let it decide, or modulate, a candidate preference. This holds in three architecturally different ways:

- an MLP XNOR (A);
- a query–key gate (B);
- a two-block gated amplifier (C).

For each, a class theorem with exact boundaries is proved, and an executed causal experiment supports it.

The causal chain is sound *within designed classes*. But it is fully explained by ordinary mechanisms: nonlinear interaction, attention composition, gating. Its assumptions say, in weight terms, that "an agreement gate exists", and the architecture implements the opposite sign just as easily. Behavior on the natural intervention family does not identify it, and per-component descriptions of it are not gauge-invariant.

**Verdict:** `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`. The implemented-mechanism theorems required for `STEP1_RESTRICTED_IMPLEMENTED_MECHANISM_PROVED` are also met. The more specific label is chosen because distinctiveness was tested and came out negative, rather than left unresolved. The final label is in `FINAL_SCIENTIFIC_DECISION.md` after cross-examination.

---

## 1. The strongest candidate and its exact causal chain

The strongest candidate is **agreement-gated evidence**, Mechanisms B and C. Unlike A, it is candidate-relative.

| Arrow | Content | Status |
|---|---|---|
| computation → relation | B: H2's score is `λx̂ŷ`, a bilinear product of a feature that layer 1 copied from A and B's feature. C: block 1's MLP writes `q = h(Ŝ) − h(D̂)` | **proved, and exact** in the construction (scores to `1e−12`) |
| relation → causal role | B: the attention weight on B is strictly increasing in `x̂ŷ`. C: the block-2 gate is nondecreasing in `q̂` at fixed `Ŝ`. Dissection: routing carries B (routing-fixed spread 0.104 against 4.238). Interchange: `q̂` is necessary (gap 3.944 → 0.182 when ablated) and nearly sufficient (0.243 → 3.928, target 4.187) | **proved** for the class; **verified** |
| causal role → candidate decision | B: unanimity; A wins in conflict iff `|v_A| > k|ŷ|w(−λ|x̂ŷ|)`; 242 reversal cells against fixed routing. C: `sign m = sign Ŝ`; agreement amplifies `|m|` at fixed `Ŝ` (4.187 against 0.243); a prior interval in which agreement decides `c` versus `c'` | **proved** for the classes with no downstream re-reader |
| decision → distinctive prediction | the matched reader-input contrast (A4) separates relation computation from net-evidence nonlinearity; the reversal band and the interval are falsifiable | predictions exist but are **not distinctive**: they are those of an interchange or gating account |

**Where the chain stops being a theory.**

1. **The assumptions encode the gate.** (B1)–(B4) and the `𝓕_C` conditions say, in weight terms, that an agreement-sensitive gate with a fixed sign exists. The conclusions follow from monotonicity in one or two lines. This is check 5 of mandate §8, and it fails in part.
2. **The direction is not architectural.** The mirrored circuit, which amplifies under disagreement, is equally implementable: `m_dis = 3.185` against `m_ag = 0.246` at matched `Ŝ`. Nothing in the architecture says that consistency should stabilize or favor a decision, which is what the original thesis asserts.
3. **Downstream re-readers defeat head-level claims.** With one re-reading MLP, 26 cells show the gated head's change and the total margin change with opposite signs (PB9). The classes exclude such readers by assumption.
4. **Identification.** On the binary family, a relation-computing circuit and a net-evidence circuit give identical margins (A2). A shallow network approximates the two-block function without the mediator (C3). Per-head and per-unit quantities change under exact gauges (ID1, ID2). The identifiable content is input-level effects plus reader compositions modulo the gauges.
5. **Not found in any trained model.** No pretrained run was authorized.

## 2. Mechanism A (positive control) in this chain

A reproduces the audit's circuit exactly (`±3.2142216080`), with three independent implementations. Its relation is **sign agreement under the reader's pairing**: `sign q = sign(xy)` for all real inputs (A1). The magnitude is product-like for weak evidence and min-like for strong.

Cosine and Euclidean distance between the source writes are constant across agreement, so they explain nothing. The rule is **not candidate-relative**: `m(a,b) = m(−a,−b)`, so sources agreeing on `c'` push toward `c`. A is therefore a consistency *classifier*, not an evidence-to-candidate bridge.

What A adds to the decision question is a lesson about controls. Its binary behavior is reproduced exactly by a reader of net evidence alone (A2). Interactions and odd–odd contrasts cannot separate the two (A3). Only a contrast matched at the reader's post-normalization input does (A4). The preregistered raw-sum match failed because pre-norm LN reads the residual norm (PA9).

## 3. Where every serious candidate fails

| Candidate | Fails at |
|---|---|
| Original `O`/`D` | causal role (Phase 1; 1A–1E): the readout and its direction are unconstrained by `D` |
| Softmax consensus and final-LN norm channel (previous package) | distinctiveness; the relation is readout-defined (previous package, adjudicated) |
| A: MLP XNOR | candidate relevance: candidate-symmetric |
| B: QK gate | distinctiveness (Q/K composition); identification; downstream re-readers (PB9) |
| C: gated amplification | distinctiveness (gating); direction not architectural (mirror); identification (C3) |

## 4. Answers to the mandate's 25 questions

See `FINAL_SCIENTIFIC_DECISION.md` §3.
