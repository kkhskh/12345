# Identifiability and reparameterization

Formal statements: `THEOREMS_AND_COUNTEREXAMPLES.md` §1 (ID). Checks: PI1–PI5. Attacks: `reviews/R3_identifiability.md` and `reviews/R2_mathematics.md`, with each reviewer's scripts and numbers as reported.

## 1. Correction of the previous package (correction B)

The earlier claim was that per-head `w_j`, `v_j`, `M` and centered writes are invariant under the architecture's symmetries. **That claim is withdrawn.**

Head redistribution between heads with identical Q, K and V preserves the function on every input. It changes per-head writes, direct effects, consensus values and patching effects:
- logits change by at most 4.4e−16;
- per-head patch effects move from `[−2.106, −2.106]` to `[−1.699, −2.514]`, while their sum is unchanged;
- the audit's consensus change (14.963 with logits fixed to 1.8e−15) reproduces exactly.

This gauge requires identical heads, so it is a non-generic case. The broader lesson holds generically: see §2.

## 2. What survives which reparameterizations

| Quantity | Fixed-parameterization observable | Interventional attribution (given θ) | Identifiable modulo the listed maps | Function-level |
|---|---|---|---|---|
| input-level effects `m(a',b) − m(a,b)` | yes | yes | yes | **yes** |
| A4 matched-`Ŝ` contrast | yes | yes | yes, given an identified reader | no |
| logits, margins | yes | — | yes | yes |
| raw attention scores | yes | yes | **no**: row-constant shifts | no |
| within-row score differences | yes | yes | yes under QK, VO, LN-bias and constant-key maps | no |
| raw patterns | yes | yes | **no**: an emptied duplicate head's pattern is arbitrary | no |
| pattern mass over positions with differing value writes | yes | yes | yes under the listed maps | no |
| Q, K, V vectors | yes | yes | **no** | no |
| per-head writes, DLA, patch effects | yes | yes | **no** (ID1) | no |
| summed same-layer writes | yes | yes | **no**: rotate under `R`; change by components along `1` or unread directions (R2, R3) | no |
| raw MLP preactivations; "which directions a unit reads" | yes | yes | **no**: cancelling linear units, ReLU rescaling, unit splitting, LN-gain folding | no |
| reader even part grouped by direction pair `±w` | yes | yes | yes under the listed maps | no |
| per-unit attributions | yes | yes | **no** | no |
| source polarity (which sign "supports `c`") | designer's input coding only | — | **no** (R3) | no |

## 3. The mandate's question for the proposed consistency quantities

**Would two mathematically equivalent implementations produce different values of the proposed consistency quantity while producing identical decisions?**

| Quantity | Answer |
|---|---|
| **`x̂ŷ`** (B, the gating relation) | **No, as a score difference.** It survives every attack in R3. **Yes for `x̂` and `ŷ` separately:** the QK sign gauge flips both. |
| **`sign(xy)`** (A) | **Yes**, unless the scale of each unit pair is fixed by its output along `Δu`. ReLU rescaling of `(3, 3, .5, .5)` makes coordinates recovered from input weights give 44/144 sign errors (R3). After readout normalization it is invariant, but only up to a joint sign flip. Sign agreement is invariant under that flip; each source's polarity is not. |
| **`q̂`** (C, C′) | invariant up to a positive scale. Within C's architecture, R3 found no exact map that removes it. |
| **"the circuit reads `x − y`"** (the weight-level description of A) | **Yes, it changes.** Cancelling linear units make even the sum-only reader "read" `x − y`, with a direct effect of about 9.5, while its function is unchanged. The invariant content is the matched-`Ŝ` contrast (A4) and the grouped even part. |
| **behavioral relation** | **Yes, on a restricted intervention family.** On the four binary cells, a relation-computing circuit and a net-evidence circuit give identical margins without final LN (A2). The exact I/O function on a continuum does distinguish C from any one-layer ReLU network (C3, revised). Finite behavior does not. |

## 4. What is identifiable, and the convention behind each claim

**No convention needed.** Input-level interventional effects on the frozen function. This includes:
- A1's sign law as a property of the function of the *coded inputs*;
- the absence of candidate-relativity in A (`m(a,b) = m(−a,−b)`);
- C′'s decision rule.

**Modulo the listed maps.** Within-row score differences, including `x̂ŷ`. Readout-normalized reader even parts. The A4 contrast at an identified reader. `q̂` up to scale.

**Convention required.**
- **Source polarity.** Fixed by the designer's input semantics. Internally it is fixed only by the source's direct path to `Δu`, which is the outcome pathway (R3). In a trained model, internal polarity identification would therefore partly use the readout. That does not make the claims circular in the response, but it means "agreement relative to `c`" is defined through `Δu`.
- **Which unit or head is "the gate".** A fixed-parameterization statement only.

**Source identification in trained models.** Unsolved here. Probes trained on evidence labels reintroduce the Phase 1 `P` problem. Directions chosen because they predict the margin would be circular. The protocol in `FINAL_SCIENTIFIC_DECISION.md` §5 fixes sources at the input level and finds readers by intervention.

## 5. Not covered

- Non-generic symmetries of trained weights.
- Exact maps that move computation between attention and MLP. That these exist only approximately in general is a conjecture.
- Identification with non-binary candidate sets: the gate relation is not agreement about the preferred candidate (R1 S2).
