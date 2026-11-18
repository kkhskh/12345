# Cross-slot causal transfer (Experiments B and C)

> **Post-review corrections** (`reviews/ADJUDICATION.md`; agent reviews, not human peer review). **W3:** the transfer does not separate H2 from the mandate's H3, and duplicating the donor code (no exchange) does not transfer (0.08–0.22), so the code acts as a unique binding label. **W4:** the layer locus is withdrawn; in A2_s704 a layer-2-only identity swap transfers 0.65–0.70 [expl, review]. **W7:** the routing transplant was content-mismatched; a matched transplant gives 0.11–0.47, and forcing baseline attention during the relabel cuts transfer to 0.02–0.32. **W11:** no slot's input content or task evidence moves, but activations do mix across slots through attention. Text withdrawn after review is kept in place and marked *[withdrawn, Wn]* so the record is preserved.


**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

## 1. What each intervention can carry (mandate §6.1, §6.3)

| intervention | what moves between slots | can it carry the donor's evidence or answer? |
|---|---|---|
| **Identity relabel (lo, hi)**: slots i and j exchange codes inside layers lo … hi − 1; every slot keeps its own content | only the slot code inside the window | **No.** Each slot's activations are computed from its own tokens throughout. This is the source-matched control the XVI patching lacked, made exact by construction |
| **Attention-routing transplant**: the recipient's attention rows are replaced by the donor's, with the two slots' columns exchanged, so the recipient attends to **its own** tokens with the donor's weights | the attention weights only. Values remain the recipient's own | **No** evidence moves. Only the routing pattern does |
| Full residual patch (XVI) | the donor's whole activation, including its evidence | **Yes.** Not used for causal claims in XVII |

**Relation to the mandated donor controls (§6.2):**
- **Relation-preserving donors, and evidence-matched against gate-different donors.** Under the relabel, donor and recipient keep their own inputs, so no evidence is shared or mismatched. The gate counterfactual is applied to the **recipient's own** gate (per-slot gate flip), and the expected decision is the exact Bayes decision for the recipient's edited input.
- **Non-relational control donors.** Exchange with another *unacquired* slot (U → U control).
- **Sham.** An empty window, or the recipient's own attention pattern; both give the baseline exactly.
- **Wrong-gate donors and reverse direction.** The donor-side F_G under the same exchange is reported for every pair (reverse direction).

**What is off-support.** Within a window, a slot's tokens carry another slot's code. Since codes are trained parameters, not data, this is an intervention on the network state, not an off-support input. The generated inputs themselves are always on-support.

## 2. Development results [dev]

**C_s202, step 1000** (A = {0, 3, 4}, U = {1, 2, 5}; 9 pairs; `logs/expl/dev_relabel_C_s202_1000.json`):

| window (layers) | recipient F_G (U slot) | donor F_G (A slot) |
|---|---|---|
| none | 0.00 | 1.00 |
| [0,1) | 0.52 | 0.01 |
| [1,2) | 0.00 | 0.66 |
| [2,3) | 0.00 | 0.96 |
| **[0,2)** | **0.96** | 0.00 |
| [1,3) | 0.01 | 0.55 |
| [0,3) (= code swap) | 1.00 | 0.00 |

**Routing transplants** (pairs (0, 1), (3, 2), (4, 5)):

| transplant | recipient F_G |
|---|---|
| layer 1, all heads, from the acquired donor | 0.08–0.11 |
| layers 0–1, all heads | 0.09–0.19 |
| layer 1, heads 0–1, evidence rows only | 0.03–0.09 |
| the recipient's own pattern (sham), or from an unacquired donor | 0.00 |

**Attention at step 1000.** Layer-1 heads 0 and 1 route evidence → gate tokens with mass 0.73–0.99 in acquired slots, against 0.01–0.10 in unacquired slots. The routing **differs**, but transplanting it is **not sufficient**.

**WB_low_s503, step 1740** (arity 2; A = {1}, U = {0, 2, 4, 5}): [0,1) 0.19 / 0.01; [1,2) 0.01 / 0.35; [2,3) 0.03 / 0.65; **[0,2) 0.68 / 0.04**; [1,3) 0.01 / 0.19; [0,3) 0.91 / 0.13. Routing transplant: ≤ 0.07.

**After full acquisition** (Experiment C, cross-position alignment without fitted maps):

| model | consistent [0,2) exchange | mixed [0,1) | mixed [1,3) | late layer [2,3) |
|---|---|---|---|---|
| C_s202, step 2400 (recipient / donor) | 0.96 / 0.92 | 0.64 / 0.60 | 0.60 / 0.61 | 0.92 / 0.95 |
| WB_low_s503, step 2500 (range over 3 pairs) | 0.58–0.73 | 0.20–0.34 | 0.04–0.36 | 0.24–0.69 |

## 3. Reading of the development evidence [dev; not confirmation]

1. **The unacquired slot can run the acquired slot's computation.** Given that slot's code in layers 0–1 only, gate use transfers (0.96 at arity 3, 0.68 at arity 2), with nothing but the code moved. *[withdrawn, W3: under Theorem P this rejects only a straw H3, and does not separate H2 from the mandate's H3]*
2. **The early-layer computation is keyed by the code, not merely routed by it.** Routing transplants carry almost nothing.
3. *[withdrawn, W4: windows with lo > 0 move only the raw code; derived identity persists]* **The late layer is largely code-agnostic at arity 3** (the donor keeps 0.96 when only layer 2 carries the other code), **less so at arity 2** (0.65).
4. **Code keying persists after acquisition.** Mixing codes across layers breaks gate use even when every slot has acquired it. The slots never become interchangeable at the level of single layers; each runs the shared computation consistently keyed by its own code.

**Necessity tests (mandate §7.3).** The relabel also tests necessity:
- An acquired slot given an unacquired code in layers 0–1 loses gate use completely (donor 0.00 at arity 3, 0.04 at arity 2).
- Given the unacquired code in layer 2 only, it keeps 0.96 (arity 3).
- So the code in layers 0–1 is **necessary** for an acquired slot's gate use. *[withdrawn, W4]* Layer 2's is largely not.

## 4. Identifiability (mandate §7.4)

- **"Shared computation, code-keyed access" and "a code-conditioned implementation that any slot can adopt by taking the code"** are observationally equivalent under these interventions. They are two descriptions of one fact: the same shared weights compute the gate for some codes and not for others.
- **What is distinguished** is the strong H3 claim, that an unacquired slot cannot use the acquired slot's computation without its own new implementation. Development **rejects** it in two models; P2 tests it in fresh models.
- **Not identified:** which features of a code make it "engaging". Codes are near-orthogonal random vectors (cosines −0.10 to 0.02) that barely move during the partial window (norm change < 0.1).

## 5. Fresh-model results [frozen]

All 6 evaluable fresh models (3 per arity) met every frozen mechanism prediction:

| prediction | result |
|---|---|
| P2 TRANSFER | [0,2) recipient 0.65–0.90 against U → U controls with model means 0.04–0.06 (W15) |
| P3 LATE_LESS_CODE_DEPENDENT | the donor keeps 0.60–0.90 under [2,3) and 0.02–0.17 under [0,2) |
| P4 ROUTING_INSUFFICIENT | routing transplant 0.05–0.17 |
| P5 CODE_KEYING_PERSISTS | after acquisition, consistent [0,2) 0.79–1.00 against mixed [0,1) 0.26–0.62 |

**Answers to Q8–Q15.**
- **Q8:** gate use can be transferred to an unacquired slot. **Q9:** the transfer survives source matching, by construction: only the code moves.
- **Q10:** attention-pattern transfer does **not** reproduce it.
- **Q11:** *[withdrawn, W4]* the causal *site* is shared, layers 0–1 for every slot. Since codes are near-orthogonal, a shared activation *direction* was not tested, and no alignment maps were fitted.
- **Q12:** yes, there is evidence for a common relational computation, available to every slot given an engaging code.
- **Q13:** yes, there is evidence for code-dependent access to it, and the access is not routing alone.
- **Q14:** distinct position-conditioned implementations in the strong sense (non-transferable) are rejected. The learning results (CROSS_SLOT_LEARNING_INTERVENTIONS.md) show each slot's access is sustained by its own training signal.
- **Q15:** "code-keyed access" and "a code-conditioned implementation adoptable by any slot" cannot be distinguished (§4).
