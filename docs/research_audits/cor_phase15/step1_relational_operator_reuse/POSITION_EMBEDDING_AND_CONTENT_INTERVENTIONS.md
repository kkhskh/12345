# Position, embedding and content interventions (Experiment A)

**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

## 1. What the architecture makes provable (Theorem P; COMPETING_NEURAL_MECHANISMS.md §1)

The network has no positional encoding and no attention mask. Slot identity enters only through E_q[q]. So:

| mandate intervention | in this architecture | consequence |
|---|---|---|
| **C. position interchange** (equivalent questions at other sequence positions, task fixed) | Moving a question's 7 tokens to different sequence positions changes nothing, because self-attention without positional information is permutation-equivariant | **no effect, by proof**. "Physical position" is not a variable of this model |
| **A. embedding interchange** (swap E_q rows of slots i and j) | Equal to permuting the two slots' contents (Theorem P) | Gate use moves with the embedding, by proof |
| **B. content interchange** (swap the two slots' question content) | With E_q fixed, slot i's content is processed with slot i's code. Contents are i.i.d. across slots | No systematic effect: ability does not follow content, because content is exchangeable [proof] |
| **D. joint swap** (content and embedding together) | Equal to a relabelling of the output indices | Exactly the original computation, permuted [proof] |

**Answer to Q7 [proof].** Relational ability follows the **question embedding E_q**, interacting with the slot's own content and the shared weights. It never follows "physical position". It follows content only through E_q.

**This alone does not distinguish H2 from H3** (mandate §5.3). A full code swap moves whatever the code does.

**Numerical verification:**
- Theorem P: max |difference| ≤ 1.2·10⁻⁶ over three permutations.
- The full-window relabel equals the E_q-row swap (3·10⁻⁷).
- At the C_s202 partial checkpoint, full-window code exchange gives recipient F_G 0.99–1.00 and donor 0.00 over 9 pairs [dev]. P1 is the frozen check of this in fresh models.

## 2. Code interpolation, with controls [dev] (`logs/expl/dev_interp_C_s202_1000.txt`)

Setting: C_s202 step 1000, with E_q[j] replaced by E_q[j] + α(target − E_q[j]):

| target | recipient F_G (α = 0.25 → 0.9) | donor F_G |
|---|---|---|
| an acquired slot's code | 0.00–0.01 → 0.09–0.17 | falls from 0.95–1.00 to 0.13–0.20 |
| another unacquired slot's code | 0.00 throughout | — |
| a random direction of equal norm | ≤ 0.04 | — |

**Reading.** Moving one code toward another does not transfer the computation. As the codes converge, the two slots become indistinguishable, and **both** lose gate use. The code carries slot identity (the binding of a slot's tokens to its readout) as well as access. That is why the clean test is the **exchange** of codes within a layer window (CROSS_SLOT_CAUSAL_TRANSFER.md), which keeps every slot's identity distinct.

## 3. Fresh-model results

After scoring: P1 in FROZEN_CONFIRMATION_AND_GENERALIZATION.md.
