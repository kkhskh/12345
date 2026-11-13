# Causal mechanism reconstruction (Workstream C1–C2)

**Interventions.** All interventions are natural forward patches at the input of the trained downstream block. A
specified set of token states is taken from the edited (post-edit) episode, and everything else from the pre-edit
episode. The block, the final LayerNorm and the readout are then recomputed in full. Every patched token value is an on-
support state produced by a real input, so no mean-ablation is used for any necessity claim.
- **Attention freeze** reruns the post-edit episode with the first downstream block's attention pattern replaced by its
  pre-edit pattern. It is surgical, not natural, and is reported as such.
- Responses are mean changes in the oriented candidate margin, using `scripts/mech_native.py`'s margin definition. Its
  Bayes values are R1 +0.36, R2 −4.11, R3 −3.45 on eval_dev pairs. They differ in scale from the evaluation-suite dm
  because the margin definitions differ, and are comparable only within this document.

**Evidence levels.** [D] development: xP81, a complete parent, plus xP80. [C] confirmed: CP5 and NP2/NP3 in
`PROSPECTIVE_PREDICTIONS.md` Part II.

## 1. Original transfer C1 (frozen pretrained layers 0–1 + fresh layer 2): the failing computation

**Token-patch shares** of the full R1 / R2 / R3 response [D, xP81]:

| patched set (post-edit values) | R1 | R2 | R3 |
|---|---|---|---|
| everything (full response) | −0.21 | −2.87 | −2.62 |
| readout token q | 0.99 | 0.66 | 0.68 |
| edited group g's 12 source tokens | 1.35 | 0.25 | 0.24 |
| g's 2 source tokens at q | 2.2 | 0.28 | 0.28 |
| the other groups' 24 source tokens | **0.08** | 0.04 | 0.01 |

**Routing.** Head-averaged attention from readout token q onto a group's source tokens:
- by own count: 0.22, 0.28, 0.38, 0.51, 0.54 for n = 0…4, then **0.18 for n = 5**;
- a perfect group gets 0.135 when no other group is perfect and 0.326 when another is;
- on F45 pairs the edited group's mass falls 0.54 → 0.32, **identically in R1, R2 and R3**.

**Attention freeze.** It removes 83 % (R2) and 91 % (R3) of the responses, but only 42 % of the wrong-signed R1
response.

**Reconstruction.**
- The relation-relevant input is the post-edit readout token: perfectness of every group, which the frozen pretrained
  layers already encode.
- The block turns it into a *routing reallocation away from the newly perfect group*. That reallocation does not depend
  on the regime.
- Since the R1 and R2/R3 regimes differ in the sign Bayes requires, but the routing change is identical across them, the
  routing pathway gives every regime the same (negative) sign. That is correct for R2/R3 and wrong for R1.
- The edited group's own tokens add a further discount through value and MLP pathways.
- **The frozen prefix carries no cross-group propagation** into other groups' tokens (share 0.08). The only place the
  conjunction could be computed is the single downstream block, and there it is computed as a regime-blind routing rule.

## 2. Native fine-tuning (full_c1): the repaired computation [D on xP81; NP1–NP4 on new parents]

| patched set | R1 | R2 | R3 |
|---|---|---|---|
| everything (full response; Bayes +0.36 / −4.11 / −3.45) | **+0.19** | −3.90 | −3.43 |
| readout tokens | −0.06 | −2.10 | −1.75 |
| edited group's source tokens (all q / at q) | −0.37 / −0.46 | −0.26 / −0.32 | −0.65 / −0.70 |
| **other groups' 4 source tokens at q** | **+0.30** | −0.87 | −0.57 |
| everything **except** those 4 tokens | **−0.57** | −2.50 | −2.60 |
| attention freeze | −0.08 | −0.05 | −0.10 |

**Reconstruction.**
1. Fine-tuning changed the pretrained layers so that **an edit to group g changes the states of the other groups' source
   tokens at the target question**. A cross-group write, absent in the frozen prefix, now happens *inside layers 0–1*.
2. Those 4 token states are
   - **sufficient** for the correct-sign R1 credit: alone they give +0.30;
   - **necessary** for it: without them R1 reverts to −0.57, the separable discount;
   - **selective**: R2/R3 keep 64–76 % of their response without them.
3. The edited group's own tokens still carry a discount (−0.37 to −0.46). The correct R1 response is the *sum of two
   opposing pathways*: a per-group discount and a cross-group credit. That is exactly the decomposition of Theorem 1,
   p ∝ S·A_y − B_y. The cross-group credit corresponds to the S·A term, which must reach every group's evidence, and the
   discount to −B_y, which is group-local. The correspondence is interpretive. No subspace was shown to *encode S*.
4. Group-level routing mass is still regime-independent (0.60 → 0.35 in all regimes; `scripts/mech_routing2.py`). Yet
   the attention freeze removes everything. The regime-dependent signal therefore travels *through* attention as a change
   in the transported values (the other groups' modified q-token states), not as a change in group-level attention mass.

**Classification (mandate B3).**
- On the development parent this is **Type III**: modified native layers learned a useful new relational computation.
  It also has the main elements of **Type IV**: a causally identified native pathway, with natural-patch necessity,
  sufficiency and selectivity.
- Whether it recurs on independent parents is tested by NP2. If it does, the Type IV claim is restricted to: token-level
  pathway identified; computation inside layers 0–1 not decomposed into heads, MLP or LayerNorm; no subspace-level
  account.

## 2b. Confirmation on independent parents (NP2, NP3, CP5; `PROSPECTIVE_PREDICTIONS.md` Part II)

**The development pathway did not recur (NP2: 0/4).** On yP130/131/132/134:
- patching only the other groups' q-tokens gives +0.01 to +0.08;
- keeping them at pre-edit values still leaves a positive R1 response (+0.11 to +0.42).

Where the R1 credit lives (exploratory, after unblinding; R1 margin response of each patch / full):

| parent | full | readout tokens (all) | edited group's tokens | all source tokens | other groups' source tokens | attention frozen |
|---|---|---|---|---|---|---|
| yP130 | +0.22 | +0.06 | −0.16 | +0.04 | +0.06 | **−0.37** |
| yP131 | +0.17 | +0.12 | −0.04 | +0.02 | +0.05 | **−0.26** |
| yP132 | +0.43 | +0.24 | +0.15 | +0.21 | +0.06 | **+0.07** |
| yP134 | +0.54 | +0.05 | +0.04 | +0.38 | +0.35 | **+0.06** |
| xP81 (dev) | +0.19 | −0.05 | −0.37 | +0.23 | +0.33 | −0.08 |

**Reading.**
- *Native implementations differ across independent parents.* On one parent the credit is carried by other groups'
  source states, on another by readout states, and on two it is distributed: no single token set carries it, and the
  parts are strongly non-additive.
- *What recurs:*
  - (i) the R1 credit depends on the downstream block's attention pattern (freeze removes it: 4/4 here, 5/5 including
    dev);
  - (ii) a frozen prefix never carries cross-group information (NP3, 4/4);
  - (iii) the C1 failure's R2/R3 responses travel through attention routing (CP5b, 6/6).
- The mandate's case of "a native mechanism that differs from the small reader's" is realised here as a set of
  *different native mechanisms with the same behaviour*. Behavioural equivalence holds. Implementation equivalence at
  the token-pathway level does not.

**CP5 (failure mechanism of C1) as frozen: falsified, 2/6 parents meet all three parts.**
- Routing carries R2/R3: 6/6.
- The freeze makes R1 less negative: 5/6.
- The perfect-group mass ratio is < 0.6 in only 3/6 (0.33–0.43; the misses are 0.61–0.67).

The routing description of §1 therefore holds qualitatively. The specific "≤ 0.6 × collapse" threshold does not.

## 3. Native-state readers (Type II)

The q-token MLP reaches Bayes on frozen native states (`PCOUNT_FAILURE_DIAGNOSIS.md` §2). Its internal mechanism was not
dissected. It establishes sufficiency and accessibility, not a native mechanism.

## 4. Bypass controls (mandate B2)

| category | example | how its dependence on native states was checked |
|---|---|---|
| Type I (external bypass) | Investigation VIII radapt | parent-mean replacement leaves the output unchanged (Investigation VIII) |
| Type II (frozen native states → new computation) | mlp_q on final states | receives only native states, so no side channel exists. Shows sufficiency; not a repair of the original computation |
| Type III/IV (modified native layers) | full_c1 | input is raw tokens only. Its correct R1 depends on native token states created by its own fine-tuned layers (necessity patch above). No component sees engineered inputs, so a bypass is impossible by construction |

## 5. Not done

- Head-level, MLP-level and LayerNorm-level decomposition inside the fine-tuned layers 0–1.
- Subspace identification of an S-like variable.
- Redundancy analysis across heads.
- Mechanism tracking across checkpoints.
- Activation-scale sensitivity.
