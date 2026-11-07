# 19 — Literature

Citations were checked against the papers named below. No numbered theorem is attributed to a paper unless this note says what the paper uses that result for. Agreement with a known identity is not a COR theorem.

## The block

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin, NeurIPS 2017, define scaled dot-product attention. The formulas in `04` and the differential in `06` are that map, plus a causal mask and an output projection. The mask derivative is zero because the mask does not depend on `X`.

Ba, Kiros, and Hinton, 2016, define LayerNorm. R1–R5 are the calculus of that definition once `ε` is kept in the variance. Zhang and Sennrich, 2019, define RMSNorm, which does not subtract the mean. This block is not RMSNorm, and the shift proof uses the mean subtraction. A positive scale is a positive multiple, so it preserves the sign of a linear form. A diagonal gain can change a sign. That is the definition of those maps. It is not reused here as an `ε>0` rank claim.

Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, and Liu, ICML 2020, PMLR 119:10524–10533, define pre-LN as LayerNorm inside the residual branch and post-LN as LayerNorm between blocks. This step analyzes one pre-LN block. It does not prove their mean-field gradient bound and does not claim the shift identity for post-LN.

## Attention paths

Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, and Olah, "A Mathematical Framework for Transformer Circuits," 2021, decompose a head into a QK circuit and an OV circuit and observe that a frozen pattern makes the head's contribution linear in the source. They also add head outputs in the residual stream.

The value-only derivative in this step is that frozen-pattern term, pushed through the rest of the chain rule. Seed 16 is the case the freeze excludes: `Q` and `K` move, and the sign changes. The first-order sum across heads is their addition, differentiated at one point. The finite leave-one-out gap on seed 4 is outside that first-order sum, because the second LayerNorm and the MLP are applied to the sum and the leave-one-out networks are different functions. Nothing here replaces their decomposition.

## Abstraction and interchange

Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard, JMLR 26(83):1–64, 2025, give the correspondence between high-level and low-level interventions. R14 is the special case in which the proposed high-level variable is a coordinate projection and the required correspondence is equality of `R`. The seed-16 pair shows the correspondence fails. That is not a new abstraction theorem.

Geiger, Wu, Potts, Icard, and Goodman, CLeaR 2024, PMLR 236:160–187, learn a distributed alignment. This step does not learn one and does not test interchange on a trained model.

Sutter, Minder, Hofmann, and Pimentel, "The Non-Linear Representation Dilemma," NeurIPS 2025, prove that without a restriction on the alignment map the abstraction relation does not separate models. The rank-one `ρ` of catalog row 7 is solved from one pair, so its residual on that token is zero by algebra and its residual on the shifted state is not a discovery about the block. The citation is a warning against that construction. This step does not re-prove their theorem.

## Path-specific effects and sensitivity

A recomputed residual edit is a total effect on `m`. A value-only clamp is a different, path-specific intervention. Geiger et al. (JMLR 2025) already list path patching and mediation as instances of their framework. This step uses the distinction and does not add a new path-specific formula beyond the chain rule.

The split of `R` into `Dm·E` and a remainder is differentiability of this `tanh` block. The seed-0 numbers show the remainder can dominate a direction the derivative kills. That is Taylor's theorem applied to one map. Phase 1 already used the same distinction against a finite-response claim. It is not repeated as a new expansion.

## Historical record

The Direction 1 letters and the Phase 1 obstruction comparisons are constraints on interpretation. They are not measurements of this block. Attribution repair AUC `0.979` against the best obstruction-using score `0.955`, and the Spearman figures about `0.905` and `0.095`, stay the figures recorded in Phase 1. They are not refit. The pair `0.936` versus `0.141` is not used.

## What would have been new

A relation fixed without `m` and without `Dm`, other than the constant-shift symmetry, that bounded a general `R` more tightly than the quantities above. No row in the register is that relation. R6 is the symmetry, and it is the pre-norm residual.
