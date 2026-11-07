# 16. Reduction to known theory

## Support functions and affine dependence

Theorem S1 is the definition of a support function on a subset of a product. Comba and Stolfi, in the SIBGRAPI 1993 paper on affine arithmetic, encode a quantity as a center plus coefficients of shared noise symbols so that the joint range can be smaller than the product of the separate ranges. Their running example is that interval arithmetic treats `x` and `10x` as independent and inflates `x(10x)`. Sharing the symbol is the linear case of refusing the product.

Theorem S4 is that statement for a Euclidean ball: the support in direction `w` is `ρ ||(P+Q)^T w||`, not the sum of the norms. A zonotope, the usual set in neural-network verification, uses an `ℓ_∞` parameter and an `ℓ_1` support. The geometry differs by that norm. The dependence principle is the same. Singh, Gehr, Mirman, Püschel, and Vechev, NeurIPS 2018, build the DeepZ verifier on zonotopes precisely because affine layers preserve the joint generators, and they give abstract transformers for tanh. Mirman, Gehr, and Vechev, ICML 2018, use differentiable abstract interpretation for certified training. Gehr, Mirman, Drachsler-Cohen, Tsankov, Chaudhuri, and Vechev, IEEE S&P 2018, is the AI2 paper those works extend. This package does not implement those transformers and does not claim their theorems.

## What the line argument adds, and what it does not

The Transformer-specific work is the reduction: fixed causal pattern, isotropic value map, rank-one MLP, invariant mean-zero line, and the sign of two scalar derivatives. After that reduction, the gap is Theorem S1. No new abstract inclusion principle is involved.

The result is not a causal abstraction in the sense of Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard, JMLR 26(83), 2025. There is no high-level variable, no interchange intervention, and no faithfulness score. It is not the distributed-abstraction statement of Sutter, Minder, Hofmann, and Pimentel, NeurIPS 2025, Theorem 1, and it is not a response field in the sense of the Olivieri and Pérez Rodríguez preprint. The architectural definitions are the pre-norm residual of Xiong et al., ICML 2020, LayerNorm as in Ba, Kiros, and Hinton 2016, and causal scaled dot-product attention as in Vaswani et al. 2017. With the pattern fixed, the value path is the linear OV path of Elhage et al. 2021. Those citations identify the block. They are not the gap.

Nothing here distinguishes a correct decision from an incorrect one, and nothing reinstates `O_c` or `D_c`.
