# Proofs

`block_math.py` is the map that the tests differentiate. It is not a proof. The written arguments are:

| claim | location |
|---|---|
| LayerNorm Jacobian, kernel, rank, shift, scale | `02_layernorm_epsilon_theory.md` |
| Both-necessary versus interaction | `03_two_path_necessity_and_interaction.md` |
| Attention differential and the mask | `06_attention_qkv_total_derivative.md` |
| Residual, second norm, MLP | `07_residual_mlp_and_normalization_chain.md` |
| Finite response versus the derivative | `08_exact_vs_linearized_causal_response.md` |
| Register, with quantifiers | `16_theorem_register.md` |

A test that matches a finite difference checks the implementation of a derivative. It does not extend the identity to another architecture.
