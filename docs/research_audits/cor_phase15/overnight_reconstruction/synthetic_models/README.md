# Synthetic models

No pretrained checkpoint is loaded. The only ensemble is `make_params` in `proofs/block.py`: NumPy Generator normals with scale `0.4`, `T=3`, `d=4`, two heads of width 2, MLP width 5, `eps=1e-5`, gains `1 + 0.05 ξ`, readout on token `T-1`.

Two further weight settings are built inside the tests, not sampled from a trained model.

- **Scaled dead attention.** `W_Q = W_K = 0`, MLP weights zero, `W_V` the split identity, `W_O = α outer(ΔX, w) / ||ΔX||^2`. This is the witness that a centered response is unbounded in the branch scale.
- **Finite kernel weights.** Start from seed 1, then project `W_Q`, `W_K`, `W_V` off the finite LayerNorm increment of one chosen non-constant `δ`, and project `W1` off the induced second LayerNorm increment. The resulting block is still a pre-norm block. It is not a draw from the Gaussian ensemble.

Both constructions are recomputed by `tests/test_reconstruction.py`. Neither is a fit to an evaluation response.
