# 22 — Claim register

| ID | Claim | Status | Where |
|---|---|---|---|
| N1 | No function of linear `Q` recovers `sign(L(Δ))` for every linear `L` with `Δ∈ker Q` and `L(Δ)≠0` | PROVED | proofs/N1_blind_statistic.md |
| N1-NL | Nonlinear orientation-reversal extension | CONJECTURE | proofs/N1 |
| N2 | Existence: joint factors sufficient when each factor is not; object is the tuple | PROVED | proofs/N2_joint_sufficiency.md |
| N3 | One leftover / residual does not determine `I` or `sign(ΔL)` | PROVED | proofs/N3; two-`I` test |
| AFF | `J h` need not equal `f(h)`; affine linearization is `f(h0)+Jδ` | PROVED | test_affine_offset… |
| FACT | Lip bounds on `z=Ph` need not bound `L` unless `L` factors through `z` | PROVED | test_lipschitz_bound_requires_factorization |
| TRUNC | Two-node truncated `do` is defined; nested `I` equals minus the parent effect | PROVED_UNDER_RESTRICTED_ASSUMPTIONS | TruncationSCM, not a general DAG |
| I-LIN | Affine `F` of two free arguments has `I=0` | PROVED | identity; 50 draws are a check, seed 20261107 |
| I-SQ | `y=(u+w)^2` has `I=2` at unit clamps while additive residual is 0 | PROVED | tests |
| PATH | Linear ablation effect equals `w·p_i` | PROVED | tests |
| MAJ | Majority vote corrects Hamming-1; pairwise disagreement does not rank correctness | PROVED | tests |
| LN | LayerNorm changes Euclidean distances and inner products | PROVED | tests |
| ATTN | Softmax mix is not a shared linear `ρ` on two patterns | PROVED | tests |
| BLOCK | A tanh-attention plus ReLU-MLP residual step is not a constant linear map | PROVED | tests |
| D-UNIV | Unrestricted `ΔD<0⇒ΔL>0` | DISPROVED | Step 1A CE-02, reproduced |
| O-RANK | Unrestricted ranking by `D` | DISPROVED | CE-01 |
| Z-SCOPE | Theorem Z ranks decisions | DISPROVED on its domain (residual is 0) | Step 1A |
| P2-PT | Pointwise Jacobian stability formula | DISPROVED | `L=1-2t^2` |
| H-SUF as COR | Sufficiency is a new obstruction mechanism | REDUNDANT_WITH_EXISTING_THEORY | causal abstraction |
| H-COMPOSE as COR | Shared-bottleneck `I` is a new obstruction mechanism | REDUNDANT_WITH_EXISTING_THEORY | Hessian / mediation |
| H-CORR | Transformers implement majority-like correction | EMPIRICALLY_UNTESTED | no model run |
| SEP | A functional of `(P,ρ,s)` separates surrogate error from causal interference | UNIDENTIFIED | file 19 Case B |
| UNIQUE-Z | A unique minimal causal state for a real checkpoint | UNIDENTIFIED | field-level |
| ENERGY | Forward pass is descent on `D` | DISPROVED as an architectural claim for a standard block | file 03 |
| SHEAF | `[ds]` is a nontrivial class for a global 0-cochain | DISPROVED | Phase 1 / MVP |

Randomized tests support I-LIN only as a check of that identity, not as a universal neural theorem. Seed 20261107, 50 draws of 3-dimensional linear maps, zero failures at `1e-10`.
