# 19 — Unresolved mechanism space

Unresolved means a class was not given a proof or a counterexample, or a proof was not matched to a trained computation. It does not mean the original signed bridge is open inside the classes already ruled out.

## Open because they were not derived

- Stacks of blocks. A coordinate dropped in layer `\ell` can re-enter through a residual skip and be read by layer `\ell+1`. The one-block fiber counterexample does not say whether a particular multi-layer projection becomes sufficient. It says sufficiency is still fiber constancy at the layer where the intervention is defined.
- Parallel attention and MLP branches that do not see each other's outputs. Interaction between those branches is then interaction in the sum, which is zero for a linear readout of the sum, and need not be zero after the next LN. That case was not written out.
- Post-norm residuals. LN after the addition mixes the skip and the branch before the next block. The sign fact still applies to that LN. The path decomposition changes.
- RMSNorm blocks. Only the closed form of one normalization was checked. Plain RMSNorm preserves linear signs; with `\gamma` it need not. The rest of an RMSNorm block was not derived.
- Position encodings other than an additive mask term `B`. Rotary maps act on `Q` and `K` before the product. They are another way residual coordinates enter the pattern. They were not expanded.
- Multi-head composition with a shared residual, and grouped or multi-query key layouts.
- The operating point of any trained model: whether a given finite intervention lands where `I` is zero, whether `\gamma` at that point reverses a particular readout, whether a particular head has `W_Q` small on the edited subspace.

## Open because identification was not solved

- A transport `\rho` fixed from weights, shared across inputs, and tested on held-out inputs, can be a definite descriptive model of a projection. Whether any such transport's residual is constant on the fibers of a named `R` is not answered by existence of the transport. This step did not search for one.
- Approximate sufficiency with an explicit modulus, on a restricted family where the exact fiber condition fails by a small gap. The two-token counterexample is not a small gap. A small-gap theory would need a Lipschitz constant of the block in the dropped coordinates. That constant is a sensitivity statement, not a residual.

## Closed, and not reopened by the items above

- Universal `D\downarrow\Rightarrow L\uparrow` and `O` as a signed cause.
- Nonlinearity as a sufficient condition for nonzero finite interaction.
- `M` as an input to `O`.
- Fixed-readout sign invariance under LayerNorm.
- `ker P_u` as the kernel of a different residual.
- A recomputed structural residual as a predictor of a non-constant response.
- A scalar relation as extra information beyond its tuple.
- Vector agreement as readout redundancy.
- An arbitrary one-point `\rho` as evidence of a computational transport.

## What a future Step 1 argument would have to add

A restriction that is checkable from the architecture or from natural activations, that does not mention `sign(R)`, and that implies fiber constancy for a statistic that is not already the Jacobian, the path effect, or the full state. Multi-layer structure is a place such a restriction could live. It does not live there automatically. Until the restriction is written and the counterexample attempt fails, the class stays unresolved rather than promising.

No item in the open list is a reason to start Step 2, and none is a reason to run a Transformer experiment inside this step.
