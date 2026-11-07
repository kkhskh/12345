# Reviewer 4 — Representation

Provenance: same subagent as Reviewer 3, [causal and representation review](b51bc858-1cf6-4f05-b506-725c1387427a). The representation half is separated here so the objection is not merged into the causal one.

## What held

Changing `M` does not change `O`. A coordinated change of coordinates, `d`, `W`, and `M` is a different operation. A scalar relation does not add information beyond the tuple. Compression and invariance were already described as coarser summaries, not as extra information.

## Defects

1. `12` said the rank-one `\rho` is identified only for the fitting pair. The same matrix also zeros `(t z_u, t z_v)` and `(z_u+u_\perp, z_v)` for `u_\perp` orthogonal to `z_u`.
2. `13` said the hidden vector changes, without a quantifier. On the inactive set of a bias-free ReLU both hidden vectors are zero. The identity also needs `b_1=0`, while the MLP in `04` has biases.
3. `12` equated "P preserves the coordinates the family reads" with fiber constancy of `R`. The first is a claim about the structural maps. The second is a claim about the response.

## Disposition

`12` now states the larger zero set of the rank-one map and separates the architectural hypothesis from fiber constancy. `13` now restricts the rescaling identity to bias-free ReLU, records that inactive units do not move, and names the tested preactivation `0.65`.
