# 9. Shared line, not a shared-subspace principle

On the structural family, `a(δ)` and `b(δ)` both lie in `span{n}`. Membership in one line does not create a support gap. The negative-gain weights of note 8 use the same line and attain the product corner.

Orthogonality is not the mechanism either. The increments are parallel. A readout orthogonal to `n` sees neither of them, so both the joint support and the marginal sum are zero. The gap vanishes because the objective vanished, not because the pathways decoupled.

The additional ingredient is orientation: the scalar coefficients of `n` have opposite monotonicity. That orientation is state-dependent off the line. In width 3, the singular value map of `proofs/examples.py` keeps only one mean-zero axis. The reflections

```text
r (cos θ n1 + sin θ n2),    r (cos θ n1 - sin θ n2)
```

at `r = 0.4` and `θ = 0.7` produce the same attention increment

```text
a = (0.93664874, -0.93664874, 0)
```

and MLP increments that are negatives of each other,

```text
b = ±(0.10776477, 0.10776477, -0.21552954).
```

Same projected attention state, different joint pairs. A fixed linear subspace of attention increments does not determine the MLP increment once the value map has a kernel.

No global linear subspace is claimed for a nonlinear block outside these constructions. The line result is global in the step and local in the direction: every magnitude along `n`, no neighborhood off `n`.
