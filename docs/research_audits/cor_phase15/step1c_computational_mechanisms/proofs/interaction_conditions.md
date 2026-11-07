# Interaction conditions

Baseline is `(0,0)`. For `F` defined on a product set,

\[
I_F(u,v)=F(u,v)-F(u,0)-F(0,v)+F(0,0).
\]

The same formula with a translated function gives the interaction of two increments around another baseline. The baseline is part of the definition.

## Separable maps

If `F(u,v)=g(u)+h(v)`, then `I_F(u,v)=0` for every `(u,v)` in the domain. The proof is substitution. No smoothness is required.

`F(u,v)=u^2+v^2` is nonlinear and separable, so `I_F` is identically zero. Checked for a grid of pairs in `test_error1_separable_square_has_zero_interaction`.

## Polynomials

Let `F=\sum_{i,j\ge 0} c_{ij} u^i v^j`. Then

\[
I_F(u,v)=\sum_{i\ge 1,\,j\ge 1} c_{ij} u^i v^j.
\]

So `I_F` is the zero polynomial if and only if every monomial that contains both variables has coefficient zero. Nonlinearity is the presence of some degree at least 2. Nonseparability is the presence of a mixed monomial. They are different conditions.

## Twice differentiable maps on a rectangle

Let `U` and `V` be open intervals containing `0`, and let `F:U\times V\to\mathbb{R}` be `C^2`. The following are equivalent.

1. `I_F` is identically zero on `U\times V`.
2. `\partial^2 F/\partial u\partial v` is identically zero on `U\times V`.
3. `F(u,v)=g(u)+h(v)` on `U\times V`.

(1) implies (2) because the three terms `F(u,0)`, `F(0,v)`, and `F(0,0)` do not depend on both variables, so the mixed partial of `I_F` equals the mixed partial of `F`. (2) implies (3) because `\partial F/\partial u` does not depend on `v`, hence `F(u,v)-F(0,v)` is independent of `v`. (3) implies (1) by the separable case.

This equivalence is about the whole rectangle. It is not a statement about one pair `(u,v)`.

## What a single pair does not say

`F(u,v)=(u+v)^3` has `I_F(u,v)=3uv(u+v)`. Therefore `I_F(1,-1)=0` and `I_F(1,1)=6`. The map is nonlinear and not separable. Vanishing at one pair is compatible with a nonzero interaction at another pair. The mixed partial at the origin is zero, so this example does not yet separate a nonzero Hessian from a zero finite interaction.

That separation is `F(u,v)=uv-u^2v-uv^2`. Its mixed partial at the origin equals `1`, and `I_F(2,-1)=0`, while `I_F(1,1)=-1`. A nonzero mixed partial at the baseline implies

\[
I_F(tu,tv)=t^2 uv\,\partial_u\partial_v F(0,0)+o(t^2)
\]

as `t\to 0`. For all sufficiently small `t\neq 0` with `uv\neq 0`, the interaction is nonzero. A finite intervention can still cancel.

## Channels that do not mix

If `F(u,v)=g(u)+h(v)` with each of `g` and `h` nonlinear, an intervention on `u` and an intervention on `v` have interaction zero. An elementwise activation applied to separate coordinates is this situation. Meeting in a later nonlinearity produces a nonzero interaction function only when the joint map is nonseparable in the two intervened arguments. Even then, a particular intervention can land where `I_F` happens to be zero.

Checked by `test_error1_channels_that_do_not_mix_stay_separable` and `test_mlp_relu_is_not_automatically_an_interaction`.
