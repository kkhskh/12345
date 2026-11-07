# LayerNorm and a fixed readout

Population LayerNorm with `eps=0`, on a non-constant vector, is

\[
\mathrm{LN}(h)=\frac{h-\mu(h)1}{\sigma(h)},\qquad
\mu(h)=\frac{1^\top h}{d},\qquad
\sigma(h)=\sqrt{\frac{\lVert h-\mu 1\rVert^2}{d}}.
\]

Learned parameters act after that map: `\gamma\odot\mathrm{LN}(h)+\beta`.

## Three operations

1. **Computation.** The block applies LN to an activation and a later linear map uses a fixed vector `w`. The scalar is `w·LN(h)`, or `w·(\gamma\odot LN(h)+\beta)` when the affine is present.
2. **Reparameterization.** An invertible linear change `h'=Ah` together with `w'=A^{-\top}w` preserves `w·h`. Both the representation and the readout move.
3. **Coordinate change of the activation only.** `h` is replaced and `w` stays. The scalar is not protected by the dual transformation.

Operation 1 is what a pre-norm block does. Operation 2 is a change of variables in an already written function. They are not the same theorem.

## Sign

Take `h=(1,0)` and `w=(1,2)`. Then `\mu=1/2`, the centered vector is `(1/2,-1/2)`, `\sigma=1/2`, and `\mathrm{LN}(h)=(1,-1)`. Hence `w·h=1` and `w·LN(h)=-1`.

The same `h` with `w=(1,0)` has `w·LN(h)=1`, still positive, but `\gamma=(-2,1)` and `\beta=0` give `w·(\gamma\odot LN(h))=-2`. The affine can reverse a sign that plain LN kept. Plain LN can reverse a sign with `\gamma=1` and `\beta=0`, which is the first example.

Checked by `test_error3_fixed_readout_need_not_keep_its_sign_under_layernorm` and `test_error3_learned_affine_can_flip_a_sign_that_plain_layernorm_kept`.

## A readout orthogonal to the all-ones vector

If `1^\top w=0`, `\gamma=1`, `\beta=0`, and `\sigma(h)>0`, then

\[
w\cdot\mathrm{LN}(h)=\frac{w\cdot h}{\sigma(h)}.
\]

The sign agrees with `w·h` for every non-constant `h`. The checked case is `h=(1,0)`, `w=(1,-1)`: the scalars are `1` and `2`. This does not restore the universal claim. `w=(1,2)` is not orthogonal to `1`, and that pair still flips.

## `eps`

The identities `LN(ah+b1)=LN(h)` and the exact kernel of the derivative are for `eps=0`. A positive `eps`, as in GPT-2, breaks exact scale invariance. The sign counterexample `h=(1,0)`, `w=(1,2)` was also evaluated at `eps=10^{-5}` and the second scalar stayed negative. That is one vector, not a theorem about every `eps`.

## Not a coordinate chart

For `a>0` and any scalar `b`,

\[
\mathrm{LN}(ah+b1)=\mathrm{LN}(h)
\]

whenever `h` is not constant. The map is not injective, so it is not an invertible coordinate transformation of `\mathbb{R}^d`.

Its derivative, where `\sigma>0`, annihilates both `1` and `h-\mu 1`. The second claim is positive homogeneity of LN in the centered vector. The Jacobian therefore has rank at most `d-2`. For `d=2` the map is locally constant off the constants: it only records which side of the diagonal the vector lies on. A finite-difference Jacobian at a non-constant vector in `\mathbb{R}^3` is checked in `test_error3_jacobian_of_layernorm_is_singular`.

## RMSNorm is a different map

`\mathrm{RMS}(h)=h/\sqrt{\mathrm{mean}(h_i^2)}` has no mean subtraction. For a positive denominator, `w·\mathrm{RMS}(h)=(w·h)/\mathrm{rms}(h)`, so the sign agrees with `w·h`. A diagonal `\gamma` removes that conclusion. The test uses `h=(1,3)` and `w=(1,-0.2)` for the plain sign, and `h=(1,1)` with `\gamma=(-1,1)` for the flip. The lemma is the positive-scale argument. It is not a claim that every normalization preserves sign.

RMSNorm was not substituted into a Transformer block in this step. The sign lemma is only the closed form above.
