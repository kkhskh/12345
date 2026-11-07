# 01 — Corrections to four Step 1B errors

Step 1B remains where it was written. This file records what in it does not follow, and the statements that replace those sentences. The checks named here are in `tests/test_step1c_math.py`. The short proofs are in `proofs/`. A passing check is agreement on a finite case. The general claims are the proofs, and they have the hypotheses written there.

The four corrections were rechecked against each other before the mechanism sections were written. Nothing below depends on a sign prediction for the original obstruction.

## Error 1 — Nonlinearity was treated as interaction

Step 1B, `04_bottom_up_mechanisms.md`, says that if the next map is a nonlinearity of a sum, mixed derivatives generally do not vanish, and that the architecture predicts nonzero interaction for parallel addends that meet in that nonlinearity.

That prediction is false as stated.

`F(u,v)=u^2+v^2` is nonlinear. Its four-term interaction at baseline zero is identically zero, because the map is a sum of a function of `u` and a function of `v`. `F(u,v)=(u+v)^3` is not separable. At `(1,-1)` the four-term interaction is zero, and at `(1,1)` it equals `6`. A third map, `uv-u^2v-uv^2`, has mixed partial `1` at the origin and interaction `0` at `(2,-1)`.

The distinctions that have to be kept separate are:

| property | what it says |
|---|---|
| nonlinearity | some derivative of order at least 2 is not identically zero, or the map is not affine |
| nonseparability | `F` is not `g(u)+h(v)` |
| nonzero mixed partial at a point | the second-order cross term at that point is nonzero |
| nonzero finite interaction | `I_F(u,v)\neq 0` at a named pair and a named baseline |
| context-dependent gating | the map applied to one argument depends on the other through a discrete or saturating rule |
| causal interaction for a pair | the four-term difference of a named intervention pair on a named output |

For a `C^2` map on an open rectangle containing the origin, the interaction function is identically zero if and only if the mixed partial is identically zero if and only if the map is separable there. Identically zero is not the same quantifier as zero at the intervention used in an experiment. A nonzero mixed partial at the baseline controls the infinitesimal interaction, not every finite pair.

An elementwise activation on separate channels is separable across those channels. A shared preactivation can be locally affine, in which case the four-term interaction of increments that stay in that piece is zero. `relu(u+v)` has interaction `0` at `(1,1)` and `-1` at `(1,-2)`.

Proof: `proofs/interaction_conditions.md`.

## Error 2 — A change of `M` was entered as a change of `O`

Step 1B, `16_representation_invariance.md`, has the cell "Euclidean `O`" under "changing `M` with `s` fixed" marked "No (CE-07)".

`O=(ds)^\top W(ds)` does not depend on `M`. With `d`, `W`, and `s` fixed, changing `M` leaves `O` where it was. The same edit can change `D=\min_{g\in\ker d}\lVert s-g\rVert_M^2`.

Exact case: `s=(1,0)`, `d=(1,-1)`, `W=(1)`, so `O=1`. For `M=I`, `D=1/2`. For `M=\mathrm{diag}(10, 0.1)`, `D=10/101`. `O` is still `1`. Replacing `W` by `(4)` changes `O` to `4` and does not change that value of `D`.

CE-07 is a different edit. It changes the matrix that enters `D` and, separately, the matrix that enters `O`. A coordinated change of the coordinates of `s`, of `d`, of `W`, and of `M` is a third operation. Step 1A's congruence calculation applies to that third operation. It is not a theorem about editing `M` alone.

The corrected row is: changing only `M` leaves `O` fixed and can change `D`. Changing only `W` can change `O` and leaves `D` fixed.

Proof: `proofs/metric_split.md`.

## Error 3 — A fixed readout was marked invariant under LayerNorm

The same table says the sign of `L` survives LayerNorm.

A fixed linear form applied before and after LayerNorm need not keep its sign. For `h=(1,0)` and `w=(1,2)`, `w·h=1` and `w·LN(h)=-1` at `eps=0`. A small `eps` does not put the second scalar back above zero.

This is LayerNorm as a computation, with the readout held fixed. It is not the dual reparameterization `w'=A^{-\top}w`, which does preserve `w·h` when `h'=Ah` and `A` is invertible. It is also not an invertible coordinate change: `LN(ah+b1)=LN(h)` for `a>0`, and the Jacobian kills both the all-ones direction and the centered vector.

Learned `\gamma` and `\beta` are part of the computation when the block has them. They can reverse a sign that plain LayerNorm kept: `h=(1,0)`, `w=(1,0)`, `\gamma=(-2,1)`, `\beta=0`.

Plain RMSNorm, with no `\gamma`, is a positive rescaling, so it does preserve the sign of every linear readout. That fact is special to the missing mean subtraction. It is not a reason to put the LayerNorm cell back to "Yes". A `\gamma` on RMSNorm can reverse the sign. No full RMSNorm block was derived here.

Proof: `proofs/layernorm_readout.md`.

## Error 4 — `ker P_u` was treated as invisible to the residual

Step 1B, `proofs/N3_leftover_does_not_label_its_source.md`, says that a kernel displacement of `P_u` does not change `r` at first order when `f` and `\rho` are linear.

The measurement of that residual is the operator `M_r=P_v A-\rho P_u`, not `P_u`. A displacement leaves `r` unchanged exactly when it lies in `\ker M_r`. Membership in `\ker P_u` cancels only the `\rho` term. With `A=I`, `\rho=0`, `P_u=(1\ 0)`, `P_v=(0\ 1)`, the vector `(0,1)` is in `\ker P_u` and changes `r` by `1`.

The narrower counterexamples stay. N1 still says that no function of `(Q(h),Q(h+\Delta))` recovers both signs when `\Delta` and `-\Delta` sit in `\ker Q` and the response is a nondegenerate linear form. Two maps can share a recomputed structural residual of zero and have interactions `0` and `2`. Those facts do not identify the source of a leftover, and they do not extend the kernel from `P_u` to `r`.

Proof: `proofs/measurement_kernel.md`.

## Internal consistency of the corrections

The four replacements do not contradict each other.

- Error 1 is a statement about a scalar function of two scalar interventions. It does not mention `M`, LayerNorm, or `P_u`.
- Error 2 is a statement about which matrix sits inside which quadratic form. The interaction of a later readout is not that quadratic form.
- Error 3 says a fixed linear form can change sign when its argument is replaced by LayerNorm of that argument. Error 1's interaction is computed on whatever function is actually being evaluated. If that function includes LayerNorm, the interaction is the interaction of the composed function, which may be nonzero even when each piece looks simple. The sign fact is not an interaction theorem.
- Error 4 says the kernel has to be the kernel of the operator that is claimed to be invariant. If the claimed invariant is an interaction, the operator is the four-term map, not `P_u`. If the claimed invariant is `O`, the operator depends on `d` and `W`, not on `M`.

No correction was used as a premise for a signed obstruction law. The original implication from a decrease in `D` or `O` to an increase in a decision margin stays false, for the reasons already recorded in Phase 1 and Step 1A.
