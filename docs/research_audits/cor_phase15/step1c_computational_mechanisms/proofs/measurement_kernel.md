# Kernel of the residual measurement

Let `A` be linear and

\[
r(h)=P_v A h-\rho P_u h.
\]

Then

\[
r(h+\delta)-r(h)=(P_v A-\rho P_u)\delta.
\]

Call `M_{r}=P_v A-\rho P_u` the measurement operator of this residual. The displacement `\delta` leaves `r` unchanged at every `h` if and only if `\delta\in\ker M_r`.

`\delta\in\ker P_u` makes the second term zero. The first term is zero only if `P_v A\delta=0` as well. A null direction of `P_u` is not automatically a null direction of `r`.

## Counterexample

`A=I`, `\rho=0`, `P_u=(1\ 0)`, `P_v=(0\ 1)`, `\delta=(0,1)`. Then `P_u\delta=0` and `r(h+\delta)-r(h)=1`.

The vector `\delta=(3,0)` does lie in `\ker M_r`, and `r` is unchanged along it. The hypothesis that has to be proved is membership in `\ker M_r`, not membership in `\ker P_u`.

Checked by `test_error4_kernel_of_one_projection_need_not_preserve_the_residual` and `test_error4_the_full_measurement_kernel_does_leave_the_residual_fixed`.

## What stays from the earlier note

Two different claims were packed into the Step 1B note `proofs/N3_leftover_does_not_label_its_source.md`.

The following remain valid and are not this kernel claim.

- If the predictor is allowed to see only `Q(h)` and `Q(h+\Delta)`, and the domain contains both `\Delta` and `-\Delta` in `\ker Q` with `L` linear and `L(\Delta)\neq 0`, then no function of that pair equals `\mathrm{sign}(L(\Delta))` on the whole domain. That is N1. Rechecked by `test_n1_narrow_counterexample_still_stands`.
- Two maps can have the same recomputed structural residual, namely zero, and different interactions. `u^2+v^2` has `I=0`. `(u+v)^2` has `I(1,1)=2`. The interaction is a property of `F` and of the intervention pair. It is not a label carried by the structural residual.

The sentence that does not survive is the claim that a displacement in `\ker P_u` leaves `r` unchanged, including the parenthetical in that note which treated linearity of `f` and `\rho` as enough. Linearity fixes the form of `M_r`. It does not put `\ker P_u` inside `\ker M_r`.
