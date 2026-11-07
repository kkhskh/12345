# `O` does not contain `M`

Fix a coboundary `ds`, a weight `W` on that coboundary, and a state `s`. The obstruction used in the earlier audits is

\[
O(s)=(ds)^\top W\,(ds).
\]

The symbol `M` does not occur. Replacing `M` by any other matrix leaves `O` unchanged.

The distance is a different function:

\[
D(s)=\min_{g\in\ker d}\lVert s-g\rVert_M^2.
\]

`D` depends on `M`. It need not take the same value at two different positive definite choices of `M`.

## Exact numbers

Take `s=(1,0)`, `d=(1,-1)`, so `ds=1`, and `W=(1)`. Then `O=1`. The kernel is spanned by `(1,1)`.

For `M=I`, the minimizer is `t=1/2` and `D=1/2`.

For `M=\mathrm{diag}(10,0.1)`,

\[
t=\frac{10}{10.1}=\frac{100}{101},\qquad D=\frac{10}{101}.
\]

The same `ds` and `W` still give `O=1`. Replacing `W` by `(4)` changes `O` to `4` and leaves `D` at `1/2` when `M=I`.

Checked by `test_error2_changing_only_M_changes_D_and_not_O`.

## What this does not say

A simultaneous change of coordinates, of `d`, of `W`, and of `M` is a different operation. Step 1A's congruence calculation remains about that simultaneous change. CE-07 changed the matrix inside `D` and, separately, the matrix inside `O`. Citing CE-07 as evidence that "changing `M`" changes `O` conflates those two edits.

Changing only `M` can reorder states by `D` while leaving their `O` values fixed. Any table that reports an `O` reversal under a pure `M` edit is wrong.
