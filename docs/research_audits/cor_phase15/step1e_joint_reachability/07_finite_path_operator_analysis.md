# 7. Finite path, not the base derivative

The raw response along a segment is the integral of its derivative. At the special-gain read-token family, with the zero base,

```text
f'(0) = 158.113883008419,    g'(0) = -50316.22776601685,
R'(0) = -50157.11388300843,  R(1) = 0.818715171479619.
```

The base slope is largely negative. The response at step `1` is positive. A certificate that freezes the base derivative, or that treats opposite base slopes as a finite cancellation, is false on this family. The loop implementation's central difference at step `10^{-7}` agrees with `R'(0)` to a relative error below `10^{-3}`. That check confirms the derivative. It does not extend it.

Theorem S2 does not use the size of `f'(0)` or `g'(0)`. It uses the sign of those derivatives at every point of the segment. The sign is uniform because `λ' > 0` everywhere and the MLP factor `(-α) sech^2` does not change sign. The magnitudes change by orders of ten as LayerNorm leaves the origin: `λ'(0) = 1/√ε = 316.227766016838`, while `λ'(1)` is on the order of `10^{-5}`.

The chain-rule factorization `J_F = (I + J_M)(I + J_A)` is the derivative of the block, not an extra mechanism. On a linear block it is Theorem S4. On this nonlinear block the finite statement is the integral of two signed scalar derivatives, which happens to be elementary because the state stays on a line.
