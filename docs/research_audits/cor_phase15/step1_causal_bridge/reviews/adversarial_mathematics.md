# Reviewer D — adversarial mathematics

Independent pass. The reviewer tried to break Theorem NFL, Theorem Z, Theorem Idle, Theorem Sign, Theorem Descent, and the stability writeup.

## Break

**Update Proposition 2, as first proved in Family F.** The writeup assumed `|L(h+δ)-L(h)| ≤ sup |J(h) η|` and treated `J(h)` as a Lipschitz bound. Take `L(t)=1-2t^2`, `h=0`, `r=1`. Then `L'(0)=0`, `|L(0)|=1` clears the printed budget, and `L(1)=-1`. The same function with `τ O(0)=0.5` still flips. A larger curvature, `3-100t^2`, defeats any fixed penalty that has already been chosen. Where an increment really is bounded by `S`, deleting `τ O` does weaken the hypothesis, and the sign still holds. `τ O` is slack beside a real bound. It is not a bound on the remainder `L(h+δ)-L(h)-∇L(h)·δ`.

## No break, with the assumption that would break each one

- **NFL.** Survives for affine readouts. Forcing `L` to be linear through the origin kills arbitrary values at the zero cochain. Forcing `L=q(D)` kills arbitrary values at a fixed `D`. The phrase "pick any `δ`" was false; existence of one `δ` was the actual claim. "Rankings" and "intervention signs" were wider than the construction.
- **Z.** Survives under the map identity and recomputation. Leaving the child unrecomputed, or checking commutation at a single sample of a nonlinear `f`, produces a nonzero residual. Those escapes are outside the hypothesis.
- **Idle.** Survives when `r` is `M`-orthogonal to `ker d`. Dropping orthogonality lets `L(s)=s_1+s_2` move.
- **Sign.** Survives for `g` in the kernel and `r` orthogonal to it. The equality `D(g+r)=D(g-r)` fails if `g` is outside the kernel.
- **Descent.** The directional-derivative identity survives. A finite step can decrease `L` while the derivative along `-∇D` is positive: `D=(s_1-s_2)^2/2`, `L=-s_1-5(s_1-1)^2` at `(1,0)`, step landing at `(0,1)`, `L` from `-1` to `-5`. The "if and only if" is first-order only.

No other named identity was given a counterexample inside its hypotheses.
