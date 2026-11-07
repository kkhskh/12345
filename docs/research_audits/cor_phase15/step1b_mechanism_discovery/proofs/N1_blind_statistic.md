# Proof N1 — Blind statistic

Classification: `PROVED`.

**Statement.** Let `Q: H → ℝ^k` be linear and let `Δ ∈ ker Q` with `Δ ≠ 0`. There is no function `f` of the pair `(Q(h), Q(h+Δ))` such that, for every linear `L: H → ℝ` with `L(Δ) ≠ 0`,

```text
f(Q(h), Q(h+Δ)) = sign(L(h+Δ) - L(h)).
```

Equivalently, for one fixed linear `L` with `L` not identically zero on `ker Q`, no function of that pair recovers `sign(L(Δ'))` for every `Δ' ∈ ker Q` with `L(Δ') ≠ 0`, because the pair does not depend on which kernel vector was chosen and `Δ' ↦ -Δ'` flips the sign.

**Proof.** `Q(Δ)=0` so `Q(h+Δ)=Q(h)` for every `h`. For any linear `L`, `L(h+Δ)-L(h)=L(Δ)`. The pair is therefore constant in `L`. If `f` of the pair equalled the sign for some `L` with `L(Δ)≠0`, the same `f` would equal that sign for `-L`, whose sign is the opposite. Contradiction. QED.

For a **single** `L` and a **single** `Δ`, `sign(L(Δ))` is one number and a constant function of the pair matches it. Non-invariance of `Q` is therefore not necessary for matching a sign on a domain that contains only one sign. The necessity is for a domain that contains both signs: either `{L, -L}` at fixed `Δ`, or `{Δ, -Δ}` at fixed `L`.

The nonlinear sketch (replace `L` by a function that runs the other way on the line) is a **CONJECTURE** about classes closed under orientation reversal. It is not marked `PROVED`.

**What this is.** A necessary condition: a statistic that does not move when `L` moves cannot determine the sign of that move.

**What this is not.** It does not say that a statistic which does move with `L` is a mechanism. `Q=L` moves with `L` and is the outcome itself.

**Scope.** Unrestricted linear readouts. If a model class forbids `L` from depending on `ker Q`, the hypothesis of the theorem is false and the conclusion is idle. That restriction is causal sufficiency of `Q`, investigated in file 07.
