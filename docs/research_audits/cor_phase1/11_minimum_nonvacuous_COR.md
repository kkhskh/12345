# 11 — Minimum nonvacuous COR

This is not a validated theory. It is the smallest package that is typed, nontrivial, and not immediately vacuous.

## 1. Surviving mathematical core

Objects: a frozen directed graph `K`, linear maps `P_v`, `ρ_e`, PD metrics `M,W` (or Euclidean with the admission that they are coordinate-relative), coboundary `d`, energies

```text
O(s)=‖d s‖_W^2
D(s)=(d s)^T (d M^{-1} d^T)^+ (d s)
```

with `λ_min^+ D ≤ O ≤ λ_max D` and `O=0 ⇔ D=0`.

This is weighted graph-consistency / sheaf Laplacian energy. It is ordinary linear algebra.

## 2. Assumptions that must be added (not in original Def. 2–4)

| Assumption | Class | If removed |
|---|---|---|
| `P,ρ` frozen on training data disjoint from evaluation outcomes | Identifiability | Attacks E,F, leakage |
| Graph `K`, ranks, `M,W` frozen before outcomes | Identifiability | Attacks C,G |
| Maps do not see evaluation `Δm` | Causal / statistical | Circular `S` |
| `O` is a **monitor residual**, not a decision sufficient statistic | Empirical hypothesis | Attacks A,B,H |
| Gauge: either transform `(M,W)` or restrict claims to one frozen basis | Representation | Attack D |
| No cohomological interpretation of `O(s)` for full 0-cochains | Mathematical regularity | Contradicts MVP §3 |

Substantive scientific claim, if any: **this frozen residual sometimes correlates with conflict beyond margin**. That is 2B, not a local-to-global decision theory.

## 3. Minimality test

Remove freeze of `P,ρ`: vacuous (E,F).
Remove freeze of `M,W`: ranking arbitrary (C).
Add “logits factor through `ker d`”: restores a causal implication, but that assumption **is** the desired conclusion in disguise unless independently tested by interchange.
Remove spectral sandwich: still have `O` as a residual; lose the `D` story, which adds nothing causal.

## 4. What must not be smuggled

“Obstruction predicts decisions whenever the model uses obstruction to decide” is empty.

The minimum package does **not** contain a noncircular route from `D` to `F_θ`. It contains a frozen quadratic residual that can be computed before an intervention. Whether that residual equals anything about decisions is an empirical hypothesis already tested, with mixed monitoring results and negative steering results.

## 5. Relation to original COR

Original COR = this Laplacian residual **plus** an unproved decision-formation thesis **plus** a response functional that mixes `Δm` **plus** sheaf language.

After the audit, only the residual plus freeze-protocol survives. The thesis does not.
