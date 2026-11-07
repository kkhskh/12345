# 10 — Counterexample catalog

All examples are finite-dimensional. Verification: `tests/test_cor_phase1_math.py`. Status tags: fatal for a universal causal claim; limiting for a restricted claim.

## Attack A — Same obstruction, opposite decisions

Claim under attack: obstruction determines the selected candidate.

Assumptions: two-node identity `ρ=1`, Euclidean `M,W`.

```text
d = [1, -1]
s = (1, 0.5)
O = (1-0.5)^2 = 0.25
w1 = (1, 0)  => L = 1
w2 = (-1, 0) => L = -1
```

Fatal for “`D`/`O` imply candidate selection” without a head-coupling assumption.

## Attack B — Lower obstruction, worse target logit

```text
s = (1, 0)     D > 0,  L = 1
s' = (0, 0)    D = 0,  L = 0
```

Reverse:

```text
s = (0, 0)     D = 0, L = 0
s'' = (4, 0)   D > 0, L = 4
```

Fatal for “reducing `D` repairs behavior” and for “increasing `D` harms the target.”

## Attack C — Metric ranking swap

Two disconnected identity edges. `s1` mismatches only edge 1; `s2` only edge 2.

```text
W_a = diag(10, 1)  => O(s1) > O(s2)
W_b = diag(1, 10)  => O(s1) < O(s2)
```

Same swap for `D` with node weights `M`. Fatal for intrinsic candidate ranking. Blocked only by a frozen joint metric chosen without outcomes.

## Attack D — Coordinate rescaling

`ρ=1`, `s=(1,0)`, Euclidean `O=1`. After `s_v'=2 s_v` and `ρ'=2`, Euclidean `O=4`. Correct `W'=1/4` restores `O=1`. Euclidean COR as coded is not gauge invariant.

## Attack E — Trivial compatibility

`s_u=(2,0)`, `s_v=(0,3)`, `ρ = s_v s_u^T / ‖s_u‖^2` gives `ρ s_u=s_v`, hence `O=0`. Unconstrained or input-dependent linear maps can zero observed discrepancy. Shared full-rank `ρ` across spanning germs is a least-squares fit, i.e. statistical alignment.

## Attack F — Trivial projection

`P=0` ⇒ `s=0` ⇒ `O=D=0` with no decision information. Fatal for unconstrained Def. 2.

## Attack G — Edge duplication

Duplicating a constraint doubles Euclidean `O`; `D` unchanged (same `ker d` up to repeated rows). Graph granularity changes `O` at constant `F_θ`.

## Attack H — Global consistency without decision content

`s=0` has `O=D=0`. Downstream logits can still be anything. `ker d` nonempty is vacuous.

## Attack I — Positive discrepancy without topology

One edge, `s=(1,0)`, `ds=1≠0`, `ker d=span{(1,1)}≠{0}`. `[ds]=0` in `H^1`. Fatal for calling `O` a cohomological obstruction.

## Attack J — Same decisions, different obstruction

Same `s=(1,0)` and logit `s_0`; `ρ=1` vs `ρ=10` changes `O` from 1 to 100. Functionally identical readouts, different scores.

## Attack K — Nonunique minimizing repair

With `M≻0`, the nearest `g∈ker d` is unique. Semantic repair is not: many `g` share the same logits (Attack A). Uniqueness of a vector is not uniqueness of a decision repair. If `M` is only PSD, even the vector minimizer need not be unique.

## Attack L — Off-manifold projection

Natural set `{(1,0),(2,1)}`. Euclidean projection of `(1,0)` onto `span{(1,1)}` is `(0.5,0.5)`, not in the set. Low-obstruction edited activations need not be attainable.

## Weakest blocking assumptions (not independently justified)

| Attack | Weakest block | Independently justified? |
|---|---|---|
| A,B | Head factors through `ker d` | NO |
| C | Frozen joint `(M,W)` | Convention only |
| D | Transform metrics | Not done in code |
| E,F | Freeze `P,ρ` on train-clean simulator targets | Protocol, not a theorem |
| G | Invariance under refinement | Not claimed; then `O` is graph-relative |
| H | Nontrivial normalized sections | Missing |
| I | Restrict cohomology to relative `H^1` | Manuscript already does; operational `O` is still not that |
| J | Fix `ρ` | Protocol |
| K | `M≻0` plus semantic uniqueness criterion | Semantic part missing |
| L | Restrict edits to image(`φ`) | Not in repair formula |
