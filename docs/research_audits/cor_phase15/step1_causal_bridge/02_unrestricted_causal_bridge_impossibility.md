# 02 — What consistency alone cannot predict

Classification of the main statement: **MATHEMATICAL THEOREM**, negative. It does not say that every restricted theory fails. It says which extra structure a restricted theory has to add, and that the extra structure is where the causal content would have to enter.

## 1. Data that obstruction actually uses

Fix a finite directed graph, stalk dimensions, linear maps `P_v` and `ρ_e`, and positive-definite matrices `M` and `W`. Write `s = P φ` for the concatenated cochain and

```text
(ds)_e = ρ_e s_u - s_v,
O(s) = ||ds||_W^2,
D(s) = min_{g: dg=0} ||s-g||_M^2.
```

`O(s)` and `D(s)` are functions of `s` and of `(ρ, M, W)`. They do not depend on any readout that is not already inside that tuple.

## 2. Theorem NFL

**Statement.** Fix `(P, ρ, M, W)` and an activation `φ_0` with cochain `s_0 = P φ_0`. Obstruction at `s_0` does not determine the decision or the sign of a recomputed intervention, in the following two senses.

1. **Value.** For every `α ∈ R` there is an affine readout `L(φ) = w·φ + c` with `L(φ_0) = α`. The cochain, and therefore `O(s_0)` and `D(s_0)`, does not depend on `(w, c)`.
2. **Recomputed sign.** Let the child be an exact function of the parent, `φ_v = f(φ_u)`, and let `ρ` satisfy `P_v f = ρ P_u`, so the residual is zero before and after recomputation (Theorem Z). Let the logit be a linear functional of the recomputed child. Replacing that functional by its negative leaves every residual at zero and reverses the sign of the intervention response.

**Proof of (1).** Choose `w = 0` and `c = α`. QED.

**Proof of (2).** The residual identity does not contain the readout. If `R` is the change in `w·φ_v` after `do(φ_u ← φ_u+δ)` and recomputation of `f`, the change in `(-w)·φ_v` is `-R`. QED.

A directional-derivative claim has a narrower proof and is not the same statement. In dimension at least 1, an affine readout `L(φ) = c + w·φ` can be given any value and any directional derivative at `φ_0` by choosing `c` and `w`. That derivative is the derivative of a functional of an assigned vector. It is not, by itself, the response under file 04's rule, which clamps one site and recomputes descendants. The identity edge makes the distinction concrete: at `(0,0)`, with `L = s_v` and parent displacement `(1,0)`, the frozen-cochain derivative is `0`, while `do(s_u ← 1)` followed by the identity map produces child `1` and response `+1`.

The earlier writeup of this theorem said that any `δ` could be corrected in an orthogonal direction. That step is false when `δ` is parallel to `s_0` and the readout is required to be linear through the origin. The statement above does not use that step.

**Scope.** Part (2) needs a fixed `f` and a free final head. It does not say that, inside one already chosen network, the sign is indeterminate. It says no function of `(O, D)` equals that sign for every final head. The theorem is silent about a model class in which `L` is forced to be a declared function of `D`. Those restrictions are file 05. They are additional assumptions.

## 3. The seven comparisons

All seven are instances of Theorem NFL or of the two-line identity edge `ρ = 1`, `M = W = I`, `d = [1, -1]`. Calculations: `tests/test_causal_bridge_math.py`.

| Claim | System | Result |
|---|---|---|
| Same obstruction, opposite ranking | `s=(1, 0.5)`, `O=0.25`, `D=0.125`, heads `(1,0)` and `(-1,0)` | `L=1` and `L=-1` |
| Same obstruction, opposite response | same `s`, displacement of the first coordinate | signs `+` and `-` |
| Same obstruction, different magnitude | heads `(1,0)` and `(3,0)` | responses differ by the factor 3 |
| Reduced obstruction, reduced logit | `(1,0) → (0,0)` | `D: 0.5 → 0`, `L: 1 → 0` |
| Increased obstruction, increased logit | `(0,0) → (4,0)` | `D` rises, `L: 0 → 4` |
| Zero obstruction, incorrect decision | `s=(1,1)`, `D=0`, head `(-1,0)` | `L=-1` |
| Positive obstruction, correct decision | `s=(1,0)`, `D=0.5`, head `(1,0)` | `L=1` while `D>0` |

## 4. What the extra structure has to do

A bridge that escapes Theorem NFL must restrict `L` so that it is no longer free once `(P, ρ, M, W, s)` is known. The restriction is the scientific hypothesis. File 05 checks the natural candidates for that restriction. The test applied to each one is the same: remove the restriction and see whether any directional prediction remains. In every case that produces a sign, the removed piece is either the output gradient or a monotone function of `D` that was assumed rather than derived.

## 5. Where the causal content enters

It does not enter through `d`, `O`, or `D`. Those are fixed in the theorem while `L` and `R` move. Any later theorem whose assumptions include a function `L = q(D)`, an alignment `∇L · ∇D < 0`, or a readout `L = w·s` with `w` supplied from the model, has moved the causal content into that assumption.

## WHERE DOES THE CAUSAL CONTENT ENTER?

In the freedom of `L` given `s`. Closing that freedom is a claim about the model, not a property of the coboundary.
