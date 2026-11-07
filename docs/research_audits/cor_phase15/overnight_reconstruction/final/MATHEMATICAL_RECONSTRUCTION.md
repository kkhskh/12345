# Mathematical reconstruction

Preserved, and rechecked rather than reopened: `O` is not `D` in general; `D` does not rank decisions; zero discrepancy is not a correct decision; a homogeneous kernel contains `0`; an exact coboundary is the zero class in `C^1/im(d)`; unrestricted alignments manufacture agreement in the sense of Sutter, Minder, Hofmann, and Pimentel, Theorem 1, under their Assumptions 1–5; a gradient is not transport of an absolute state; nonlinearity does not imply a nonzero four-term interaction; changing only `M` does not change `O`; `ε>0` LayerNorm is not scale-invariant and its Jacobian rank is `d-1` when every gain is nonzero, while `ε=0` and non-constant input give rank `d-2` with kernel `span{1,z}`; the pair `c1=1`, `c2=-0.5` does not make both deletions necessary; a direction invisible to LayerNorm still passes through the skip; a raw residual identity does not survive a final LayerNorm on the readout.

The constant-shift identity of step 1C-R is a lemma. For every residual edit `E` and a one-token constant `C`, `H2(H+E+C)=H2(H+E)+C`, so `R` changes by `a 1^T w` only if that token is read. The centered part depends on the branch weights. `Dm=0` does not imply `R=0`.

## Definitions

```text
do(H_t <- H_t+δ): replace token t, recompute the pre-norm block.
m(H) = w · H2[r], r = T-1.
R(H,t,δ) = m(do(...)) - m(H).
Dm: directional derivative of m. Not R.
```

Budget B0: architecture, `ε`, LayerNorm gains and biases, `w`, read index, `H`, `(t,δ)`. No branch weights, no `R`, no `Dm`.

Budget B1: B0 plus branch matrices and the LayerNorm increments they are multiplied against. No `R`, no `Dm`.

Budget B2: values or derivatives of `m`. Using B2 to predict `R` is evaluation or the chain rule.

## Proved boundaries

**Theorem F.** If `ε>0` and at most one gain is zero, `LN(h+δ)=LN(h)` if and only if `δ` is constant. Two zero gains: the algebraic pair in the counterexample catalog.

**Theorem U.** For non-constant `δ`, the explicit scale family has `R = α ||w||^2 / T` plus the skip dot when the edited token is read. `|R|` is unbounded on B0.

**Theorem L.** A Lipschitz constant `Λ` of `m` on a convex neighborhood of a differentiable point satisfies `Λ ≥ ||∇m||`, hence `Λ||δ||` is at least the Cauchy–Schwarz number. Seed 0: `Λ = 2894603.0377523545`, `||g||=3.548311334245126`.

**Theorem K.** If the finite increments lie in the branch input kernels, `R` equals the skip dot. The constant line always qualifies. A manufactured non-constant point has `R=0`, Cauchy number `0.4554966651252395`, and `g·δ=0.1066527641320657`. Injective `QKV`, which holds on seeds 0–11, plus Theorem F, leaves only constants.

**Theorem J.** Kernel membership of `Jδ` matches the linear skip and not the finite response. At step `0.8` the gap is about `0.00741` and increases at larger steps.

## Unified dependency map

Arrow labels are only: exact equation, theorem, conditional result, checkable assumption, conjecture, measurement, unsupported.

```text
block weights, ε, gains, H
    --exact equation--> X, H1, H2, m

do(H_t <- H_t+δ)
    --exact equation--> R

LN shift invariance
    --theorem--> R(H, E+C) = R(H, E) + κ(C)
        [lemma from step 1C-R, rechecked]

ε>0 and at most one zero gain
    --checkable assumption--> hypothesis of Theorem F
Theorem F
    --theorem--> LN(h+δ)=LN(h) iff δ constant

non-constant δ
    --theorem--> Theorem U: no finite upper bound from budget B0

m differentiable, Λ a Lipschitz constant
    --theorem--> Theorem L: Λ ||δ|| ≥ ||g|| ||δ|| ≥ |g·δ|

explicit weight-norm majorant
    --theorem--> a valid Λ
    --measurement--> seed 0 ratio about 8.16×10^5 against ||g||

ΔX, ΔU and the branch matrices
    --checkable assumption--> kernel products vanish
kernel products vanish
    --conditional result--> Theorem K: R = skip dot

stacked QKV injective and full gains
    --measurement--> seeds 0..11, smallest singular value > 0.5
    --conditional result--> Theorem K adds no non-constant δ

Jδ in the kernels
    --theorem--> Dm matches the skip derivative
    --theorem--> the finite response need not match the skip (Theorem J)

exotic branch zero off the factor kernel
    --conjecture--> not excluded; search minima positive on seed 0

O, D, or the class [ds]
    --unsupported--> sign(R), a comparison of decisions, or a bound tighter than Theorem L

metric built from ∇m
    --exact equation--> a function of the linear response

final LN on H2
    --theorem--> constant skip contribution to that readout is 0

value-only derivative
    --exact equation--> OV path at a frozen pattern
    --measurement--> seed 16, opposite sign from Dm
```

The word "counterexample" in Theorem J's arrow is the existence proof recorded as `COUNTEREXAMPLE_FOUND` in the register. It is not an eighth label. The logical arrow is a theorem that the infinitesimal hypothesis does not imply the finite conclusion.

## Name

| piece | does the name COR fit |
|---|---|
| Original equations `O=‖ds‖_W^2`, `D=min_{g in ker d} ‖s-g‖_M^2` | As names of those two quadratics, yes. They are not a response theory. |
| Original hypothesis that lower `D` raises `L`, or that `D_c<D_{c'}` ranks decisions | No. Both implications are false without an extra assumption that already contains the ranking. |
| Broader ambition that identifiable computational relations constrain responses | Only through symmetries and unread subspaces already described by the block and by causal abstraction. That ambition does not need these equations. |
| A restricted successor | No survivor in the examined classes. |
| Empirical validation | Not claimed. The historical comparisons stay unfavorable to obstruction scores and were not refit. |

Phase 2 remains NO GO. `OBSTRUCTION_READY` remains NO. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` remains the Phase 1 label and is not rewritten.
