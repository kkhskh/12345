# 02 — Formal definitions and type checks

Authoritative formulas: original PDF §5, Update PDF §5, `docs/math_map_v0.md` §2, `docs/theory_mvp_theorem_set.md` §0.

Classification tags follow the audit protocol.

## 1. Object inventory

| Object | Intended type | Depends on | Operational? |
|---|---|---|---|
| `X` | input/prompt space | task | Observed |
| `c=(a,b)` | candidate pair | chosen answers | Chosen by analyst |
| `y(x)` | latent state (Update only) | simulator or unknown world | Observed only in toys |
| `M_θ` / `F_θ` | frozen Transformer | weights | Observed I/O; internals if open |
| `K_x=(V_x,E_x)` | directed graph of sites | analyst graph choice, possibly `x` | Chosen |
| `φ_v: X → H_v` | activation map | model, site | Manipulable by patching |
| `F_c(v) ≅ R^{r_v}` | stalk | `c`, analyst rank | Chosen |
| `P^c_v : H_v → F_c(v)` | “projection / probe / feature map / logit-lens” | `c`, fit data | Fitted or chosen |
| `s_c(v;x)=P^c_v φ_v(x)` | local germ | all of the above | Computed |
| `ρ^c_{u→v}: F_c(u)→F_c(v)` | linear map in experiments | `c`, train-clean activations or Jacobian | Fitted |
| `C^0=∏_v F_c(v)` | 0-cochains | graph, ranks | Defined |
| `C^1=∏_{e:u→v} F_c(v)` | 1-cochains valued in **head** stalks | graph | Defined in MVP |
| `d_c` | coboundary `C^0→C^1` | `ρ` | Defined |
| `H^0=ker d_c` | global sections | `ρ` | Defined, always contains 0 |
| `M` | inner product on `C^0` | analyst | Chosen; often Euclidean |
| `W` | inner product on `C^1` | analyst edge weights | Chosen |
| `O_c=‖d_c s_c‖_W^2` | residual energy | all of the above | Computed on some LMs |
| `D_c=min_{g∈ker d} ‖s-g‖_M^2` | distance to kernel | all of the above | Synthetic only |
| `m_c=z_a-z_b` | margin | model output | Observed |
| `R_c` / `S_u` | response score | `ΔO`, `Δm`, cost, `α,β,γ` | Computed in 2C/2D |
| `do(h_v ← h_v+δ)` | activation intervention | site, scale | Manipulable |

Circularity risk: if `P`, `ρ`, `M`, `W`, `α,β,γ`, or the graph are selected using the same intervention outcomes later used to “validate” obstruction, the loop is not a test. The manuscripts recommend clean-train fits; they do not forbid outcome-tuned metrics, ranks, or `β`.

## 2. Candidate-specific local states

Equation (Update/original Def. 2):

```text
s_c(v;x) = P^c_v φ_v(x) ∈ F_c(v)
```

Answers required by the audit:

1. Domain of `P^c_v`: `H_v`, the activation space at site `v`. DEFINITION ONLY.
2. Codomain: `F_c(v) ≅ R^{r_v}`. DEFINITION ONLY.
3. Linearity: **not required** by the sentence “projection, probe, feature map, or logit-lens contrast.” Experiments use linear maps. CONDITIONAL.
4. Algebraic projection: **not required**. Ridge/PCA maps need not satisfy `P^2=P`. DEFINITION ONLY.
5. `(P^c_v)^2=P^c_v`: NOT ASSERTED. Often false.
6. Orthogonal vs oblique: UNSPECIFIED. Euclidean PCA is orthogonal in the ambient inner product used to compute it.
7. Inner product: UNSPECIFIED in the definition; Euclidean in code.
8. Rank fixed: NO. 2B searches `r∈{4,8,16}`.
9. Rank candidate-dependent: MAY BE. Not forbidden.
10. Candidate identity: a pair of answers `(a,b)`, or in Update a latent `y` then `s_c=Q^c_v s_y`. Chosen by the analyst.
11. Selection without future causal outcomes: possible if frozen on training-clean data. Not implied by the definition. CONDITIONAL RESULT on protocol, not on mathematics.
12. What makes `s_c` a “decision germ” rather than an arbitrary feature: **nothing in the equation.** The name is postulated. DEFINITION ONLY.

Counterexample (unjustified interpretation): take `P^c_v=0`. Then `s_c=0`, `O_c=D_c=0` for every input, including wrong answers. The definition still holds. Verified in `tests/test_cor_phase1_math.py::test_attack_f_zero_projection_collapses_obstruction`.

A learned probe is not a causal state. A subspace is not candidate-specific merely because it is labelled `c`.

## 3. Compatibility maps

```text
ρ^c_{u→v}: F_c(u) → F_c(v)
(d_c s)(e:u→v) = ρ^c_{u→v} s_c(u) - s_c(v)
```

Type check: both terms live in `F_c(v)` if and only if `ρ` lands in the head stalk. That matches MVP `C^1=∏_e F(head(e))`. PROVEN UNDER STATED ASSUMPTIONS (finite-dimensional linear maps).

| Question | Finding | Status |
|---|---|---|
| Linear? | Required for `d` to be a linear coboundary and for `O=s^T L s`. Experiments use linear ridge/Jacobian. Nonlinear forward maps are **replaced** by a linear surrogate. | CONDITIONAL |
| Invertible? | Not required. | DEFINITION ONLY |
| Vary with `x`? | Definitions write `ρ^c_{u→v}` without `x`. Jacobian option is input-dependent. | VERSION CONFLICT |
| Vary with `c`? | Yes, by notation. | DEFINITION ONLY |
| Depend on observed decision? | Not in the formula. Can leak if fitted on the evaluation condition. | IDENTIFIABILITY |
| Learned from natural activations? | Experimental recipe: ridge on `T_clean`. | EMPIRICAL PROTOCOL |
| Causally identified? | NO theorem. Statistical alignment of germs is not causal influence. | UNSPECIFIED |
| Composition law? | **Not imposed.** Residual-stream computation does not make learned `ρ` a functor. | DEFINITION ONLY |

Without a composition law, the structure is a **directed graph of linear constraints**, not a sheaf in the sense that restriction maps compose. Calling it a sheaf is terminology. The legitimate mathematical object is a collection of maps and a linear operator `d`.

Linear alignment ≠ nonlinear forward computation ≠ residual transport ≠ causal influence ≠ statistical predictability ≠ consistency of assigned local states.

## 4. Discrepancy / coboundary block form

For edges `e:u→v` ordered, and concatenated 0-cochain `s=(s_v)_{v∈V}`,

```text
(d s)_e = ρ_e s_u - s_v
```

Block row for `e` is `[… ρ_e … | … -I_{r_v} …]`. Implemented in `run_08_theory_numerical_checks.py` lines 69–79 and `src/cor_phase1_math.py`.

This construction:

- handles different stalk dimensions (rectangular `ρ`);
- handles branching (multiple outgoing block rows);
- does **not** encode residual skip connections, attention mixing, or MLP nonlinearity except insofar as those effects are already baked into `φ_v` and then linearly transported;
- is a linearized surrogate of the Transformer, not the Transformer.

Status: PROVEN UNDER STATED ASSUMPTIONS as a typed linear operator. It is NOT the model’s computational graph unless `ρ` is exactly the true Jacobian of `φ_v` with respect to `φ_u` and the graph lists every causal parent. The manuscripts allow learned ridge maps that need not equal any Jacobian.

## 5. Metrics

MVP requires `M` positive definite on `C^0` and `W=diag(w_e I)` with `w_e>0`. Original Def. 3 writes `‖·‖_M^2` without proving `M≻0`. If `M` is only semidefinite, the minimizer of `D` need not be unique (Attack K). If `W` is only semidefinite, `O=0` need not imply `ds=0`.

## 6. Dependency diagram (textual)

```text
analyst: {c, K, ranks, M, W, α,β,γ, fit protocol}
    → P, ρ
model + x: φ_v(x), F_θ(x), m_c(x)
    → s_c = P φ
    → d s, O, D
intervention δ:
    → Δφ, Δs, ΔO, Δm, ΔCost, S
```

If the analyst objects depend on `Δm` from the evaluation set, `S` is not a prospective prediction.
