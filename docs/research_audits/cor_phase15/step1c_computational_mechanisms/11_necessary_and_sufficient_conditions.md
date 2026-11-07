# 11 — Necessary and sufficient conditions

Each row states one direction when only one direction holds. "Iff" is used only where both directions are proved. Domains are part of the claim.

| claim | direction | hypotheses that are doing the work |
|---|---|---|
| `F(u,v)=g(u)+h(v)` implies `I_F\equiv 0` | sufficient for zero interaction | the product domain of the identity. No smoothness |
| For `C^2` on an open rectangle about `0`, `I_F\equiv 0` iff the mixed partial is identically zero iff `F` is separable | iff | `C^2`, rectangle, the quantifier "identically" |
| `I_F(u_0,v_0)=0` at one pair | necessary for separability, because a separable map has `I_F` identically zero. Not sufficient for separability. Neither necessary nor sufficient for a zero mixed partial at `0` | one pair. `(u+v)^3` has `I_F(1,-1)=0` and is not separable. `uv-u^2v-uv^2` has mixed partial `1` at `0` and `I_F(2,-1)=0` |
| Nonzero mixed partial at `0` | sufficient for nonzero `I_F(tu,tv)` at all small `t\neq 0` with `uv\neq 0`. Not sufficient for a named finite pair | `C^2` Taylor expansion of `I_F` |
| Clamped `A` and `Y=AVW_O` | sufficient for zero interaction of value edits, and for equality with the linear map. Not necessary: `u^2+v^2` has interaction zero and is not of this form | `A` and `W_O` do not depend on the edited `V` |
| `W_Q=0` in the bias-free map `Q=XW_Q`, with no LayerNorm | sufficient for zero interaction of residual edits of that map. Not a theorem about the pre-norm block, and not a theorem once `Q` has an additive bias | the test `test_zero_query_map_makes_residual_edits_additive`. The bias counterexample is `test_query_bias_keeps_scores_dependent_on_keys_when_W_Q_is_zero`. Within one token, constant-pattern LayerNorm edits need not be additive: `test_constant_pattern_through_layernorm_is_not_additive_inside_one_token` |
| `R=g(P(h),\delta)` on a family | iff `R` is constant on the fibers of `P` in that family | the family is part of the statement. Exact, not approximate |
| Sign determination by `Q` for all linear heads that are nonzero on `\ker Q` | impossible when both orientations of a kernel vector are in the domain | N1. One orientation, one head, is not this claim |
| `\delta` leaves the linear residual `r` invariant | iff `\delta\in\ker(P_v A-\rho P_u)` | linearity of `A` and `\rho`. Not equivalent to `\delta\in\ker P_u` |
| Recomputed structural residual equals zero | if the child is assigned by `h_v:=f(h_u)`, the residual is zero. The converse is false: a clamp equal to `f(h_u)` has the same residual and was not that assignment. Zero on the whole recomputed family does not constrain a non-constant `R` | the intervention row in `05` |
| Sign survives deletion of path 1 in `m=c_1+c_2` | iff `c_2` has the margin's sign, or both are zero | linear sum, deletion means setting that term to zero, the other term not recomputed |
| `O` invariant under replacement of `M` | iff the formula of `O` is used, so `d`, `W`, and `s` are the objects held fixed | not a claim about simultaneous coordinate changes |
| Fixed `w` keeps its sign under plain LN, for every `w` and every non-constant `h` | false. The counterexample is `h=(1,0)`, `w=(1,2)` | universal quantifier over `w` |
| The same sign claim, restricted to `1^\top w=0`, `\gamma=1`, `\beta=0`, `\sigma>0` | true for every non-constant `h`, because `w·LN(h)=(w·h)/\sigma` | this restriction does not restore the universal claim |
| Fixed `w` keeps its sign under plain RMSNorm | true | positive scale, `rms>0`, no `\gamma` |
| Projected transport `P_v J=\rho P_u` plus matched offsets | sufficient for agreement of the first-order projected expansion. Not sufficient for the finite map | the remainder term |
| A scalar relation tightens the Jacobian bound using strictly less information | not shown. No sufficient condition was found | would need a proof that was not obtained |
| `D` or `O` determines `sign(ΔL)` | false as a universal claim | Phase 1. No new restriction in this step removes the counterexamples without assuming the sign |

## Local versus global

The rectangle theorem is global on that rectangle and local to it: it says nothing about a pair outside, and nothing about one pair inside. The infinitesimal interaction statement is local at the baseline. Value-linearity is global in `V` and conditional on `A` being the same matrix at every edited `V`. Fiber sufficiency is global on the set `H` named in the claim and silent outside it. The LayerNorm sign failure is an existence result. It does not say most vectors flip.

## Necessary conditions for a directional prediction

For a statistic `Q` to determine `sign(R)` on a domain:

1. `sign(R)` is constant on each fiber of `Q` in that domain.
2. The domain, not a later filter, is where the prediction is claimed.
3. The intervention that defines `R` is the intervention named in the theorem.
4. If the fibers were defined with a different operator than the one `Q` computes, the argument is about a different statistic.

A relational residual meets (1) only under a factorization that is not implied by the residual's definition. Assuming (1) by saying "the response improves when the residual falls" repeats the conclusion. No independently checked computational restriction in this step implies (1) for `O` or `D`.

## Sufficient conditions that do determine a response

- The full residual, the weights, and the intervention, by recomputation. This is the definition of `F_θ`.
- A projection that is injective on the coordinates the named family reads, by the fiber theorem.
- `A`, `V`, `W_O`, and `w`, for value-only edits, by linearity.

All three are existing descriptions of the block. They do not pass through a coboundary.
