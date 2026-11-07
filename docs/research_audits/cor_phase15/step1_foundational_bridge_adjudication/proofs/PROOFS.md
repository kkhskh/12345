# Proofs

Each theorem is stated with its full hypotheses. "Verified" means checked by `tests/test_bridge.py`. A passing test is not offered as a proof of a universal claim. The only claims proved by computation are the existence claims D and LN, and their numbers are far from zero relative to floating-point error.

## 0. Setting and intervention semantics

**Architecture.** Pre-norm residual Transformer in the GPT-2 layout. The input is `X ∈ R^{T×d}`, and `H^0 = X`. Layer `ℓ` computes:

```
Z    = LN1(H)                                   (per row, ε>0)
c_h  = A^h (Z W_V^h) W_O^h,  A^h = softmax_row(mask(Z W_Q^h (Z W_K^h)^T / √d_h))
Hmid = H + Σ_h c_h + b_O
H'   = Hmid + MLP(LN2(Hmid))
```

The read position is `r`. Logits are `L = W_U^T LN_f(H^L[r]) + b_U`, with `LN_f(h) = γ⊙(h−μ(h)1)/σ(h) + β` and `σ(h) = (‖h−μ1‖²/d + ε)^{1/2}`. A candidate is a vocabulary index. For a pair `(c,c')`, the margin is `m = L_c − L_{c'}`. Margins are invariant to candidate-independent logit shifts, so candidate scores are compared on one shared scale.

**Interventions.** An intervention replaces the value of a named site: an input row, a head output `c_h`, the MLP output, attention scores, or the final residual. Every downstream structural equation is then recomputed. The response is `R = m(do) − m`. An input substitution (replacing the token at one position) involves no internal choice at all.

**Canonical variables.** These are input tokens, per-component writes `c_k[t]`, attention scores and weights, residuals `H^ℓ[t]`, and the frozen readout. The architecture itself defines them. None is a learned projection `P` or a fitted transport `ρ`.

---

## Theorem SS — same-cut sum screening

**Hypotheses.** Let `𝒦` be a set of sites whose outputs `c_k ∈ R^{T×d}` enter one additive cut `S = B + Σ_{k∈𝒦} c_k`. Here `B` and every `c_k` are computed upstream of `S`, no `c_k` reads another member of `𝒦`, and every downstream variable is a function of `S` and of variables upstream of `S`. Examples: all heads of one attention layer (`S = Hmid`); attention and MLP in a parallel-residual block.

**Claim.** (i) For an edit `c_k ↦ c_k + δ_k` with `Σ_k δ_k = 0` (as `T×d` matrices), every downstream variable is unchanged, including every logit. (ii) For any edit, `R` depends on `(δ_k)` only through `Σ_k δ_k`. (iii) Let `ψ` be any statistic of the tuple `(c_k)` that is not a function of `Σ_k c_k`, for example a pairwise cosine. Then `ψ` has no causal effect on any decision once the sum is held fixed. Equal `ψ` is also compatible with different decisions.

**Proof.** (i) and (ii): the downstream variables are functions of `S`, and `S` depends on the tuple only through its sum. (iii) The fiber `{(c_k + δ_k) : Σδ_k = 0}` contains tuples with every value of a non-sum statistic. For two writers in `R^d` with `d ≥ 2` and nonzero sum, the cosine ranges over `[−1, 1]` on that fiber. Every logit is constant on the fiber by (i). Scaling all writers by `λ > 0` preserves every cosine and changes the sum. ∎

**Verified.** Two layers and three heads: 20 random zero-sum edits leave all logits fixed to `1e-10`. Scaling both heads by 3 preserves their cosine and changes the margin.

**Scope.** Writers at different cuts are not covered. A later writer reads the earlier sum, so their "relation" is a path-specific question. Downstream of the later cut, the sum screening applies again.

---

## Theorem NC — the final-normalization norm channel

**Hypotheses.** Final LayerNorm with `ε > 0`, a frozen unembedding, and a candidate set `C`. For a pair, `Δu = u_c − u_{c'}`, `P = I − 11^T/d`, `ũ = P(γ⊙Δu)`, `a(h) = ũ^T h`, and `κ = Δu^T β + b_c − b_{c'}`. The decision-null space is `N_C = {v : 1^T v = 0, (γ⊙(u_c−u_{c'}))^T v = 0 ∀ c,c' ∈ C}`, with orthogonal projector `Π_N`. Write `h̃ = Ph`.

**Claim.**

1. `m(h) = a(h)/σ(h) + κ` for every `h`.
2. For `v ∈ N_C`: `a(h+v) = a(h)` and `σ(h+v)² = σ(h)² + (2h̃^T v + ‖v‖²)/d`. Hence `R = a(h)(1/σ(h+v) − 1/σ(h))` and `sign R = −sign(a)·sign(2h̃^T v + ‖v‖²)`. The statement is exact and finite.
3. For `c_0 ∈ C`, every `L_c − L_{c_0}` restricted to `h + N_C` equals `a_c τ + κ_c` with one common `τ = 1/σ(h+v)`. Each pairwise order is a single-crossing function of `σ`, with the crossing at `τ* = −κ/a`. It therefore changes at most once along any path on which `σ` is monotone. *Correction after review B (S5):* an earlier draft said "along any path". That is false. An out-and-back path in `N_C` flips a flippable pair twice (`test_NC_two_flips_on_a_non_monotone_path`).
4. `{σ(h+v) : v ∈ N_C} = [σ_min, ∞)` when `dim N_C ≥ 1`, where `σ_min = (‖h̃ − Π_N h̃‖²/d + ε)^{1/2}`.
5. A pair's sign can be reversed by some `v ∈ N_C` if and only if `aκ < 0` and `σ* = −a/κ > σ_min`. As `‖v‖ → ∞`, every margin in `C` tends to `κ`: the ranking set by the final-LN bias and the unembedding bias.
6. `dim N_C ≥ d − |C|`. If `{1} ∪ {γ⊙(u_c−u_{c_0})}` spans `R^d`, then `N_C = {0}`. This holds generically for the full vocabulary whenever `|V| ≥ d`.

**Proof.**
1. `μ(h)1 = (11^T/d)h`, so `Δu^T(γ⊙(h−μ1)) = (γ⊙Δu)^T P h = (P(γ⊙Δu))^T h`, because `P` is symmetric. Divide by `σ(h)` and add the bias terms.
2. `1^T v = 0` gives `Pv = v`, so `ũ^T v = (γ⊙Δu)^T P v = (γ⊙Δu)^T v = 0`. Also `P(h+v) = h̃ + v`, so `‖h̃+v‖² = ‖h̃‖² + 2h̃^T v + ‖v‖²`. The sign follows because `t ↦ 1/t` is decreasing on `t > 0`.
3. Apply 1 and 2 to each pair `(c, c_0)`. `τ` does not depend on the pair. A nonconstant affine function of `τ` has at most one zero.
4. `N_C ⊂ 1^⊥`. Write `h̃ = (h̃ − Π_N h̃) + Π_N h̃`. The first term is orthogonal to `N_C`, so `‖h̃+v‖² = ‖h̃ − Π_N h̃‖² + ‖Π_N h̃ + v‖²`. The second term takes every value in `[0, ∞)` as `v` ranges over `N_C` (at `v = −Π_N h̃ + s n` with unit `n ∈ N_C` and `s ≥ 0`). Continuity and unboundedness give the interval.
5. `m = aτ + κ` with `τ ∈ (0, 1/σ_min]`. If `aκ ≥ 0`, then `m` keeps one sign (or is zero) on `τ > 0`. If `aκ < 0`, `m` changes sign at `τ* = −κ/a > 0`, which is reachable iff `τ* < 1/σ_min`, that is, `σ* > σ_min`. The limit is `τ → 0`.
6. `N_C` is the kernel of `|C|` linear forms (`1` and the `|C|−1` differences). ∎

**Verified.** Three candidates, 50 random null edits, all three pairs. Flip criterion against a continuous sweep of `σ` over `[σ_min, ∞)` on seeds 5, 6, 8, and 12, covering flippable, same-sign, and disagreeing-but-unreachable pairs. The limit equals `κ` at `‖v‖ = 10^5`. `N_C = {0}` for `|V| = 20 > d = 8`.

**Withdrawn relational reading (reviews B S4, C SERIOUS-2, D).** An earlier draft read `d(σ² − ε) = Σ_{k,l} c̃_k^T c̃_l` as "writer alignment". But this sum equals `‖PΣ_k c_k‖²`, a function of the sum at one cut, so by Theorem SS it is not a relation among writers. NC is the **norm channel**: a temperature (common `τ`) together with a pull toward the bias ranking `κ`. It is not a consistency mechanism.

**Scope additions (review D).**
- NC constrains the order *within* `C`. A null edit for `C` can let a token outside `C` overtake.
- With an effective, approximate null space, single crossing and the `κ` limit hold only up to the leakage terms.
- **RMSNorm analogue** (LLaMA, Mistral, and similar): there is no centering and no bias, and the unembedding has no bias. So `m = a/ρ(h)` with `κ ≡ 0`, null edits rescale every margin, and no pair can ever flip. It is pure temperature (`test_NC_rmsnorm_analogue_never_flips`).
- In GPT-2 and Pythia, `κ = Δu^T β_f` is exact and generically nonzero. Whether it matters depends on the unmeasured ratio `|κ|/|a/σ|`.

---

## Theorem CI — softmax consensus identity

**Hypotheses.** Finite index set `J`, weights `w_j ≥ 0` with `Z = Σ w_j > 0`, scalars `v_j`, and `M = Σ w_j v_j / Z`. Any new `w'_j ≥ 0` with `Z' > 0`, and any new `v'_j`.

**Claim.**

1. `M' − M = Z'^{-1} Σ_j w'_j (v'_j − M)`.
2. If only index `j` changes: `M' − M = [w'_j(v'_j − M) − w_j(v_j − M)] / Z'`.
3. If source `j` is added (`w_j = 0 < w'_j`): `sign(M' − M) = sign(v'_j − M)`, for every `w'_j > 0`. *Scope (reviews B S2, C SERIOUS-3, D SERIOUS-5):* a softmax head never gives an unmasked position zero weight. So a token *substitution* is case 2 (first layer) or case 1 (later layers), never case 3.
3*. **(Reallocation split.)** For a substitution at `j` with every other index unchanged: `M' − M = [w_j(v'_j − v_j) + (w'_j − w_j)(v'_j − M)] / Z'`. The first term depends on source `j` alone. The second, the **only term in which other sources enter**, is the *weight change* times the deviation from the consensus. If the weight does not change, the dependence on the other sources is a positive rescaling `w_j/Z'` and the sign is relation-free. *Proof:* expand `w'_j(v'_j−M) − w_j(v_j−M)` in case 2.
4. `min_j v_j ≤ M ≤ max_j v_j`. If every attended position, including BOS and the read token, has `v_j > 0`, then `M > 0`.

**Proof.**
1. `Z'(M' − M) = Σ w'_j v'_j − M Σ w'_j = Σ w'_j (v'_j − M)`.
2. In 1, the unchanged indices contribute `Σ_{i≠j} w_i(v_i − M) = −w_j(v_j − M)`, because `Σ_i w_i(v_i − M) = 0`.
3. Substitute `w_j = 0` into 2.
4. A convex combination. ∎

**Transformer instance.** Fix a head `h` in layer `ℓ`, the read position `r`, and any fixed linear functional `ũ`. Put `w_j = exp(s_{rj})` for `j ≤ r` and `v_j = ũ^T (Z_j W_V^h W_O^h)`. Then `ũ^T c_h[r] = M`. Any recomputed intervention upstream of layer `ℓ` changes `(w, v)` to `(w', v')`, and `Δ(ũ^T c_h[r])` is given by 1 exactly. With `ũ = P(γ⊙Δu)`, this is the head's direct contribution to the numerator `a` of Theorem NC.

**Corollary CI-5 (exact margin decomposition, last layer).** `Δm = Δa/σ' + a(1/σ' − 1/σ)`, with `Δa = Σ_h Δ(ũ^T c_h[r]) + ũ^T ΔMLP[r] + ũ^T ΔH^{L−1}[r]`. Proof: `a` is linear in the final residual, and the final residual is the sum of those terms (Theorem SS with `B = H^{L−1}`). Then apply NC.1 at both states.

**Verified.** 200 random single replacements; two layers and three heads with single-row residual edits whose effects propagate to every weight and value at layer 1; 30 random states for the convex hull; the exact total decomposition to `1e-10`; the reallocation split in Construction D.

**Prior work (review F).** CI.1–4 are the weighted-mean update. CI.3 is the influence function of the mean (Hampel 1974). CI-5 is algebraically the LayerNorm self-repair decomposition of Rushing and Nanda (2024).

**What CI signs (review B S3, review D SERIOUS-6).** CI signs the head's term in the *numerator* `a`, not the margin. The `σ` term in CI-5 is a second channel. In random one-layer, one-head models, the head term in `a` and the margin change have opposite signs in 591 of 4000 substitutions (review B). Construction D escapes this only because its head writes along `ũ` alone.

**What the identity says about consistency (corrected).** Other sources enter a head's direct numerator term only through attention *reallocation* (CI.3*). The relation that then sets the sign is the source's direct-logit attribution minus the head's (`v'_j − M`), both defined by the readout `ũ`. This is ordinary softmax normalization. Every head satisfies it by algebra, whatever the head's function, so it does not identify a dedicated reconciliation mechanism (review C FATAL-3). From layer 1 on, a token substitution moves every weight, every value, and the query, and the quantity becomes the head's DLA change (review D SERIOUS-4).

---

## Theorem RB — ordinal relation-blindness

**Definition.** `A`, `B` are finite level sets of two independently manipulable evidence variables, and `m : A×B → R`. `m` is **relation-blind** if there are `f : A → R`, `g : B → R`, and a strictly increasing `ψ` such that `m(a,b) = ψ(f(a) + g(b))`. Otherwise `m` is **relation-sensitive** (nonremovable interaction).

**Claim.**

1. (Scott 1964 criterion.) `m` is relation-blind iff the LP `max t` subject to `f_i+g_j−f_k−g_l ≥ t` whenever `m_ij > m_kl`, `f_i+g_j = f_k+g_l` whenever `m_ij = m_kl`, and `|f|,|g| ≤ 1`, has optimum `t > 0`.
2. (Invariance.) For any strictly monotone `φ` and any relabeling of levels, `m` is relation-blind iff `φ∘m` is. *Scope (review B S1):* this covers transforms of **one fixed scalar**. With more than two candidates, probability, log-probability, and logit are not monotone functions of the pair margin. The decision scalar, including the named competitor, must be frozen. *Consequence (review C FATAL-1):* because the verdict is invariant to relabeling, RB cannot tell agreement from conflict. It is a generic test for nonremovable interaction. Its role here is a **necessary condition**: if `m` is relation-blind, no relation between the sources can set the direction of either source's effect (RB.3).
3. (Independence.) If `m` is relation-blind, then for all `a, a'` the sign of `m(a,b) − m(a',b)` does not depend on `b`. The same holds with the factors exchanged. Hence a crossover implies relation-sensitivity.
4. (2×2.) Every 2×2 table without a crossover is relation-blind.
5. (Beyond crossover.) Some 3×3 tables with no crossover are relation-sensitive.
6. (Prospective forcing.) Given a training set `𝒯` of observed strict comparisons that is itself representable, relation-blindness *forces* `m_p > m_q` on an untested pair iff the LP `min (f_p+g_p) − (f_q+g_q)` subject to separations `≥ 1` on `𝒯` has a positive optimum.

**Proof.**
1. (⇒) Take the representation. The finitely many strict differences of `f+g` have a positive minimum. Scale `f, g` into the box. (⇐) A solution with `t > 0` orders the cell values `f_i + g_j` exactly as `m` orders them, with ties preserved. On the finite set of values, the map `f_i + g_j ↦ m_ij` is then a well-defined, strictly increasing bijection. Extend it to a strictly increasing `ψ : R → R` by piecewise-linear interpolation, with slope-1 extensions beyond the end values.
2. `φ∘ψ` is strictly monotone. For decreasing `φ`, use `(−f, −g)` and `x ↦ φ(ψ(−x))`. Relabeling permutes the variables.
3. `ψ` is strictly increasing, so `sign(m(a,b) − m(a',b)) = sign(f(a) − f(a'))`, which does not depend on `b`.
4. Without crossover, one sign `s_A ∈ {−1,0,1}` describes `m(a_1, b) − m(a_0, b)` for both `b`, and one sign `s_B` likewise. Take `f_0 = g_0 = 0`, `f_1 = s_A x`, `g_1 = s_B y` with `x, y ≥ 0` (`x = 0` iff `s_A = 0`, `y = 0` iff `s_B = 0`). This matches the four within-row and within-column comparisons. If `s_A = 0` or `s_B = 0`, those comparisons and transitivity determine the remaining two, and the representation matches them. If `s_A s_B > 0`, the diagonal pair `(a_0,b_0)` against `(a_1,b_1)` is forced by transitivity. The off-diagonal pair `(a_1,b_0)` against `(a_0,b_1)` is free, and `f_1 + g_0 − f_0 − g_1 = s_A(x − y)` matches it by choosing `x` greater than, equal to, or less than `y`. If `s_A s_B < 0`, the off-diagonal pair is forced, the diagonal pair is free, and `f_1 + g_1 = s_A(x − y)` matches it by the same choice. In every case exactly one comparison is free, and one scalar choice matches it.
5. The table `[[3, 6, 6.5], [5, 7, 8], [7, 7.5, 8.5]]` is monotone in both factors with no crossover. Double cancellation requires: `m(a_0,b_1) ≥ m(a_1,b_0)` and `m(a_1,b_2) ≥ m(a_2,b_1)` imply `m(a_0,b_2) ≥ m(a_2,b_0)`. Adding `f_0+g_1 ≥ f_1+g_0` and `f_1+g_2 ≥ f_2+g_1` gives `f_0+g_2 ≥ f_2+g_0`. Here `6.5 < 7`, so double cancellation fails.
6. *(Prior work, review F: this is the necessary preference relation of robust ordinal regression, Greco, Mousseau, and Słowiński 2008, computed by the same LP. See also the signed difference analysis of Dunn and James 2003.)* Strict representations form a cone up to the gauge. Any strict representation can be scaled so that every training separation is at least 1. If the minimum is positive, every representation orders `p` above `q`. If it is not positive, a feasible point exists with `f_p+g_p ≤ f_q+g_q`. The code bounds coordinates by `100`. With coefficient matrices in `{−1,0,1}` and at most four free coordinates (designs up to 3×3), the Hadamard bound limits vertex coordinates to 16, well inside that box. The code refuses larger designs (review B, M2). ∎

**Verified.** Additive tables pass under four monotone transforms. Random crossovers are always rejected. 2000 random 2×2 tables agree with claim 4. The double-cancellation table is rejected, and the forced held-out order fails on it.

**Prior work.** Claims 1–5 are standard additive conjoint measurement: Luce and Tukey (1964); Scott (1964); Krantz, Luce, Suppes, and Tversky (1971); and, for claim 4, Loftus (1978) and Wagenmakers et al. (2012). They are restated here because they give the consistency question an identification standard that is free of metrics and link functions.

---

## Construction D — dilution crossover in a full pre-norm block

**Claim (existence).** This is the real architecture with *designed* weights. It is an existence proof that the architecture can express the pattern, not evidence about trained weights: solving six LN'd tokens in `d = 8` can realize any key and value table (review D). Some parameters of one pre-norm layer exist (LN with `ε = 10^{-5}`, one causal softmax head, a GELU MLP that is off or random with gain 1, final LN, bias-free unembedding) such that the 2×2 margin table over two evidence slots is

| A \ B | neutral | strong support |
|---|---|---|
| neutral | −2.4681 | 3.2162 |
| weak support | −1.2184 | 2.7083 |

(attention only). Weak support raises the margin by `+1.25` when B is neutral and lowers it by `−0.51` when B is strongly supporting, which is *agreeing* evidence. The table has a crossover and is relation-sensitive by RB.3.

**Proof.** The parameters are specified in `bridge_math.dilution_model`. The value, key, and query maps are solved exactly on the six LN-normalized token vectors (an exact linear system, asserted to `1e-9`), so the design is a statement about the computation. CI.3* gives the head's direct term. The own-change term is small and positive; the reallocation term `(w'_W − w_N)(e_W − M)` carries the sign. With B strong, `M ≈ 180/93 ≈ 1.935 > e_W = 0.5`, so the sign is negative; with B neutral, `M = 0 < 0.5`, so it is positive (`test_CI_star_relation_enters_only_through_attention_reallocation`). The crossover therefore depends on N having low salience (`w_N/w_W = e^{−3}`). With equal salience it disappears (review C, probe a). Raising the BOS sink weight to about 0.8 also removes it (review D). `test_D_crossover_locus_is_the_consensus` checks that, for every B level, the sign of the total margin change equals the sign CI predicts for the direct effect. The reported margins are evaluated in double precision. The smallest effect is 0.5 in absolute value. ∎

**Counterexample D′ (multi-path).** Add one MLP neuron that reads the evidence direction after LN2 and writes `−0.5 e_out`. Then the direct effect of `N→W` with B strong is negative while the total effect is positive (`test_D_downstream_mlp_reverses_the_direct_prediction`). CI fixes the head's direct term, not the decision.

## Construction LN — norm-coupling crossover without attention

With final LN only, `h = h_0 + c_A + c_B`. A weak-support write `0.5e + 3n` has a large null component `n`. It raises the margin when B is neutral and lowers it when `B = 3e`, where `e` is the decision direction. The table is relation-sensitive. The mechanism is NC.1: the null component raises `σ`, which shrinks a large content term more than the small added content raises it.

---

## Theorem MIX — the mixed repair objective

**Hypotheses.** A local quadratic consistency term with gradient `g` and Hessian-plus-regularizer `A ≻ 0`, a local linear margin with gradient `J`, and the objective `½δ^TAδ + g^Tδ − βJ^Tδ`.

**Claim.** `δ*(β) = −A^{-1}(g − βJ)`. The linearized margin change splits as `J^Tδ* = −J^TA^{-1}g + β J^TA^{-1}J`. The second term is nonnegative, and positive when `J ≠ 0` and `β > 0`. The first term can have either sign. For every `(A, g, J ≠ 0)` there is a `β_0` beyond which `J^Tδ* > 0`, whatever `g` is.

**Proof.** First-order condition, then `A^{-1} ≻ 0`. ∎ This is standard algebra. Its consequence is the protocol in `OUTCOME_LEAKAGE_AND_FREEZING.md`: only the `β = 0` term is evidence about consistency.

---

## Proposition CORR — correctness is not an internal property

For a fixed model and input set, the internal computation does not depend on the world labeling `y*`. Take any input on which the prediction equals `y*(x)`, and replace the labeling by one with `y'*(x) ≠ y*(x)` at that input. Every internal statistic, every consistency relation, and every response is unchanged, while correctness flips. So no internal consistency relation determines correctness without an assumption about how training tied `y*` to the computation. ∎ (Trivial. It is stated because Section 4 of the mandate asks whether consistency can hold for a wrong decision. It can, always.)

---

## Proposition FIB — what information determines the sign

(Elementary; Step 1C N1. An earlier draft attributed a "fiber condition" to Geiger et al. 2025. Review F did not find it there, and the attribution is withdrawn.) A statistic `Ψ` of `(state, intervention)` determines `sign R` exactly on a family `𝒟` iff `sign R` is constant on the fibers of `Ψ` in `𝒟`. Applications:

1. The semantic agreement indicator (both sources support `c` / they conflict) does not determine the sign. Example: in `m = e_A + e_B`, the conflict cells `(e_A, e_B) = (2, −1)` and `(1, −2)` have margins `+1` and `−1`.
2. Under sum aggregation, the source's own change in its contribution determines the sign of its direct effect.
3. Under softmax aggregation at layer 0, the sign of `w_j(v'_j − v_j) + (w'_j − w_j)(v'_j − M)` determines the head's numerator term (CI.3*). The source alone does not, as Construction D shows: the same switch `N→W` produces both signs.
4. Neither 2 nor 3 determines the sign of the total effect when other paths read the changed head (Counterexample D′).
