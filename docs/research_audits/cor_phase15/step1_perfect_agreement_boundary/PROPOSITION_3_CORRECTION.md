# Correction of Proposition 3 (and Proposition 1's quantifier)

The Investigation V statement is in `../step1_curriculum_response_identification/FORMAL_DECISION_EQUIVALENCE.md` §7. It is
**not edited**; this file supersedes it.

> **Post-review [R1].** Findings R1-1 to R1-6 (REVIEWS_AND_ADJUDICATION.md) are applied in this text:
> - A3 is restated as a connectivity condition.
> - A2 is dropped for φ_col; mixed gates are now proved non-invariant.
> - A remark on limits of the family is added.
> - A1 is restated and its pattern count corrected.
> - Edit definitions and τ are fixed.
> - One false sentence in §4 is removed.
>
> Pre-review text: commit 9389496.

**Status labels**
- **[P]** proved here.
- **[N]** checked numerically by `scripts/exact_boundary.py`, with output in `logs/exact_boundary.json`.

## 1. The original claim and the counterexample

**Original claim (Inv. V, Prop. 3).** Take f_θ ∈ 𝔽_mix, with log-weight φ_rel(n_g) + φ_col(n_h) on configuration (g reliable,
h colluder). Then f_θ responds 0 to every minimal Bayesian-null edit **iff** φ_rel is affine on {0,…,4} **and** the gate is
hard (φ_col(n) = −∞ for n < 5).

**Counterexample (mandate §3).** Take φ_rel(n) = 0 for n ≤ 4, φ_rel(5) = 1, and φ_col ≡ 0.
- The gate is finite everywhere, yet the model is null-invariant.
- [N] Over 96 natural null pairs the maximum |response| is exactly **0.0**.
- [N] The same member responds up to **0.63** to 4→5 edits, so it is not trivial on the boundary.

The "hard gate is necessary" direction is therefore **withdrawn**. The proof sketch's step "with a finite gate … the output
changes for some X_q" is false when φ_col is constant on 0…4. In that case the configurations whose colluder is a
sub-perfect group carry weights that do not change under the edit.

## 2. Setting

- **Class.** 𝔽_mix is the set of maps p_θ(y | X) ∝ [Σ_{(g,h) ∈ CFG} w_{gh}(n) · J(x_q, y | g, h)]^τ, where:
  - τ > 0 is an output temperature (as in `cri.mix_logits`); it multiplies every response and does not affect invariance;
  - log w_{gh} = φ_rel(n_g) + φ_col(n_h);
  - φ_rel, φ_col : {0,…,5} → ℝ ∪ {−∞};
  - n = n^{(−q)} are the leave-q agreement counts;
  - J is the exact own-question likelihood.
- **Natural support.** At least one group has n = 5 and agrees at q, which is the only way a natural T4 episode can have
  positive probability.
- **Minimal Bayesian-null edit.** On natural support with exactly one count-5 group h* (|H| = 1), the two other counts
  (a, b) ≤ 4 both move by the same ±1. One report changes and the exact posterior over configurations is unchanged, since
  π(g, h*) ∝ RHO^{n_g} is multiplied by a common factor. The edit changes **two** reports, one per moved group
  (`eqv.pi_preserving_edit`) [R1-5].

**Nondegeneracy (A1) [R1-3].** A1 is a joint condition, not a per-X_q one: a single X_q gives only 3 equations for 6
weights. Fix h*, and let X_q range over the own-question patterns compatible with the edit (h* agrees at q). Then, for every
w ≥ 0 with w ≠ 0, the only weight changes that leave every predictive Σ_cfg w_cfg J(x_q, · | cfg) unchanged up to
normalisation are common rescalings. Equivalently, the linear system has a one-dimensional kernel (projective dimension 0).
- [N] The original script check used a single dense w over all **513** positive-probability patterns (rank 518; the
  pre-review text said "518 patterns" in error) and found a null dimension of 1.
- [N, reviewer] Restricted to the 243 compatible patterns for each h*, with both a dense and a hard-gate w: null dimension 1,
  and the six J-functions have rank 6.
- A1 is a property of the generator, not of θ.

**Finiteness (A2) [R1-2].** φ_rel is finite on {0,…,5}. No regularity of φ_col is assumed: the gate may be hard, finite or
mixed on {0,…,4}.

**Edit coverage (A3) [R1-1].** Form a graph on {0,…,3} with an edge between a and b whenever some natural null edit moves
counts (a, b) to (a+1, b+1), or the reverse. A3 requires this graph to be connected.
- Covering each step k → k+1 is *not* sufficient: edits with a = b would cover every k and constrain nothing.
- [N] All 96 abstract (a, b, ε) pairs used by `exact_boundary.py` are enumerated, and naturalness does not depend on (a, b).
  The graph is therefore complete, and A3 holds.

## 3. Proposition 3′ [P under A1–A3]

Under A1–A3, f_θ ∈ 𝔽_mix responds 0 to every minimal Bayesian-null edit on natural support **iff** one of the following
holds. A gate that is finite on a proper, non-empty subset of {0,…,4} (a *mixed* gate) is never invariant.
- **(a)** the gate is hard and φ_rel is affine on {0,…,4};
- **(b)** the gate is finite and **both** φ_rel and φ_col are constant on {0,…,4}.

In both cases φ_rel(5), φ_col(5) and hence s = φ_rel(5) − φ_rel(4) are unrestricted. Null invariance therefore carries no
information about the boundary parameter.

**Proof.** Fix a null edit with colluder candidate h*, the other groups g₁ and g₂ with counts a and b, and both counts
moving by ε ∈ {+1, −1}. Write α = φ_rel(a+ε) − φ_rel(a), β = φ_rel(b+ε) − φ_rel(b), γ = φ_col(a+ε) − φ_col(a) and
δ = φ_col(b+ε) − φ_col(b).

The six log-weights change by:

| configuration | change |
|---|---|
| (g₁, h*) | α |
| (g₂, h*) | β |
| (h*, g₁) | γ |
| (h*, g₂) | δ |
| (g₁, g₂) | α + δ |
| (g₂, g₁) | β + γ |

The term φ_col(5) or φ_rel(5) of h* is unchanged.

By A1, the output is unchanged for every X_q compatible with the edit iff all *finite* weights are multiplied by one
common factor.

*Hard gate.* Only (g₁, h*) and (g₂, h*) are finite, so invariance requires α = β, i.e. d_a = d_b, where
d_k = φ_rel(k+1) − φ_rel(k), for every co-moving pair (a, b). By A3 (connectivity) all d_k are equal. That is affinity on
{0,…,4}.

*Mixed gate [R1-2].* Let S ⊊ {0,…,4} be the non-empty set where φ_col is finite. Some edit moves a count from k ∈ S to
k ± 1 ∉ S. That sends the weight of (h*, g₁) from finite to 0, while (g₁, h*) stays finite. This is not a common factor, so
by A1 the response is nonzero for some compatible X_q.
- [N, reviewer] Five choices of S, each with constant and Bayes φ_rel: maximum null response 2.3–5.0 in every case.

*Finite gate.* All six weights are finite, so α = β = γ = δ = α + δ = β + γ. The equation α + δ = α forces δ = 0. Then
α = β = γ = δ = 0. By A3 every increment of φ_rel and φ_col on {0,…,4} is zero, so both are constant there.

The converse is immediate: under (a) or (b) the changes are respectively (α, α, −∞-terms) and all zero. ∎

**Numerical check [N]** (5 random members per class, max |response| over natural null pairs):

| class | max \|response\| |
|---|---|
| exact Bayes | 2.2e-15 |
| hard gate + affine φ_rel | ≤ 3.6e-15 |
| finite gate + both constant | 0 |
| hard + non-affine | 1.2–2.7 |
| finite + affine φ_rel / constant φ_col | 0.29–0.80 |
| finite + constant φ_rel / affine φ_col | 0.17–1.04 |

**Excluded degenerate cases and limits.**
- *Mixed gates* are covered by the theorem and are never invariant. The pre-review claim that they "can be invariant
  under weaker conditions" was false [R1-2].
- *Limits [R1-2].* Branch (a) is a **non-uniform limit** of non-invariant members. With affine φ_rel and a constant finite
  gate c, the maximum null response is 0.86 at c = −5, 0.51 at c = −10 and 6e-5 at c = −20 (reviewer computation). A
  fitted surrogate with a large negative but finite gate is therefore approximately, not exactly, null-invariant.
- If A1 fails for a different generator, (b) may not be the only finite-gate solution.
- Members with φ_rel = −∞ somewhere are excluded as degenerate.
- Off natural support no statement is made. Investigation IV's D2 shows Bayes-optimal models can differ there.

**What survives from Inv. V.** The conclusion "s does not enter null responses; s is decision-relevant" survives, and is
strengthened, because it holds in both branches. What is withdrawn is the claim that null invariance *implies a hard gate*
or *implies the exact-Bayes form*. Branch (b) is a non-Bayesian, null-invariant operator that ignores every sub-perfect
count and reacts only to which groups are at 5.

## 4. Proposition 1, corrected quantifier [P]

**Original statement.** "For any κ > 0 and any B > 0 there is a model with E KL ≤ κ …". This used δ → 0, which a finite
generator does not allow.

**Corrected.** Let δ_min > 0 be the smallest probability of an episode x₀ that is an endpoint of a natural π-preserving pair
(finite domain). Tilt the logit of ŷ = argmax p*(x₀) by t > 0 at x₀ only. Then:
- the argmax is unchanged everywhere;
- the centred-logit response on the pair (x₀, x₀′) has magnitude (2/3)t in the ŷ coordinate;
- E KL(p* ‖ p_M) = δ_min · KL_{x₀}(t) with KL_{x₀}(t) = −t p*_ŷ + log(p*_ŷ e^t + 1 − p*_ŷ) ≤ t(1 − p*_ŷ).

Hence **for every B > 0 and every κ ≥ δ_min · B · (1 − p*_ŷ(x₀)) · (3/2)** there is a model with E KL ≤ κ, Bayes-optimal
argmax on all of D, and |Δ_M| ≥ B on a pair of probability ≥ δ_min. Equivalently, for every κ > 0 the construction gives a
response of size up to (2/3)·κ / (δ_min (1 − p*_ŷ)).

The claim "arbitrarily small κ with fixed B" is withdrawn for this construction. The pre-review sentence "no single-episode
tilt reaches B below the floor" was false and is deleted [R1-6]: a *downward* tilt of a non-argmax candidate y keeps the
argmax, gives the same (2/3)t response, and has KL ≤ t·p_y, which can be smaller. No lower bound on κ is claimed.
