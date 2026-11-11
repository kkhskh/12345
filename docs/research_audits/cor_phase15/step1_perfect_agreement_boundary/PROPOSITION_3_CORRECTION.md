# Correction of Proposition 3 (and Proposition 1's quantifier)

The Investigation V statement is in `../step1_curriculum_response_identification/FORMAL_DECISION_EQUIVALENCE.md` §7. It is
**not edited**; this file supersedes it.

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

- **Class.** 𝔽_mix is the set of maps p_θ(y | X) ∝ Σ_{(g,h) ∈ CFG} w_{gh}(n) · J(x_q, y | g, h), where:
  - log w_{gh} = φ_rel(n_g) + φ_col(n_h);
  - φ_rel, φ_col : {0,…,5} → ℝ ∪ {−∞};
  - n = n^{(−q)} are the leave-q agreement counts;
  - J is the exact own-question likelihood.
- **Natural support.** At least one group has n = 5 and agrees at q, which is the only way a natural T4 episode can have
  positive probability.
- **Minimal Bayesian-null edit.** On natural support with exactly one count-5 group h* (|H| = 1), the two other counts
  (a, b) ≤ 4 both move by the same ±1. One report changes and the exact posterior over configurations is unchanged, since
  π(g, h*) ∝ RHO^{n_g} is multiplied by a common factor.

**Nondegeneracy (A1).** For each X_q in the reachable set, the six vectors J(x_q, · | cfg) determine the mixture weights up
to a common positive scale. Equivalently, the linear map w ↦ Σ_cfg w_cfg J(x_q, · | cfg), taken jointly over the
reachable own-question patterns, has kernel of projective dimension 1 (only the overall scale).
- [N] Verified over all 518 own-question patterns of positive probability: null dimension = **1**.
- A1 is a property of the generator, not of θ.

**Gate regularity (A2).** φ_col is either −∞ on all of {0,…,4} (a *hard* gate) or finite on all of {0,…,4} (a *finite*
gate). φ_rel is finite on {0,…,5}.

**Edit coverage (A3).** For each consecutive pair (k, k+1) ⊂ {0,…,4}, some natural null edit moves a non-colluder count
from k to k+1. [N] All four pairs occur.

## 3. Proposition 3′ [P under A1–A3]

Under A1–A3, f_θ ∈ 𝔽_mix responds 0 to every minimal Bayesian-null edit on natural support **iff** one of the following holds:
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

*Hard gate.* Only (g₁, h*) and (g₂, h*) are finite, so invariance requires α = β. Ranging over all reachable (a, b, ε)
(A3) gives φ_rel(k+1) − φ_rel(k) equal for all k ∈ {0,…,3}. That is affinity on {0,…,4}.

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

**Excluded degenerate cases.**
- *Mixed gates* (−∞ on part of {0,…,4}) violate A2. There the conditions apply piecewise to the counts with finite
  weight, and invariance can hold under weaker conditions.
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

The claim "arbitrarily small κ with fixed B" is withdrawn: below the floor δ_min · KL_{x₀}(3B/2) no single-episode tilt
reaches B. A tighter lower bound on κ would need an optimisation over all tilts, and it is not claimed.
