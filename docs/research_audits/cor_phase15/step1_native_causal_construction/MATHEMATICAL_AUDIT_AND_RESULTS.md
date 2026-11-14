# Mathematical audit of Theorem X1 and corrected results (Investigation XI, Workstream A)

Scripts: `scripts/audit_x1.py` (→ `logs/audit_x1.json`, checks C0–C7) and `scripts/audit_x1_extra.py`
(→ `logs/audit_x1_extra.json`, checks E1–E4). Both are deterministic and run in under 10 s.

**Verdict.**
- Theorem X1 is **corrected and partially retracted**.
- Its central qualitative point stands, in a sharper form: the same input–output function admits implementations
  whose realizable interventions differ.
- Retracted: the I2-VS proof sketch, and the claim "impossible by every linear interchange".
- Strengthened: I1-VE and I3 are impossible for *every* function of the base and source states, not just for linear
  interchange.
- Investigation X's reading of F3 (S-interchange fails, E-interchange succeeds) must be qualified, because VS is not a
  legitimate high-level intervention (§4).

## 0. Setting

Exact C = 1 Bayes rule (Investigation IX, Theorem 1). Per target question q:
- e_g ∈ {0,1} is 1 when group g is perfect (n_g = 5) and its two sources agree at q;
- S = Σ_g e_g;
- U_g = W_g L_g (a K-vector);
- p ∝ Σ_g (S − e_g) U_g.

On the support of the T4 generator, S ≥ 1, because the colluder is always eligible.

There are three implementations:

| implementation | state z | readout |
|---|---|---|
| I1 | (S, A, B), with A = Σ U_g and B = Σ e_g U_g | p ∝ S·A − B |
| I2 | (m, U), with m_g = S − e_g | p ∝ Σ m_g U_g |
| I3 | T_{gh} = U_g e_h for g ≠ h | p ∝ Σ T_{gh} |

There are two high-level interventions, both taking W and L from the base:
- **VS**: S := S_s, keep e_b. The resulting high-level state has S ≠ Σe.
- **VE**: e := e_s, and therefore S := Σ e_s. This intervention is consistent.

Three classes of low-level realization of an intervention (z_b, z_s) ↦ z′:
- **R_any**: any function of (z_b, z_s);
- **R_lin**: z′ = z_b + M(z_s − z_b) for a fixed linear M;
- **R_proj**: as R_lin, with M idempotent (an orthogonal or oblique projection). This is the class that DAS / interchange
  interventions search; DAS uses orthogonal QQᵀ.

R_proj ⊂ R_lin ⊂ R_any.

## 1. The error in Investigation X's I2-VS argument

The sketch relied on "taking m_b = m_s fixed while S_b varies (possible because e_b and e_s can differ with equal m
patterns)". This is false, because m determines e and S:
- Σ_g m_g = 3S − S = 2S, so S = ½ 1ᵀm;
- e_g = S − m_g.

Check C0 confirms that no on-support pair has equal m and different S (0 pairs over all configurations), and that the
identity S = Σm/2 holds exactly. The argument is therefore **retracted**. Its conclusion is partly true and partly
false, as §2 shows.

## 2. Corrected Theorem X1′

| | VS (set S, keep e_b) | VE (set e) |
|---|---|---|
| **I1** (S, A, B) | **∈ R_proj**: copy the S coordinate (C1, error 3e-16) | **∉ R_any** (C5; proved below) |
| **I2** (m, U) | **∈ R_lin** via M = ½·11ᵀ on m (C3, error 0); **∉ R_proj**, for every orthogonal or oblique projection on the full (m, U) space (proved; E1, E2) | **∈ R_proj**: copy the m-coordinates (C2, error 0) |
| **I3** (T_{gh}) | **∉ R_any** (C6) | **∉ R_any** (C6) |

### Proofs

**I1-VE ∉ R_any (C5).** There exist two pairs with identical (z1_b, z1_s) but different VE targets (L1 difference
0.52). The reason is that z1 = (S, A, B) does not determine e: B = Σ e_g U_g pools the groups. No function of the two
states can therefore produce the target. A natural-distribution witness (C7): 500 natural episode pairs with identical
I1 states, of which 96.2% have different VE targets (median L1 0.032).

**I3 ∉ R_any for VS and VE (C6).** Same method. There are pairs with identical (z3_b, z3_s) and different targets:
- VE: L1 difference 0.197;
- VS: L1 difference 0.090.

T_{gh} = U_g e_h stays zero whenever e_h = 0, so I3 loses U_g for every g that has no eligible partner h.

**I2-VS ∈ R_lin.** VS needs m′_g = S_s − e_{b,g} = m_{b,g} + (S_s − S_b). Since S = ½1ᵀm, this equals
m_b + ½11ᵀ(m_s − m_b), a fixed linear map of the difference. It holds for every on-support pair (E1), with U′ = U_b.

**I2-VS ∉ R_proj (full space, any projection).** Let z = (m, U) and z′ = z_b + P(z_s − z_b), with P idempotent.

(a) VS keeps U, so U′ = U_b for every pair. The U-rows of P must annihilate every difference, and these differences span
the space. Hence P_Um = 0 and P_UU = 0.

(b) Pairs with e_s = e_b and U_s ≠ U_b have VS target equal to the base. This forces P_mU ΔU = 0 for arbitrary ΔU, so
P_mU = 0.

(c) What remains is the m-block P_mm. Take base e_b = (1,0,0) and sources (0,1,0), (0,0,1), (1,1,0). The Δm vectors are
(1,−1,0), (1,0,−1), (1,0,1). Their determinant is 2, so they span ℝ³. VS requires P_mm Δm = (½1ᵀΔm)·1 for all three,
which forces P_mm = ½11ᵀ (E1). Its eigenvalues are {3/2, 0, 0}, so it is not idempotent. Contradiction. ∎

The dense numerical search over projections of ℝ³ (C4) agrees: the minimum residual over ranks 0–3 is ≥ 0.476, and the
best is the symmetric 11ᵀ/3 at 0.471.

## 3. Restricted general results

**Proposition A1 (realizability is a property of the implementation, not of the function).** For the C = 1 rule,
I1 and I2 compute the same function, but their sets of realizable interventions differ:
- I1: {VS} by projection;
- I2: {VE} by projection, and {VS} only by a non-projective linear map.

I3 realizes neither, under any function. This is the corrected content of X1. It is an explicit, task-specific instance
of the known non-uniqueness of causal abstraction; we do not claim the general phenomenon as new.

**Proposition A2 (exact characterization of projection-realizability).** Let the low-level state be z = φ(V, R),
where V is the intervened variable and R is everything else. Suppose the pair support is a product set (any V_s can be
combined with any R_b). Then "set V := V_s" is realizable by a projection interchange z_b + P(z_s − z_b), with P
idempotent, **if and only if** z decomposes as a direct sum:

φ(V, R) = a(V) + b(R), with a(V) ∈ range P and b(R) ∈ ker P.

*Proof.* The patched state is z′ = (I − P)z_b + Pz_s, and the target is φ(V_s, R_b).
- (⇒) Apply (I − P) to both sides: (I − P)φ(V_s, R_b) = (I − P)φ(V_b, R_b) for all V_s, so (I − P)φ does not depend on
  V. Apply P: Pφ(V_s, R_b) = Pφ(V_s, R_s) for all R, so Pφ does not depend on R. Set a = Pφ and b = (I − P)φ.
- (⇐) Substituting the decomposition gives z′ = b(R_b) + a(V_s) = φ(V_s, R_b). ∎

Corollaries:
- I2-VE is realizable, since m depends only on e and U only on the rest.
- I1-VS is realizable, since A and B do not depend on S once e is held fixed.
- I1-VE fails the decomposition, because B mixes e and U multiplicatively.

Checked numerically on random implementations (E4):
- a separable φ is exact under the oblique projection (error 1e-16);
- adding a multiplicative cross term breaks it (projection error 0.98; even the best *linear* map leaves 31% residual).

**Corollary A3 (what reparametrization preserves).** Projection-realizability is invariant under:
- any **linear** invertible re-mixing T of the state, with P → TPT⁻¹ (E4, error 7e-15);
- any coordinate-wise nonlinearity *inside* the V-block or the R-block.

It is **not** invariant under nonlinear mixing across blocks (E4: best linear map residual 0.43). Normalization is a
case of this. An implementation that stores only the normalized posterior (z = p) mixes V and R nonlinearly through the
normalizer. It realizes neither VE nor VS by projection, although it computes the same function. Whether an intervention
is "found" therefore depends on the representation format as well as on the computation. This is the mathematical
reason why interchange success or failure is weak evidence about an algorithm (compare §5 and Workstream C).

## 4. Intervention legitimacy: VS is not a valid high-level test

VS sets S while keeping e. The resulting high-level state violates the model's own constraint S = Σ_g e_g, so it is
not the image of any input.

In I2 terms the conflict is concrete. The VS target m′ corresponds to e′_g = (S_s − S_b)/2 + e_{b,g}, which is
non-integer whenever S_s − S_b is odd. That is 49% of on-support pairs (E3). VS therefore asks I2 to represent states
that no input produces.

**Consequence for Investigation X's F3.** In the networks, the S-interchange failed (IIA 0.47–0.55) and the
E-interchange succeeded (0.85–0.97). This was read as a "per-group signature". The corrected reading:
- The E result (a consistent intervention) is evidence for a per-group-separable eligibility representation.
- The S result is evidence only that the network does not hold S in a separately projectable coordinate. It does not
  show that S is absent, since an I2-like implementation contains S linearly (S = ½1ᵀm) and still fails VS-by-projection.
- "S-interchange fails" must not be cited as a falsification of a pooled algorithm. That claim depends on an invalid
  high-level intervention.

The F3 *data* are unchanged; the *interpretation* is downgraded from "rules out pooled-S implementations" to "consistent
with per-group storage; VS is not a valid discriminating test".

## 5. Proved vs numerical-only

| claim | status |
|---|---|
| I1-VS ∈ R_proj; I2-VE ∈ R_proj | proved (construction) + numerical 0 |
| I2-VS ∈ R_lin | proved (construction) + numerical 0 |
| I2-VS ∉ R_proj, full (m, U) space, orthogonal or oblique | **proved** (§2) |
| I1-VE ∉ R_any; I3-VS, I3-VE ∉ R_any | **proved** by explicit witnesses (C5, C6, C7) |
| Proposition A2 (characterization) | proved; illustrated numerically (E4) |
| DAS (gradient search) fails to find the exact I1-VS alignment (Investigation X) | numerical only: an optimization fact, not a realizability fact |
| Network-level "E realizable, S not" | empirical (Investigation X F2/F3), now qualified (§4) |

## 6. Effect on Investigation X conclusions

- "Different implementations realize different interventions": **upheld**, with corrected scope.
- "I2-VS impossible by linear interchange": **retracted**. It is possible by a linear map, and impossible only by
  projection.
- "I1-VE impossible by linear interchange": **strengthened** to impossible by any function.
- "I3 near-failure": **strengthened** from numerical to proved impossibility.
- F3's interpretation: **restricted** (§4).
- The identification caveat (a failed DAS search is weak evidence): **upheld and generalized** by Proposition A2.
  Search failure can also reflect a representation-format mismatch (Corollary A3), not just optimization.
