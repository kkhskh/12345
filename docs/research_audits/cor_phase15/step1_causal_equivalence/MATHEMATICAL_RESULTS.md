# Mathematical results (Investigation X)

Tags: **[P]** proved (elementary); **[N]** numerically verified (script and log given); **[K]** known. Bayes' rule is not
renamed, and no COR theorem is claimed.

## 0. The rule and its assumptions (carried over and re-verified)

Investigation IX, Theorem 1. For the role model:
- G groups of 2 sources;
- one reliable group and C colluding groups, assigned uniformly;
- β = 1 (colluders always agree);
- noise sources uniform;
- K candidates and Q questions, with independent uniform Y_q.

The posterior is p(y | X) ∝ binom(S, C)·A_y − binom(S−1, C−1)·B_y, where:
- e_g = 1[n_g = Q−1]·1[g agrees at q];
- S = Σ_g e_g;
- A_y = Σ_g W_g L_g(y) and B_y = Σ_g W_g e_g L_g(y).

It was verified against brute force for (G, C) up to (6, 4) in Investigation IX (`logs/verify_math2.json`, max |Δp| ≤ 1.9e-13).

**Assumptions that matter here.**
- **β = 1 is essential.** For β < 1 the hard gate is replaced by soft likelihood ratios (Theorem 3 of Investigation IX).
- **Support.** At least one eligible group always exists (the true colluder). Any input with S = 0 has probability 0,
  so it is **off-support**. This is used in §3.

## 1. Notions of equivalence between two models M, M′ computing decisions at question q

1. **Behavioural equivalence on the support D:** M(x) = M′(x) for all x ∈ D.
2. **Functional equivalence under a family 𝓔 of input edits:** the responses M(x′) − M(x) and M′(x′) − M′(x) are equal for all
   (x, x′) ∈ 𝓔.
3. **Causal-abstraction equivalence with respect to a high-level model H and a variable V:** each model admits an
   alignment, meaning a site and a map, under which interchanging V between inputs produces H's counterfactual output.
   Both share the abstraction (H, V) [K: causal abstraction, Geiger et al.].
4. **Implementation equivalence:** a structure-preserving correspondence of internal components, such as tokens, heads
   and subspaces.

**Proposition 1 [P]. Behavioural equivalence on D implies functional equivalence under every edit family inside D.** If
M = M′ on D, then for every on-support edit pair the responses are equal.

*Consequence.* Two models that both compute the exact Bayes function can be distinguished **only** by:
- (a) off-support inputs;
- (b) internal interventions;
- (c) their approximation errors.

The mandate's preferred demonstration, "agree on ordinary episodes but differ under a valid relational intervention", is
impossible between exact models. It is possible only through differences in approximation, and that is what §4 measures.

## 2. Theorem X1: interchange abstraction depends on the factorization, not on the function [P; N]

Three exact implementations of the C = 1 rule (`scripts/math_abstraction.py`, `logs/math_abstraction.json`), with
U_g = W_g L_g:

| implementation | intermediate z | readout |
|---|---|---|
| I1 pooled | (S, A, B) | p ∝ S·A − B |
| I2 per-group | (m_0, m_1, m_2, U_0, U_1, U_2), m_g = Σ_{h≠g} e_h, computed pairwise with no S | p ∝ Σ_g m_g U_g |
| I3 pairwise | T_{gh} = U_g e_h for ordered g ≠ h | p ∝ Σ_{g≠h} T_{gh} |

All three compute the same posterior (max |Δ log p| = 2.5e-12). Two high-level interventions are considered:
- **VS:** set S := S_source, keep base e_g and U;
- **VE:** set the eligibility vector e := e_source, keep base U.

| | VS (set S) | VE (set e) |
|---|---|---|
| I1 | **exact**: interchange coordinate S (error 0) | impossible by linear interchange (the S coordinate gives 0.62) |
| I2 | impossible (m-coordinates give 0.62; m·1 = 2S, but VS needs Σm′ = 3S_src − S_base) | **exact**: interchange the 3 m-coordinates (error 0) |
| I3 | best subspace found ≥ 0.98, against 1.02 with no intervention | best ≥ 0.77, against 0.78 with no intervention |

The error is the mean L1 distance in probability to H's counterfactual, over 4000 pairs with S_src ≠ S_base.

**Proof sketch.**
- *I1-VS:* replacing S leaves A and B intact, which is exactly VS.
- *I2-VE:* m depends only on e, and U is kept.
- *I2-VS fails for every linear interchange subspace:* VS requires m′_g = m_g + (S_s − S_b) for every g, a shift that
  depends on *both* episodes. A linear interchange copies the source's coordinates along R. Along any direction it
  realizes values of the form P_R m_s + (I − P_R) m_b. Taking m_b = m_s fixed while S_b varies (possible because e_b and
  e_s can differ with equal m patterns) shows that no R satisfies VS for all pairs. The numerical optimum (≥ 0.45 for
  k = 1–4) agrees.
- *I1-VE fails:* B = Σ_g U_g e_g mixes base U with source e, which is a product of base and source quantities. No
  linear interchange of (S, A, B) produces it.

**Statement.** For the same input-output function, the set of high-level interventions realizable by linear interchange
differs between implementations: {VS} for I1, {VE} for I2, and none tested for I3. **A failed interchange alignment for a
variable therefore does not show that the function or algorithm differs, and a successful one does not establish a
unique intermediate.** This is a constructive, task-specific instance of the known non-uniqueness of causal
abstractions [K].

**Identification caveat [N].** Gradient-based subspace search (Adam, 4 restarts, k = 1–4) did **not** find the exact I1-VS
alignment: the best was 0.26, although 0 exists. It did find I2-VE (0.0018 at k = 3). A failed DAS search is therefore
weak evidence even about one implementation. This is why the neural analysis uses (i) several variables, (ii) random-
subspace controls and (iii) a behavioural test that needs no search (§4).

## 3. Proposition 2: support restrictions on relational edits [P]

An edit that removes the agreement at q of the only eligible group produces S = 0. That input has probability 0 under the
role model.

Investigation X's first version of edit family D1 included such edits (5,501 of 6,000). They were detected by a NaN in
the Bayes reference and removed. D1 is restricted to S ≥ 2, giving n = 499. Results on off-support inputs are not
interpreted.

## 4. Algorithm discrimination by on-support edits (the testable consequence of §1)

Candidate rules share W and L and differ in the eligibility factor:

| rule | eligibility factor |
|---|---|
| B (Bayes) | f_g = S − e_g |
| A1 | perfect-only: e = [n = 5], ignoring agreement at q |
| A2 | threshold: e = [n ≥ 4][agree at q] |
| A3 | separable: f_g = 1 − e_g |
| A0 | none: f_g = 1 |

**Separation table [P, computed exactly].** Mean L1 distance between rules' predicted responses (centred logits):

| family | A1 vs B | A2 vs B | A3 vs B |
|---|---|---|---|
| D1 (break q-agreement of an eligible group, S ≥ 2) | 1.12 | ≈ 0.01 | 1.65 |
| D2 (create q-agreement of a perfect group) | 1.05 | 0.014 | 1.80 |
| R1 | 0.002 | 0.11 | 1.80 |
| R2 | 0.000 | 2.05 | 1.68 |
| R3 | 0.004 | 1.93 | 1.72 |

Each rule pair has at least one family on which it is separated by ≥ 1 (A2 vs B only through R2/R3). So the families
jointly identify the rule among these candidates. This is the design basis of the confirmation tests.

## 5. Interaction order is not a Transformer lower bound

Investigation IX, Theorem 2 (interaction order 1 + C of the evidence function) restricts *evidence-form polynomial*
readers. It does not bound what a Transformer needs. Finite-domain sum decompositions are universal (Investigation IX,
§3). No Transformer-wide lower bound is claimed.
