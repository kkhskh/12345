# 01 — Authoritative source register

Phase 1.5, Step 1. This directory does not replace `docs/research_audits/cor_phase1/`. No experimental letter is changed. Repository `HEAD` at the time of the audit is `f95d134fee8bfd870561e21bc57ada18c46248be`. The Phase 1 audit and the two COR PDFs are in that tree. Later uncommitted manuscript files under `docs/papers/` are not sources of COR definitions.

The PDF text below was read from the files in `docs/notes/`. Equation numbers are the original manuscript's. Proof status "MISSING" means this repository still has no proof; this step does not invent one and then attribute it to the manuscript.

## 1. Sources

| ID | Path | What it is | Use here |
|---|---|---|---|
| PDF-0 | `docs/notes/Causal_Obstruction_Response_theory.pdf` | Original COR manuscript, 15 pages | Definitions 1–4, Theorem 1 target, original Propositions 2–6 |
| PDF-1 | `docs/notes/Causal_Obstruction_Response_theory_Update.pdf` | Update, 26 pages | Renumbered propositions, latent-state germs, Propositions 6–7 |
| MVP | `docs/theory_mvp_theorem_set.md` | In-repo proofs of the spectral sandwich, the quadratic repair identity, and `[ds]=0` | Proof status of those three results only |
| MAP | `docs/math_map_v0.md` | Theorem targets, not proofs | Confirms the stability target was already conditional on beating margin and attribution |
| P1 | `docs/research_audits/cor_phase1/` | Prior audit | Failure classes F1–F30 and frozen empirical letters. Not re-litigated as experiments |
| CODE | `src/cor_phase1_math.py`, `tests/test_cor_phase1_math.py` | Finite-dimensional algebra | What the 19 synthetic tests establish |
| NEW | `tests/test_causal_bridge_math.py` | This step's calculations | Counterexamples below. Not a Transformer |

## 2. Original objects, as written

Candidate-specific local state (both PDFs, Definition 2):

```text
s_c(v;x) = P^c_v φ_v(x) ∈ F_c(v)
```

`P^c_v` is described as "a candidate-specific projection, probe, feature map, or logit-lens contrast." The sentence does not require `P^2=P`, an inner product, a fixed rank, or independence from the evaluation response.

Compatibility (both PDFs, §5.4): `ρ^c_{u→v}: F_c(u) → F_c(v)`. Two constructions are offered: ridge regression on `T_clean`, and the local Jacobian formula

```text
ρ^c_{u→v} ≈ P^c_v (∂φ_v/∂φ_u) (P^c_u)^†.
```

The Jacobian formula right-multiplies by the pseudoinverse of `P_u`. It reproduces `P_v J` on the row space of `P_u` and discards `ker P_u`. That fact is used in `03_exact_vs_approximate_computation.md`. No composition law `ρ_{v→w} ∘ ρ_{u→v} = ρ_{u→w}` is stated.

Coboundary (both PDFs, §5.5): `(d_c s)(u→v) = ρ^c_{u→v} s_c(u) - s_c(v)`.

Distance (Definition 3):

```text
D_c(x) = min_{g ∈ H^0} ||s_c(x) - g||_M^2,    H^0 = ker d_c.
```

Residual energy (Definition 4):

```text
O_c(x) = ||d_c s_c(x)||_W^2 = Σ_e w_e ||ρ s_u - s_v||^2.
```

The PDFs also define a normalized energy and `Frag_c = Õ_c / (|m_c|+ε)`. Fragility divides by the margin. It is not a derivation that obstruction causes the margin.

Intervention notation in the PDF is `h ↦ h + B_u δ_u`, with linearized margin response `Δm_c ≈ J^m_{c,u} δ_u` and `J^m = ∂m_c/∂h_u`. Descendants are not given a separate recomputation rule in the definition. This step's semantics are in `04_intervention_semantics_and_outcomes.md` and are marked as an audit convention, not as a manuscript theorem.

## 3. Propositions that matter for a bridge

Update Proposition 1, original Proposition 2, states a minimizer of an objective that already contains `-β m_c`. MVP proves the quadratic-program identity under `A ≻ 0`. The identity does not derive `m_c` from `D_c`.

Update Proposition 2, original Proposition 3, as printed:

```text
Suppose the margin response is locally Lipschitz under perturbations of norm at most r.
If |m_c(x)| > sup_{||δ||≤r} |J_m δ| + τ O_c(x),
then the candidate ordering is stable under the allowed perturbation class.
```

There is still no proof in the repository. The statement is analyzed, not adopted, in file 05. The slack term `τ O_c` is not needed for the implication: `τ = 0` is a weaker hypothesis and yields the same conclusion. The quantity that does the work is the gap between the current margin and the output Jacobian.

Update Proposition 3, original Proposition 4, says that multiple low-overlap compatibility structures "may" induce similar obstruction-response predictions. That is an existence remark. It is not an identification theorem, and it has no proof in the repository. Counterexamples CE-07 and CE-10 exhibit the phenomenon the remark points at.

Update Propositions 6 and 7 are a local Taylor remainder and a superposition non-uniqueness statement. Neither supplies a map from `D_c` to `L_c`.

## 4. Conflicts that this step does not resolve by choosing a favorite

| Object | Conflict | Treatment |
|---|---|---|
| Repair proposition number | Original Prop. 2 vs Update Prop. 1 | Both cited. Same objective. |
| Stability proposition number | Original Prop. 3 vs Update Prop. 2 | Quoted from the Update text above. |
| `O` versus `D` | PDFs call the relationship equivalence; MVP proves a spectral sandwich | Sandwich only, unless `W = (d M^{-1} d^T)^+`. |
| `ρ` independent of `x` | Definition has no `x`; Jacobian option does | Both kept. Input-specific `ρ` is an identifiability attack, not a repair. |
| Sheaf | No composition law in the PDFs | The object used here is a directed graph of linear maps and the operator `d`. |

## 5. What the existing synthetic tests establish

`tests/test_cor_phase1_math.py` checks the coboundary, the pseudoinverse formula for `D`, the spectral sandwich, and the Phase 1 attacks A–L on small matrices. Passing those tests shows that those finite-dimensional statements are calculated correctly. It does not show that `D_c` predicts a Transformer decision.

## 6. What is still missing

- A proof, in the original sources, of Update Proposition 2.
- A uniqueness rule for `P`, `ρ`, `M`, `W`, the graph, and `τ`.
- A theorem deriving `L_c` or `R_c` from `(P, ρ, d, M, W)` without an extra assumption that already relates them.
- Any source in which exact natural computation is shown to produce nonzero candidate-specific obstruction for a reason other than projection loss, an omitted parent, or an approximate `ρ`.

Those absences are inputs to this step. They are not repaired by renaming `D`.
