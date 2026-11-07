# 14 — Prosecution, defense, adjudication

## PASS 1 — Prosecution

COR, as currently formulated, does not establish a mechanism of Transformer decision formation.

Five most serious vulnerabilities:

1. **No causal bridge.** Distance to `ker d` has no derived relationship to `F_θ`. Attacks A and B are linear counterexamples. The repair formula assumes `-β m` in the objective.

2. **Identifiability failure of `P` and `ρ`.** Definitions allow the zero projection and per-input maps that zero `ds`. Ridge on clean data is a protocol, not a uniqueness theorem. Update Proposition 3 already admits non-uniqueness; the actual gap is worse.

3. **`O` is not a new global quantity.** `D=(ds)^T (d M^{-1} d^T)^+ (ds)` and `O=(ds)^T W (ds)`. Special `W` makes them equal. “Global” means projection onto a linear kernel.

4. **No cohomology.** The paper proves `[ds]=0`. Operational obstruction is a graph residual. Trees can have `O>0`. The local-to-global slogan is not a topological theorem.

5. **Empirical record of the mechanism is negative where it was actually tested.** IOI obstruction-negative; 2D obstruction does not beat attribution at repair; Othello obstruction was never computed (`OBSTRUCTION_READY=NO`); the factorial compatibility test that independently set visible and invisible target identity was negative.

A residual energy plus attribution is not a theory of how decisions form.

## PASS 2 — Defense

Legitimate, non-rewriting defense:

1. Theorem 1 is correct under PD metrics. Using a Laplacian residual as a proxy for distance to a constraint kernel is standard and well-typed.

2. The cohomology correction is a real contribution *against* a mistaken construction. The authors did not ship `[ds]∈H^1` as the operational score.

3. Non-identifiability and superposition are **acknowledged** (Update Props. 3 and 7; math_map theorem targets). The reviewer-safe thesis is weaker than the slogan thesis.

4. Empirical work did produce independently interesting facts: probe-invisible causal information in Othello; a Pythia-1B residual monitor of factual conflict; documentation of IOI as a negative control rather than a hidden win.

5. A freeze-protocol (train-only maps, disjoint evaluation) makes `O` computable prospectively as a **monitor**, which is the only role `docs/paper_claims_after_2d.md` still defends.

The defense may not: rewrite VNF or SYM; treat NST as a germ; treat steering as obstruction; change `W` to reverse negatives.

## PASS 3 — Adjudication

| Claim | Outcome |
|---|---|
| `d`, `O`, `D` are typed linear-algebra objects | SURVIVES |
| Theorem 1 sandwich | SURVIVES |
| `O` equals `D` in general | FAILS (only special `W`) |
| `[ds]` is a useful `H^1` class | FAILS (authors already proved this) |
| Sheaf / local-to-global topology | FAILS as more than metaphor |
| Euclidean `O` is representation-invariant | FAILS |
| Unconstrained definitions identify a unique obstruction | FAILS |
| Low `D` implies correct / preferred candidate | FAILS |
| Reducing `D` repairs `L` | FAILS |
| `S` is a noncircular obstruction predictor | FAILS when `β` uses `Δm` |
| Decision germ is a sufficient / unique causal state | FAILS |
| Probe-invisible Othello causality | SURVIVES as empirical fact, **not** as COR |
| 2B conflict monitoring | SURVIVES as residual correlation, **not** as mechanism |
| Obstruction beats attribution | FAILS |
| Visible/null compatibility interaction | FAILS (completed negative) |
| Symmetric switching | FAILS |
| `OBSTRUCTION_READY=NO` | SURVIVES and remains correct |

Claims that should be removed from any empirical paper: “decisions are local-to-global consistency events”; “`O` is equivalent to global obstruction”; sheaf-cohomology framing of computed scores; “obstruction-response predicts repair beyond attribution” as a general result.

Claims retainable as **conjectures only**: some frozen residual energies may monitor conflict; some interchange-style factorizations may exist in unspecified models.

The prosecution wins on the scientific mechanism. The defense wins on a small algebraic toolkit and on keeping the negative empirical record honest.

This adjudication is not balanced. Algebra ⊂ theory ⊂ mechanism. Only the first is in hand.
