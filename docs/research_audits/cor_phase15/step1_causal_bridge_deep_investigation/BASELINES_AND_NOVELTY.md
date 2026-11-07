# Baselines and novelty

For each result, the table gives the strongest non-COR explanation. The mapping and the references follow `reviews/R4_baselines_novelty.md`, an independent review that ran in a separate context and searched the literature. URLs are in that file. References marked † were taken from it and not re-fetched for this document.

| Result | Strongest non-COR explanation | What consistency framing adds |
|---|---|---|
| A1: MLP agreement law `sign m = sign(xy)` | An XNOR/XOR unit. The small-signal regime is the polarization identity `xy = ((x+y)² − (x−y)²)/4`, which is the standard way ReLU networks multiply (Yarotsky 2017†). `|x+y| − |x−y| = 2 sgn(xy) min(|x|,|y|)` is a textbook ReLU construction | nothing beyond naming the pairing |
| A2: binary indistinguishability | Mechanism non-identifiability from finite behavior (Méloux et al., ICLR 2025†) | nothing |
| A3: interaction and odd–odd component | Scale and link dependence of interaction (Loftus 1978; Ai and Norton 2003†). Concretely, `q` is exactly twice the odd–odd component of the single-index function `h(x+y)` | nothing |
| A4: matched post-LN sum contrast | An interchange or matched-statistic intervention (causal abstraction, Geiger et al. 2021†, JMLR 2025), performed at the reader's actual input | nothing. The failed PA9 is an example of matching at the wrong site |
| B1: QK gate | Q/K composition and content-based routing (Elhage et al. 2021). Copy-suppression heads (McDougall et al. 2023†) are a trained-model example with the opposite sign. PB9 is the familiar gap between direct and total effects (path patching, backup heads, Hydra effect: McGrath et al. 2023) | nothing |
| C1–C3: two-block reconciliation | A gated multiplicative interaction (GLU-style gating; Jayakumar et al. 2020†). C1.3 is the intermediate value theorem. C2 is screening-off. C3 is Méloux et al.† again | nothing |
| ID1, ID2: gauges | Standard function-preserving weight symmetries (Chen, Lu and Hecht-Nielsen 1993†; Godfrey et al. 2022†). Attention weights are not unique in general (Brunner et al. 2020). ID1 requires identical heads and is **non-generic** | nothing; the corrections are restorations of known facts |
| SS*: no-bypass screening | Screening-off in a causal graph | nothing |

## The mandate's §10 checklist

- **A more restricted theorem?** The theorems are correct and restricted. They follow from monotonicity of a designed gate in one or two lines (`THEOREMS_AND_COUNTEREXAMPLES.md` §5, check 5). A gating or composition account states them just as well.
- **A reusable prediction?** Only for the designed classes. A mirrored circuit that amplifies under *dis*agreement is equally implementable in the same architecture (supplementary log: `m_dis = 3.185` against `m_ag = 0.246` at matched `Ŝ`). So no architectural fact predicts that consistency stabilizes decisions, which is the original thesis's direction (`docs/notes/Causal_Obstruction_Response_theory_Update.pdf` §2: "stable decisions tend to exhibit low candidate-specific incompatibility").
- **An independently identified structural condition?** Weight conditions are checkable modulo the gauge groups, and only on designed weights. They are not found in any trained model.
- **A simpler or more invariant explanation?** No. The invariant objects (input-level effects, reader compositions) are the ordinary objects of interchange analysis.
- **A mechanism that attribution misses?** Interaction indices and first-order attribution do miss the difference between "relation" and "net-evidence nonlinearity" (A2, A3). Interchange interventions catch it (A4). This is a known limitation of attribution, and the known remedy works.
- **A new distinction between causally different systems?** One minor one: a candidate-symmetric *agreement detector* (A) versus *agreement-gated evidence* that favors a candidate (B, C). Standard terms state it without loss.

**Conclusion.** The examples do not support a distinctive theory. The mechanisms are real and implementable, and they are fully explained by ordinary nonlinear interaction, gating, and attention composition.
