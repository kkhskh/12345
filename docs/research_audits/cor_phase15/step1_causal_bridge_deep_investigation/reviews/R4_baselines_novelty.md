# R4: Baselines and novelty review

Reviewer role: adversarial, baselines and novelty. Inputs: `sources/MANDATE_deep_investigation.txt` §10, §13, §17; `THEOREMS_AND_COUNTEREXAMPLES.md`; `logs/experiments.txt`. I re-derived the identities that matter. I did not rerun the code. Citations are ones I found by web search during this review. Where I rely on memory, I say so.

**Bottom line.** Every mechanism and every identification result in the package has a complete standard explanation, with nothing left over. The consistency framing adds one modest taxonomic distinction (candidate-relative vs candidate-agnostic agreement), but no theory, prediction or identification result that standard tools lack. **Recommended verdict: `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`.** `STEP1_RESTRICTED_IMPLEMENTED_MECHANISM_PROVED` is defensible only if the package also says that distinctiveness was examined and found absent, not that it "remains unresolved".

---

## 1. Strongest non-COR explanation, by item

### Mechanism A (A1): agreement-sensitive MLP

- **Standard reading.** A1 is a textbook XOR/XNOR unit built from the polarization identity.
  - `q = h(c(x+y)) − h(c(x−y))` with `h` even.
  - For a quadratic `h`, this is exactly `xy = ((x+y)² − (x−y)²)/4`. That is the standard way ReLU/MLP networks multiply via squaring; see Yarotsky 2017, https://arxiv.org/abs/1610.01145.
  - For ReLU, `|x+y| − |x−y| = 2 sgn(xy) min(|x|,|y|)` is the standard min/abs construction.
  - Parity/XOR learning by MLPs is well studied; see Barak et al. 2022, https://arxiv.org/abs/2207.08799.
- **Nothing is candidate-relative.** A1.2 (`m(x,y) = m(−x,−y)`) says the unit is XNOR, not "consistency toward a candidate". The package says this itself.
- **A1.3 adds nothing.** Sign invariance under separate positive rescalings is a property of `sgn(xy)`, not a new invariance.
- **Residual: none.**

### A2: binary indistinguishability from a sum-only reader

- **Standard reading.** A finite behavioral family does not identify the mechanism. This is the known non-identifiability of mechanistic explanations from behavior: Méloux et al., ICLR 2025, https://arxiv.org/abs/2502.20914. A related point, that interchange-intervention fits are not enough without constraints on the alignment map, is in Sutter et al. 2025, https://arxiv.org/abs/2507.08802.
- **Residual: none.** It is a correct, clean instance.

### A3: interaction and odd-odd component identity

- **The package's framing understates how trivial this is.** For even `h`:
  - `E_oo[h(x+y)] = [2h(x+y) − 2h(x−y)]/4 = q/2`.
  - So the designed circuit is *exactly twice the odd-odd ANOVA projection of the sum-only reader*. The two differ only in components that are even in `x` or in `y`.
  - Any contrast built from sign-flip symmetry therefore cannot separate them. That is a one-line ANOVA fact.
- **Known in statistics.** "Nonzero interaction ≠ a relation is computed" is the standard point that an interaction depends on the scale or link function.
  - Loftus 1978 on removable interactions, as surveyed by Wagenmakers et al. 2012: https://pmc.ncbi.nlm.nih.gov/articles/PMC3267935 (I found this page by search only; I did not fetch it).
  - Ai & Norton 2003 on interactions under nonlinear links: https://ideas.repec.org/a/eee/ecolet/v80y2003i1p123-129.html.
- **Precision note.** `h(x+y)` is *not* Loftus-removable by a monotone transform, because `h` is not monotone in `S`. It is a single-index model, so the right baseline is "a nonlinear link of an additive index", not Loftus removability strictly.
- **Weight-based interaction detectors** would also flag both models as interacting. See Tsang et al. 2018, https://arxiv.org/abs/1705.04977, and Shapley-Taylor, Sundararajan et al. 2020, https://proceedings.mlr.press/v119/sundararajan20a.html.
- **Residual: none.**

### A4: matched post-normalization sum criterion

- **Standard reading.** This is an interchange-intervention / causal-abstraction test. Hold the high-level variable `Ŝ` fixed at the site the reader actually consumes, vary `D̂`, and ask whether a high-level model with only `S` is faithful.
  - Geiger et al. 2021, https://arxiv.org/abs/2106.02997.
  - Geiger et al., JMLR 2025, https://jmlr.org/papers/v26/23-0058.html.
- **PA9 is the textbook failure of matching at the wrong site.** The package matched the raw sum, but the reader sees the LayerNorm-scaled sum. That failure plus the Loftus scale point fully explains PA9.
- **Residual.** A methodological lesson: match at the reader's input, after LN. It is useful, not distinctive. It is also post hoc.

### Head-redistribution gauge (ID1)

- **Standard reading.** Function-preserving weight symmetries make per-component attributions non-identifiable. See Chen, Lu & Hecht-Nielsen 1993, https://mlanthology.org/neco/1993/chen1993neco-geometry, and Godfrey et al. 2022, https://arxiv.org/abs/2205.14258. ID1 is a direct consequence of SS*, which is ordinary functional composition / screening-off.
- **Weakness.** ID1 needs *identical* `W_Q`, `W_K`, `W_V` in two heads, which is a measure-zero case.
  - The generic version of the concern is Brunner et al. 2020, https://arxiv.org/abs/1908.04211: attention weights are not uniquely determined by the head output when the sequence is longer than the head dimension.
  - The package does not cite or engage it.
- **Residual: none.**

### QK/VO gauges and ReLU rescaling/splitting (ID2)

- **All standard.**
  - Only the products `W_Q W_K^T` and `W_V W_O` matter. Elhage et al. 2021, https://transformer-circuits.pub/2021/framework/index.html, treats the QK and OV circuits as the meaningful objects. *Uncertain:* the secondary summary I fetched did not quote the exact `W_QK` definition. That the paper defines it is my recollection.
  - ReLU positive rescaling and unit splitting are classical; see the Chen et al. and Godfrey et al. links above.
- **Concern.** The package lists "attention scores and patterns" as identifiable. That is true only modulo the listed group `G`, **not** at function level.
  - A head whose OV output is never read can have any pattern.
  - The Brunner et al. null-space argument also applies.
  - Mechanism B's relation `C = x̂ŷ` *is* an attention score, so B's "independent identification" inherits this caveat. The package should say "identifiable modulo the stated `G`" and nothing stronger.

### Mechanism B (B1): two-layer QK gate

- **Standard reading.** Q/K-composition (Elhage et al. 2021, link above): a later head's query reads a feature an earlier head wrote, and the bilinear QK score routes attention to the source whose key matches. This is content-based routing.
- **A known trained-model analogue exists.** Copy-suppression heads (McDougall et al. 2023, https://arxiv.org/abs/2310.04625) attend according to a match between a feature from earlier layers and a source token, then move a vote. Their sign is the reverse of B's, but it is the same QK-gated vote structure.
- **B1.4's reversal band is a direct softmax calculation.** PB4's non-additivity is the bilinearity of the score.
- **PB9 (downstream re-reader breaks the decision claim)** is the standard gap between direct and total effects, and self-repair. See path patching, Goldowsky-Dill et al. 2023, https://arxiv.org/abs/2304.05969; IOI backup and negative heads, Wang et al. 2022, https://arxiv.org/abs/2211.00593; and the Hydra effect, McGrath et al. 2023, https://arxiv.org/abs/2307.15771.
- **Residual: none.**

### Mechanism C (C1–C3): two-block reconciliation

- **Standard reading.** Block 1 is Mechanism A writing an XNOR feature. Block 2 is a ReLU gate on the conjunction "`|Ŝ|` large AND `q̂` large", i.e. a multiplicative/gated interaction.
  - Jayakumar et al., ICLR 2020, https://iclr.cc/virtual/2020/poster/1710.
  - GLU-style gating, Shazeer 2020, https://arxiv.org/abs/2002.05202.
- **C1.3 is the intermediate value theorem** applied to a constant prior shift.
- **C2 is SS*, i.e. screening-off.**
- **C3 (behavior does not reveal depth)** is again Méloux et al. 2025.
- **Side note on C3.** Even without LN, exact non-representability by one hidden ReLU layer should follow from a standard fact: the kink set of a shallow ReLU net in 2D is a union of full lines, while C's kink set is bounded or a polyline. The package could cite or prove this instead of the analytic-zero-set sketch. *Uncertain:* I have not checked the exact construction.
- **Residual: none.**

---

## 2. Does the consistency framing contribute anything (mandate §10)?

| §10 criterion | Finding |
|---|---|
| More restricted theorem | No. A1, B1 and C1 are correct, but each reduces to the polarization identity or the monotonicity of a ReLU/softmax gate under designed weights. A circuits paper would state them identically without "consistency". The package concedes that check 5 ("not the output sign in disguise") partly fails. |
| Reusable prediction | No distinctive one. B1.4's band and C1.3's prior interval are one-line consequences of the designed gate. No prediction spans a family of interventions that a QK-composition or gating account would not also make. |
| Independently identified structural condition | Only modulo the stated `G`, and only as weight conditions on a designed model. Scores and patterns are not function-level identifiable (see ID2 above). Nothing is identified in a trained model, which §13 correctly does not claim. |
| Simpler or more invariant explanation | No. The standard descriptions (XNOR unit, QK-gated vote, gated conjunction) are simpler and carry the same invariances. |
| Mechanism ordinary attribution misses | Partly. Interaction indices and odd-odd contrasts miss the sum-vs-relation difference (A2, A3). Standard interchange interventions at the reader's input catch it (A4). So it is missed by naive attribution, not by established causal abstraction. |
| New distinction | One, and minor. The split between *agreement detector* (A: even in the evidence, candidate-agnostic) and *agreement-gated evidence* (B, C: odd in the evidence, candidate-relative) is clean and useful. It is fully expressible in standard terms (the parity of the gated output in the evidence) and gives no new theorem. |

**Conclusion.** No distinctive theory is supported. Under §10, the package should say so plainly: "the example does not support a distinctive theory".

## 3. Verdict (mandate §17)

- `STEP1_CAUSAL_BRIDGE_IDENTIFIED`: **no.** It lacks a defensible distinction from existing methods, and identification holds only modulo the listed groups.
- `STEP1_CONDITIONAL_CAUSAL_BRIDGE_SURVIVES`: **no.** The surviving theorems are implemented-mechanism identities, not a bridge with substantive assumptions.
- `STEP1_EXAMINED_MECHANISMS_FAIL`: **no.** The mechanisms work within their stated classes.
- `STEP1_INVESTIGATION_INCOMPLETE`: **no.** All three families were examined (MLP, moving Q/K attention, two-block rereading), and the identifiability checks were done. Brunner-type generic non-identifiability of attention is a missing engagement, not a skipped computational class.
- `STEP1_RESTRICTED_IMPLEMENTED_MECHANISM_PROVED` vs `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`: the facts satisfy both preconditions. "Explicit circuits implement relation-sensitive rules" is true. But RESTRICTED says distinctiveness "remains unresolved", and from the baselines side it is resolved, negatively: every item above has a complete standard explanation with zero residual. ONLY_KNOWN is the more informative and more accurate label.

**Recommendation: `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE`.** State the A/B/C proofs as correct implemented-mechanism results, and record the candidate-relative vs candidate-agnostic distinction as the only framing-level contribution. If the authors choose RESTRICTED anyway, they must write that distinctiveness was tested against these baselines and failed. Leaving it "unresolved" would overstate the open question.

## 4. Requested fixes (text only)

1. Add the baseline mapping in §1 to the package, with citations: polarization identity / XOR; single-index link function (A3); interchange intervention (A4); Q/K-composition and copy suppression (B); gated multiplicative interaction (C); Méloux 2025 (A2, C3); path patching and self-repair (PB9).
2. In A3, state that `q = 2·E_oo[h(x+y)]`: the designed circuit is the odd-odd projection of the sum-only reader.
3. Qualify "attention scores and patterns are identifiable" as "modulo the listed `G` only". Cite Brunner et al. 2020. Note the consequence for B's relation `C = x̂ŷ`.
4. Note that ID1 is non-generic (it needs identical heads).
