# Proof N2 — Joint versus individual sufficiency

Classification: `PROVED_UNDER_RESTRICTED_ASSUMPTIONS`.

**Definition.** `P: H → Z` is value-sufficient for `L` on a set `S` if `P(h_1)=P(h_2)` implies `L(h_1)=L(h_2)` for all `h_1,h_2 ∈ S`. For linear `P`, `P(h+δ)=P(h)` iff `δ ∈ ker P`. Intervention-sufficiency for a family `Δ` is then `ker P ∩ Δ ⊆ ker L`.

If `Δ` is the Euclidean rowspace of a linear `P`, then `ker P ∩ Δ = {0}`, so every such `P` is intervention-sufficient for rowspace edits, for every `L`. That is a tautology and is not used as a theorem. The contentful split is: value-sufficiency on a set `S` that is not closed under kernel translations does not imply invariance of `L` under kernel edits. Direction 1 is that pattern empirically; it is not re-proved here.

**Theorem.** There exist linear maps `P_1`, `P_2` and a linear `L` such that neither map is value-sufficient for `L` on `ℝ^2`, while `(P_1,P_2)` is.

**Proof.** `L(h)=h_0+h_1`, `P_1(h)=h_0`, `P_2(h)=h_1`. Then `P_1(1,4)=P_1(1,-3)` and the logits differ, so `P_1` is not sufficient. `P_2(1,4)=P_2(2,4)` and the logits differ, so `P_2` is not sufficient. The joint map is the identity, hence sufficient. QED.

**What the joint object is.** The pair `(P_1 h, P_2 h)`, not a scalar relating them. The residual `P_1 h - P_2 h` discards `P_1 h + P_2 h`, which in this example is `L` itself.

**Identification.** Declaring `P_1` and `P_2` sufficient because `L` was used to choose them is circular. An independent test is interchange or kernel intervention that does not see evaluation `ΔL` during the choice of `P`. That test is causal abstraction (Geiger et al., 2025). This theorem does not replace it.

**Transformer content.** Direction 1 is an empirical kernel-edit pattern, not a proof in this directory. Joint sufficiency of complementary coordinates is the identity map. A residual relating the two parts is not required for that sufficiency.
