# Candidate H2: causal-sufficiency consistency

**Question.** Can several partial representations satisfy a relation that constrains the preservation of candidate-relevant causal information?

**Definition.** A representation `Z = q(state)` is causally sufficient for `R` on a family `𝒟` iff `R` is constant on the fibers of `(Z, intervention)` in `𝒟` (Geiger et al. 2025, fiber form; 1C Q6). Partial representations `Z_1, …, Z_k` are jointly sufficient iff the tuple is.

**Result 1 (relation adds no information).** For any relation `r(Z_1, …, Z_k)`, the fibers of the tuple refine the fibers of `r`. If `r` determines `R` on `𝒟`, so does the tuple. The converse fails. Consistency is never more informative than the joint values (1C Q9; data-processing form).

**Result 2 (at a single cut, only the sum matters).** If the `Z_i` are the writes of same-cut components, then by SS the decision is a function of their sum. Any relation not determined by the sum is causally inert on sum fibers. Jointly sufficient but individually insufficient writers combine by addition, and "consistency" among them is not a mechanism.

**Result 3 (relation can matter only through a separate read).** A relation among partial representations has a causal role only where some operation reads them separately and nonlinearly: attention across positions, LN across features, or an MLP. That role is then the H1 or H3 object, not an independent sufficiency mechanism.

**Same-statistic, opposite-response counterexample.** SS fiber: identical cosine between two heads, different sums, different margins (`test_SS_same_relation_different_sum_changes_decision`). 1C-R Q14: same projection, opposite responses.

**Does an independently selected relation exclude these?** No. Any relation that is not the sum is excluded from causal efficacy at a cut by SS.

**Comparison.** Approximate abstraction and interchange intervention accuracy (Geiger et al. 2025). The fiber condition is theirs. Nothing distinct remains.

**Identifiability.** The sum is canonical. Partial representations defined by learned subspaces are not (Sutter et al. 2025).

**What it predicts.** Response stability on sum fibers (exact invariance). That is a prediction about the architecture, not about consistency.

**Status.** `REDUNDANT_WITH_KNOWN_THEORY` as a mechanism. Results 1 and 2: `PROVED`. Candidate `REJECTED` as a distinct COR mechanism.
