# Cross-family mathematical structure: T4 and GATE

**Post-review version.** Corrections from mathematical review A (`reviews/REVIEW_A_MATHEMATICS.md`) are marked in place; the adjudication is in `reviews/ADJUDICATION.md`.

**Evidence labels.**
- **[proof]** mathematical proof under the stated assumptions;
- **[num]** exact enumeration or numerical verification (script and log named);
- **[spec]** conjecture or interpretation.

Nothing here is claimed for "AI systems" in general. Every statement is about these two generators, under the stated symmetry assumptions.

## 1. The two environments, from the generator code

### T4

- **Generator.** `step1_relational_computation_atlas/src/gen_family.py`, with `Fam(pc=1, rho=1)`. The native line uses `NV.ENV = Env(T4, K=3, Q=6, alpha=0.8)` with the exact posterior `genlib.posterior_closed`.
- **Episode.** Q = 6 questions with answers Y_q ~ U{0,1,2}, independent. There are six sources in three fixed pairs G0 = {0,1}, G1 = {2,3}, G2 = {4,5}.
- **Configuration.** A latent configuration Z = (g, h) is drawn uniformly from the 6 ordered pairs of distinct groups, and is **shared by all questions**.
- **Roles.** Pair g is reliable: each source equals Y_q with probability α, otherwise it is uniform over the other K − 1 values. Pair h colludes: both sources equal a common c_q ~ U(K) that is independent of Y_q (ρ = 1, exact copying). The remaining pair is noise, with independent uniform values.
- **Decision.** P(Y_q | X) for every q.
- **Sufficient statistics.**
  - Within-pair agreement indicators a_{q,G} = [x_{q,G,1} = x_{q,G,2}] across all questions are sufficient for Z: Theorem S, `step1_latent_reliability`, and ledger M1.
  - The answer for q needs P(Z | X) together with the own-question reports x_q.

### GATE

- **Generator.** `step1_foundational_atlas_continuation/src/tasks_ext.py::sample_gate`.
- **Questions.** Q = 6 independent questions; nothing is shared across questions.
- **Per question.** Y ~ U{0,1,2}. The gate g ~ Bernoulli(½). x₀ ~ U(K); x₁ = −x₀ (mod 3) if g = 1, otherwise uniform over the two other values. If g = 1, sources 2 and 3 report Y with probability β = 0.9 (otherwise uniform over the other values) and sources 4 and 5 are uniform. If g = 0 the two pairs swap roles.
- **Exact posterior.** P(y | x) ∝ ∏_{s∈S(g)} (β if x_s = y else e), with S(1) = {2,3}, S(0) = {4,5}, e = (1 − β)/2, and g = [x₀ + x₁ ≡ 0 (mod 3)] is a deterministic function of (x₀, x₁).
- **Sufficient statistic per question.** (g, x_{S(g)}).

### The relational objects

| | T4 | GATE |
|---|---|---|
| sources that carry answer evidence | all six | 2–5 |
| latent trust variable Z | configuration (g, h), episode-level, **uncertain** given X | gate g, question-level, **deterministic** given (x₀, x₁) |
| relation that identifies Z | within-pair equality, aggregated over Q questions | modular sum of two non-answer sources |
| a second relational cue | none | within-pair agreement of the evidence pairs (informative marginally, screened off by g) |
| when information is lost | a single disagreement removes the colluder hypothesis for that pair (exact copying); with ρ < 1 this softens | never within a question: g is always identified |

## 2. Relational invisibility [proof] (both environments)

**Assumption A1.** For every source s, the distribution of its report(s) does not depend on the trust latent Z.

**Proposition 1 (A1 holds in both generators).**

*T4.* Take a single source s and its sequence (x_{1,s}, …, x_{Q,s}). Questions are independent given Z, so it suffices to show that each x_{q,s} is U(K) under every role.
- Reliable: P(x = v) = (1/K)·α + (1 − 1/K)·(1 − α)/(K − 1) = 1/K.
- Colluder: x = c_q ~ U(K).
- Noise: U(K).

So the law of the whole sequence is i.i.d. uniform whatever Z is. ∎

*GATE.* x₀ ~ U(K) for both values of g. For x₁, if g = 1 then x₁ = −x₀ is uniform because x₀ is; if g = 0 then x₁ is uniform over the two values other than −x₀, and averaging over x₀ gives U(K). For an evidence source, the reliable law is uniform by the same computation as in T4, and the noise law is uniform. ∎

The exact mutual information confirms this (`logs/g_rules_exact.json`):
- I(g; x_s) = 0 for every s;
- I(g; (x₀, x₁)) = ln 2 = H(g);
- I(g; [x₂ = x₃]) = 0.124 nats;
- I(g; ([x₂ = x₃], [x₄ = x₅])) = 0.220 nats.

**Corollary 1 (source-wise trust stays at the prior).** Suppose a trust inference combines per-source likelihoods, P̃(Z | x) ∝ P(Z) ∏_s P(x_s | Z). Under A1 every factor is constant, so P̃(Z | x) = P(Z). The decision rule built on it is the "no-relation" rule. The best decision that uses only single-source evidence is the source-wise rule. Its exact values:

| | Bayes | best order-1 (source-wise) | source-wise naive Bayes | gate-free mixture (pairwise) |
|---|---|---|---|---|
| GATE accuracy | 0.900 | 0.801 | 0.801 | 0.801 |
| GATE NLL | 0.182 | 0.462 | 0.489 | 0.430 |
| T4 (native) NLL | 0.541 | 0.872 (own question) / 0.872 (all 36 sources) | — | — |
| T4 agreement with Bayes argmax | 1 | 0.61 | — | — |

Sources: [num] `logs/g_rules_exact.json`, `logs/g_order_ceilings.json` (exact), and `logs/t4_order1.json` (logistic fit on 120k sampled questions, evaluated on 30k).
- In T4, adding the 30 other-question sources at order 1 changes nothing (0.8721 against 0.8724), as Proposition 1 predicts: those sources carry the trust information only jointly.
- **Scope of A1.** A1 is a symmetry of these generators, not a law. It fails, and source-wise summaries become informative, if roles change single-source marginals. Two examples are biased noise and unequal answer priors (§7, counterexample X3).

## 3. Interaction order of the decision [proof] [num]

**Definition.** A rule has interaction order ≤ k if its logits are logit_y(x) = Σ_{A ⊆ sources, |A| ≤ k} w_A(x_A, y). This is a log-linear (ANOVA-type) expansion with arbitrary tables w_A.

**Proposition 2 (GATE's Bayes rule has order exactly 3).**
- log P(y | x) = g(x₀, x₁)·ℓ₂₃(y) + (1 − g(x₀, x₁))·ℓ₄₅(y) − log Z(x), where ℓ_S(y) = Σ_{s∈S} log(β·[x_s = y] + e·[x_s ≠ y]).
- The normaliser does not depend on y. Each product g·log(…x_s…) is a function of (x₀, x₁, x_s). So the order is ≤ 3.
- The order is not ≤ 2, by Proposition 3: the Bayes rule has gate response 1, and every order-≤ 2 optimum has gate response 0. ∎

Numerically, the expected-NLL-optimal order-3 rule reaches KL 8·10⁻⁹ (`logs/g_order_ceilings.json`).

**Proposition 3 (every expected-NLL-optimal rule of order ≤ 2 ignores the gate).**

Let Q*(y | x_E) be the optimal order-≤ k rule that uses only the evidence sources, for k ∈ {1, 2}. Extend it to all sources by putting zero weight on every term that contains x₀ or x₁.

The expected NLL is convex in the weights, so stationarity is enough. Stationarity means moment matching: E_data[f] = E_{Q*}[f] for every indicator feature f(x_A, y) with |A| ≤ k.

1. **Features without x₀ or x₁.** These are matched by the definition of Q*.
2. **Features (x₀ = a, y) and (x₀ = a, x_s = c, y).** x₀ is independent of (g, Y, x_E). This follows from Proposition 1 and the generator: x₀ ~ U(K) independently of g, and (Y, x_E) depend only on g. Both moments therefore factor as (1/3)·[the corresponding evidence-only moment], and those are matched.
3. **The same features for x₁.** Marginally over x₀, x₁ is also independent of (g, Y, x_E): P(x₁, g, Y, x_E) = P(g)·P(Y, x_E | g)·Σ_{x₀} P(x₀)·P(x₁ | x₀, g) = P(g)·P(Y, x_E | g)/3. The same factorisation applies.
4. **Feature (x₀ = a, x₁ = b, y).**
   - The data moment is P(x₀ = a, x₁ = b)/3, because Y is independent of (x₀, x₁).
   - The model moment is P(x₀₁ = ab)·E[Q*(y | x_E) | g(a, b)].
   - The law of (Y, x_E) given g is invariant under a common relabelling π of the candidates applied to Y and to the evidence values only. The gate is *not* invariant under relabelling (x₀, x₁), so the relabelling must not act on them.
   - The expected NLL is strictly convex in the centred logits on the support, and the order-≤ k class is a linear space of logit functions. The optimal centred logit function is therefore **unique**, hence equivariant under π. So E[Q*(y | x_E) | g] = 1/3.

Hence Q* is *the* optimum among all order-≤ k rules: it is unique as a predictive rule, though not necessarily as a weight vector. It does not depend on (x₀, x₁), so its gate-flip response is exactly 0. ∎

**Stronger statement (review A, F6).** By exact enumeration, *every* order-≤ 2 rule has gate-flip slope G exactly 0, not only the optimum: the Bayes change under a gate flip is orthogonal to every change an order-≤ 2 rule can make. The uniqueness step and the correct scope of the relabelling were added after review A.

Exact enumeration agrees:
- order 1: gate slope −2·10⁻¹⁸;
- order 2: gate slope −1·10⁻¹⁸;
- order 3: gate slope 0.99994.

**Corollary 3 (the swap test).**
- The order-1 optimum is exactly invariant to swapping x₃ ↔ x₅.
  - The induced law of (x_E, Y) is invariant under the within-pair swaps (2 3) and (4 5) and under the pair exchange (2 4)(3 5); the exchange is paired with g ↔ 1 − g, which the order-1 optimum ignores by Proposition 3.
  - The optimum is unique, so its four evidence tables are equal.
  - The rule is therefore symmetric in the four evidence sources, and the swap preserves the multiset of evidence values. (The earlier reason, "identical marginal laws", was insufficient; review A, F7.)
- The order-2 optimum, which is essentially the gate-free mixture (NLL 0.4314 against 0.4305), is not invariant. It uses within-pair agreement, and the swap changes the pairing.

Corollary 3 and Proposition 3 together separate three stages by two observable responses:

| stage | gate-flip slope G | swap-agreement index A (frozen) | partial index A_p (amendment) |
|---|---|---|---|
| order 1 | 0 | 0 | 0 |
| order 2 | 0 | ≈ 0.96 | 0.96 |
| gate-free rule | 0 | 1 | 1 |
| Bayes | 1 | **0.71** | 0 |

These are the measures of the frozen prediction X1 (CROSS_FAMILY_CONFIRMATION.md).

**Correction after review A (F5).** A responds to gate use as well as to pairwise-agreement use: Bayes has A = 0.71, and a model moving from order 1 towards Bayes has A ≈ 0.71·G. The partial index A_p, registered as an amendment before any X1 output was inspected, separates the two.

**Proposition 4 (T4: where the high order comes from; corrected after review A, F2).** [proof]
- **Trust evidence is low order in the log domain.** log P(Z | X) = Σ_q log P(x_q | Z) + const, and for q' ≠ q each term depends only on the Q within-pair equality indicators of question q'. Every indicator is an order-2 relation. With exact copying the colluder term is −∞ after one disagreement: a hard constraint, still a sum of per-question order-2 terms (Q relations over 2Q sources).
- **The decision's high order comes from marginalising over an uncertain configuration.** P(y | X) = Σ_Z P(Z | X)·P(y | x_q, Z) is a log-sum-exp mixture, whose log-odds has interactions among all questions' indicators.
- **GATE has the same effect.** The gate-free mixture (marginalising g) has interaction order 4, exceeding the Bayes rule's 3 (review A).
- **What survives.** In GATE the trust variable is identified exactly by one order-2 relation, so no marginalisation is needed and the decision has order 3. In T4 trust is uncertain and must be marginalised.
- The earlier statements that T4's trust is an "order-12 relation" and that "P(y | X) can be zero" were wrong and are withdrawn.
- **Learnability [spec, with counter-evidence].** The earlier link between the order of the trust relation and T4's failure to train from scratch (0/3, M Transformer) is not supported. DeepSets learned T4 to Bayes in 3/3 seeds, and the source attributes the Transformer failure to an optimisation barrier and the non-monotone agreement map (ledger N2, M3).

## 4. Screening-off [proof]

**Proposition 5.** In GATE, for every x, P(y | x) depends on the untrusted pair x_{S(1−g)} not at all: not on its values and not on its agreement.

Within-pair agreement is informative about g marginally (0.124 nats per pair), but it is screened off by the deterministic g.

In T4 there is no deterministic trust relation. Agreement *is* the trust evidence, and its map to trust is non-monotone (perfect agreement indicates the colluder; ledger M3). The T4 heuristic "agreeing sources are trustworthy" is therefore:
- wrong in T4 at perfect agreement;
- weakly valid in GATE (the gate-free mixture);
- superseded in GATE by the gate.

## 5. The combination operator is non-identifiable in GATE [proof]

In T4 the two organisations are *fitted* by different nested power means: ρ ≈ 0.54–0.62 for early readout and 0.87–0.98 for mixture (ledger N20). XIII-B itself notes that this reading is not unique. Whatever its reading, a power-mean family can be fitted non-trivially only because the trust weights w_Z = P(Z | X) are non-degenerate.

**Proposition 6.** For any weights w on {pair 23, pair 45}, write M_ρ(w; L) for the nested power mean of the components (L₂₃, L₄₅). If w is a point mass, M_ρ(w; L) = L_{S(g)} for every ρ. In GATE, w = (g, 1 − g) is always a point mass. So every ρ gives the same rule on every input, and ρ is not identifiable from behaviour. ∎

**Consequence.** For Bayes-optimal behaviour, the power-mean exponent cannot be identified in GATE. A non-optimal model's fitted ρ could still differ; this was not measured. T4's fitted operator split therefore depends on trust uncertainty, which GATE lacks. This is a T4-specific feature (H4). Review A notes that the proposition is near-trivial.

## 6. Shared symmetry, not a common principle (corrected after review A, F3)

**Verdict.** The proposed "common principle" is **rejected as a principle with cross-family predictive content**. What survives:
- (i) a shared symmetry, A1, verified in both generators;
- (ii) an exact diagnostic calculus for GATE: order ceilings and the G, A and A_p reference values;
- (iii) a GATE-specific staircase hypothesis, X1, which rests on an external simplicity-bias assumption.

The pre-review statement is kept below, so the correction is visible.

**The pre-review restricted principle (both environments).**
1. The variable that determines which evidence to trust has no single-source component (A1, Proposition 1).
2. It is identified only by relations among sources. Given A1 this is a restatement.
3. It enters the decision multiplicatively with the evidence, so the decision has interaction order ≥ (order of the trust relation) + 1. This is proved only for GATE, needs a non-degeneracy condition, and is ill-posed for T4 (Proposition 4).

**Non-trivial consequences that hold in both:**
- (a) Exact source-wise ceilings far below Bayes (§2).
- (b) **For sources that carry no answer evidence for the decided question** (GATE gate tokens; T4 sources at non-target questions): an edit that changes a trust relation while preserving every single-source marginal can change the decision, and an edit that changes values but preserves every trust relation cannot. For answer-bearing sources the second half is false, because their values are votes (review A, F8).
  - GATE: (x₀, x₁) → (x₀ + k, x₁ − k) leaves Bayes exactly invariant.
  - T4: any relabelling of the values within a pair at a non-target question, preserving within-pair equality, leaves Bayes for q exactly invariant. For q' ≠ q, P(x_{q'} | Z) depends only on the equality indicators.
- (c) **[spec]** Under a simplicity-bias account of gradient training, learning should pass through order-1 rules before relational ones. This is tested prospectively in GATE as X1. Ledger N12 (Investigation VIII) is **not** a T4 counterpart: its readers received per-group agreement counts, which are already relational at source level (review A, F3).

**What the principle does not say.** It does not fix a neural implementation. It does not imply that the trust variable is stored at source tokens, as it is in T4 mixture models, or at the readout. It does not predict the combination operator (§5).

## 7. Limits and counterexamples [proof by construction]

- **X1, the source-wise-sufficient GATE variant.** Let g = [x₀ = 0], so the gate depends on one source. A1 fails, the trust variable is order 1, the Bayes decision has order 2, and source-wise summaries suffice for trust. The principle is therefore not a property of "gated" tasks as such. It depends on the gate being an invisible relation.
- **X2, observationally equivalent relational descriptions.**
  - GATE: g = [x₀ + x₁ ≡ 0] is the same relation as [x₁ = −x₀], an equality after recoding. No input distinguishes the "sum" description from the "recoded equality" description, so claims about which one a model "computes" are not identifiable behaviourally.
  - T4: pooled S against per-group m (ledger M6).
- **X3, A1 failure in a T4-like generator.** Let the noise pair's values be non-uniform, for example drawn from (0.5, 0.25, 0.25). Then a single noise source's marginal reveals its role, and order-1 rules acquire trust information. The continuation's "biased errors" family is **not** an instance: its errors go to (y + 1) mod K, which keeps A1 (review A, F8; corrected).
- **X4, a previously sufficient representation becomes insufficient.**
  - T4: per-group statistics are sufficient for C = 1 but insufficient for C = 2 with shared copying (ledger M4).
  - GATE: (g, evidence) is sufficient. If the trusted pair were defined by a 3-way gate over three pairs, the binary g would become insufficient, and a representation storing one bit per evidence token would fail.
- **X5, the principle fails as a claim about implementations.** The three original GATE models differ in where the gate is computed (layer 0 or 1) and in where the decision is formed (layer-1 or layer-2 readout attention). T4 models differ again (ledger N19). A shared mathematical principle does not imply a shared circuit.
