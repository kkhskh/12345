# Source relations and generative models (mandate B and E)

"Consistency" has been used for several different mathematical objects. This document separates them.

**Coverage key.**

| label | coverage |
|---|---|
| **[X]** | exact calculation in this package |
| **[L]** | learned-model evidence (this or an earlier package) |
| **[A]** | analytic statement only |
| **[—]** | not examined |

## 1. The relation families, defined independently of model logits

Notation: s and t are sources and x_s, x_t their reports in one question. All definitions use inputs and the generator only.

| relation | definition | semantics it needs | what it keeps | what it discards |
|---|---|---|---|---|
| R1 statistical agreement | 1[x_s = x_t] within a question | none | equality pattern | values, source identity across groups |
| R2 candidate-relative agreement | #{s : x_s = c} for each candidate c (vote count) | candidate labels | support per candidate | which sources agree with which |
| R3 reliability evidence | P(role(s) \| X) under a stated generator | the generator's role model | credibility of each source's process | anything the generator calls irrelevant |
| R4 constraint compatibility | 1[(x_s, x_t) ∈ 𝒞] for a stated constraint 𝒞, e.g. x_s + x_t ≡ y (mod K) | the constraint | joint satisfiability | everything outside 𝒞 |
| R5 redundancy / dependence | the evidence that s and t share a common origin, e.g. that they agree on *false* values (copy detection) | an error model, plus a notion of "false" | dependence structure | — |
| R6 contradiction | 1[x_s ≠ x_t] given that both are claimed reliable | a reliability claim | conflict to be reconciled | — |
| R7 semantic and causal compatibility | agreement with a specified model of meaning or intervention | an external semantic or causal model | — | — |

**Key distinction.** R1 is a deterministic function of the input. R3 and R5 are *inferences* that use R1-type statistics, and their sign and meaning depend on the generator.

In the latent-reliability family:
- the sufficient statistic for R3 and R5 is the across-question **count** of R1 per group (Theorem S, extended);
- R2 alone is insufficient (bottleneck I3, 0.636 against 0.743).

## 2. Relation by relation

### R1 → R3 (agreement as reliability evidence) [X][L]

**When it matters.** It matters when roles differ in agreement probability and are shared across questions. Theorem U gives the shape of the response, and the phase diagram gives where it reverses.

**When it is irrelevant.**
- **Reliability is known (T0).** The Bayes value of cross-question agreement is zero. The learned T0 model shows no cross-question dependence: 8% of argmaxes change under cond, at near-ties only, and no tag forms.
- **All sources are exchangeable (T5).** The same holds: 6% of argmaxes change, and the tag content R² is 0.00.
- **Reliability is per question rather than per episode.** Agreement elsewhere carries no information about q. [A]: by independence, π_{−q} = prior.

**When its interpretation reverses.**
- **Collusion or copying.** a_coll > a_rel produces the non-monotone response (§3 of the theorems document). It reverses at full agreement in a region of (ρ, p_c): see the phase diagram. It needs a_coll > a_rel (Corollary U2), and it is not a simple function of p_c·ρ (review R1-l).
- **Agreement rates coincide.** When a_rel = a_coll, agreement **cannot** identify which group to trust (I2). Its decision value collapses to averaging both groups: Bayes 0.566 vs own-question 0.557.

**Can a Transformer respond to it?** Yes [L]:
- T1 and T4 models learn it;
- PP1 confirms that the response direction matches Bayes in all implementations.

**How it differs from an ordinary nonlinear feature.**
- *Extensionally* it is an ordinary nonlinear function of the reports: the tag is explained by a flexible count lookup, with R² 0.76–0.95.
- What makes it a *relation* is its **invariance profile**:
  - invariant to aggregate-changing, relation-preserving edits (G3, 3.6–5.8% change);
  - sensitive to relation-changing, aggregate-preserving edits (G5, 27–35% change).
- A histogram-based nonlinear feature has the opposite profile.

### R5 (redundancy / copying) [X][L]

**Prior work.** Bayesian copy detection in truth discovery (Dong, Berti-Equille and Srivastava, PVLDB 2009) already uses the key fact that *agreement can be evidence of dependence rather than of truth*, and infers dependence and truth together. T4 is a special case:
- colluders copy a common value with probability ρ, independent of the truth;
- because Y is uniform and never observed, "agreement on a false value" is not observable per question;
- dependence is detected only through *excess* agreement across questions, so only through counts.

**Structure added here.** Theorem U (unimodality) and the phase boundary (p_c, ρ). We have not found these stated in that literature, but we have not done an exhaustive search.

**Correlated honest errors (κ).** These raise a_rel. When a_rel reaches a_coll the roles are exchangeable (I2). In that regime "redundant honest sources" and "colluders" are the *same* statistical object, so the distinction requires source semantics beyond the reports.

### R2 (candidate-relative agreement) [X][L]

- **Bayes.** R2 is sufficient for T5 (exchangeable sources) and insufficient for T1 and T4.
- **Learned models.** `mlp_hist` (R2 only) reaches its ceiling (0.638 vs 0.636). It is the canonical *aggregate-only* computation, and its invariance profile is the opposite of R3's.

### R4 (constraint compatibility) [L, earlier package; X analytic]

**Setting.** In REL-XOR3 (`../step1_learned_mediator_identification`), the candidate is a deterministic function of a relation among sources: the relation **is** the target computation.

**Learned evidence.** The learned models encoded the target itself, or per-source content, and **no intermediate relation variable distinct from the target** was found.

**Analytic reason [A].**
- When the relation fully determines the candidate in the *same* question, any representation of the relation is, after one step, a representation of the target.
- There is no reuse across questions to favour a separate intermediate.

**Contrast with R3.**
- **R3 (latent reliability):** the same relational summary serves 6 questions (Theorem M), and a separable tag was found in most seeds.
- **R4:** the relation is consumed once.

The hypothesis that **reuse across decisions drives explicit relation intermediates** fits both packages. It has **not been tested by a controlled manipulation**; see the open problems, OP-3.

### R6 (contradiction) [A]

- In the reliability family, contradiction (a disagreeing pair) is simply the complement of R1.
- A contradiction that must be *reconciled* rather than down-weighted needs a model in which both sources are claimed reliable but measure different things, for example multi-hop reconciliation.
- **Not constructed. [—]**

### R7 (semantic and causal compatibility) [—]

Not examined. It needs source semantics beyond categorical reports.

## 3. Error-generating processes (mandate E): what is informative

| process | optimal rule (or reference) | informative relation | simple agreement sufficient? | learned evidence |
|---|---|---|---|---|
| independent noise + one reliable group (T1) | exact Bayes; log-linear weights in counts | R1 counts | yes: monotone weighting by counts is near-optimal (0.736 vs 0.743, latent-reliability package) | T1 Transformers at Bayes; tag and routing |
| fixed collusion, ρ = 1 (T4) | exact Bayes; colluder constraint | R1 counts, non-monotone | **no**: "most-coherent group" scores 0.359 | Bayes-level only under curriculum; PP1 non-monotone response confirmed |
| partial collusion ρ < 1 | exact Bayes (family) | R1 counts, interior-peak response | no | T4 models were *not* trained on it. Under shift they mostly do not follow either Bayes or the training plug-in exactly (PP2) |
| low collusion prevalence p_c | exact Bayes | R1 counts, drop at full agreement when ρ is high | no | **not trained** |
| correlated honest errors κ | exact Bayes | R1 counts (a_rel rises) | yes, unless a_rel ≈ a_coll | **not trained** |
| honest-copy vs collusion, equal rates | Bayes ≈ own-question | **none distinguishes the roles** (I2) | — | no estimator can succeed |
| unknown pairing | exact Bayes over the 15 pairings | R1 over *all* source pairs; the pairing itself | — | from scratch: fails (PP4, 0.640 < own 0.654) |
| biased sources (non-uniform errors) | breaks (U); values matter | R1 plus values | **no**: counts not sufficient | not examined [—] |
| adaptive / adversarial collusion | not derived | — | — | not examined [—] |
| reliability drift within an episode | π_{−q} → prior as drift → ∞ | weighted R1 (recency) | — | not examined [—] |

**Natural versus off-support inputs.** Every row above uses its generator's support. Off-support stress tests are reported separately:
- PP1: 31% of edits;
- G4 perm hybrids: 98% off support in T4.

## 4. What is left of "consistency" after this analysis

**Mathematical role.** Within-group agreement counts are the Bayes-sufficient statistic for source roles, whenever error models are symmetric (assumption U).

**Decision consequence.** The consequence is set by the generator's role semantics:
- positive under independent noise;
- non-monotone under copying or collusion;
- null when roles are agreement-indistinguishable (I2), or when the latent is not shared.

**No generator-free sign.** There is no generator-free sign or ranking theorem (D2). Any "consistency theory" with predictive content must therefore carry the source semantics. Once it does, it coincides with Bayesian latent-role inference (H1) **at the behavioural level**.

**Where H1 is incomplete.** The behavioural level does not determine implementation (E3–E5 in the equivalence document). Off the support, learned implementations diverge from the Bayes plug-in in organised ways (PP2: global versus local), which no Bayes-level statement predicts.
