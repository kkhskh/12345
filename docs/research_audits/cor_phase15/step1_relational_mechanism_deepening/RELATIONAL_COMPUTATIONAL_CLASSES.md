# Relational computational classes: T4, GATE, TRANS, ADAPT (Workstream E)

> **Post-review [R].** RC membership criteria tightened (FM10): interchange condition requires IIA ≥ 0.5 on informative pairs (stated post hoc; T4 seed 12 at 0.47 is therefore not a member); "TRANS: no" is re-graded "not shown" because condition 2 was not tested for one-hop σ mediation and σ/τ are not functions of the reports as condition 1 requires. "Relation variable at source tokens" is a property of answer-supervised curricula, not of the relation (CA10). Superseded pre-review statements are kept in ~~strikethrough~~ next to their
> corrections; details in `reviews/ADJUDICATION_R1.md`.

Purpose: decide, with interventions and counterexamples, what (if anything) the relation families share beyond "the
input–output map depends on source relations". Evidence grades as in THEOREMS_AND_COUNTEREXAMPLES.md.

## 1. A candidate class, made restrictive (E1)

**Definition RC (relation-mediated candidate computation).** A model M on a source generator P is in RC with relation R if:
1. **(independent relation)** R is a function of the reports defined from the source semantics alone (no reference to M or
   to candidate logits): T4 — within-pair agreement across questions; GATE — x₀ + x₁ ≡ 0 (mod K); TRANS — the maps σ, τ;
   ADAPT — question-indexed agreement.
2. **(identifiable mediator)** there is a site and a variable V = v(R(X)) such that an intervention setting V to its
   counterfactual value v(R(X′)) — with X′ an on-support counterfactual that changes R — reproduces M's counterfactual
   decision (interchange sufficiency ≥ a stated level), **and**
3. **(non-aggregate signature)** some on-support input change that leaves every per-candidate aggregate (vote counts per
   candidate) fixed but changes R changes M's decision in the Bayes direction — excluding pure aggregate-only computations;
4. **(margin consequence)** the sign of the margin change under the intervention is fixed by the generator semantics
   (Theorem D for T4; the gate rule for GATE; T1/T2 for TRANS).

## 2. Membership, by family (E2)

| condition | T4 (A′ Transformer seeds) | GATE (3 seeds) | TRANS (full / aux runs) | TRANS (holdout run) | ADAPT (curriculum) |
|---|---|---|---|---|---|
| 1 independent relation | agreement pattern | modular gate | σ, τ | σ, τ | positional agreement |
| 2 mediator, interchange | 2-D source-token tag: 0.77 (s10) to 0.47 (s12); readout count variable in P-count transfer models 0.60–0.72 | gate state at evidence tokens; readout-attention output carries the flip at 1.00 (each seed's decisive layer) | **no**: C-only decisions do not follow map interventions (0.31–0.36 ≈ chance) | inputs: σ/τ interventions followed 0.64 / 0.63; internal mediator not localised | not localised (input-level history intervention only) |
| 3 non-aggregate signature | count-preserving vs count-changing edits (atlas PP1, OD) | gate flip with evidence fixed (aggregates fixed) changes answer: 1.00 | — | holds at input level | order permutation with counts fixed: follows exact 0.88 |
| 4 signed margin consequence | Theorem D; PP1 sign agreement 0.996+ | gate rule; flip follows Bayes 1.00 | T2 (chance) | partially (restricted prior) | exact posterior |
| **in RC?** | **yes** (with seed-dependent mediator) | **yes** (seed-dependent layer) | **no** | input-level only | input-level only |

## 3. What is shared and what is not (E3) — answers to questions 22–26

**Shared by every member found (T4 A′ seeds, GATE seeds 0–2):**
- (S1) the relation-derived variable is computed by **attention at source/evidence tokens** and stored there; (S2) the
  decision is formed by **readout-token attention to those tokens** (T4: pattern-mediated routing — frozen-pattern tag
  ablation 0.02 in seed 10; GATE: the decisive readout head's output carries the flip at 1.00); (S3) readout MLPs never
  carry the decision effect in GATE (≤ 0.07).
  Grade: interventionally supported per family, *but compared structurally*: no intervention can map a T4 variable onto a
  GATE input, so "the same abstract variable" is not testable across families — only "the same computational shape".
**Genuinely different:**
- **Scope of the relation variable**: T4/ADAPT episode- or question-indexed cross-question reliability; GATE strictly
  within-question.
- **Code and site of the transferable variable**: P-count transfer models use an additive one-hot count code at the
  readout (abstract-variable interchange 0.60–0.72 over 5 seeds); T1-transfer models carry ~~the cross-question information at
  source tokens in a non-count code (0.04–0.08 for the same abstraction~~ **[R: CA1, R2]** a count code at the source tokens
  with slot-specific directions (per-slot encoder 0.66–0.82; shared encoder 0.04–0.08), full-site 0.84–0.87 over 3 parents).
- **Layer and redundancy**: GATE's computation layer (0 vs 1), decisive readout layer (1 vs 2), and redundancy (cross-layer
  backup, within-layer pair, none) all vary by seed.
- **Composition**: no learned TRANS model satisfied RC; aux supervision gives represented-but-unused maps — a separation
  between "represented relation" and "relation in the decision" that has no counterpart in T4/GATE models.

**Invariance (Q25).** RC membership and S1–S2 are invariant under exact head splits (pattern-class level, continuation R1)
and under linear reparameterisation of the site (the abstract-variable test refits its encoder). They are **not**
invariant under changes of seed at the level of layers/heads (GATE), nor under architecture change (DeepSets has no readout
attention: T4 DeepSets solves T4 by pooling, so S2 fails — RC-membership must then be shown with a pooled mediator; not
done here).

## 4. Counterexamples limiting unification (E5) — each realised

| attack | realised instance | what it shows |
|---|---|---|
| represented relation not used | TRANS aux run: σ, τ decodable at the statistical ceiling, composite near its ceiling; C-only at chance, unaffected by removing either map's evidence | decodability ≠ mediation; condition 2 is needed, condition 1 alone is vacuous |
| salient attention not the mechanism | GATE seed 0: 94 % gate-dependent routing in L2-h0, ablation harmless (backup) | attention patterns alone cannot define the class |
| same task, different circuits | GATE seeds 0/1/2 (layers, redundancy differ); T4 seeds 10 vs 12 | a class defined by components is not seed-invariant; RC defines it by mediated variables |
| ~~same relation, different codes~~ count variables at different sites and codes [R: CA1, R2] | P-count (readout) vs T1 (source tokens, slot-specific) transfer models, both carrying the additive leave-q count code | ~~a shared abstract variable … fails across curricula~~ a count variable is causally used under both curricula, but its *site* and code differ — site-based class definitions fail |
| prior-fitting instead of computation | TRANS holdout: above full-Bayes on training pairs, below chance on held-out pairs, matching training-prior Bayes | success on a relation family can be inference within a memorised hypothesis class; generalisation tests are required |
| Bayes-closeness explains the behavioural signature | T4 PP1 sign agreement (continuation) | condition 4 alone adds nothing beyond accuracy |
| relation statistic that discards information | ADAPT: static counts are insufficient (spread 0.35) | an aggregate-relation definition (counts) is not the general mediator |

## 5. Conditions for candidate-margin consequences (Q27, Q28)

- A margin consequence follows **at the behavioural level** whenever the model is pointwise close to Bayes on the
  intervention's endpoints (continuation C1 coverage) — no mechanism needed.
- A **directionally predictable internal** consequence (intervening on a mediator predicts the sign of the margin change)
  is established only where condition 2 holds with high interchange: GATE (all seeds, at the decisive head), T4 A′
  (tag/routing), P-count transfer models (readout count code). The direction is fixed by the generator semantics in every
  case (no generator-free sign; continuation D2).
- Where a relation is represented but not mediated (TRANS aux), intervening on it changes outputs (21–51 %) **without** a
  predictable direction (follows Bayes 0.32–0.36 ≈ chance).

## 6. Status

RC is a definition with restrictive conditions and a membership table, not a theorem. The only positive cross-family
statement (S1–S2) is a structural regularity observed in two families and five Transformer models, not a law; it fails for
DeepSets by construction and for TRANS because no member exists.
