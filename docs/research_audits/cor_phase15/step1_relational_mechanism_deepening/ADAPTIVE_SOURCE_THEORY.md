# Adaptive and changing sources (Workstream F)

First investigation of non-stationary source relations in this programme. Exact references are by enumeration; learned
models are the 3-layer Transformer. Code: `src/tasks3.py` (ADAPT), `scripts/adapt_theory.py`, `eval_adapt.py`,
`adapt_state_probe.py`, `adapt_curriculum*.py`. Logs: `logs/adapt/`.

## 1. Generative definitions

**ADAPT (dynamic reliability, abrupt change).** Three source pairs, Q = 6 questions, K = 3. One pair is reliable (accuracy
0.8 per source) for questions q < c and another pair for q ≥ c; the remaining pairs are noise. Prior: no change (c = 6) w.p.
1/3; otherwise c uniform on 1–5, the first reliable group uniform, the second uniform on the other two. Exact posterior by
enumeration of the 33 configurations (g₁, g₂, c), leave-q-out as in all packages.
**Static reference.** The T1 Bayes rule (one reliable group for the whole episode), which is count-sufficient (atlas S).
**Triggered copier.** Roles (reliable g, copier h, noise); in each question the copier pair copies one common uniform value
w.p. ρ_t = 0.9 **only when the reliable pair disagrees**, otherwise reports independently. **Static copier** with the same
marginal agreement rate: copies w.p. ρ_s = 0.306 in every question.
**Camouflaged copier.** A static copier whose copy rate makes its agreement rate equal to the honest pair's (ρ_c = 0.49).

## 2. Exact results

| result | value | type |
|---|---|---|
| ADAPT Bayes accuracy (all / changed / unchanged episodes) | 0.706 / 0.692 / 0.735 | [N] |
| static T1 reference (all / changed / unchanged) | 0.690 / 0.660 / 0.748 | [N] |
| count insufficiency: episodes with equal per-group counts and own reports, posterior spread | max 0.35, median 0.04 (2,536 classes) | [P, N] (THEOREMS AD1) |
| posterior mass on the currently reliable group vs distance q − c | 0.50 (far before) → **0.36** (q = c−1, c) → 0.45 → 0.50–0.52 (≥ 2 after) | [N] (AD4: decision lag around the change) |
| camouflaged copier vs honest pair, per-question law | max \|Δ\| = 2·10⁻¹⁷ | [P, N] (AD2: non-identifiable) |
| triggered vs static copier: mean P(triggered) when true / when false | 0.66 / 0.34 | [N] (AD3) |
| accuracy cost of assuming the wrong copier mechanism | ≤ 0.007 | [N] |

Decision boundaries: (i) under a change point, the optimal decision uses **where** a pair agrees, not how often — static
count summaries lose information (on changed episodes the static rule loses 0.03); (ii) the static rule is *better* on
unchanged episodes (0.748 vs 0.735) because it does not pay for the change prior — a real trade-off, not a free gain;
(iii) a copier that matches honest statistics question by question is observationally identical to an honest pair (no
estimator, hence no network, can detect it); (iv) a copier whose behaviour depends on the honest sources is detectable only
through cross-group co-occurrence, and only weakly with six questions.

## 3. Learned models

| model | accuracy all / changed / unchanged | Q3-E1 | Q3-E2 (follows exact Bayes in position-decided count classes) | history intervention: follows new exact answer (change rate) |
|---|---|---|---|---|
| `adapt_s0`, from scratch, 6000 steps (frozen Q3-E) | 0.657 / 0.656 / 0.660 | **fail** | **0.49 — fail** | 0.48 (0.12) |
| `adapt_fromT1_s0`, 3000 steps from T1 s0 (**exploratory**, post hoc) | 0.706 / 0.691 / 0.736 | pass | 0.88 | 0.88 (0.79) |
| `adapt_fromT1p1_s1`, 3000 steps from T1 s1 (**frozen Q3-E′**, confirmation) | 0.706 / 0.691 / 0.736 | **pass** | **0.88 — pass** | **0.88 — pass** (0.78) |

- From scratch the model stays near the own-question level (0.657), below even the static rule: an optimisation failure
  of the same kind the atlas found for T4 from scratch.
- With a T1 curriculum the model matches exact Bayes on changed and unchanged episodes and uses **positional** evidence:
  permuting the order of the other questions (counts unchanged, so the static rule is unchanged) changes its answer where the
  exact posterior changes, in the predicted direction 88 % of the time — a causal intervention on histories, prospectively
  confirmed on a second parent.
- Latent-state representation (`state_probe_*.json`): the question-specific reliable group is linearly decodable at readout
  token q at 0.62 (exact-posterior MAP ceiling 0.65; static MAP 0.60); after a change 0.58 (ceiling 0.61; static 0.53). The
  from-scratch model: 0.43 after a change. The curriculum model therefore represents a **changing** latent source state,
  close to the statistical limit.

## 4. Mechanistic comparison (F4)

- ADAPT's optimal computation is reliability routing with a **question-indexed** reliability variable; the curriculum
  model inherits T1's machinery and specialises it per question. This is an extension of the T4/T1 class (relation →
  per-source reliability → routing), not a new computational class, on present evidence — but the routing variable's
  localisation was not tested causally (only probes and input-level history interventions).
- It differs from GATE (within-question constraint, no cross-question state) and from TRANS (no map composition).
- Not run: drifting reliability, reliability conditioned on outcomes, learned models of triggered or camouflaged copiers,
  architecture comparison (DeepSets) on ADAPT.
