# Causes of neural solution selection (Workstream A, and the learning-transition question of §8)

COR Phase 1.5, Step 1 (provisional ≈ 25%). Evidence labels: [dev], [conf] (FROZEN_CONFIRMATION), [expl].

> **Review 3 corrections (accepted; `reviews/ADJUDICATION.md`).**
> - **(M1)** The split cells show only that the layer-2 initialisation does *not* carry the init effect. They cannot separate the readout
>   map from a readout-map × data interaction; see §Post-review.
> - **(M2)** "Data effect ≤ 0.02" holds in development only. In confirmation, data moves the continuous hybrid residue (data 0.06–0.09,
>   interaction 0.11–0.18). The *binary* head event follows the initialisation.
>   **Post-review run (exploratory):** uP163 L21/**W22**/D21 ended early-readout, (0.818, 0.168). That is 0.08 from the WA-seed diagonal
>   s22 and 0.86 from the L2/data-seed diagonal s21. The readout-map initialisation decides here even against both other factors. It is
>   one run in one parent.
> - **(M3)** The uP161 l2q run is under-trained and is excluded from interpretation.
> - **(R2)** The interventions show path dependence and lock-in after release. Active competition in unmasked runs is not shown.



**Statistical units.** Every run here is a *fine-tuning run* of one of three pretrained parents (uP160, uP161, uP163). Within a parent,
runs are paired by construction (identical initialisation tensors or identical data streams). Cells have n = 1, so the factorial effects
below are **descriptive contrasts, not tests**, and no confidence interval over parents is computed. Episode-level sampling error of the
measures is small: route_Q has sd 0.03 across streams (CAUSAL_CONTROLS §D7).

## A1. Every source of randomness in fine-tuning, separated

Audit of XIII-B `native_train.py` (spec full_c1). The fine-tuning seed s enters in three places:
1. the fresh layer-2 block, via `N.init(s + 7777)`;
2. the fresh readout map WA, via the same `N.init(s + 7777)` call;
3. the training stream `default_rng(1000003(s + 1) + 31782)`. This one generator produces both the episodes and their order; there is no
   separate shuffle.

Nothing else in training is random: no dropout or augmentation, AdamW is deterministic, and embeddings, layers 0–1 and the final LN are
inherited. Floating-point order is fixed by single-threaded XLA.

`scripts/ft_factorial.py` takes separate seeds L (layer 2), W (readout) and D (data). Two checks:
- L = W = D = s reproduces XIII-B's seed-s run **bit for bit over the first 50 steps** (`logs/repro_uP161_22_50.json`, 44 leaves, max diff 0). The full 4000-step replay was descoped; the diagonal cells are XIII-B runs from the original script.
- After a container restart killed the first runs at 07:23Z, the trainer was made resumable, and a killed-and-resumed run is bitwise
  identical to an uninterrupted one (`logs/repro_resume.json`).

## A2. Factorial design

| role | parent | cells (L = W, D) | diagonal references (XIII/XIII-B seeds) |
|---|---|---|---|
| development | uP161 | (21, 22), (22, 21) | s21 mixture, s22 early |
| development | uP160 | (21, 23), (23, 21) | s21 early, s23 mixture |
| confirmation R1 | uP161 | (20, 23), (23, 20) | s20 mixture, s23 hybrid |
| confirmation R1 | uP163 | (21, 22), (22, 21) | s21 hybrid, s22 early |
| confirmation R1b | uP163 | L22/W21/D21, L21/W22/D22 | L2-init vs WA-init split |

Diagonal cells are the existing seed runs, valid because of the bitwise reproduction. Each parent therefore has a complete 2 × 2:
- 4 new development runs plus 4 diagonals;
- 4 new confirmation runs plus 4 diagonals;
- 2 split cells.

The development diagonals were chosen because they differ in class. That is outcome-based selection of *which* contrasts to run, and it is
disclosed; it does not bias the direction of the result.

## A2 results (development)

v = (lik_RO1, mix_mass) at step 4000, measured on the fixed stream 86_100_000.

| parent | cell | v | v of init diagonal | v of data diagonal | route_Q | ρ (MSE) |
|---|---|---|---|---|---|---|
| uP161 | L21/D22 | (0.007, 0.965) | s21 (0.006, 0.966) | s22 (0.78, 0.20) | 0.37 (s21 0.39) | 0.85 (s21 0.87) |
| uP161 | L22/D21 | (0.779, 0.215) | s22 (0.78, 0.20) | s21 | 0.08 (s22 0.22*) | 0.53 (s22 0.56) |
| uP160 | L21/D23 | (0.806, 0.167) | s21 (0.81, 0.17) | s23 (0.03, 0.97) | 0.09 | 0.62 (s21 0.59) |
| uP160 | L23/D21 | (0.025, 0.968) | s23 (0.03, 0.97) | s21 | 0.78 | 0.91 |

\* route_Q is ill-defined when the relational transfer is mostly outside route Q (early units).

**Effect sizes**, from `scripts/analyze_A.py` on the development 2 × 2s:
- the init main effect on lik_RO1 is ±0.77–0.78 in both parents;
- the data main effect and the interaction are ≤ 0.02.

Even the continuous internal quantities that XIII-B thought were "seed-dependent routing proportions" follow the initialisation:
- route_Q 0.37 vs 0.39;
- the mixture-head share;
- ρ (0.85 vs 0.87).

**Initialisation × parent, not the seed value.** Seed 21's initialisation produces a mixture model in uP161 and an early-readout model in
uP160. What decides is how a particular fresh layer 2 (and readout) meets a particular parent's layer-0/1 features.

**No step-0 predictor** [dev, 17 known fine-tunes]. Correlations with final lik_RO1:

| step-0 feature | r |
|---|---|
| max readout→q attention mass of the fresh heads | −0.41 |
| layer-2 / layer-0 gradient ratio | −0.34 |
| first-order mass change | −0.23 |

The outcome is determined by the initialisation, but it is not readable from simple statistics of it.

## A3. Emergence: competing pathways

These trajectories come from development runs only. XIII-B seed units were measured at 7 checkpoints; the factorial and intervention cells
at 14 (`logs/unit_metrics_traj_*`).

**What every run shows:**
1. **Every run starts both pathways.** Early readout grows in all runs (lik_RO1 0.12–0.39 by steps 111–246).
2. **Mixture-destined runs form a q-mass head early.** mix_mass is 0.77–0.87 at step 246 (uP160 s23: 0.35 → 0.89 between 246 and 546).
   Their early readout then **declines to ≈ 0** (uP161 s21: 0.16 → 0.06 → 0.01): suppression, not mere coexistence.
3. **Early-destined runs never form the head.** mix ≤ 0.25 at every checkpoint, while lik_RO1 rises to 0.78–0.81.
4. **The hybrid** (uP161 s23) forms the head late (0.47 at 546, 0.89 at 1211), after lik_RO1 had reached about 0.49. The two then
   **coexist** (final 0.42 / 0.92).
5. **The behavioural transition** to the correct rule (slope_e ≥ 0.5) occurs at steps 546–1211 in every class. It is not a class signature.

**Causal test of the competition (development).** ro0 masks layer-0 readout→source attention for the first 600 steps, applied to the
early-destined uP161 L22/W22/D22:
- a mixture head forms by step 246 (mix 0.69) and 546 (0.91);
- after release, early readout never returns: lik_RO1 −0.001 at step 4000 against 0.78 unintervened;
- ρ_MSE is 0.87 against 0.56;
- the run is a mixture model.

The organisation is therefore **not fixed by the initial weights as such**. It is decided by which pathway wins an early race, and the
initialisation biases that race. [conf R2] tests the converse: suppressing mixture-head formation (l2q) should leave early readout in
place.

## A4. Competing hypotheses

| hypothesis | evidence |
|---|---|
| H-A1 fresh initialisation determines | init effect 0.33–0.79, data 0.00–0.09 [dev + conf R1a]. **Within the initialisation, the readout-map init and not the layer-2 init** [conf R1b, falsifying the a-priori L2 prediction] |
| H-A2 data order determines | rejected in development (data effect ≤ 0.02); [conf R1] |
| H-A3 interaction | interaction ≤ 0.02 [dev] |
| H-A4 early competition between readout and source-token routes | trajectories (suppression and coexistence) [dev]; ro0 flips an early-destined run to mixture [dev]; [conf R2a, R2b] |
| H-A5 continuous outcome rather than a discrete choice | see A5 |

H-A1 and H-A4 are compatible. The initialisation is the cause, and the early race is the mechanism through which it acts. The ro0 result
shows the race can be overridden.

## A5. Discrete, continuous, or both

Final measures over 17 XIII/XIII-B fine-tunes, the 4 development cells and the ro0 cell (stream 86_100_000):

- **mix_mass is bimodal:** 0.16–0.29 or 0.87–0.97, with nothing in between at step 4000.
- **lik_RO1 has a tight early cluster** (0.78–0.81) and a continuum among mixture-head units (0.00, 0.03, 0.05, 0.14, 0.22, 0.32, 0.42,
  0.52). The value tracks how late the head formed.

The organisation is therefore *one binary event* (whether a q-mass head forms) plus *a continuous residue* (how much early readout survives
it). It is a combination: two attractor-like end states for the head, connected by hybrids whose position depends on the timing of head
formation.

uP164's XIII-B threshold sensitivity fits this picture: lik 0.52 with a head (seed 20), early (seed 21, 22).

## §8. The learning transition

**Development finding.** The internal event that predicts the *organisation* is the formation of a q-mass head by step 546. It is an
attention-only quantity; the outcome, final lik_RO1, is an intervention-based readout measure.

**No internal predictor of the behavioural transition** to the correct decision rule is found:
- its timing (546–1211) is shared across classes;
- every candidate (mixture-head formation, the rise in readout organisation, ρ) moves at the same time as, or after, the behaviour.

The frozen prediction R2c tests only the organisation predictor. **The behavioural-transition predictor remains a negative result.**

## Confirmation (frozen; FROZEN_CONFIRMATION_AND_STATISTICS.md)

**R1a passed in all four new crossed cells** (two new parents' 2 × 2s).
- d_init is 0.01–0.20 and d_data 0.28–0.89, with non-overlapping bootstrap intervals.
- In the uP161 confirmation 2 × 2 the hybrid diagonal's residue moves with the data (lik 0.42 → 0.24). The data effect is 0.06–0.09 and
  the interaction 0.11–0.18 in the confirmation parents: small, not zero.

**R1b failed, and the failure changes the conclusion.** In both split cells (layer-2 init from one seed, readout-map WA init and data from
the other) the organisation followed **WA + data**, not layer 2:
- distance to the WA + data reference 0.08–0.10;
- distance to the L2-init reference 0.70–0.87.

R1a excludes data alone. Under additivity, the initialisation of the **fresh readout map WA (a 64 × 3 linear map)** decides which
organisation emerges, not the fresh layer 2 in which the mixture heads live. This is new and unexpected. It is still compatible with a
WA × data interaction, which one run would settle (review 3): L = W from one seed, D from another, only WA varied.

**R2a and R2b passed in all three runs.**
- ro0 turned an early-destined uP160 run into a mixture model (lik 0.83 → −0.001).
- l2q turned two mixture-destined runs into early-readout models (lik 0.007 → 0.71; 0.034 → 0.82).
- The flipped organisation persisted for 3400 steps after release.

The uP161 l2q run trained badly (KL to Bayes 0.21), so it shows that the organisation changed, not that a normal solution was reached.

**R2c (step-546 mixture mass as a predictor) failed in 2 of 5 diagnostic units.** The early event predicts clear cases and fails near the
boundary.

**Revised answers.**
- **Q1.** In the four tested parents the *initialisation*, not data order, decides the organisation. Among the initialisation components,
  the **readout-map initialisation** carries it (R1b; interaction caveat).
- **Q2.** An identifiable early event exists: a race between early readout and q-mass head formation in layer 2. Blocking either side for
  600 steps flips the outcome in 3/3 runs [conf]. As a *prediction from step-546 internals*, however, it failed near the boundary.
- **Q3.** A binary event (head forms or not) plus a continuous hybrid residue that is partly data-sensitive.
