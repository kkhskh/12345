# Open foundational research map (after Investigation III)

Ordered by expected leverage; each item states its dependency and the resolving evidence.

| # | open problem | depends on | resolving evidence |
|---|---|---|---|
| 1 | **Learned composition.** No model composed σ and τ on the full 36-pair task; aux supervision gives represented-but-unused maps; the 24-pair model fits the training prior | TRANS §2–§5 | runs beyond 12k steps and multiple seeds; σ-only → composite curricula with intermediate composition questions; held-out pairs designed so the restricted prior cannot help; a pathway intervention showing where (if anywhere) σ and τ are combined |
| 2 | **Causal test of graded vs saturated features** (why SAT/GCD predict extrapolation within Transformers) | STRUCTURE §3 | a probe-free intervention on natural computation (e.g. models trained with a pooled graded-count channel injected), with a frozen directional prediction and ≥ 10 independent Transformer seeds (independent T1 parents) |
| 3 | **A pooled-architecture mediator for RC** | CLASSES §3 | identify T4 DeepSets' pooled relation variable with abstract-variable interchange; decides whether S1–S2 is architecture-specific |
| 4 | **Why T1 and P-count place the aggregate differently** | PCOUNT A3 | move the P-count targets to source tokens and test whether the site moves |
| 5 | **Nonlinear abstract alignment for T1** (reliability-posterior encoder) | PCOUNT A5 | abstract-variable interchange with a posterior-valued encoder at T1's source tokens |
| 6 | **GATE generalisation** (larger K, noisy or multiple gates, changing gates, other constraints) and whether stage-1/readout-attention structure persists | GATE §6 | trained variants + the same circuit pipeline |
| 7 | **Learned models under unknown / changing groups** (T4-perm etc.) | GENERALIZATION §4 | a model beating own-question Bayes with hidden pairing, then mediator identification |
| 8 | **Adaptive sources beyond one change point**: drift, outcome-conditioned reliability, learned triggered/camouflaged copiers | ADAPTIVE §4 | exact references exist for the copier models (`adapt_theory.py`); learned models and history interventions |
| 9 | **Finer dynamics** (every 100 steps, several seeds) and path-specific necessity for the transferred code | PCOUNT A4, A6 | as stated |
| 10 | OD2 for K = 2; sufficient statistics for change-point histories (AD1) | THEOREMS §6 | proofs |

Directions not examined at all in this package: more than three candidates; larger source groups; correlated-error and
partial-copying learned models; longer episodes; architectures other than the 3-layer Transformer and DeepSets.
Results resting on single runs: every TRANS regime; the dynamics run; the ADAPT curriculum (two parents).
