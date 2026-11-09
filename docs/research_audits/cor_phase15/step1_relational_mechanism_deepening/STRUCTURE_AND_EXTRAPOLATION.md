# Structure and extrapolation (Workstream D)

> **Post-review [R].** Q3-D held-out Transformers are parent-shared replications (SD7); the GCD sign-reversal narrative is withdrawn (SD8); EX1 rules out on-support behaviour *determining* extrapolation, not on-support features *predicting* it among trained models (FM9). Frozen verdicts unchanged: D1 passed, D2 failed, D3 passed. [R2] The parent-sharing caveat also applies to D3: A2's held-out ρ 1.00 is matched by parent lookup; A2 beats it only on MAE (0.019 vs 0.042). Superseded pre-review statements are kept in ~~strikethrough~~ next to their
> corrections; details in `reviews/ADJUDICATION_R1.md`.

The continuation's frozen internal predictor P-A1 failed (MAE 0.126 vs architecture 0.091; held-out Transformers ranked
backwards). That verdict stands. This document records why, what can be said formally, a second frozen test (Q3-D), and
additional shifts. Code: `src/graded_sig.py`, `scripts/extrap_dev.py`, `scripts/eval_extrap.py`. Logs: `logs/extrap/`.

## 1. Formal limits (D1)
- **EX1 [P]** (THEOREMS §4): agreement on the training support constrains nothing off support without a class restriction.
- **Realised**: two Transformers with natural KL 0.0008 and 0.0019 have p_c = 0 local indices 0.87 and 0.30.
- Hence any predictor must use architecture, internal structure, or off-support queries. Only the last is guaranteed to
  carry the information, and then partly tautologically (the continuation's A2 probe is a milder off-support shift).
- What the distinctions buy: behavioural agreement on S = functional equivalence on S (here, up to KL 10⁻³); neither
  implies internal causal similarity (GATE seeds; T4 seeds 10 vs 12) nor agreement under off-support queries.

## 2. Why P-A1 failed (D2) — discriminating evidence
P-A1 used CCP_3, a probe trained on natural T4 and applied to **off-support** degraded episodes. The continuation's probe
control showed that such probes read which agreement statistic is decodable, not a validated variable. Here (post hoc,
`extrap_dev.json`, 14 earlier models) the on-support versions of the two candidate features were measured:
- **GCD** — decodability of the *graded* agreement counts of the two non-colluder groups from the episode representation;
- **SAT** — decodability of the colluder identity (the saturated, T4-sufficient feature).
On-support, DeepSets carry *less* graded-count information (GCD 0.49–0.51) than Transformers (0.61–0.89) — the opposite of
the continuation's off-support reading ("DeepSets carry graded counts"). So the earlier signature's **off-support meaning
was not its on-support meaning** (explanation "off-support representations having different meanings" supported; "wrong
site" and "sample size" not excluded). Within development Transformers GCD and SAT correlated −0.12 and +0.63 with the
local index.

## 3. Second frozen test (Q3-D) — 6 new Transformers (new T4 trajectories from existing T1 parents) + 10 new DeepSets

| predictor (frozen fits, dev models only) | uses | within-TF Spearman (n = 6) | within-TF MAE | all-16 MAE |
|---|---|---|---|---|
| constant (within TF) | nothing | — | 0.166 | 0.288 (global constant) |
| architecture | label | — | — | **0.087** |
| GCD (on-support structure) | natural T4 only | **−0.83** | 0.178 | 0.110 |
| SAT (on-support structure) | natural T4 only | **+0.71** | **0.123** | 0.289 |
| A2 behavioural probe | off-support inputs | **+1.00** | **0.019** | **0.016** |

Verdicts: Q3-D1 passed (10/10 DeepSets < 0.30, max 0.215; Transformer mean 0.62). **Q3-D2 — the negative prediction that
on-support structure would not predict the within-Transformer spread — FAILED**: both on-support signatures carried strong
rank information in the predicted-by-hypothesis directions of the continuation (more graded-count information → more
global; more saturated colluder code → more local), and SAT's frozen within-TF fit beat the constant on new models.
Q3-D3 passed.

Reading. (i) Architecture still predicts the cross-architecture direction better than any structural signature. (ii) Within
Transformers, ~~an on-support structural measurement (SAT) predicted unseen models' extrapolation prospectively (ρ = 0.71,
MAE 0.123 vs 0.166) — the first structural predictor in this programme to beat its baseline on held-out models.~~ **[R: SD7]** SAT's
held-out association (ρ 0.71, MAE 0.123) replicates the parents' development values: each held-out TF shares a T1 parent with a
development TF, and a parent-lookup predictor gives ρ 1.00, MAE 0.042. Exact null p 0.068; no positive SAT prediction was frozen. Its strength
is limited: n = 6, the six share T1 parents with development models, and a behavioural off-support probe is far better.
(iii) ~~GCD's within-TF sign reversed between development and held-out~~ **[R: SD8]** On the same six parents in development GCD's ρ is already −0.66; the positive frozen slope came from one leverage model (tf_s14). Under the frozen signed wording GCD complied with Q3-D2; SAT alone failed it [R: SD9].
No causal test (D3) of graded vs saturated features was run, so the association is not a mechanism.

## 4. Several shifts (D5)

Disagreement rate between true Bayes and the T4 plug-in (competing hypotheses must disagree to be tested): α 0.8 → 0.7:
7.1 %; ρ 1 → 0.9: 3.1 %; p_c 1 → 0.5: 22.6 %. Agreement with true Bayes on those questions (held-out models):

| shift | Transformers: true / plug-in | DeepSets: true / plug-in |
|---|---|---|
| source error rate (α 0.7; overlapping support) | 0.38 / 0.49 | 0.38 / 0.51 |
| partial copying (ρ 0.9) | 0.50 / 0.50 | 0.38 / 0.62 |
| collusion prevalence (p_c 0.5) | 0.54 / 0.42 | 0.10 / 0.84 |

- An error-rate shift with overlapping support behaves like miscalibration: both architectures stay near the plug-in.
- Partial copying and prevalence shifts separate the architectures as p_c = 0 does; within Transformers the p_c = 0.5 result
  ranks models identically to the p_c = 0 local index (Spearman 1.0), the α and ρ shifts do not (0.26, 0.23).
- So "local vs global" is a property of the colluder-absent region, not a general extrapolation trait.

## 5. Answers (questions 29–31)
29. **Why P-A1 failed:** its off-support probe signature did not mean, on new models, what it meant on development models;
    on-support measurements of the same idea reverse the cross-architecture reading. Small n and parent-sharing remain.
30. **Which structural measurements predict unseen behaviour:** on-support colluder-code decodability (SAT) predicted the
    within-Transformer ranking on 6 new models (ρ 0.71), beating the constant; graded-count decodability ranked strongly
    but with the sign opposite to its frozen fit; architecture predicts the DeepSets/Transformer split best. None beats a
    behavioural off-support probe.
31. **Architecture vs learned computation:** the DeepSets/Transformer split (15/15 DeepSets global over three studies) is
    architectural; the within-Transformer spread is learned-computation dependent and partly visible in on-support
    representation.
