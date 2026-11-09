# TRANS: learned relational composition (Workstream C)

> **Post-review [R].** Corrections: holdout-run probe ceilings must use the 24-pair prior (σ 0.694, τ 0.703, composite 0.540; the model is below all three) (CN1); aux-run decodability is by construction at the supervised token (CN2); map-intervention following ≈ 0.64 equals the training-prior Bayes rule's own following rate (0.65–0.74), and 78 % of held-out-episode counterfactuals land on training pairs (CN3); under the 24-pair prior Bayes without τ evidence scores 0.429 and prior + own C reports 0.392, so much of the 0.562 needs no composition (CN4, FM2); T1a holds only under the product prior; T3 corrected (FM3: non-chained single pairs leave 4 candidates); one seed per regime with confounded budgets (CN6); argmax ties bias near-chance references slightly (CN7). Superseded pre-review statements are kept in ~~strikethrough~~ next to their
> corrections; details in `reviews/ADJUDICATION_R1.md`.

Status: the statistical results are exact; the learned-model results are three training runs (one seed each, one per
regime), evaluated with the criteria frozen in PROSPECTIVE_PREDICTIONS Q3-C before any run finished. Code: `src/tasks3.py`,
`src/train3.py`, `scripts/run_job3.py` (`trans_*`), `scripts/eval_trans.py`, `scripts/trans_identifiability.py`,
`scripts/trans_probe_controls.py`, `scripts/trans_restricted_bayes.py`. Logs: `logs/trans/`.

## 1. The statistical problem (C1) — exact

Generator: σ, τ independent uniform on S₃; A reports Y, B σ(Y), C τσ(Y); source accuracy β = 0.9; each question is AB,
BC or C-only (uniform). THEOREMS §1:
- **T1 (factorisation).** P(X_{−q} | σ, τ) = L_AB(σ)·L_BC(τ): σ is informed only by AB questions, τ only by BC
  questions, C-only questions carry no map information; σ and τ are a-posteriori independent.
- **T2.** If either map's posterior is uniform, the composite's is uniform, so the C-only Bayes answer is at chance however
  well the other map is known. Entropy of the composite ≥ that of each map.
- **T3 (coverage, β = 1).** One observed value pair leaves 2 of 6 permutations; two identify the map.
- Exact numbers (`identifiability.json`): C-only Bayes accuracy at chance whenever nAB = 0 or nBC = 0 among the other five
  questions; rising with both ((1,1) 0.43, (2,2) 0.59, (3,2) 0.72 at β = 0.9). Marginal C-only Bayes 0.487, MAP accuracy of
  the composite 0.394, of σ 0.617, of τ 0.626.
- Consequence: any model that ignores the C reports of BC questions has *no* τ information and must be at chance on C-only
  questions; "partial composition" is impossible without τ.

## 2. Learning experiments (C2, C5, C6)

| run | training distribution | steps | AB acc (Bayes 0.949) | BC acc (0.660) | C-only acc (0.489) | Q3-C "learned" |
|---|---|---|---|---|---|---|
| `trans_full_s0` | all 36 (σ, τ) pairs | 12,000 (3× the earlier run) | 0.948 | 0.640 | **0.336** | **no** |
| `trans_aux_s2` | all 36 pairs + auxiliary σ and τ supervision at the episode token | 8,000 | 0.948 | 0.639 | **0.332** | **no** |
| `trans_holdout_s1` | 24 training pairs (12 held out) | 8,000 | 0.952 | 0.726 | **0.562** | **yes** (on training pairs) |
| same model, held-out pairs | — | — | 0.941 | 0.480 | **0.246** | no (below chance) |

Learning curves (`trans_full_s0`, snapshots every 2000 steps): BC reaches 0.64 by step 6000 and stays; C-only stays at
0.33–0.34 through 12,000 steps; the loss plateaus at ≈ 0.59–0.62 (cosine schedule to 0). Stopping was fixed in advance;
no run was extended after inspection.

## 3. What the networks represent (C3) — with controls

Probe accuracy (chance 0.167) from the final [readout-mean, episode token] representation, held-out episodes:

| | σ (lin / MLP) | τ (lin / MLP) | τσ (lin / MLP) |
|---|---|---|---|
| statistical ceiling (exact MAP) | 0.617 | 0.626 | 0.394 |
| random network | 0.27 / 0.37 | 0.28 / 0.38 | 0.16 / 0.18 |
| full run | **0.63 / 0.60** | 0.21 / 0.28 | 0.16 / 0.20 |
| aux run | **0.62 / 0.61** | **0.62 / 0.62** | 0.31 / 0.37 |
| holdout run (training-pair episodes) | 0.66 / 0.63 | 0.65 / 0.64 | 0.39 / 0.44 |

- Full run: σ at the statistical ceiling; τ *below* the random-network baseline — training discarded the τ evidence.
- Aux run: **both maps are represented at the statistical ceiling and the composite almost at its ceiling, yet C-only
  decisions are at chance and are unaffected by removing τ or σ evidence.** This is a direct demonstration that this
  Transformer can represent σ and τ (and even decodable composite information) **without composing them into the decision**.

## 4. Causal use (C4)

Input-level map interventions on valid episodes (σ replaced by σ′ in the AB questions only; or τ by τ′ in the BC
questions' C reports), scored on C-only questions whose unique Bayes answer changes (`map_interventions`):

| run | σ change: follows new Bayes / model changes | τ change: follows / changes |
|---|---|---|
| full | 0.33 / 0.40 | 0.31 / 0.10 |
| aux | 0.32 / 0.51 | 0.36 / 0.21 |
| holdout (training-pair episodes) | **0.64 / 0.75** | **0.63 / 0.77** |
| holdout (held-out-pair episodes) | 0.67 / 0.75 | 0.68 / 0.75 |
| chance (follows) | ≈ 0.33 | ≈ 0.33 |

Masking criteria (Q3-C ii, iii) in the holdout run on training pairs: removing τ evidence 0.562 → 0.416; removing σ
evidence → 0.432. The holdout model's C-only decisions **causally depend on both maps' evidence**, with the predicted
direction in about two thirds of changed cases. The full and aux runs' C-only outputs do not follow either change.

## 5. Held-out compositions (C6): the holdout model behaves as the training-prior Bayes rule

Exact Bayes with the prior restricted to the 24 training pairs (`restricted_bayes.json`):

| episodes | model C-only | restricted-prior Bayes | full-prior Bayes |
|---|---|---|---|
| training pairs | 0.562 | 0.591 | 0.490 |
| held-out pairs | 0.246 | 0.205 | 0.493 |

The model exceeds full-family Bayes on training pairs and falls *below chance* on held-out pairs — and both numbers are
close to the training-prior Bayes rule. The held-out failure is therefore what a learner that has absorbed the training
pair distribution should do (cf. the low-task-diversity regime of Raventós et al. 2023): it does not show inability to
compose, nor ability; it shows that success on training pairs was achieved by inference *within the 24-pair hypothesis
class*, which with 24 of 36 pairs makes the composite partly predictable from σ alone. Memorising full pairs (as a prior)
explains the held-out behaviour; generalising composition was not observed.

## 6. Answers (C7)

- **Can TRANS be learned under a completed schedule?** Not with the full 36-pair distribution in 12,000 steps (C-only at
  chance), nor with auxiliary map supervision in 8,000. A restricted 24-pair distribution is learned to near its Bayes rule
  in 8,000 steps.
- **σ identified?** Yes, at the statistical ceiling, in every run. **τ?** Only with auxiliary supervision or the restricted
  distribution; the unsupervised full run discards it. **Composite?** Decodable only where the restricted prior or the aux
  loss makes it so.
- **Causally used?** σ and τ evidence: yes, in the restricted-prior model (interventions follow Bayes in ≈ 0.64; masking
  drops 0.13–0.15). In the full and aux runs neither reaches C-only decisions.
- **Held-out compositions?** No generalisation; behaviour matches the training-prior Bayes rule.
- **Curriculum.** Auxiliary map supervision makes the maps *represented* but not *composed*; restricting the pair
  distribution makes C-only decisions learnable within the training class. No curriculum produced generalising composition.
- **Limitation type.** Statistical: none for the full task (Bayes 0.487 is attainable). The failure is an optimisation /
  inductive-bias failure under this architecture and budget; it is **not** an impossibility result (one seed per regime,
  fixed budgets). ~~The aux result rules out "the maps cannot be represented"; the failure lies in combining them.~~ **[R: CN2]** σ/τ decodability at the episode token is guaranteed by the auxiliary head (a linear head on that token); at decision tokens τ reaches only 0.27–0.44 and the composite is at chance; composing the head's two predictions gives 0.396 (the composite ceiling), so composite decodability is the probe composing. Routing failure and combining failure are not distinguished.
- **Not run (compute):** more seeds; longer runs beyond 12,000; σ-only or τ-only pretraining; intermediate composition
  questions; larger alphabets.
