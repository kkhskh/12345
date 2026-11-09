# R1 — Adversarial review: COMPOSITION and SCIENTIFIC NOVELTY (TRANS)

Scope: TRANS_COMPOSITION_RESEARCH.md, BASELINES_AND_NOVELTY.md, MAIN_RESEARCH_SYNTHESIS.md (TRANS items); code
`src/tasks3.py`, `src/train3.py`, `scripts/eval_trans.py`, `scripts/trans_probe_controls.py`,
`scripts/trans_restricted_bayes.py`, `../step1_foundational_atlas_continuation/src/tasks_ext.py`; logs `logs/trans/*.json`.
Reviewer checks were run from `/tmp/r1/` (`stat_checks.py`, `model_checks.py`, `aux_layers.py`, `prior_only.py`,
`agree2.py`). They only do forward passes and exact Bayes, on the package's own evaluation stream (9_950_000; intervention
stream 9_950_500, reproduced exactly). Nothing was trained and no package file was changed.

Severity: **Major** = a stated conclusion is wrong or unsupported as worded; **Moderate** = the conclusion needs
qualification or the number is misread; **Minor** = presentation, or a robustness gap.

---

## F1 (Major). The probe ceiling for the holdout-run row is wrong, so "at the ceiling" claims there do not hold

The MAP ceiling in `trans_probe_controls.py` is computed on 36-pair episodes under the uniform 36-pair prior. Using the
full 6-question posterior (not leave-one-out) is **correct** for an episode-level probe: the marginal MAP of σ is the
Bayes-optimal classifier from X. I reproduced it (0.616 / 0.621 / 0.403 on 24-pair episodes under the 36-prior; 0.617 /
0.626 / 0.394 in the log). The flaw is that the probes are fitted on episodes from the evaluated pair set. They therefore
learn that set's prior, and their ceiling must use the same prior:

| episodes (probe trained and tested on them) | prior-matched MAP ceiling σ / τ / τσ | holdout-model probe lin / MLP σ, τ, τσ |
|---|---|---|
| 24 training pairs | **0.694 / 0.703 / 0.540** | 0.66/0.63, 0.65/0.64, 0.39/0.44 |
| 12 held-out pairs | 0.798 / 0.801 / 0.747 | 0.76/0.74, 0.75/0.74, 0.67/0.67 (not reported in the doc) |

So the holdout model is **below** its ceiling on all three variables, and well below it on the composite (0.39–0.44 vs
0.54). The table in §3 sets the holdout row next to the 36-pair ceiling. That makes σ and τ look at or above the ceiling
and the composite look "at" it. Synthesis Q15 ("σ … at the statistical ceiling, every run") is false for the holdout run
under the correct ceiling. The unreported held-out-episode row shows the same artefact: a composite probe accuracy of 0.67
on episodes where the model's C-only accuracy is 0.25. Under the 12-pair prior, I(τσ; σ) = 1.36 bits, so the probe is
largely reading the prior.

Other probe-fairness points:
- The random-network control is a single initialisation (seed 123).
- An MLP probe on the random network reaches 0.37–0.38 on σ/τ. An MLP on raw one-hot input reaches 0.49–0.51. The MLP
  probe therefore does much of the inference itself.
- Selectivity in the sense of Hewitt & Liang (2019) should be reported. The full-run σ value (0.63 vs 0.617) is within one
  standard error (about 0.011 with n_test = 2000).

## F2 (Major). Aux run: "maps represented at the ceiling" is partly by construction, and the "composite information" comes from the probe composing

- The auxiliary loss (`train3.loss`, `trans_aux`) is a **linear softmax head on the final-layer episode token
  z[:, 42]**. The probe input is [readout-mean, episode token], so it contains the vector that was directly supervised.
  Decodability of σ/τ there is guaranteed by the training objective. Its head accuracy is σ 0.623 and τ 0.621, the
  ceiling itself. It is not an independent finding about representation.
- **Composite.** Composing the aux head's own argmaxes analytically, τ̂∘σ̂, gives **0.396**, the composite MAP ceiling
  (0.394). This is expected, because the posterior factorises (T1). So the 0.31 (linear) / 0.37 (MLP) composite
  decodability needs no composite representation at all: a nonlinear probe reading two independent map codes composes
  them itself. The claim "even decodable composite information" (§3) is an artefact of the probe.
- **Where τ is.** Layer-wise linear probes, from `aux_layers.py`:

  | | σ | τ | τσ |
  |---|---|---|---|
  | episode token, layer 1 | 0.44 | 0.63 | — |
  | episode token, layer 2 | 0.62 | 0.63 | — |
  | readout tokens, layer 1 | 0.62 | 0.27 | — |
  | readout tokens, layer 2 | 0.61 | 0.44 | 0.185 (chance) |

  τ reaches the decision tokens only partially, and only in the last layer, where no further layer remains to compose it
  with the question's C report. The full run has τ ≈ 0.17–0.22 at every layer.
- **Verdict.** The causal null is reproduced: masking leaves C-only at 0.333 / 0.334. "Represented but not composed" is
  defensible only in the weak form: "τ is represented at the auxiliary site and only partially routed to the readouts; the
  decisions do not use either map". §6's "the failure lies in combining them" and BASELINES' "clean new observation" are
  overclaims. The data do not separate a *routing* failure (τ never reaches the readouts with depth to spare) from a
  *combination* failure. Synthesis Q17 ("composite decodable … under supervision") and Q26 ("represented-but-unused maps")
  should be reworded.

## F3 (Major). Map interventions: the episodes are valid, but the held-out-episode row mostly measures training pairs, and "≈ two-thirds follows" is the ceiling of a training-prior learner

**Validity of `regen`: confirmed.**
- The regenerated B reports in AB questions have an error rate of 0.103 (β = 0.9 expected), with the original noise
  positions kept.
- Relabelling the non-AB questions under the new world, Y′ = σ′⁻¹σ(Y), the new Bayes C-only accuracy is 0.500, which is
  in distribution (0.49).
- The τ branch keeps the B reports and regenerates only the C reports of BC questions, which is correct.
- Targets are computed by `bayes_trans` on the new X, and tied argmaxes are excluded by `uniq`. This is correct for the
  36-pair prior.

**The problems:**
- σ′ ≠ σ (or τ′ ≠ τ) is drawn uniformly from the other five maps, ignoring the pair set.
  - From **held-out-pair episodes**, 78% (σ) and 77% (τ) of counterfactual worlds are **training pairs**. Row "holdout
    (held-out-pair episodes) 0.67 / 0.68" is therefore mostly an in-distribution measurement. It is not evidence about
    held-out compositions, and §4 does not say so.
  - From training-pair episodes, 38% of counterfactuals land on held-out pairs.
- For the holdout model the target is the 36-prior Bayes answer, which is the wrong reference. The exact **24-prior
  (training) Bayes "follows the new 36-prior Bayes"** at the following rates; the model matches it:

  | intervention, episodes | model | restricted Bayes |
  |---|---|---|
  | σ, training-pair episodes | 0.64 | 0.65 |
  | τ, training-pair episodes | 0.63 | 0.67 |
  | σ, held-out-pair episodes | 0.67 | 0.70 |
  | τ, held-out-pair episodes | 0.68 | 0.74 |

  Split by landing class:

  | counterfactual lands on | model σ / τ | restricted Bayes σ / τ |
  |---|---|---|
  | a training pair | 0.72 / 0.70 | 0.75 / 0.74 |
  | a held-out pair | 0.52 / 0.53 | 0.50 / 0.54 |

  "The predicted direction in about two thirds of changed cases" is therefore the **ceiling** of a learner that matches
  the training prior, not partial causal use. Restate it relative to restricted Bayes.

## F4 (Moderate). Shortcut quantification: under the 24-pair prior a large part of C-only accuracy needs no composition

Exact computations (`stat_checks.py`, `prior_only.py`, training-pair episodes):
- I(τσ; σ) = I(τσ; τ) = 0.52 bits. Given σ exactly, the composite is uniform over 4 of the 6 maps, so max P = 0.25.
- The composite marginal is non-uniform. Counts over the 24 pairs are 5, 4, 5, 2, 3, 5 (the 3-cycles PERMS[3] and [4]
  are rare).
- **Prior plus the question's own C reports only, with no other question, reaches 0.392 C-only accuracy.** Chance is
  0.333.
- Restricted Bayes with τ evidence masked: **0.429**. This is above Q3-C threshold (i) of 0.42 without any τ
  information. With σ evidence masked: 0.436. The model's masked scores, 0.416 and 0.432, match these.

Q3-C was calibrated on the 36-pair task, where masking either map gives chance (T2). It is materially weaker in the holdout
regime. Criteria (ii) and (iii) still discriminate: a σ-only model would show no τ-drop, and the model drops by 0.15 and
0.13. So "learned (on training pairs)" stands formally. But the composition-attributable increment is ≈ 0.56 − 0.43, not
0.56 − 0.33.

T1/T2 (a-posteriori independence; a uniform factor implies a uniform composite) assume the product prior. They **do not
hold** under the 24-pair prior, and the document applies them implicitly when it reads the masking drops.

§5 says the composite is "partly predictable from σ alone". That is correct, but it should be quantified as above.

## F5 (Moderate; mostly supportive). Training-prior-Bayes interpretation of the holdout run: supported, and more strongly than the document shows

C-only questions where the restricted and full Bayes argmaxes are both unique and differ (n = 1892 on training-pair
episodes, n = 2757 on held-out):

| | model agrees with 24-prior Bayes | model agrees with 36-prior Bayes |
|---|---|---|
| training-pair episodes | 0.64 | 0.26 |
| held-out-pair episodes | 0.65 | 0.25 |

- Mean KL(Bayes ‖ model) on C-only questions: restricted 0.087 vs full 0.166 on training pairs; 0.129 vs 0.322 on
  held-out pairs.
- Masked held-out accuracy: model 0.166 / 0.161 vs restricted Bayes 0.141 / 0.139.

This item could be upgraded from a matching of two marginal accuracies (0.246 vs 0.205) to per-question agreement
evidence.

Caveats:
- One seed.
- The model is not exactly the restricted Bayes rule (0.246 vs 0.205; BC 0.48 vs 0.41).
- "Inference within the 24-pair hypothesis class" does not require computing τσ. A 24-way hypothesis-enumeration solution
  is consistent with every observation, so even the training-pair success is not evidence of a compositional mechanism.

## F6 (Moderate). Design confounds and n = 1 limit the curriculum and regime conclusions

- The three regimes differ in several variables at once:
  - pair set: 36 / 36 / 24;
  - steps: 12k / 8k / 8k;
  - seed: 0 / 2 / 1;
  - task difficulty: Bayes C-only 0.49 vs 0.59; the restricted task also gives a prior-only signal of 0.39 (F4), which
    may supply the gradient that starts learning.

  "Restricting the pair distribution makes C-only decisions learnable" is a single-seed observation with an alternative
  explanation: the C-only gradient is no longer zero before any map is learned. That is a different claim from "the
  restricted class is easier to compose".
- "No curriculum produced generalising composition" rests on two curricula.
- The full-run plateau (loss ≈ 0.6, C-only at chance through 12k steps) is the regime where the grokking and implicit-
  reasoning literature reports late composition (Wang et al. 2024, which needs training far beyond fit). The hedge "not an
  impossibility result" is appropriate. The budget statement should cite this literature.

## F7 (Minor). Tie-breaking in argmax on exactly uniform posteriors

`np.argmax` on (numerically) uniform posteriors returns an index that depends on float rounding. It is not random.
- Example: the 36-prior with no other-question evidence gives 0.352 C-only accuracy on training-pair episodes and 0.295 on
  held-out episodes, instead of 0.333.
- This slightly biases the "no_tau" and Bayes reference accuracies and the masked-input accuracies near chance (e.g.
  0.3347 vs 0.3320).
- The interventions correctly exclude ties (`uniq`).
- Recommendation: break ties at random, or report cross-entropy alongside accuracy (the logs already contain CE).

## F8 (Moderate). Novelty: missing literature and overclaims

BASELINES_AND_NOVELTY says no web search was run. I ran a limited one and confirmed these works exist:
- Wang et al. 2024, *Grokked Transformers are Implicit Reasoners* (arXiv 2405.15071);
- Balesni et al. 2024, *The Two-Hop Curse* (arXiv 2411.16353);
- Raventós et al. 2023 (arXiv 2306.15063, NeurIPS);
- Sanford, Hsu & Telgarsky 2024, *Transformers, parallel computation, and logarithmic depth* (ICML; k-hop induction);
- Hewitt & Liang 2019, *Control tasks* (arXiv 1909.03368);
- Belinkov 2022, probing survey (arXiv 2102.12452).

**Missing and directly relevant:**
1. **Two-hop / latent composition failures.** Balesni et al. 2024; Wang et al. 2024, where composition fails out of
   distribution and needs prolonged training; Press et al. 2023, compositionality gap [m]; Biran et al. 2024, "hopping too
   late" [m].
2. **Depth requirements for in-context chaining.** Sanford et al. 2024. With L = 3 this is a relevant structural
   explanation for F2's "τ arrives at the readouts only in the last layer".
3. **Learned permutation and group composition.** Liu et al. 2023, *Transformers learn shortcuts to automata*
   (S₃-type composition) [m]; Chughtai et al. 2023, group-operation circuits [m]; Zhang et al. 2022, LEGO chained
   assignments [m].
4. **Task-diversity transitions and plateaus in in-context learning.** Kirsch et al. 2022 [m]; Singh et al. 2023,
   transient in-context learning [m]. These sit beside Raventós for the 24-vs-36 pair contrast.
5. **Probe methodology.** Hewitt & Liang 2019; Belinkov 2022; amnesic probing (Elazar et al. 2021), which is already
   cited [m]. Systematic holdout splits: Keysers et al. 2020 (MCD) [m].

**Classification adjustments:**
- "Represented-but-uncomposed maps … clean new observation" should be downgraded to an **instance**. The representation is
  partly by construction, the composite is decodable only via the probe, and the routing-vs-combination question is
  unresolved (F2).
- The holdout result is correctly an **instance** of Raventós-type prior fitting. Raventós concerns task *count* in
  regression. Here the split is structured (a compositional holdout of a finite group family), so systematic-generalisation
  benchmarks are the closer analogue.
- T1–T3 are correctly listed as known mathematics.

## F9 (Minor). Synthesis wording

- Q15: "σ at the ceiling, every run". False under the prior-matched ceiling for the holdout run (F1).
- Q17: "composite decodable under supervision". Probe artefact (F2).
- Q18: "follows Bayes 0.64". Should be read against the restricted-Bayes 0.65 ceiling (F3).
- Q19: tags [E] for the interpretation. F5's per-question agreement would justify keeping it [E] but with stronger
  wording.
- Q26: "represented-but-unused". Needs the routing caveat (F2).
- The §1 statements of T1/T2 should state that they hold under the product prior only (F4).

## What holds

- The statistical analysis of the 36-pair task: T1–T3, the C-only Bayes value 0.487, and the MAP ceilings for the
  36-prior.
- The Q3-C outcomes as frozen.
- The full-run discard of τ: below the random network at every layer, and 0.17 at the readouts.
- The causal nulls in the full and aux runs.
- The validity of the `regen` episodes.
- The no-generalisation verdict on held-out pairs.
- The training-prior-Bayes reading (strengthened by F5).
