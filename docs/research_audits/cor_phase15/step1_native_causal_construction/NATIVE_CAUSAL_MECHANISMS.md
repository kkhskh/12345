# Native causal construction of the relational computation (Investigation XI, Workstream B)

**Question.** Where, and how, do the fine-tuned layers 0–1 combine *perfectness* (n_g = 5) with *agreement at the
target question q* into the eligibility conjunction e_g = perfect_g ∧ agree_g(q)? And is that construction what drives
the decision?

## Models and evidence status

- **Development:** xP81, yP130, yP131, yP132, yP134 (full_c1, seed 0), xP81 and yP130 seed 1, and C1 controls xP81,
  yP130, yP131.
- **Held-out:** wP140–wP143 (full_c1 and C1). These are pre-existing Investigation X confirmation models, never analysed
  in XI before the freeze; see the frozen predictions.
- Every number below is labelled **[dev]** or **[conf]**. The restoration test on held-out models was not frozen and is
  labelled **[conf-exploratory]**.

## 1. Why decodability is not enough

Held-out logistic probes on the q7 tokens [dev], from `mech_construct.py`, decodability by stage:

| stage | agree(q) | perfect | e |
|---|---|---|---|
| resid0 (embeddings) | 0.67 | 0.63 | 0.64 (= majority) |
| after layer-0 attention | 0.78–1.00 | 0.63–0.79 | 0.64–0.81 |
| resid1 | 1.00 | 0.72–0.79 | 0.74–0.81 |
| after layer-1 attention | 1.00 | 1.00 | 1.00 |
| resid2 | 1.00 | 1.00 | 1.00 |

In the **C1** model xP81 (frozen layers 0–1, which does *not* follow the Bayes rule), e is also decodable at 0.99. AND
is linearly separable, so decodable e is no evidence that the conjunction is *computed*. The decodability results
establish only **propagation**:
- agreement-at-q reaches the q7 tokens through layer-0 attention and MLP0;
- perfectness, an aggregate over the five other questions, arrives only through layer-1 attention.

## 2. Combination: a within-episode 2 × 2 factorial (P1)

`factorial.py`. For a group h that disagrees at q:
- **A** makes h agree at q;
- **P** makes h agree at its one remaining disagreeing question, so that n_h goes 4 → 5.

Bayes e_h is 1 only in the cell xAP (sanity check: ΔS = 0, 0, 0, 1). The control design starts from n_h = 3, so e = 0
in all four cells.

The double difference dd = z(xAP) − z(xA) − z(xP) + z(x00) is computed per episode, so no alignment is needed. A purely
additive representation has dd = 0.

| | full_c1 [dev, 8 models] | full_c1 [conf, wP140–143] | C1 [dev, 3] | C1 [conf, 4] |
|---|---|---|---|---|
| E-subspace interaction ratio TRUE/CTRL | 2.73–9.81 | **5.27 / 7.02 / 5.54 / 5.35** | 0.95–1.48 | **0.99 / 0.75 / 1.05 / 1.19** |
| rel. interaction after layer-0 attention (mid0) | ≤ 0.02 | ≤ 0.014 | ≤ 0.04 | ≤ 0.022 |
| rel. interaction after layer-1 attention (mid1) | 0.14–0.44 | 0.32–0.43 | ≤ 0.06 | ≤ 0.09 |
| layer-0 share of the E-interaction | ≤ 0.01 | ≤ 0.001 | — | — |
| layer-1 attention / MLP1 share | 0.17–0.70 / 0.30–0.83 | 0.23–0.58 / 0.42–0.77 | — | — |

**Finding.** The conjunction is **formed in layer 1** (attention and MLP1) and is absent before it. C1 models contain
no conjunction at all, although e is decodable in them.

Residual additivity makes the per-component split of dd(resid2) exact. The split between layer-1 attention and MLP1
varies across models.

## 3. Necessity: removing the conjunction from one layer (P2)

`factorial_ablate.py`. In the xAP run, each component's output is replaced by its **additive completion**
c(xA) + c(xP) − c(x00). The component keeps both main effects and loses only its conjunction. Everything downstream is
recomputed.

The reference is the "e_h = 0" counterfactual, cf_e0: the Bayes formula with group h treated as ineligible.

| removal | E-interaction left | KL(Bayes ‖ model) | KL(cf_e0 ‖ model) | fraction closer to cf_e0 |
|---|---|---|---|---|
| none [conf] | 1.00 | 0.003–0.010 | 0.39–0.40 | ≤ 0.06 |
| **layer 0** [conf] | 1.00 | unchanged (Δ = 0.000) | unchanged | unchanged |
| layer-1 attention only [conf] | 0.06–0.12 | — | — | — |
| MLP1 only [conf] | 0.31–0.62 | — | — | — |
| **layer 1 (attention + MLP1)** [conf] | **0.00** | **0.53–0.58** | **0.029–0.042** | **0.67–0.72** |
| layer 1, C1 models [conf] | — | Δ ≤ 0.025 | — | — |

Development: identical pattern in 8/8 models (layer-1 removal leaves ≤ 0.01; pref_e0 0.53–0.74).

**Finding.** Without layer 1's conjunction, the network decides **as if group h were not eligible**, not merely
"worse". The layer-0 conjunction is causally irrelevant.

The two layer-1 routes overlap. MLP1 recomputed from conjunction-free attention retains part of the interaction, and
attention-only removal leaves 0.05–0.37. This is a **serial composition** (attention partly forms the conjunction, and
MLP1 sharpens and writes it), not two independent backups.

## 4. Mediation and restoration: the 6-dim E-subspace carries the decision-relevant conjunction

`restore.py`. Starting from the layer-1-ablated xAP state, restore **only** the 6 E-subspace coordinates (out of 448 at
the q7 tokens) of the intact state, then recompute layer 2.

KL(Bayes ‖ model) after each restoration:

| model | ablated | restore E (6 dims) | restore random 6 dims | restore complement (442 dims) |
|---|---|---|---|---|
| dev, 7 full models | 0.39–0.66 | **0.012–0.124** | 0.37–0.65 | 0.22–0.57 |
| conf-exploratory, wP140–143 | 0.53–0.58 | **0.036 / 0.076 / 0.157 / 0.052** | 0.52–0.58 | 0.38–0.55 |

The random 6 dims recover nothing (≤ 0.05 recovery). The complement recovers a minority.

**Finding.** The conjunction that drives the decision passes through the E-subspace identified by Investigation X's
DAS. The other 442 q7 dimensions carry little of it. This is a **sufficiency** result for the subspace given everything
else, at the level of the decision.

Direct-write decomposition at resid2 [dev] (`mech_construct.py`):
- MLP1 plus layer-1 heads write ≥ 85% of the E-subspace displacement for the agreement edit (D2) and the count edit
  (CNT). An exception is yP132 on D2, where MLP0 writes 0.37. This is a propagation route, because MLP0 carries
  agreement, not the conjunction (§2).
- Selectivity is D2c/D2 = 0.25–0.35, against 0.76 in C1. One development outlier: yP132 0.79.

## 5. The causal chain

1. **Input → propagation (layer 0).** Layer-0 attention and MLP0 bring the agreement-at-q bit of each group to the q7
   tokens. There is no interaction (mid0 rel_I ≤ 0.02), and removing layer 0's interaction has no effect.
2. **Relational combination (layer 1).** Layer-1 attention gathers perfectness, which spans the other five
   questions, and forms the conjunction jointly with the agreement bit. MLP1 completes it. Both are necessary jointly
   (P2).
3. **Downstream representation (resid2, E-subspace).** The conjunction is written into a 6-dim subspace at the q7
   tokens:
   - restoring that subspace alone restores the decision (§4);
   - mapping it from another model's states transfers it (Workstream C, P3).
4. **Candidate update (layer 2).** Layer 2 reads the subspace. Group h's eligibility changes the weights (S − e_g) of
   the *other* groups' candidate evidence. That is why the decision under conjunction removal tracks cf_e0.

What is **not** shown: the weight-level circuit, i.e. which QK features let layer-1 attention depend jointly on both
bits, and whether MLP1 is a thresholded AND. See the open questions.

## 6. Comparing implementations across parents

`perm_compare.py` [dev]. These are the E-interaction shares of the four layer-1 heads and MLP1.
- Same-parent seed pairs: raw L1 distance 0.42, best-permutation 0.40. Head identities are inherited from the
  pretrained parent.
- Cross-parent pairs: raw 0.91, best-permutation 0.62.

So the abstract computation (the conjunction in layer 1, carried by E) is shared. Its allocation across heads and MLP1
is parent-dependent, and partly seed-dependent: yP130 seed 0 vs seed 1 gives MLP1 0.83 vs 0.61. In held-out models,
attention-only removal already eliminates 88–94% of the interaction. That differs from several development models,
where MLP1 dominated. This is further evidence that the construction is *functionally* but not *componentially*
invariant.

## 7. Emergence during fine-tuning (P6)

`dyn_factorial.py`, using the full-space interaction ratio at resid2, which needs no subspace.

| model | interaction ratio ≥ 2 first at step | first D2-Bayes-closest step (Investigation X) |
|---|---|---|
| xP81 [dev] | 246 | 813 |
| yP130 [dev] | 546 | 1211 |
| yP131 [dev] | 1211 | 4000 |
| yP132 [dev] | 246 | 1211 |
| yP134 [dev] | 1211 | 1803 |
| wP140 [conf] | **366** | 1211 |
| wP141 [conf] | **813** | 1211 |
| wP142 [conf] | **366** | 813 |
| wP143 [conf] | **366** | 546 |

**Finding.** The representational conjunction appears **before** the behavioural switch from the A1/A2 heuristics to
the Bayes rule, in 9/9 models, by one to three checkpoints. Behaviour lags mechanism. During the lag, the internal
conjunction exists but the decision is still heuristic-dominated.

The cause of the lag (readout learning versus magnitude) is not established.

## 8. Cross-model interchange and false-positive diagnostics (Workstream C)

### Mapped interchange

`cross_interchange.py`. A ridge map is fitted A → B on q7 states of natural episodes only. The mapped source is then
patched into B's E-subspace.

Source-specific gain = (KL_mean − KL_x) / (KL_mean − KL_within). "Mean" patches B's mean state, a source-free control.

| pair | map R² | mapped gain | shuffled-map gain | mean-patch IIA_disc (base) |
|---|---|---|---|---|
| full→full, cross-parent [dev, 4 pairs] | 0.94–0.98 | 0.92–1.00 | −0.16 to −0.06 | 0.75–0.95 (0.18–0.20) |
| full→full, same parent, seed 0↔1 [dev, 3 pairs] | 0.99–1.00 | 0.99–1.00 | −0.12 to −0.10 | 0.76–0.95 |
| full→full, cross-parent [conf, 4 pairs] | 0.92–0.99 | **0.97 / 1.00 / 0.89 / 1.02** | ≤ 0.05 | 0.56–0.85 (0.19–0.20) |
| C1→full, same parent [dev 1, conf 2] | 0.95–0.97 | **0.58; 0.49 / 0.71** | — | — |
| full→C1, same parent [dev 1, conf 2] | 0.95–0.99 | 1.00 | — | — |

Readings:
1. Between models that compute the conjunction, the variable transfers through a linear map fitted without any
   counterfactual information.
2. C1 states are equally predictable (R²) but carry the variable only partially in a linearly mappable form.
3. **Plain IIA is inflated**: most of the discriminating-pair accuracy is reached by a source-free patch.

### Decoy DAS: false positives and the diagnostics that expose them

`das_decoy.py`. Same site and training as Investigation X.

| model class | rule | IIA | preserve_KL | selectivity D2c/D2 |
|---|---|---|---|---|
| full [dev 2 + conf 4] | **E (true)** | 0.85–0.97 | **0.006–0.021** (wP142: 0.060) | **0.21–0.44** |
| full | A1 decoy (perfect-only) | **0.83–0.93** | 0.16–0.21 | 0.36–0.51 |
| full | PERM decoy (wrong group assignment) | 0.75–0.84 | 0.10–0.19 | 0.44–0.58 |
| full [dev] | RANDE decoy (random targets) | 0.69–0.74 | 0.35–0.47 | 0.44–0.62 |
| full | random subspace (untrained) | 0.46–0.50 | 0.001–0.008 | 0.75–0.95 |
| C1 [dev 2 + conf 4] | E | 0.83–0.92 | 0.11–0.17 | 0.70–0.98 |
| C1 | A1 | 0.83–0.92 | 0.15–0.26 | 0.80–1.01 |

**Distinguishing variable identification from causal explanation.**
- A subspace can reproduce a wrong rule's counterfactuals with high accuracy (A1 in Bayes models). In other words,
  DAS finds output-matching subspaces for rules the model does not follow.
- The true mediator is distinguished by three properties:
  - near-zero effect when the variable's value is unchanged (preservation);
  - response to the defining input changes and not to their controls (selectivity);
  - the restoration result of §4.
- In C1 models no rule passes preservation. This is consistent with C1 having no conjunction (§2). It also explains
  Investigation X's F4, where C1's discriminating IIA was high.
