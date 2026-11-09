# Causal identification and invariance — Investigation III

> **Post-review [R].** Corrections: §3 GATE pair is contributing, not jointly necessary (CA4); §4 T1 source result reversed by a per-slot encoder (CA1); necessity controls must be per-position (CA3); "2-D contrast subspace" should read k = 4 (CA11); invariance holds per token, not under cross-slot weight-sharing assumptions, and not across ablation operators (CA1, CA4). Superseded pre-review statements are kept in ~~strikethrough~~ next to their
> corrections; details in `reviews/ADJUDICATION_R1.md`.

This document states the identification framework used in this package, the specific failure modes it was built to
avoid (each was a real error in an earlier package), and the results that the framework produced.  Earlier frameworks
(atlas intervention taxonomy; continuation quotient-game attribution R1) are referenced, not restated.

## 1. Three kinds of evidence, kept separate

| kind | what it establishes | instrument used here | what it cannot establish |
|---|---|---|---|
| **decodability** | information about a generator variable is (linearly or nonlinearly) present at a site | held-out logistic / ridge / small-MLP probes against generator labels, with random-network and raw-input controls | use; necessity |
| **necessity under replacement** | the computation degrades when a site's content is replaced (mean, resample) | mean-ablation of components or subspaces, always with a **matched control** (another head of the same layer; random and variance-matched subspaces) and with the **direction** of damage (KL to the own-question posterior) | that the replaced content is what the computation *uses* (replacement is off-manifold; redundancy can hide use) |
| **sufficiency under interchange** | changing a site's content to its value under a counterfactual input reproduces the counterfactual decision | interchange from on-support counterfactual episodes (cond pairs; gate flips; map replacements), with full downstream recomputation; IIA on informative pairs | uniqueness of the route (redundant routes can each be sufficient) |

**Abstract-variable interchange (used in Workstream A5).** Fit, on natural data and with no reference to the model's
decisions, a linear *encoder* E from a generator-defined variable v (here the one-hot leave-q agreement counts of the three
groups) to the activation at a site. Intervene a′ = a + E(v_cf) − E(v_base), where v_cf is the variable's value in a
counterfactual episode. If the decision follows the counterfactual, the model uses v *in E's coordinates* at that site.
Because E is defined by v (not by the model's outputs), this is a test of a stated abstraction, not a fitted explanation;
because the same procedure is applied to every model, it gives a common coordinate system for comparing implementations
whose raw activations live in different places.

## 2. Failure modes the framework guards against, with the real instances

1. *Attention pattern taken for mechanism.* GATE's last-layer head 0 routes 94–95 % of readout attention to the informative
   evidence pair, yet removing it changes nothing (continuation R3). Resolved here: it is a **backup** route (§3).
2. *Ablation insensitivity read as non-use.* GATE layer-1 head 2's output at the readout is individually dispensable
   (ablation changes nothing) yet carries the decision (interchange 0.66 alone; layer-1 attention at the readout 0.97).
3. *Mean-ablation read as natural necessity.* In the P-count transfer model, ablating the count subspace at the readout
   removes 0.65 of the relational gain, but a variance-matched control removes 1.33 and a random subspace 0.14; the count
   ablation also moves the output *away* from the own-question posterior (KL 0.36 → 0.71). Magnitude-based necessity is
   therefore not specific here; interchange is (§4).
4. *Probe success read as representation.* A randomly initialised TRANS network makes τ more decodable (MLP probe 0.38)
   than the trained network (0.28) — the trained network's τ "signal" is below the random-feature baseline (§5).
5. *Off-support probe meaning.* The continuation's "graded vs saturated" PP2 signature, measured off support, is not
   reproduced by an on-support measurement of the same idea (STRUCTURE_AND_EXTRAPOLATION §2).

## 3. Redundant pathways: GATE (seed 0; seeds 1–2 in GATE_FULL_CAUSAL_CIRCUIT.md)

| intervention | result |
|---|---|
| mean-ablate layer-1 head 2 (all positions) | Bayes agreement 1.000 (no effect) |
| mean-ablate layer-2 head 0 (all positions) | 1.000 |
| mean-ablate **both** | **0.883** (accuracy 0.900 → 0.799) |
| the 15 other cross-layer head pairs (matched controls) | 1.000 each |
| patch layer-1 attention output at readout q from a gate-flipped run | flip-transfer 0.97 |
| patch layer-2 attention output at readout q (layer 1 intact) | 0.08 |
| ablate layer-1 attention at readout q in both runs, then patch layer-2 head 0 at readout q | **1.000** |
| same, patching layer-2 heads 1, 2 or 3 instead | 0.000 each |

Identification statement: the gate-conditional decision reaches the readout through either of two attention routes —
primary L1-h2 and backup L2-h0 — each sufficient in its regime, neither individually necessary, ~~jointly necessary~~.
**[R: CA4]** Not jointly necessary: with both mean-ablated the model still follows 86 % of gate flips; the 0.883 agreement
is below the gate-blind level 0.890 only marginally. Resample ablation reverses single-head results (L1-h2 alone 0.62), so
conclusions depend on the operator. The supported statement is "jointly contributing, with further routes" — under mean
ablation; under resample ablation the pairs drop agreement to 0.37–0.39, below the gate-blind 0.890 [R2], because resampled
heads inject another episode's evidence values.
Ablation of either alone cannot detect use; the pair (joint ablation + conditional interchange) does.

## 4. Abstract-variable identification of the transferred count computation (Workstream A)

Transferred T4 models (frozen layers 0–1 from pretraining), site = last-layer input (`logs/pcount/causal_*.json`).
Counterfactual: cond pairs (the question's own reports kept, other questions resampled on the generator).

| arm (seed 0) | full-site interchange, readout q | full-site interchange, source tokens | **abstract count variable**, readout q | abstract count variable, source tokens |
|---|---|---|---|---|
| P-count | 0.78 | 0.15 | **0.69** (KL rec 0.66) | 0.06 |
| P-group | 0.72 | 0.23 | 0.24 | 0.04 |
| T1 (reference) | 0.10 | **0.87** | 0.08 | 0.04 |
| P-local | 0.54 | 0.44 | 0.20 | 0.06 |

(IIA on informative pairs; P-local's counterfactual behaviour is itself weakly relational, so its IIA is about a small effect.)

- In the P-count transfer model, writing *only the change of the leave-q counts* into the readout token, in the encoder's
  coordinates, reproduces 69 % of the counterfactual decision changes: the readout **count variable is causally used**,
  in an approximately additive one-hot code.
- In the T1 transfer model, the counterfactual information is at the source tokens (full-site 0.87; contrast
  subspace k = 4 [R: CA11] 0.72). ~~The same linear count abstraction fails there (0.04) … not interchangeable.~~
  **[R: CA1, R2]** The 0.04 used one encoder shared by all source slots; a per-slot linear one-hot count encoder gives
  0.82 / 0.80 / 0.66 (3 parents), against a mismatched-count floor of 0.46 and capacity-matched nulls of 0.07–0.08. The two
  curricula produce causally used count variables at different sites (readout vs source tokens), in a shared vs slot-specific
  code; the T1 data fit an own-group-count-per-source reading equally well.
- Selectivity: on count-preserving edits (every other question's six reports cyclically relabelled; counts and Bayes
  unchanged) the abstract intervention is zero by construction, and the count-subspace patch changes 1.7 % of P-count's
  answers; but the models themselves are not relabel-invariant (they change 7–15 % of answers on such edits [R: CA11], Bayes 0 %),
  i.e. both use some content beyond counts.

## 5. Map representation in TRANS (Workstream C3)

Probe accuracy for σ, τ and τσ (chance 1/6) from the final-layer [readout-mean, episode] representation
(`logs/trans/eval_*.json`, `probe_controls.json`):

| representation | σ linear / MLP | τ linear / MLP | τσ linear / MLP |
|---|---|---|---|
| exact posterior MAP (statistical ceiling) | 0.617 | 0.626 | 0.394 |
| raw input one-hot | 0.17 / 0.51 | 0.17 / 0.49 | 0.16 / 0.17 |
| random network | 0.27 / 0.37 | 0.28 / 0.38 | 0.16 / 0.18 |
| trained, full run 12k steps | **0.63 / 0.60** | 0.21 / 0.28 | 0.16 / 0.20 |

σ is represented at the statistical ceiling; τ is represented *below* the random-network baseline (training removed it);
the composite is not represented. See TRANS_COMPOSITION_RESEARCH.md for the causal tests.

## 6. Invariance under reparameterisation

The continuation's R1 (quotient-game attribution over exact pattern classes is invariant under exact head splits) applies
to the GATE redundancy result: the ~~jointly necessary~~ contributing set {L1-h2, L2-h0} [R: CA4] is a set of *pattern classes* (no two GATE heads
share a pattern), so the redundancy statement is split-invariant. The abstract-variable interchange is invariant under any
invertible linear reparameterisation of the site (the encoder is refitted in the new coordinates), but not under nonlinear
ones; its negative result for T1 is therefore relative to the linear alignment class (continuation I7).
