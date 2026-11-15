# Alternative native decision algorithms (Workstream A, with D1)

Investigation XIII-B, Phase 1.5 Step 1. **[dev]** = XIII seed-20 models (uP160–165) and the earlier pools; **[conf]** = frozen seed-21
replicates (FROZEN_CONFIRMATION_AND_STATISTICS.md). All interventions are natural interchanges (class II: activations of a real donor
episode) unless stated otherwise.

## 1. Two families of on-support edits

- **Relational pairs** (`src/relpairs.relational_pairs`). The donor differs from the base only in one group's source pair at one question
  q' ≠ q (agree ↔ disagree). The values at q are identical, so the Bayes change is purely relational: it acts through W_h and, in 11% of
  pairs, through e_h.
- **Likelihood pairs.** One source value **at q** changes, which moves candidate evidence L_g(y).

Fidelity slope: the observed change in centred log-probabilities is regressed on the Bayes change. 1 means the patched component carries
the whole change; 0 means none of it.

## 2. Where the decision is formed (`scripts/locate.py`, `scripts/path_chain.py`) [dev]

**Availability at the readout token**, measured by transplanting the donor's residual state at the readout token:

| model | candidate evidence after layer 0 (lik resid1@RO) | candidate evidence at layer-2 input (lik resid2@RO) | relational evidence at layer-2 input (rel resid2@RO) |
|---|---|---|---|
| uP160 | **0.84** | 0.91 | 0.49 |
| uP163 | **0.85** | 0.90 | 0.44 |
| uP164 | 0.53 | 0.56 | 0.34 |
| uP161 | 0.00 | 0.02 | 0.27 |
| uP165 | 0.00 | 0.03 | 0.29 |

**Causal routes of the relational effect.** The upstream intervention is the edited group's layer-0 outputs at q', taken from the donor;
it transfers 0.86–0.95 of the effect in every model. The table gives the share of that transfer lost when a downstream component is frozen
to its base value:

| frozen component | uP160 | uP163 | uP164 | uP161 | uP165 |
|---|---|---|---|---|---|
| L1 attention → q source tokens (route Q) | 0.04 | 0.03 | 0.24 | **0.70** | **0.65** |
| L1 attention → readout token (route R) | **0.55** | **0.56** | 0.37 | 0.28 | 0.38 |
| L1 MLP at readout | 0.37 | 0.32 | 0.21 | 0.10 | 0.13 |
| L2 attention at readout | 0.68 | 0.63 | 0.90 | **1.00** | **0.99** |
| L2 MLP at readout | **0.76** | **0.78** | 0.63 | 0.27 | 0.34 |
| L2 attention + MLP at readout | 0.97 | 0.91 | 0.99 | 1.00 | 1.00 |

Shares are not additive: the components lie on serial and overlapping paths.

**Reading** [dev]. The "pre-assembled" label from XIII is inaccurate. In uP160 and uP163 the decision is **not** complete before layer 2:
freezing layer 2 at the readout token removes 91–97% of the relational effect. What differs is the routing.

- **Early-readout class** (uP160, uP163 and seven earlier models).
  - Candidate evidence: the layer-0 readout heads read the values at q directly (mass 0.8–0.95 on the q source tokens).
  - Relational evidence: layer-0 agreement features at every question are pooled into the readout token by broad layer-1 readout heads.
    These heads put 0.7–0.8 of their mass on other-question tokens, with per-group weights uncorrelated with Bayes responsibilities
    (|corr| ≤ 0.1; `scripts/attn_profile.py`).
  - Combination: both are combined *at the readout token* by layer 2, mostly by its MLP.
- **Late-readout (mixture) class** (uP161, uP165 and eleven earlier models).
  - Candidate evidence stays at the q source tokens until layer 2.
  - Relational evidence is written by layer-1 attention *into the q source tokens* (route Q).
  - Layer-2 attention at the readout token reads the per-group candidate likelihoods (values) with relationally determined weights.

## 3. What operation is performed (`scripts/fit_family.py`, `scripts/algorithm_discrimination.py`) [dev]

Two explicit algorithm families were fitted to each model's centred log-probabilities, with held-out test R²:
- **MIX**: p(y) ∝ Σ_g w_g L_g(y)^τ with log w_g = α log W_g + κ log f_g. This is the exact Bayes form at α = κ = τ = 1 with f_g = S − e_g.
- **POE**: log p(y) = Σ_g c_g log L_g(y) + const, with reliability-dependent exponents. This is log-linear pooling.

| | Bayes | uP160 | uP163 | uP164 | uP161 | uP165 |
|---|---|---|---|---|---|---|
| MIX R² | 0.9999 | 0.985 | 0.983 | 0.986 | 0.995 | 0.995 |
| POE R² | 0.827 | 0.836 | 0.837 | 0.828 | 0.824 | 0.823 |
| MIX (α, τ, κ) | 1.01, 1.00, 1.62 | 0.95, 0.98, 1.30 | 0.93, 0.98, 1.30 | 0.95, 1.00, 1.36 | 1.02, 1.00, 1.48 | 1.00, 1.00, 1.46 |

κ on the Bayes posterior itself is 1.62 because log f_g is floored at ε = 1e-3. Lower κ means softer eligibility exclusion.

**Eligibility variable.** On on-support edits that flip e but not A1 (D1/D2), every model is closest to the E rule:
- distance to B is 0.24–0.28 (early) and 0.10–0.11 (late);
- distance to A1 is ≥ 1.0.

**Approximation errors are class-specific.** Errors relative to Bayes correlate:
- early–early: 0.64–0.72;
- early–late: 0.06–0.35;
- late–late: 0.18–0.30.

The incomplete-pretraining uP162 correlates 0.41–0.67 with the early models.

**Answer to "same algorithm or different?"** At the level of the input-output map, both classes implement the **same** computation: a
posterior mixture over "which group is reliable", with E-based eligibility, reliability weighting by W, and candidate likelihoods. Neither
class implements log-linear pooling.

At the level of implementation they are **genuinely different**:
- the location of candidate and relational evidence (readout token vs source tokens);
- the operator that combines them (layer-2 MLP at the readout token vs layer-2 attention-softmax weighting of per-group values);
- a class-shared approximation signature (softer exclusion, larger and correlated errors in the early class).

This is a statement about two implementations of one function family. It is **not** a claim that they compute the same intermediate
variables: for example, no early-class layer carries attention weights that resemble responsibilities.

## 4. Continuous class structure (`scripts/class_features.py`; 26 development models) [dev]

| feature | distribution |
|---|---|
| lik_RO1 (candidate evidence at the readout token after layer 0) | **bimodal with a tight upper cluster**: 0.82–0.83 in 8 models (all without a mixture head); 0.00–0.15 in 11 models; an intermediate band 0.20–0.52 in 7 models (uP162 0.27, uP164 0.52, yP131 0.25, wP143 0.20, vP153 0.29, vP154 s11 0.25, vP156 0.38) |
| layer-2 maximum q-mass | bimodal: 0.16–0.23 vs 0.86–0.97; every intermediate-lik_RO1 model has a high-mass head |
| route_Q (share via layer-1 attention into the q tokens) | ≥ 0.45 in all 11 late-cluster models; ≤ 0.28 in all 8 early-cluster models; intermediate models 0.08–0.62 |

So the structure is **a discrete early-readout cluster plus a continuum of mixture-head models**. Within the continuum, candidate evidence
is partly read early and the mixture head carries less of the decision. uP164 (lik_RO1 0.52, route_Q 0.20) is a **hybrid**, not a
thresholding accident.

Seed replication in development: both seeds of vP150 are early (0.83 / 0.82); both seeds of xP81 are late (0.01 / 0.00); yP130 0.15 / 0.01;
vP154 0.08 / 0.25. Q2 freezes the prediction for new seeds.

## 5. Necessity of layer 1 (graded and resampling ablations, `scripts/l1_graded.py`) [dev]

**Behavioural residual** = relational sensitivity remaining after ablation / intact.

| model | L1 attention, mean ablation 0.5 / 1.0 | L1 attention, resample 0.5 / 1.0 | L1 MLP, mean 1.0 | L1 attention + MLP, mean 1.0 |
|---|---|---|---|---|
| uP161 | 0.51 / 0.00 | 0.25 / −0.02 | 0.32 | 0.00 |
| uP165 | 0.53 / 0.00 | 0.22 / 0.00 | 0.43 | 0.00 |
| uP160 | 0.62 / 0.03 | 0.39 / 0.05 | 0.36 | 0.01 |
| uP163 | 0.61 / 0.03 | 0.37 / 0.09 | 0.29 | 0.00 |

- In the mixture models, the mixture heads keep their q-token mass (0.96–0.99) under every ablation, but their alignment with Bayes
  responsibilities falls with the residual: corr 0.90 → 0.02 under full attention ablation.
- No ablation revealed a hidden alternative route: the residual decays monotonically to about 0 when layer-1 attention is removed. Accuracy
  then falls to the candidate-likelihood-only level (0.57–0.65).
- Resampling is more destructive than mean ablation at equal strength. Part of the mean-ablation residual at 0.5 is therefore a property
  of the mean, not preserved computation.

## 6. Confirmation

See FROZEN_CONFIRMATION_AND_STATISTICS.md (Q1–Q3).
