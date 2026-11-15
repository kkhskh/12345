# Causal-control and identifiability audit (Workstream E)

COR Phase 1.5, Step 1, Investigation XIII-D. This report covers interpretability results on small trained transformers. It does not validate COR and makes no claim about the O/D operators.

XIII-C left five control problems open. This report takes each in turn:

- E1: the eligibility mediation result could be an artefact of patching the edited tokens themselves.
- E2: the decoy (SHUFD) row control was scored against the wrong reference, and NOOP had no explicit float tolerance.
- E3: it was unknown whether the parallel routes Q/R/O are independent, redundant or additive.
- E4: the source of the early-class decision error was unidentified.
- E5: the identifiability results needed to be preserved.

**Unit roles.**

- Development units: uP161 s21 (mixture) and s22 (early), plus uP160 s21 and s23.
- Confirmation units: frozen in freeze 2 (commit `971f07f`). No confirmation unit appears in any XIII-D development output.
- Disclosed caveat: one row unit, uP161 s20, was analysed with the old row scoring in XIII-C development (COMPUTE_LOG 18:49:50Z).

## E1. Eligibility mediation at sites that exclude the edited tokens

**Problem.** The XIII-C site ALL2 patched every token, including the edited source tokens. A DAS effect there could be pure re-insertion of the edit.

**Change.** `d_das2.py` adds two sites:

- **ALLX2:** every layer-2-input token except the edited source tokens.
- **ALLXQ2:** additionally excludes every token of the edited questions.

It also adds a graded rule, G: e = agree_q·(n/5)^4. G competes with:

- E: the exact eligibility rule;
- A1: other-question agreement only;
- A2: own-question only.

**Metrics.**

- own_share: the fraction of the full interchange effect recovered by the k = 8 DAS subspace.
- R²_X: how well rule X predicts the per-episode DAS interchange effect, per edit family. The families are E5, E3, QF5 and QF4.

### Development (ALLX2, k = 8)

| unit | family | DAS own | SRCFREE own | R²_E | R²_A1 | R²_A2 | R²_G |
|---|---|---|---|---|---|---|---|
| uP161 s22 (early) | E5 | 0.77 | 0.57 | 0.84 | 0.83 | 0.61 | 0.38 |
| | E3 | 0.69 | 0.09 | 0.63 | 0.62 | 0.01 | 0.57 |
| | QF5 | 0.79 | 0.68 | 0.87 | 0.37 | 0.87 | 0.86 |
| | QF4 | 0.81 | 0.53 | 0.94 | 0.94 | 0.22 | 0.38 |
| uP161 s21 (mixture) | E5 | 0.91 | 0.52 | 0.92 | 0.92 | 0.53 | 0.31 |
| | E3 | 0.69 | 0.21 | 0.78 | 0.77 | 0.00 | 0.64 |
| | QF5 | 0.78 | 0.62 | 0.82 | 0.23 | 0.81 | 0.81 |
| | QF4 | 0.55 | 0.40 | 0.66 | 0.66 | 0.10 | 0.20 |

**ALLXQ2 (development only): the two classes differ.**

- In the early unit the effect survives removal of the edited questions: DAS own 0.62–0.78.
- In the mixture unit the q-flip families vanish: QF5 0.30, QF4 0.03.

The early class carries eligibility information on tokens of *other* questions. The mixture class carries it on the edited question's own tokens. Because this was seen only in development, it is reported as an observation, not a confirmed result.

### Confirmation (P6, E1 arms)

| unit | family | DAS own | SRCFREE own | R²_E | R²_A1 | R²_A2 | R²_G |
|---|---|---|---|---|---|---|---|
| uP164 s21 (early) | E5 | 0.89 | 0.56 | 0.87 | 0.87 | 0.69 | 0.47 |
| | E3 | 0.53 | 0.50 | 0.30 | 0.30 | 0.00 | 0.31 |
| | QF5 | 0.87 | 0.67 | 0.91 | 0.38 | 0.90 | 0.90 |
| | QF4 | 0.89 | 0.56 | 0.94 | 0.94 | 0.21 | 0.37 |
| uP165 s20 (mixture) | E5 | 0.89 | 0.55 | 0.96 | 0.95 | 0.59 | 0.37 |
| | E3 | 0.71 | **0.92** | 0.79 | 0.79 | 0.00 | 0.58 |
| | QF5 | 0.80 | 0.61 | 0.85 | 0.29 | 0.85 | 0.84 |
| | QF4 | 0.58 | 0.38 | 0.67 | 0.67 | 0.14 | 0.25 |

The rule comparison replicates. E beats A2 on E5, E3 and QF4, beats A1 on QF5, and beats G on E5 and QF4, by ≥ 0.15 R² in both units. One weakness: in uP164 s21, E and G are tied on E3, at a low level (0.30 / 0.31).

**The pre-declared specificity criterion failed in the mixture unit.** The criterion was a mean excess of DAS own-share over the source-free control of ≥ 0.15:

- uP164 s21: 0.23, pass.
- uP165 s20: 0.13, fail. In this unit's E3 family the source-free interchange alone recovers 0.92 of the effect, more than DAS (0.71).

**P6 is therefore FALSIFIED.** The eligibility rule is supported at sites without the edited tokens. But in one of two units, an eligibility-specific subspace is not distinguishable from a source-free interchange. For the E3 family of that unit, the effect is carried by tokens that the source-free control also patches. That family's mediation cannot be attributed to an eligibility subspace.

## E2. Corrected decoy scoring and explicit NOOP tolerance

**Change.** `d_rows.py` adds two arms:

- **SHUFD_vs_borrowed:** the decoy-shuffled row interchange is scored against the change that the borrowed pair would itself produce, `dB[perm_d]`. XIII-C scored it against the base pair's change.
- **SHUF_vs_borrowed:** the same correction for the plain shuffle.

**NOOP** now reports max |Δlogit| against the unmodified forward pass, with an explicit tolerance of 1e-5 and a within_tolerance flag.

| unit | mixture heads | TRUE R² (slope) | SHUFD R² | SHUFD vs borrowed R² | SHUF vs borrowed R² | NOOP max abs Δlogit |
|---|---|---|---|---|---|---|
| dev uP161 s21 | 1, 3 | 0.961 (0.93) | 0.003 | 0.005 | 0.001 | 7.2e-7 |
| dev uP160 s23 | 2, 3 | 0.963 (0.89) | 0.002 | 0.000 | 0.000 | ≤ 1e-6 |
| conf uP161 s20 | 3 | 0.898 (0.72) | 0.014 | 0.003 | 0.001 | 9.5e-7 |
| conf uP165 s23 | 1 | 0.911 (0.79) | 0.014 | 0.001 | 0.001 | 7.2e-7 |

**The true row interchange explains ≥ 0.90 of the change.** Shuffled rows explain nothing, whether scored against the base pair or against the borrowed pair. So no generic signal transfers with a row; the effect is specific to the correct row content. NOOP is within tolerance by more than an order of magnitude. The P6 rows arms pass in both confirmation units.

## E3. Parallel routes: independent, redundant or additive?

**Change.** `d_routes2.py` uses two complementary methods.

- **Path patching** on small and large (two-question) edits, with all interaction terms.
- **Functional blocking** of each subset of three routes, measuring KL, accuracy and relative slope:
  - R: layer-1 readout→source attention;
  - Q: layer-1 q→other attention;
  - O: layer-2 readout→other attention.

**Confirmation (P8).** "Single open" is the relative slope with only that route left unblocked, divided by the unblocked slope.

| unit | path-patching sufficiency Q / R / O (small) | max abs interaction (small) | single open Q / R / O | block R: slope (KL) | block Q | block O |
|---|---|---|---|---|---|---|
| uP165 s20 (mixture) | 0.66 / 0.27 / 0.02 | 0.04 | 0.39 / 0.18 / 0.03 | 0.36 (0.135) | 0.23 (0.211) | 0.95 (0.003) |
| uP163 s22 (early) | 0.25 / 0.30 / 0.52 | 0.07 | 0.39 / 0.30 / 0.38 | 0.39 (0.207) | 0.77 (0.026) | 0.76 (0.084) |
| uP164 s21 (early) | 0.11 / 0.43 / 0.50 | 0.04 | 0.16 / 0.51 / 0.32 | 0.31 (0.404) | 0.82 (0.034) | 0.63 (0.151) |

**The two methods give different pictures.**

- **Path patching looks additive:** the shares sum to the total and the interactions are ≤ 0.07.
- **Blocking shows interdependence:** no single route, left open alone, supports more than 0.51 of the behaviour. Blocking R costs 0.6–0.7 of the slope even where path patching gives R only 0.27–0.43.

The routes are additive in how a perturbation propagates. Functionally, they are not independent backups: each needs the others' normal operation. "Redundant" is ruled out in every unit. "Independent" is ruled out by the blocking.

**Class difference.** Route O carries nothing in the mixture unit and about half of the effect in the early units. That replicates development (P8 CONFIRMED).

## E4. Source of the early-class decision error

**Hypothesis tested.** H_pos: the Bayes-invariant MOVE error comes from the positional asymmetry of the layer-1 readout attention rows. MOVE is the error when an agreement moves between two other questions, which leaves Bayes exactly unchanged; the invariance is checked to 1.1e-16.

**Intervention.** Symmetrise the two questions' readout rows (SYM1, SYM2, SYM12), or permute them (PERM1, PERM12).

| unit | base MOVE | SYM1 MOVE (reduction; KL) | SYM12 MOVE | PERM1 MOVE | row asymmetry (L1, L2) |
|---|---|---|---|---|---|
| dev uP161 s22 | 0.048 | 0.085 (−0.79; 0.079) | 0.083 | 0.180 | 0.52, 0.59 |
| dev uP160 s21 | 0.028 | (−1.06; 0.037) | | | |
| dev uP161 s21 | 0.028 | (−1.49; 0.19) | | | |
| conf uP164 s22 | 0.0275 | 0.0293 (−0.06; 0.047) | 0.027 | 0.138 | 0.43, 0.50 |
| conf uP163 s20 | 0.0359 | 0.0933 (−1.60; 0.106) | 0.095 | 0.174 | 0.48, 0.45 |
| conf uP160 s20 | 0.0274 | 0.0295 (−0.08; 0.023) | 0.028 | 0.093 | 0.40, 0.39 |

**H_pos is rejected (P7 CONFIRMED, as a predicted negative).**

- Symmetrising never reduces MOVE; it leaves it unchanged or increases it.
- Symmetrising always increases KL to the posterior.
- Permuting the rows increases MOVE 2.4–5×.

The asymmetric rows are part of the *correct* computation, not the source of its error.

**The source of the early-class MOVE error remains unidentified** (see the open questions document). What has been excluded:

- positional asymmetry of the readout rows (this test);
- the parametric decision families of XIII-C, which miss it by construction.

## E5. Identifiability results preserved

The following XIII-C results are unchanged, and nothing in XIII-D bears on them:

- the ρ non-identifiability proof (`math_identifiability.py`);
- the E identification;
- the exact LOCALQ gap (`c_math.py`, review-1 exact LOCALQ).

They are a dependency of this package, listed in `logs/DEPENDENCIES.json`, not re-derived here.

## Verdicts (freeze 2, E workstream)

- P6 FALSIFIED: the E1 source-free excess fails in uP165 s20; every other P6 arm passes.
- P7 CONFIRMED: the predicted negative.
- P8 CONFIRMED.
