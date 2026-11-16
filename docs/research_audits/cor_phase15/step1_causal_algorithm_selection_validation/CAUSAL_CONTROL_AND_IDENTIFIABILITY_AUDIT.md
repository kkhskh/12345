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

**Caveat (review 2, F11; the XIII-C M6 objection persists).** The DAS rule-discrimination R² values at ALLX2 are at or below the model's own *behavioural* R² in every family. For example, in uP164 s21:

| family | DAS R²_E / R²_A2 | behaviour R²_E / R²_A2 |
|---|---|---|
| E5 | 0.87 / 0.69 | 0.914 / 0.685 |
| E3 | 0.30 (E) | 0.606 (E) |

So "E beats A2" restates the model's behaviour; it is not a property of the subspace. The supported statement is narrower: an 8-d subspace at sites free of edited tokens transmits the model's own eligibility response, with own-share 0.53–0.89. Patch-minus-behaviour margins are not reported.

Also, R²_E = R²_A1 exactly in E3 and QF4, so E and A1 are discriminated only by QF5.

**The pre-declared specificity criterion failed in the mixture unit.** The criterion was a mean excess of DAS own-share over the source-free control of ≥ 0.15:

- uP164 s21: 0.23, pass.
- uP165 s20: 0.13, fail. In this unit's E3 family the source-free interchange alone recovers 0.92 of the effect, more than DAS (0.71).

**P6 is FALSIFIED as scored, but the specificity criterion is ill-posed (review 1, F4).**

- **own_share is not a fraction of the effect.** It is a regression slope of the patched change on the model's own change, and it is not normalised by the size of the patched change. In uP165 s20 E3, the source-free patch moves the output about 9 times as much as DAS (|d| 1.70 vs 0.19) with weak alignment.
- **The threshold is not anchored.** An ideal carrier already scores 0.40–0.50 on this control (`reviews/review1_scripts/srcfree_e1.py`), so the ≥ 0.15 excess threshold is not tied to anything.
- **Both arms patch the same tokens.** The source-free arm uses an unrelated donor at the same site, so the earlier "token-location" reading is withdrawn.

The falsification is therefore of an ill-posed criterion. It is not evidence against eligibility specificity.

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

**The true row interchange explains ≥ 0.90 of the change, and decoy-shuffled rows scored against the base pair explain ≤ 0.014.**

**The borrowed-pair arm (SHUFD_vs_borrowed) has no power by construction (review 1, F7).** A perfect row carrier also scores R² 0.003 on it, because rows act through the base episode's values. Its passing therefore adds nothing; the corrected scoring did not produce a usable control. NOOP is within tolerance by more than an order of magnitude.

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

The routes are additive in how a perturbation propagates. Under functional blocking, no single route supports the behaviour.

**Caveat (review 2, F11).** The blocking uses −1e9 attention masks with renormalisation. These are off-distribution interventions, and the single-route-open arms block two routes at once. BLOCK_R also masks *all* 36 sources, including q's own, so it removes non-relational candidate evidence as well (KL 0.135–0.404).

The discrepancy between path patching and blocking is therefore equally consistent with "blocking damages the substrate". P8's "interdependent" is a blocking-specific description. Resample-ablation of each route's attention output was not run.

**Class difference.** Route O carries nothing in the mixture unit and about half of the effect in the early units. That replicates development (P8 CONFIRMED as scored).

**Limit (review 1, F9).** The P8 criterion rules out *redundancy*: no single route suffices. It cannot distinguish "independent additive" from "interdependent" routes, because the single-open arms are off-distribution (KL up to 0.40).

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

**P7 CONFIRMED as scored, but the test cannot exclude H_pos (review 1, F8; review 2, F11).**

- The criterion requires SYM1 to raise KL, so the intervention's damage is part of the success condition. SYM1 raises KL 5.8–22×, and a damaging intervention can raise MOVE generically.
- SYM equalises only the attention *weights*. The value vectors keep their positional content, so H_pos was tested only in a weights-only form.
- **Number corrections:**
  - SYM2 does reduce MOVE, by 7.2% (uP164 s22) and 4.8% (uP160 s20); SYM12 reduces it by 1.7% (uP164 s22).
  - Development SYM2 on uP161 s21 leaves KL at base.
  - PERM1 raises MOVE 3.4–5.0× in confirmation (3.8–7.8× in development).

Supported statement: SYM1 did not reduce MOVE, at a KL cost of 0.02–0.10.

The symmetrising interventions also damage the decision (SYM1 KL 0.023–0.106). The supported statement is therefore only that symmetrising the rows did not reduce MOVE and did damage accuracy (review 2, F11). It does not follow that the asymmetric rows are part of the correct computation.

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

- P6 FALSIFIED: the E1 source-free excess fails in uP165 s20, but that criterion is ill-posed (see above). The E2 borrowed-pair arm passes but has no power.
- P7 CONFIRMED as scored. It cannot exclude H_pos.
- P8 CONFIRMED as scored. It establishes non-redundancy only.
