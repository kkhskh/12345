# Causal equivalence and mechanisms (Workstreams A and B)

**Systems.**
- **Development:** the five Investigation IX full-fine-tuned models (xP81, yP130, yP131, yP132, yP134; "full_c1"), their
  frozen-prefix originals ("C1"), and two seed-1 fine-tune replicates (yP130/s1, xP81/s1).
- **Confirmation:** wP140–143 (`FROZEN_PREDICTIONS_AND_RESULTS.md`).

**Evidence levels.** [D] development; [C] confirmed; [F] falsified; [X] exploratory.

## 1. The question and the hypotheses

Investigation IX found that native repair recurs behaviourally (4/4) but the development model's token pathway does not
(NP2, 0/4).

| hypothesis | statement |
|---|---|
| H1 | a shared abstract variable |
| H2 | shared, but distributed differently |
| H3 | genuinely different algorithms |
| H4 | partial or unidentifiable |

`MATHEMATICAL_RESULTS.md` §1–2 constrain what can be shown.
- Two exact models are indistinguishable by any on-support input test.
- Interchange abstractions depend on the factorization. Three exact implementations of the same rule support the "set S"
  intervention, the "set e" intervention, or neither.
- Gradient search can miss an exact alignment that exists.

The analysis therefore combines three independent kinds of evidence:
1. **behavioural algorithm identification** on discriminating on-support edits;
2. **interchange abstraction** for two variables (S and the eligibility vector E), with random-subspace and
   discriminating-counterfactual controls;
3. **selectivity** of the found subspace to the input edits that define the variable.

## 2. Behavioural (functional) algorithm identification [D; F1 confirms]

`scripts/algorithm_discrimination.py`. Five candidate rules differ only in the eligibility factor (math §4). The table
gives the mean response distance on the discriminating families D1/D2 (change q-agreement of a perfect group) and R1.

| model | D1: B / A1 / A3 | D2: B / A1 / A3 | closest on D1, D2, R2, R3 |
|---|---|---|---|
| xP81 full_c1 | **0.10** / 1.07 / 1.63 | **0.11** / 1.03 / 1.79 | B, B, B, B |
| yP130 full_c1 | **0.15** / 1.10 / 1.65 | **0.14** / 1.04 / 1.79 | B, B, B, B |
| yP131 full_c1 | **0.18** / 1.06 / 1.56 | **0.19** / 0.99 / 1.69 | B, B, B, B |
| yP132 full_c1 | **0.21** / 1.10 / 1.56 | **0.22** / 1.04 / 1.68 | B, B, B, B |
| yP134 full_c1 | **0.25** / 1.14 / 1.64 | **0.27** / 1.05 / 1.71 | B, B, B, B |
| xP81 C1 | 0.61 / 0.83 / 1.44 (A2) | 0.61 / 0.81 / 1.51 (A2) | — |
| yP130 / yP131 / yP134 C1 | A1-closest | A1-closest | — |
| yP132 C1 | A2-closest | A2-closest | — |

**Reading.**
- The five repaired models are **functionally equivalent to the Bayes eligibility rule** on every discriminating
  family. They are equally far from the alternative rules, so behaviourally they implement the same relational
  algorithm.
- The test has power: the original C1 models are identified as *different* algorithms. They are closer to
  "perfect-only" (A1, ignoring agreement at q) or to a threshold rule (A2).
- An earlier version of the test contained **off-support** D1 edits (5,501 of 6,000, which create S = 0). They were found
  via the NaN of the exact reference and removed (math §3).

## 3. Causal abstraction by interchange (DAS) [D; F2–F5 confirm]

`scripts/das.py`. A learned k-dim subspace at the residual stream before layer L, over token set T, is interchanged from
a source to a base episode. Its output is compared with the high-level counterfactual of Theorem 1, using held-out pairs
with the variable changed. Here IIA is interchange argmax accuracy, KLcf is KL(counterfactual ‖ patched), and the random
control is an untrained subspace of the same k.

| model | E, L2/q7/k6: IIA / KLcf | random IIA | KL without intervention | S, L2/q7/k2: IIA |
|---|---|---|---|---|
| xP81 full_c1 | **0.967** / 0.017 | 0.49 | 0.885 | 0.55 (all tokens 0.58) |
| yP130 full_c1 | **0.954** / 0.029 | 0.50 | 0.883 | (development queue) |
| yP131 full_c1 | **0.953** / 0.023 | 0.48 | 0.878 | — |
| yP132 full_c1 | **0.915** / 0.079 | 0.49 | 0.866 | (development queue) |
| yP134 full_c1 | **0.881** / 0.122 | 0.49 | 0.872 | — |
| xP81 C1 | 0.892 / 0.103 | 0.48 | 0.844 | — |
| yP130 C1 | 0.892 / 0.133 | 0.51 | 0.844 | — |
| yP131 C1 | 0.851 / 0.182 | 0.51 | 0.795 | — |

**Site scan on xP81 (E):** L2/q7 0.967; L2/all 0.973; L3/q7 0.613. For S: L1/q7 0.39; L2/q7 0.55; L2/all 0.58; L3 0.34–0.40.

**Localization (E, L2, k = 6):**

| model | readout token only | 6 source tokens only | both (q7) |
|---|---|---|---|
| xP81 | 0.757 | 0.856 | 0.967 |
| yP130 | 0.756 | 0.867 | 0.954 |

**Discriminating counterfactuals.** These are the pairs where the Bayes and separable (A3) counterfactual decisions
differ (`scripts/das_eval.py`). The table gives IIA_B / KL_B.

| parent | full_c1 | C1 | random |
|---|---|---|---|
| xP81 | **0.962** / 0.041 | 0.822 / 0.181 | 0.21 |
| yP130 | **0.948** / 0.054 | 0.718 / 0.304 | 0.22 |
| yP131 | **0.960** / 0.036 | 0.635 / 0.396 | 0.20 |

**Selectivity** (`scripts/subspace_shift.py`) is the displacement inside the learned E-subspace under natural edits that
do (D2, CNT) or do not (D2c, CNTc) change eligibility.

| model | D2c / D2 | CNTc / CNT | D2 / CNT |
|---|---|---|---|
| full_c1 (3 parents) | 0.24–0.28 | 0.29–0.41 | 0.94–1.02 |
| C1 (3 parents) | 0.76–0.99 | 0.85–1.02 | 1.30–1.62 |

**Reading.**
1. **A shared, causally interchangeable abstract variable exists in all five repaired models: the per-group eligibility
   vector E.** It sits at the same layer boundary (the downstream input, after the fine-tuned layers 0–1) and is
   distributed over the readout and source tokens of the target question. Neither token type alone suffices (0.76 /
   0.86 vs 0.95–0.97). This **reconciles** Investigation IX's NP2 failure: no fixed token *subset* carries the R1 credit
   across parents, yet the same abstract variable is present, distributed over the same 7 tokens. This is H2.
2. **The abstraction has the per-group (I2) signature, not the pooled (I1) one.** "Set E" interchanges nearly exactly
   (0.88–0.97), "set S" only partially (0.55–0.58 on xP81). By Theorem X1 this is what a per-group m_g-like
   representation predicts. It does *not* by itself prove there is no S variable, because of the identification caveat.
3. **The repaired and original models differ in the *computation* of E, not only in its use.**
   - The repaired models' E-subspace moves strongly and selectively when eligibility changes, through either input
     route (q-agreement or counts).
   - The original C1's subspace is non-selective: it moves as much for control edits.
   - Even so, interchange on C1 gives moderate IIA (0.85–0.89 overall, 0.64–0.82 on discriminating pairs). The
     downstream block can partly reconstruct relational information from whatever is swapped.
   - **Moderate IIA without selectivity is a false-positive mode of DAS.** It is guarded against here by the
     selectivity test.
4. **Mechanistic chain for the original failure** [D].
   - The frozen prefix provides perfectness information (from counts) but no selective encoding of agreement at q.
   - The downstream block therefore implements the A1 or A2 rule behaviourally.
   - Fine-tuning the prefix creates a selective, distributed E representation at the downstream input, and the
     behaviour becomes Bayes-like.

## 4. Component-level profiles (heads, MLPs) [D, X]

`scripts/head_profile.py` patches each component's output (all tokens) with its post-edit value, in a total-effect
variant, and records the share of the response. `scripts/profile_perm.py` compares profiles across models.

**Layer-0 MLP.** It carries ≈ 100 % of the R1 response in every repaired model (share 0.91–1.09) and 0.57 in C1. It is
the first nonlinearity reading the edited source tokens. As a *total effect* of an early component this is expected and
says little about the relational step.

**Heads in layers 0–1.**
- Head indices are arbitrary across independently pretrained parents. Cross-parent comparisons must therefore quotient
  the per-layer head permutation symmetry.
- With the permutation maximized, R1 head-effect profiles of the four Investigation IX parents are 0.81–0.98 similar.
  The null (random profiles under the same maximization) has mean 0.61–0.63 and 95th percentile 0.90–0.94, so this is
  **only weak evidence** of a shared head-level organization. xP81 (Investigation VI parent) is less similar (0.32–0.61).
- **Within a parent**, head identities are inherited from pretraining. Two fine-tuning seeds of yP130 have similar
  profiles without any permutation search: R1 0.78, D1 0.87, D2 0.43, against a null centred at 0.
- **Interpretation (H5-like):** much of the implementation detail is inherited from the pretrained parent. Which heads
  are available to move agreement information is fixed by pretraining. Different parents therefore realize the same E
  abstraction through different heads and token subsets.
- No subspace-level identification inside a head was attempted.

## 5. Learning trajectories [D; F7 confirms]

`scripts/traj_alg.py` records the closest rule on D2 at each of 12 checkpoints.
- All five repaired runs start A1-like (perfect-only; 6/6 earliest checkpoints).
- They switch to B at step 813, 1211, 1211, 1803 and 4000.

The recurrent learning order is "**perfectness first, q-agreement integration later**". Investigation IX's R1-based
order (NP4, falsified) could not detect it, because A1 and B make identical R1 predictions. The timing differs widely
across parents; the order does not.

## 6. What remains open

- Head-internal and MLP-internal mechanisms of how E is computed: which components write agreement-at-q into the q7
  tokens.
- Cross-model *mapped* interchange (state alignment between models). Not run.
- Whether an S variable exists that DAS failed to find (identification caveat).
