# Prospective predictions (Investigation IX)

## Part I — frozen before any confirmation outcome was observed

**Freeze statement.** Part I and `logs/PROTOCOL_HASHES.txt` were committed before any T4 transfer, probe, reader or
evaluation touched the confirmation parents yP130–yP135. They were also committed before any generalization reader with
seed ≥ 10 or the confirmation evaluation streams had been run.

- The confirmation parents were pretrained under `STAGE0_COHORT_FREEZE.md`. Only their `pre.json` was read: the P-count
  pretraining count accuracy.
- That pretraining evaluation is the declared **completeness covariate**, fixed here:
  - **complete**: yP130 (0.99995), yP131 (0.99999), yP132 (1.0), yP134 (0.99994);
  - **incomplete**: yP133 (0.517), yP135 (0.555).
- Every method choice, metric and threshold below was set on development material only: xP80 (incomplete) and xP81
  (complete, Investigation VI), the Investigation VIII lineages zP100–107 (used retrospectively), `eval_dev.npz`, and
  generator seeds 0 and 1.
- Part I is not edited after the freeze. Later changes go into Part III.

### I.1 Material and code

- **Evaluation set.** `logs/eval_conf9.npz` is built by `scripts/build_eval_conf9.py` from stream **19_900_000**, never
  used before, with Investigation VII's `build_eval` unchanged. Exact Bayes evaluated as a model gives max |E| = 0.
- **Native experiments.** For each parent yP130–yP135, run the commands below; they are listed in
  `logs/queue_conf_native.txt`.
  1. `scripts/decode_compose.py <parent> 3 logs/eval_conf9.npz` and the same with `2`.
  2. `scripts/native_reader.py <parent> mlp_q 8000 0 label 3 logs/eval_conf9.npz`.
  3. `scripts/native_train.py <parent> c1 4000 0 label logs/eval_conf9.npz`.
  4. `MECH_EVAL=logs/eval_conf9.npz scripts/mech_native.py <parent> c1 4000 0 _eval_conf9`.
- **Generalization experiments** (`logs/queue_conf_gen.txt`). Run `GEN_EVAL_OFFSET=5000000 scripts/gen_readers.py`
  with:
  - readers ds_sum, attn and mlp;
  - ID = 0 for ds_sum and attn, and ID = 0 for mlp (its identity is implicit in the slot);
  - HOLD = 1;
  - 12000 steps;
  - seeds 10, 11 and 12.
- **Metrics.** As defined in `src/native.py:summarize`, mandate §11.2.
  - **absE** is the per-episode mean absolute response error and is primary.
  - **E** is the signed mean.
  - |E| R2∪R3 is the n-weighted mean of the per-episode absE.

  Generalization uses KL to exact brute-force Bayes on 30,000 decisions (`indist_bigN`): KL_pair01 on decisions where
  groups 0 and 1 are both eligible (the held-out pair), and KL_S2_not01 on the other S ≥ 2 decisions. Size and semantics
  shifts use 6,000 decisions per setting.
- **Unit of inference.** The unit is the parent, or the seed for the generalization readers. Exact one-sided sign or
  binomial counts are reported. No pooling over decisions is used for inference.

### I.2 Predictions

**CP1 (native information, decode-then-compose).** Linear probes read counts from readout tokens and reports from source
tokens of the frozen final pretrained states (UPTO = 3). Their output is plugged into the exact formula.
- For **all 4 complete** parents: KL ≤ 0.005 and absE_R1 ≤ 0.05.
- For **both incomplete** parents: KL ≥ 0.03.

**CP2 (completeness separates the original transfer's KL).** The largest C1 KL among complete parents is below the
smallest C1 KL among incomplete parents.

**CP3 (the R1 failure persists when the information is present).** For the C1 transfer, E_R1 ≤ −0.20 in **at least 3 of
4** complete parents.

**CP4 (a native-state-only reader uses the relation).** Take mlp_q on the frozen final pretrained states: readout token
plus the 6 source tokens of question q, labels, 8000 steps, no side information.
- On complete parents, |E_R1| ≤ 0.15 in **at least 3 of 4**.
- absE_R1(mlp_q) < absE_R1(C1) in **4 of 4**, paired by parent.

**CP5 (mechanism of the C1 failure: per-group routing away from perfect groups).** Holds in **at least 5 of 6** parents:
- (a) in the C1 block, the head-averaged attention mass from readout token q onto a group's 12 source tokens, averaged
  over natural decisions with that group's count = 5, is < 0.6 × the mass at count = 4;
- (b) freezing the attention pattern to its pre-edit value reduces the magnitude of the R2∪R3 response (the
  n-weighted mean of the R2 and R3 "full" responses) by ≥ 50 %;
- (c) under the same freeze, the R1 response is less negative than the full R1 response.

**GP1 (held-out eligible pair, identity-free readers).** For ds_sum and attn with ID = 0 and HOLD = 1, KL_pair01 ≤ 0.005
in **3 of 3** seeds each.

**GP2 (held-out pair, concatenation MLP).** For mlp with HOLD = 1, KL_pair01 ≥ 5 × KL_S2_not01 in **3 of 3** seeds.

**GP3 (no size extrapolation despite exact representability).** For ds_sum and attn with ID = 0, KL at G = 4 is ≥ 0.01 in
**3 of 3** seeds each. Theorem 1 shows a fixed (φ, ρ) is exact for every G, so this prediction says the *learned* readers
do not find that solution.

**GP4 (semantic shift: partial copying).** For all three readers, KL at β = 0.7 is ≥ 0.10 in **3 of 3** seeds. Theorem 3
says the hard gate stops being the Bayes rule, and readers trained at β = 1 are predicted to keep it.

### I.3 Rules

- **Intention to treat.** Every listed run is reported. A crashed run is rerun once with the identical command, and the
  crash is disclosed.
- **Development-only.** These are not confirmed by Part I:
  - the native-learning variants (full_c1, full_keep3, LoRA, unf0, keep3, keep3x2, deep2, scratch3);
  - learning-dynamics trajectories;
  - the native mechanism of the successful native reader.

  If a native-learning method is promoted to confirmation, a Part I addendum is committed before its confirmation runs.

## Part II — Results

**Integrity.**
- Part I is byte-identical to its freeze commit `1ee69c2`, and Addendum A1 to commit `55eae73`. Verify with `git show`.
- Verdicts are computed by `scripts/conf9_evaluate.py` and `scripts/conf9_evaluate_A1.py`. Their outputs are
  `logs/confirmation_results.json` and `logs/confirmation_results_A1.json`.
- **Deviations** (`logs/COMPUTE_LOG.txt`):
  - background runs were stopped several times when a reply ended (first misread as environment restarts);
  - each lost run was rerun once with its identical command, and none was dropped;
  - `scripts/mech_native.py` gained post-freeze additions (disclosed in A1). The c1 path is unchanged.

| prediction | verdict | evidence |
|---|---|---|
| **CP1** native information is linearly sufficient in complete parents, insufficient in incomplete ones | **confirmed** | decode-then-compose KL 0.0001 / 0.0000 / 0.0000 / 0.0001 with absE_R1 0.000 (complete); KL 0.326 and 0.184 (incomplete); count decoding 1.000 vs 0.62–0.68 |
| **CP2** completeness separates C1 KL | **confirmed** | complete 0.043 / 0.076 / 0.032 / 0.074 < incomplete 0.192 / 0.170 |
| **CP3** C1 fails R1 even with the information present | **confirmed** (4/4) | E_R1 −1.02 / −0.91 / −0.76 / −0.97; incomplete −0.85 / −0.82 |
| **CP4** a native-state MLP reader uses the relation | **falsified** | mean-E criterion met in 1/4 (E_R1 −0.42 / −0.34 / +0.00 / −0.44). Paired improvement over C1 holds in 4/4 (absE 0.44–0.52 vs 0.81–1.05). The development success on xP81 did not replicate at 8k label steps |
| **CP5** per-group de-attention mechanism of C1 | **falsified as frozen** (2/6 meet all three) | (a) perfect/n=4 mass ratio < 0.6: 3/6 (0.42, 0.33, 0.43; the misses are 0.61, 0.63, 0.67). (b) attention freeze removes ≥ 50 % of the R2∪R3 response: **6/6** (53–89 %). (c) the frozen R1 response is less negative: **5/6** |
| **GP1** identity-free readers generalize to the unseen eligible pair | **confirmed** (6/6) | KL_pair01 0.0009–0.0027 |
| **GP2** the concatenation MLP does not | **confirmed** (3/3) | 0.37–0.40 vs 0.005 on seen pairs |
| **GP3** no size extrapolation | **confirmed** (6/6) | KL at G = 4: 0.025–0.053 |
| **GP4** partial copying breaks every reader | **confirmed** (9/9) | KL 0.20–0.37 |
| **NP1** native fine-tuning repairs R1 | **confirmed** | paired absE_R1 better 4/4 (0.21 / 0.30 / 0.21 / 0.35 vs 1.05 / 1.01 / 0.81 / 1.06). \|E_R1\| ≤ 0.2 in 3/4 (−0.09, −0.18, +0.14; +0.26 overshoot). KL ≤ 0.005 in 3/4 (0.002 / 0.003 / 0.003 / 0.007) |
| **NP2** the development pathway (other groups' q-tokens) recurs | **falsified** (0/4) | others_q patch +0.01 to +0.08 (positive but small). The complement keeps the positive credit (+0.11 to +0.42), so these tokens are not necessary on new parents |
| **NP3** the frozen prefix has no cross-group propagation | **confirmed** (4/4) | others_src share −0.02 to −0.04 |
| **NP4** separable first, then cross-group, in native fine-tuning | **falsified** (1/4) | yP130 min −0.77 at step 546 → −0.09. yP131 min −0.47 at 1803 → −0.17. yP132 / yP134 shallow early minima (−0.30, −0.24) |

**Exploratory (after unblinding, not a confirmation).** Per-parent R1 credit in the fine-tuned models is located
differently on each parent:
- yP134: through the other groups' source tokens at all questions (0.35 of 0.54);
- yP132: split between readout tokens (0.24) and source tokens (0.21);
- yP130/yP131: no tested token set carries it, and the parts sum to far less than the whole, so it is distributed.

**What does recur, 4/4, is attention dependence:** freezing the first downstream block's attention removes the R1 credit
(0.22 → −0.37, 0.17 → −0.26, 0.43 → 0.07, 0.54 → 0.06).

## Part III — Addenda

- `PROSPECTIVE_PREDICTIONS_ADDENDUM_A1.md`: native fine-tuning predictions NP1–NP4, frozen before their runs.
- Disclosure: `scripts/mech_native.py` post-freeze additions (A1). Added `scripts/mech_routing2.py` (exploratory).
