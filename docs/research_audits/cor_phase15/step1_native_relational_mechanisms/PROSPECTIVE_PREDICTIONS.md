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

(Filled in after the runs.)

## Part III — Addenda

(None yet.)
