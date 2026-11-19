# XIX pre-experiment review B: computational graph and optimization

**Reviewer.** B (computational graph and optimization). This is an AI-agent review, not a human peer review.

**Scope.** I checked every claim against the code and the saved states, not against mode names.

**Material read.**
- `sources/MANDATE_XIX.txt` §3.1–3.2, §4–7
- `DESIGN_XIX.md`
- `COMPUTATIONAL_GRAPH_AND_GRADIENT_AUDIT.md`
- `logs/graph_table19.csv`, `logs/graph_audit19.json`
- `scripts/xix_train.py`, `scripts/graph_audit19.py`, `scripts/compose19.py`, `scripts/conf_setup19.py`, `scripts/score19.py` (provenance and matching only), `scripts/distance19.py`
- `logs/conf/queue_c1.txt`, `logs/conf/queue_c2.txt`, `logs/conf/design19.json`
- `logs/DEV_NOTES19.md`
- `../step1_cross_family_relational_foundations/src/maskfwd.py`

**Where `maskfwd.py` comes from.** The harness's `sys.path` resolves `maskfwd` and `native` from `step1_cross_family_relational_foundations/src`. `step1_pathway_selection_causal_identification/src` does not contain `maskfwd.py`.

**Isolation.** Every check below was cheap and single-threaded (`XLA_FLAGS` as specified). Nothing was written into `checkpoints/` or `logs/conf/`. The one training check was an 11-step branch run in a throw-away sandbox under `/tmp`, which I deleted afterwards.

---

## 0. Independent checks performed (results)

| # | Check | Result |
|---|---|---|
| V1 | Is the location of the Adam mu/nu blocks by shape matching unambiguous? | The optimizer state has 90 leaves: `count`, 44 × mu, 44 × nu, then the schedule `count`. **Exactly one** start offset matches (i0 = 1). `param_names19.json` equals the harness's `tree_flatten_with_path` order. Within the block, same-shape tensors (WQ/WK/WV, the LN vectors) are mapped by tree position, not by shape, so they are not confused. The optimizer chain is `ScaleByAdamState`, `EmptyState` (weight decay), `ScaleByScheduleState`. There is no other state. |
| V2 | Are the composed full states equal to their sources? | `tp_E801_fullB` equals B and `tp_E801_fullI` / `tp_M801_fullI` equal I, **bitwise**: all 44 parameters, all 90 optimizer leaves and the RNG. Only `hist` differs: it is always B's, it is a loss log, and it is not used in the computation. |
| V3 | Is a branch continuation from a training arm's own step-100 window state identical to the uninterrupted windowed run? | Branch from `…D801_vsgown0T100_stop100_s100dev:100`, 11 plain steps, compared with `…vsgown0T100_stop1211_dev/step111.npz`: **max abs diff 0.0 (bitwise)**. Step-100 parameters of the two runs are also bitwise equal. `step_plain` never enters `block_m`, `XM`, `LNZ` or `FRZ`, so a continuation without `FX_INTERV` computes exactly what the uninterrupted run computes for i ≥ 100. Note: the "step-150 branch check = 0.0" cited in DESIGN §1 has no artefact in `logs/`. |
| V4 | Does `frz` freeze what it claims? | In the dev state `…frzL0_QKT100_stop100_s100dev`: `pre/0/WQ` and `WK` are equal to init (0.0), and their mu and nu are exactly 0. All other layer-0 tensors moved and have non-zero moments. Both Adam counts are 100. The RNG equals B's. Weight decay is discarded for frozen leaves, because `t2` is restored after `apply_updates`. |
| V5 | Post-window behaviour of the frozen leaves | See objection 1. They move **1.7–2.4×** as far as the same tensors in the matched B continuation. |
| V6 | Do the forward attention shares change under the controls? (fixed batch, stream 97 900 000) | See objection 3. |
| V7 | vsg gradients per tensor (`graph_audit19.json`, step 0) | Target-layer WV, g1 and b1 change, by 0.07–0.17 relative. **WO is unchanged (0.0)** on both bases, and WQ/WK are unchanged. See objection 2. |
| V8 | Is the `--perturb` composition reproducible? | It is **not**. See objection 4. |
| V9 | Queue tags against harness tag construction | All 48 c1 tags and 90 c2 run/branch/measure paths match the harness's tag grammar (`_gofown0b1e9T100`, `_frzL0_QKT100`, `_mskoq0T100`, `_gofoq2b1.0T100`, `_stop1211_br100_<pair>_<arm>`) and `design19.json`. STEPS defaults to 4000 in both the window runs and the continuations, so the warmup and cosine schedule is the same everywhere, and 1211 is a snapshot step. |

---

## 1. Verification of the audit's claimed effects

- **vsg.**
  - Forward unchanged (≤ 1.2e-6).
  - d loss/d bias_T and d loss/d bias_C equal plain.
  - WQ/WK gradients are unchanged at the target layer.
  - Only the gradient into v (WV, LN1, embeddings, and on M the upstream layers) changes.
  - **Confirmed, with the WO caveat in objection 2.**
- **gsg.**
  - T-logit gradient = 0.
  - Complement gradient = plain (+0.0079 on E, +0.0061 on M).
  - WV is unchanged.
  - Because of softmax normalisation, T's share still receives the plain-direction signal through the complement logits.
  - **Confirmed.** The C1 withdrawal in the audit is correct.
- **vrs / vrg.** The forward changes and every tensor's gradient changes, as the audit says. **Confirmed.**
- **frz.**
  - Forward and gradient at fixed parameters are identical to plain, because `mask_for` returns `None` for `frz`.
  - Only the parameter update, the weight-decay step and the mu/nu change of the frozen leaves are discarded.
  - The Adam `count` is shared and keeps advancing.
  - **The audit does not cover frz at all.** See objection 5.
- **Optimizer-state claims in DESIGN §1.** Confirmed (V1).

---

## 2. Objections

### 1. MANDATORY: frz is followed by a bias-correction catch-up artefact that favours P5_frz_nil

**What happens.**
- During the window, the frozen leaves keep mu = nu = 0 while the shared `ScaleByAdamState.count` reaches 100.
- When plain training resumes, these leaves get Adam's bias correction for count ≈ 100 applied to moments that have only seen k steps.
- For a steady gradient, the step-size factor m̂/√v̂ is about 1.0 at k = 1, about 2.1 at k = 10, about 2.0 at k = 30 and about 1.4 at k = 100. A fresh Adam gives about 1.
- This happens at warmup LR ≥ 5e-4.

**What I measured.** ‖θ_s − θ_100‖ of the frozen tensors, frz continuation against the matched B continuation:

| base | tensor | step 111 | step 150 | step 200 |
|---|---|---|---|---|
| E | `pre/0/WQ` | 0.292 vs 0.175 | 0.877 vs 0.561 | 1.10 vs 0.75 |
| M | `new/0/WQ` | 0.192 vs 0.079 | 0.536 vs 0.284 | 0.847 vs 0.545 |

- The tensors that were not frozen (WV, the other layers) move by the same amount as B's, to within 5%.
- On E, the frz step-100 state has the **same plain-forward T share as I** (0.137 against I's 0.134; B's is 0.235). The routing deficit is therefore real at step 100. What follows is an inflated catch-up that pushes back toward B.
- So P5_frz_nil ("routing learning is not the cause") is biased **toward being confirmed**.

**Correction.**
- (a) Add `FX_FRZ_OPT=track` to `mkstep`. It restores only the frozen *parameters*, not mu/nu, so the moments see the gradients at the frozen parameters and stay consistent with the shared count. Use it as the primary frz arm, or as a frozen co-primary.
- (b) P5_frz_nil is SUPPORTED only if both variants fall in [−0.25, 0.25].
- (c) Record ‖θ_s − θ_100‖ of the frozen tensors at steps 111, 150 and 200 against the matched fullB continuation for every pair, and report it.
- (d) State in the audit and the design that the current frz discards weight decay and Adam-moment updates for the frozen leaves, and that the count is shared.

**P5_gsg_exceeds_frz** inherits this bias and must use the same rule.

### 2. MANDATORY: vsg does not remove value learning through T, because WO still learns T-routed content

**What the code does.**
- `ctx_T = (a·XM) @ sg(v)` is then multiplied by `lay["WO"]` without a stop-gradient.
- So WO's gradient contains the full T-routed content term. The audit json shows rel_diff(WO) = 0.0 on both bases.
- Downstream LN2, MLP and the later layers also keep training on T content, because the forward is unchanged.
- Only the v side of the OV circuit (WV, LN1, embeddings, upstream) is cut. That is 7–17% of those tensors' gradient.

**Why it matters.**
- DESIGN §3 reads vsg as "value gradient through T removed" and predicts that X3 gives a "switch".
- P4_vsg_partial claims to reject "value-learning-only".
- A hybrid is exactly what an *incomplete* removal of value learning would produce under X3. The prediction therefore cannot reject X3.

**Correction.**
- Restate vsg everywhere as "stop-gradient on the value *vectors* read through T (WV / LN1 / upstream side). WO and the downstream readers of T content keep learning."
- Remove "X3 (value-learning-only predicts a switch)" from P4_vsg_partial's rejectable list.
- Or add an arm `vsgo` with `x += ctx_C @ WO + ctx_T @ sg(WO)` together with sg(v) on T. Audit it, then attach the X3 rejection to vsgo instead.
- Add an explicit 3.2 limitation: the downstream use of T content cannot be stopped without changing the forward.

### 3. MANDATORY: the matched non-targeted controls are edge-disjoint but share softmax rows with T, and they raise T's share and remove task-relevant content

**Disjointness.** `oq0`/`oq2` (30 edges per readout row) and `own0`/`own2` (6 per row) are edge-disjoint by construction (`maskfwd.edges`). However, they are **the same softmax rows** at the **same layer**.

**Measured forward head-mean shares on readout rows** (fixed batch, parent initialisation):

| base | control | T share: plain → control | oq share: plain → control | targeted block (I): T → oq |
|---|---|---|---|---|
| E | msk oq0 | 0.134 → **0.332** | 0.469 → 0 | T 0, oq 0.552 |
| M | gof oq2 b = 1.0 | 0.142 → **0.241** | 0.661 → 0.431 | — |

**Measured learned (plain-forward) T share at step 100:**

| base | B | I | ctl |
|---|---|---|---|
| E | 0.235 | 0.134 | 0.178 |
| M | 0.169 | 0.139 | **0.217** |

On M, the control *strengthens* the targeted route beyond B.

**Why this fails mandate D2.** The controls are therefore not "non-targeted" with respect to the target route's forward share. They push it in the opposite direction to I.

**Task-relevant content.** `step1_cross_family_relational_foundations/CROSS_FAMILY_MATHEMATICAL_STRUCTURE.md` l.74 states that the other-question sources carry trust information jointly. `COR_TEN_FAILURE_MATRIX.md` lists "other-question content" as an internally represented relation. msk oq0 deletes this content during the window. A nil outcome is therefore also predicted by X2, by suppressing a competitor's input, so P6 does not discriminate X8 from X2.

**Correction.**
- (a) Add to the audit table forward T share and oq share, at step 0 and as a window trajectory, for every ctl, and the step-100 learned T share.
- (b) Either replace ctl with a row-disjoint, share-neutral control and verify it per D2, or downgrade P6.
  - Replacement: a new `edges()` kind on non-readout rows of the same layer, e.g. source → other-question-source, so that no T softmax row is touched. It must pass the same distance gate. Its T-share change must be ≤ 10% relative.
  - Downgrade: restate P6 as descriptive and state that "X8 remains unresolved" (mandate D3: "If a properly matched control cannot be constructed, leave the general-disturbance explanation unresolved").

### 4. MANDATORY: the randnorm composition is not reproducible and uses the same seed for every pair

**Not reproducible.**
- `compose19.py` iterates `for g_ in G:` over a Python **set of strings**. Iteration order depends on `PYTHONHASHSEED`, which is randomised per process.
- So the order in which the groups draw from `default_rng(19)` changes between invocations.
- Verified: two compositions of `tp_E801_randnorm_L0` with PYTHONHASHSEED = 1 and 2 give different `step100.npz` hashes. The maximum parameter difference is 0.034.

**Consequences.**
- (i) The mandate's archive-stage "reproduce transplant outcomes" cannot be met.
- (ii) Silent provenance breakage on lane restart. If a c2 line is re-run after a crash, `compose19` overwrites `tp_*` and `provenance.json` with a *different* state. Meanwhile the run directory resumes from its own `resume.pkl`, which was built from the first composition. The scorer's provenance check compares `tp_*` outputs with the new `provenance.json` and passes, although the continuation came from different bytes.

**Same seed for every pair.** Every pair uses seed 19, so once the ordering is fixed all pairs would share one random direction. That makes A7 a single draw, not three.

**Correction.**
- Iterate `for g_ in sorted(G):`.
- Use a per-pair seed, e.g. `perturb = 19_000 + D` (encode it in `design19.json`; the scorer already checks `perturb_seed`).
- Make `compose19.py` refuse to overwrite an existing `tp_*` whose `provenance.json` exists, and make the c2 line skip compose when the continuation's run directory already has `resume.pkl`.
- Record the composed-state hash in the continuation directory at branch time (e.g. in `xix_train.py` after `branched from`). Have the scorer compare that hash with `provenance.outputs`.

### 5. MANDATORY: the machine-readable audit table does not cover every confirmation intervention or the mandate's per-component columns

**Missing rows.**
- `graph_table19.csv` has no row for **frz**, **msk oq0** or **gof oq2 b = 1.0**. All three are confirmation arms.
- Mandate §3.1 requires a table "for each proposed intervention".

**Wrong granularity.** The update and optimizer-state columns are a single maximum over all tensors (`update_rel_diff_max`, `optstate_rel_diff_max`). They are not reported per component (targeted routing / targeted value / complementary routes / downstream / shared), as the mandate's table requires.

**Correction.**
- Add these rows at step 0 and step 50: `frz:L0_QK` (E), `frz:L2_QK` (M), `msk oq0` (E) and `gof oq2 b = 1.0` (M).
- Report per-component gradient, update and mu/nu changes.
- For frz: show that the gradient equals plain, that the update and mu/nu change of the frozen leaves are 0, and that every other leaf's update equals plain.
- Include the T-share and oq-share columns from objection 3.
- Add the vsg WO row from objection 2.

### 6. ADVISORY: the matching quantity for ctl (total step-100 parameter distance)

**Acceptable as a magnitude gate.** On E801 the per-group ratios are close to 1 (L0 groups 0.75–1.2). Magnitude in layer 0 is already shown to be inert by randnorm_L0.

**But it is direction-agnostic.** Mandate D1 asks for the matching measure to be predeclared *per explanation*. For X8 ("optimization disturbance"), the loss disturbance and the update or moment displacement are the more relevant quantities.

For E801 they disagree. The ctl state's plain-forward loss at step 100 is 1.206, I's is 1.071 and B's is 1.014. So ctl is *more* disruptive than I on loss. That is good, because it is not a weak control, but it should be reported.

**Correction.**
- Predeclare cumulative window loss excess over B, the Adam mu-displacement ratio and per-group distance ratios as reported descriptors next to the gate.
- State which explanation each one addresses.

### 7. ADVISORY: artefact for the bitwise branch check

- DESIGN §1 cites a "step-150 branch check = 0.0" that is not present in `logs/`.
- My V3 check is bitwise at step 111.
- Commit a small json comparing, for at least one training arm and for fullB/fullI, the step-111/150 branch states with the uninterrupted run (or V2's byte equality).

### 8. ADVISORY: no step-100 window self-check for the training arms

- The `branch_src` record hashes the source state, but nothing checks that the window state was produced as intended.
- Add a cheap c1 post-check for each training arm. frz: the frozen leaves equal init and their mu/nu = 0. All arms: count = 100 and the RNG equals B's.

### 9. ADVISORY: role labels in the I-base transplants

- For `IfromB_L0`, compose is called with `bdir = I` and `idir = B`. `provenance.json` then labels I's hashes as "B/…" and the parameter list as `params_from_I`.
- The scorer handles the swap correctly (`src_dirs`), so this is not a bug.
- Add `base: "I"` to `provenance.json` so that human readers and the archive auditor are not misled.

### 10. Verified correct, so no objection

- mu/nu location (V1).
- `follow` semantics: moments follow parameters; with `--perturb`, B's moments are used.
- RNG, step and schedule state are taken from B and asserted equal to I's.
- fullB and fullI are bitwise natural (V2).
- Branch continuation is bitwise equal to the uninterrupted windowed run (V3).
- Every queue tag, directory and `measure19` call (V9).
- The scorer's provenance cross-checks for transplant inputs and outputs and branch sources.
- The gsg complement-gradient claim.
- The vrs/vrg classification.

---

## 3. Summary for the design authors

**What is sound.**
- The transplant machinery and the branch-continuation machinery are sound and bitwise exact.
- The queues are correct.

**What must change.**
- Three interventions are not what the design says they are:
  - frz is followed by an inflated catch-up (objection 1);
  - vsg leaves WO learning on T content (objection 2);
  - ctl raises the target route's share and removes task-relevant content (objection 3).
- Each of these biases or invalidates one of the eliminative predictions P5, P4_vsg and P6.
- The randnorm composition is non-reproducible (objection 4).
- The audit table is incomplete (objection 5).

All of these can be fixed before freezing without new science: harness flags, re-labelling, one replacement control or an explicit "unresolved" statement, and two compose fixes.

CLASSIFICATION: Approved with mandatory corrections

---

## Round 2

**What I re-reviewed.**
- `DESIGN_XIX.md` (v2) and `PRE_EXPERIMENT_IDENTIFIABILITY_REVIEWS.md`.
- In code: `xix_train.py` (vsgo, frzp, pnz, lbl, FX_WLOSS, run_config.json, FX_NOFALLBACK, BRANCH_DIR), `compose19.py`, `graph_audit19.py` v2, `graph_table19.py` and `logs/graph_table19.csv`, `conf_setup19.py`, the generated `queue_c1.txt` / `queue_c2.txt` / `design19.json`, and the provenance and matching parts of `score19.py`.

**How I checked.**
- Short runs only (≤ 30 steps), in a `/tmp` sandbox that was deleted afterwards.
- Nothing was written to `checkpoints/` or `logs/conf/`.

### R2.0 Checks performed

| # | Check | Result |
|---|---|---|
| C1 | Queue tags against the harness's tag grammar | I executed the harness's own `tag =` line with each queue line's environment. All 54 c1 tags equal the s100c19 directories that `design19.json` implies (B, I and the 7 training-arm windows per pair). All 99 c2 continuation tags equal the scorer's `run` templates. Every `measure19` directory equals its run tag. Every training-arm `FX_BRANCH` names that pair's own s100c19 window directory. |
| C2 | pnz and lbl leave the data stream untouched | After 30 steps, the data-stream RNG state of plain, lbl (p = 0.5) and pnz runs is **identical**. lbl draws from its own `LBL_RNG` and pnz from its own `PNZ_RNG`. |
| C3 | Forward pass and moments under the new modes | (a) lbl and pnz call `mask_for` → `None`, so the forward is plain. (b) pnz adds noise to θ after `apply_updates`, so the Adam moments are never perturbed directly. The audit's pnz row agrees: the step's update, mu and nu differences are 0, and the noise is 0.3–14.5× the plain update per tensor. (c) vsgo: forward diff 1.4e-6. WQ/WK gradients are identical. WV changes by 0.267 and **WO by 0.262** (it was 0 under vsg). Downstream is unchanged. (d) frzp at step 25: WQ/WK held, with non-zero mu/nu (6.8e-3 / 3.0e-5). The audit gives frzp's mu/nu as identical to plain, while frz's mu differs. |
| C4 | compose19 | Two compositions under PYTHONHASHSEED 1 and 2 give **identical** output hashes. Re-running with the same arguments exits 0 ("already composed (identical)"). Re-running with a different seed refuses to overwrite and exits 1. Per-pair seeds in the queue: E 19821–19823, M 20321–20323. They agree with the scorer's `pseed` rule. |
| C5 | Graph table v2 | It has per-component update, mu and nu columns from the mode's real training step. My first reading of very large vsg "routing" values was my own string truncation of numbers like 8.6e-06. The table is correct. |
| C6 | run_config and FX_NOFALLBACK | Every run writes `argv`, the FX_* environment, the hashes of the harness, `maskfwd` and `native`, `branch_dir`, and `branch_src` (the hashes of the step-100 files actually loaded). The scorer checks, for each continuation: `FX_NOFALLBACK = 1`; no `FX_INTERV`; the branch directory equals the expected tp/s100c19 directory; the `branch_src` hashes equal the composed outputs or the window state; and the window-state run_config's WKEYS environment equals `design19.wenv`, including `FX_PNZ_REF` for pnz. With NOFALLBACK there are no external alternative directories. |
| C7 | Determinism of a pnz/lbl run resumed inside the window | **Fails.** See R2.1. |

### R2.1 Resolution of the round-1 mandatory items

| Item | Status | Notes |
|---|---|---|
| 1 (frz catch-up) | **Resolved** | frzp is implemented correctly (C3d). The design now treats frz against frzp as a moment-artefact contrast, and states that X4a is not eliminated on E. |
| 2 (vsg / WO) | **Resolved** | vsgo stops the gradient of both v and WO for the T term. The downstream-reader limitation is stated. |
| 3 (same-row controls) | **Resolved** | msk oq0 / gof oq2 are dev-only. pnz and lbl do not touch edges or rows. Gates fail closed, and a failed gate means "X8 unresolved". |
| 4 (randnorm reproducibility) | **Resolved** | See C4. Composed-state hashes recorded at run start are checked by the scorer. |
| 5 (audit table) | **Partially resolved** | Rows for frz, frzp, pnz, vsgo and ctl_same_row, with per-component columns, are present. **lbl has no row.** See R2.3. |

### R2.2 Remaining objections

#### R2.2. MANDATORY: pnz and lbl window runs are not reproducible when resumed

**The problem.**
- `PNZ_RNG` and `LBL_RNG` are neither saved to `resume.pkl` nor restored from it. Only the data RNG is.
- The window has snapshots at 25, 50 and 74. A window run restarted after a crash or a container restart (this happened before, at 07:23Z) therefore re-seeds the noise RNG mid-window.
- The resulting step-100 state differs from an uninterrupted run, and nothing records that.

**Measured.** lbl with seed 7:
- uninterrupted against a run killed at step 25 (`FX_DIE_AFTER = 25`) and resumed: step-30 parameter maximum difference **2.5e-4**;
- for scale, lbl against plain at step 30: 2.5e-3;
- step 25 was identical.

pnz uses the same code path.

**Why the scorer cannot catch it.** `run_config.json` is written only once, so it does not record the resume either.

**Correction.**
- Store `PNZ_RNG.bit_generator.state` / `LBL_RNG.bit_generator.state` in `resume.pkl` (and in `opt_step*.pkl`), and restore them in the resume branch.
- Alternatively, refuse to resume a pnz/lbl run when `start < iw1`: delete it and restart from step 0.
- In both cases, append a `resumed_at` entry to run_config (or a sidecar file) on every resume.

#### R2.3. MANDATORY: no graph-table row for lbl

- lbl is a confirmation arm, so mandate §3.1 requires its measured row.
- Expected content: forward unchanged; every gradient changes; per-component update, mu and nu.
- **Correction.** Add `lbl` (`FX_LBL_P = 0.5`) to `base_modes()` in `graph_audit19.py`. It must use the mode's actual noised labels: the audit currently calls `lossf` with the clean Y, so the label substitution has to be reproduced in the audit, e.g. by applying the same `LBL_RNG` mask to Y. Then regenerate the CSV.

#### R2.4. ADVISORY: the pnz wait loop has no timeout

- `until [ -f …B/opt_step100.pkl ] && [ -f …I/opt_step100.pkl ]` blocks the lane forever if the B or I window run fails.
- Add a time limit (e.g. 30 min) that exits non-zero.

#### R2.5. ADVISORY: describe what the pnz audit row measures

- The pnz update/mu/nu columns are 0 because the noise is applied outside the step function.
- State in the audit text that pnz's effect is carried entirely by the noise term (column `pnz_noise_over_update_*`) and by the trajectory.

#### R2.6. ADVISORY: lbl gate has no upper bound

- The lbl gate is `hi = 1e9`.
- On M, the design says lbl is far stronger than the block, so an M nil rejects X8b against an over-matched disturbance. That is acceptable.
- Report the step-100 accuracy and KL of lbl next to the gate, so readers can see the competence cost (mandate D3).

#### R2.7. ADVISORY: label vsgo as a synthetic gradient intervention

- vsgo is a synthetic gradient intervention (mandate B3).
- On M it also changes the gradients of all 24 upstream tensors.
- The P4 reading ("combined V+O learning through T") should say "under a synthetic backward rule".

#### R2.8. ADVISORY: housekeeping

- pnz and lbl seeds are equal for E and M at the same D (9100 + D, 9200 + D). This is harmless, because bases are never pooled.
- compose19's refuse-overwrite path leaves a `mkdtemp` directory behind.

Both are cosmetic.

### Round-2 verdict

The round-1 mandatory items 1–4 are resolved and verified in code. Item 5 lacks only the lbl row.

There are two small mandatory corrections (R2.2 resume determinism of pnz/lbl, R2.3 lbl audit row). Neither changes the design's identifying logic, and both can be fixed before the freeze without new development runs.

CLASSIFICATION: Approved with mandatory corrections

---

## Round 3 (verification only)

**Method.** Light checks in a `/tmp` sandbox, which was deleted afterwards. No run exceeded 30 steps. Nothing was written to `checkpoints/` or `logs/conf/`.

### R2.2 (mid-window resume of pnz/lbl): resolved

**The fix.**
- `xix_train.py` l.243–246 discards a `resume.pkl` whose step is below the window end for pnz/lbl.
- It removes the stale `run_config.json` and `window_loss.json`, restarts from step 0, and logs `discarded_resume_at` to `resume_log.json`.
- Every ordinary resume is logged too (`resumed_at`, l.251).
- Completed window states (step = 100) are kept. Continuations (no `FX_INTERV`) are unaffected.

**Sandbox test.**
- I ran lbl (p 0.5, seed 7) uninterrupted to step 30.
- I ran the same configuration killed at step 25 (`FX_DIE_AFTER`), then re-run.
- The re-run logged `[{"discarded_resume_at": 25, ...}]`, rewrote `run_config.json` with `start: 0`, and produced a step-30 state **bitwise identical** to the uninterrupted run. In round 2 the resumed run had differed by 2.5e-4.

### R2.3 (lbl audit row): resolved

- `graph_audit19.py` audits lbl with noised labels (own RNG, p 0.5, K = 3). The noised labels feed the gradient, the bias derivatives and the actual training step.
- `graph_table19.csv` now has lbl rows for E/M × step 0/step 50:
  - forward unchanged (diff 0.0);
  - all 44 gradients changed;
  - per-component update, mu and nu values present (e.g. E step 50: routing update 0.18, value mu 0.23).

### R2.4 (pnz wait timeout, advisory): adopted

- All 6 pnz lines in `queue_c1.txt` now exit with status 7 after 2400 s without B/I.
- Syntax check: ok. A simulated expired wait returns rc = 7.
- `design19.json` was regenerated: `harness_sha256` equals the current `xix_train.py` (96f9fb9e…). The queues still have 54 + 99 lines.

### Verdict

Both round-2 mandatory items are resolved. I have no remaining mandatory objections. The round-2 advisories R2.5–R2.8 stand as advisory.

CLASSIFICATION: Approved
