# Adjudication of the pre-run design reviews, round 2

The reviewers are AI agents, not human peer reviewers. Their round-2 reviews are in this folder:
`PRE_RUN_REVIEW_A_causal_identification_v2.md`, `PRE_RUN_REVIEW_B_mathematics_functional_v2.md` and `PRE_RUN_REVIEW_C_reproducibility_integrity_v2.md`.

Round-2 verdicts:
- A: approve with required changes. A targeted check of the fixes is sufficient.
- B: approve with required changes. Track C is approved; the Track B classification cannot be frozen until N1 is fixed.
- C: reject in the v2 state; approve after N1, N2 and the majors are fixed and checked before the freeze.

Status codes: FIXED / DOCUMENTED / DEFERRED / DECLINED. Each fix is followed by the test that checks it.

## Blockers

| id | issue | decision |
|---|---|---|
| A-N-B1 | Precision matching removes HB_e, so B3 can never be the headline | **FIXED.** Three parts: (1) the *functional* status (difference / equivalence / neither) must agree between the matched and final levels; (2) the locus is read at the matched level; (3) the precision flag is read at the final level. Check: `tests/test_trackb_oc.py` (precision-only scenario → B3, 10/10 at 4 and 6 parents). |
| A-N-B2 | Intervention history alone produces a functional difference; with a single base the sign check is vacuous | **FIXED.** The per-parent HB_c is now the average of the within-unit contrast and the history-matched same-D cross-unit contrast, which cancels history. A parent without both bases is flagged history-confounded. If every parent is confounded, a difference is reported as "FUNCTIONAL_DIFFERENCE (class and intervention history not separable)" and never as B1. The native/induced sign check must be True; a missing check does not pass. Checks: OC history-only scenario gives no B1; planted case E-only gives the confounded label. |
| B-N1 | HB_c_dir is computed against tie-break artefacts | **FIXED.** Rule decisions are tie-aware (`t4.decide`, −1 = tie). The discriminating set and HB_c_dir use only rules whose *unique* decisions differ from Bayes. Tie-aware disagreement with Bayes on natural episodes is local 0.127, noelig 0.530, and nocount / maprole / softrole / Bayes(α′ = 0.7) 0.000, so HB_c_dir and the equivalence test use local and noelig only. The equivalence table has a category column. |
| C-N1 | The frozen Track B code hash (one file) could never equal the records' full hash set, so Track B scoring always failed provenance | **FIXED.** `scripts/freeze.py` stores `trackb_run.code_hashes()`, the full set. The freeze hash covers the files and both code-hash sets. |
| C-N2 | `ALLOW_IDENTITY_CHANGE` too permissive; no per-attempt identity; no mixed-identity label | **FIXED** (round 3 X3: the RESULTS banner and the verdict table carry the MIXED-IDENTITY label); a per-pair MIXED_STACK split of the analyses is **DEFERRED**. A changed freeze can never continue a run root (hard stop). An allowed identity change does three things: (1) it records `mixed_identity`; (2) it supersedes PREFLIGHT, RESUME_TEST and the benchmark, so the preflight must be rerun on the new identity; (3) it attaches every attempt's identity hash in the registry. `run()` requires a preflight that passed under the *current* identity. Strictness: GPU model (not driver patch), Python major.minor, package and nvidia-* wheel versions, flags, precision, freeze. |

## Majors

| id | issue | decision |
|---|---|---|
| A-N1 | HA10a/b is a QK-weights vs non-QK-weights test | **FIXED (relabelled).** Added BfromVSGO_L2QK (layer-2 QK only; exploratory XA7). The composed states at step 100 are measured. Interpretation table: DESIGN §1. |
| A-N2 | HA10 conditioned on the block, not on vsgo | **FIXED.** HA10a/b are evaluated only in pairs whose vsgo arm is I-like (otherwise CONDITION_NOT_MET, counted). |
| A-N3 | The OC test skipped its own evaluability criterion | **DOCUMENTED** (DESIGN §1 and COMPUTE_BUDGET.md). At 4 parents × 4 data orders with 80% of pairs evaluable, P(GEN \| strong effect) = 0.42 (INCONCLUSIVE 0.55). This is stated in DESIGN and COMPUTE_BUDGET: the Standard preset is underpowered when pairs are lost; Extended (6 data orders) gives about 0.90. The gated criteria remain FAILS ≤ 0.05 at 80% evaluability and P(GEN \| null) ≤ 0.01. |
| A-N4 | A whole null parent could be the "one exception" | **FIXED.** A parent CONTRADICTS iff C ≥ ⌈n/2⌉ and C > S. With 8 parents × 6 data orders and one contradicting null parent, P(GEN) = 0.075 (gated ≤ 0.15). A *silently* null parent (mostly MIDDLE) can still be the single allowed exception (P(GEN) = 0.87). That is the frozen "all but one" policy; exceptions are listed per hypothesis. |
| A-N5 | Compatibility check incomplete | **FIXED.** The flag now applies to every transplant arm, including forward ones, and uses the step-100 composed-state KL versus the source window states as well as steps 150 and 300. Any supporting outcome of a flagged state is relabelled INCOMPATIBLE. The 0.10 threshold is uncalibrated, so verdicts at 0.05 and 0.20 are also reported. |
| B-N2 | Directed statistic needs a precision adjustment | **FIXED.** d_k = P(decision = k's) − P(decision = the third option). Errors that only reflect precision land uniformly on the two non-Bayes options, so d_k = 0 in expectation. |
| B-N3 | The OC simulation could not produce the failures | **FIXED.** The OC is rebuilt on the real tie-aware discriminating set (9 353 inputs), with dev-calibrated error rates (2.6% / 1.4%), both bases per parent, and native and induced organisations. Scenarios: precision only, directed, history only. |
| B-N4 | Equivalence table raw argmax, no category | **FIXED** (tie-aware; categories). |
| C-N3 | Failed pretraining frozen into eligibility on the first failure | **FIXED.** Failed pretraining is reported from the registry, never frozen. A screening job that has failed 3 attempts is recorded as SCREENING_FAILED. |
| C-N4 | resume.pkl written before opt_step | **FIXED** (opt_step is written first; resume_log is atomic). |
| C-N5 | Decision npz unhashed | **FIXED.** The battery JSON (hashed in the registry) records each npz's sha256, and the scorer verifies it. |
| C-N6 | nvidia wheels; CPU fallback; precision | **FIXED** (nvidia-* versions in the identity; `JAX_PLATFORMS=cuda` in GPU mode, so a fallback fails loudly). **DOCUMENTED:** the wheels are recorded, not pinned, because their versions cannot be resolved from here. |
| C-N7 | Preflight ignores checks 01/03; caches forever | **FIXED.** Pass requires 01 and 03. The resume test is re-run whenever the identity changes, and the preflight is tied to the identity. |
| C-N8 | Storage underestimated; Drive trash | **FIXED** (1.5× margin; a missing measurement fails the check). **DOCUMENTED:** deleted files go to the Drive trash, which must be emptied. |
| C-N9 | Freeze coverage; clean-extraction script; dirty commit | **FIXED.** tests, fixtures and requirements are frozen; `scripts/clean_extraction_test.py` is added; the build records the source commit and the working-tree status. |

## Minors

| item | decision |
|---|---|
| graph-audit check inside a comment | **FIXED** |
| report KeyError | **FIXED** |
| tests writing results into the package | **FIXED** (`$COR_TEST_OUT` or a temporary directory) |
| measurement hashes missing modules | **FIXED** (every vendored `.py` is hashed) |
| α grid vs null range, and α* selected on the evaluation half | **FIXED** (grid widened; fit on A1, select on A2, evaluate on B) |
| HB_c_dir text vs code | **FIXED** (the protocol text is updated) |
| the runner lock is not exclusive across machines | **DOCUMENTED:** one runner per Drive folder; heartbeat lock |
| orphan processes after a kernel crash | **DOCUMENTED:** restart the runtime; stale records are reset only when the process is not alive on this machine |

## Unchanged deferrals

- prsg (projected routing stop).
- A cross-model / abstract-variable interchange. Without it, B1 is functional only and B2 is a locus statement.
- A second init seed per unit.
