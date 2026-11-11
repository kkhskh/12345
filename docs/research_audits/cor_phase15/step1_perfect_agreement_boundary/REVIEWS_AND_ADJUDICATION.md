# Reviews and adjudication

Five independent reviewers each examined the final pre-review package (commit 9389496), with full access to code, logs and
checkpoints. None of them edited package files. Their reports are kept verbatim in `reviews/R1–R5_*.md`, and their check
scripts and outputs are in `reviews/checks/`. The corrections are in the post-review commit. Each corrected document carries
a "Post-review" banner naming the findings applied. Pre-review text remains available at 9389496.

**No frozen file was edited** (protocols, frozen scripts, hashes). Where a review exposed a problem in frozen material, the
problem is recorded below as a deviation note.

## Verdicts

| reviewer | verdict |
|---|---|
| R1 formal mathematics | accept with required revisions. Core repair sound; 5 major, 8 minor |
| R2 statistical design | freeze intact; analysis correct and reproducible. The R1 result stands; P1(a) to be downgraded. 5 major, 8 minor, no critical |
| R3 causal identification | causal sections need major revision. Behavioural results and the "not Outcome B" classification stand. 8 major, 3 minor |
| R4 Transformer computation | major revision of the interpretation, none of the data. Includes exploratory path-split and symmetric-site analyses |
| R5 adversarial theory and novelty | the R1 sign reversal holds and is not implied by Bayes, gain or the pooling families. It is well described as structured, worst-group approximation error under frozen-feature transfer. 3 major, 3 moderate, 2 minor |

## Findings and actions

**Labels.** A = accepted and corrected. A-dev = accepted; it concerns frozen material, so it is recorded as a deviation
note. P = partially accepted.

### R1 — Formal mathematics

| # | finding | action |
|---|---|---|
| 1 | A3 as stated does not imply affinity; it needs connectivity of the co-moving-pair graph, and the coverage check was not logged | A: A3 restated; holds because all 96 pairs are enumerated (PROPOSITION_3_CORRECTION §2) |
| 2 | Mixed gates are never invariant, so the pre-review claim was false; limits of the family were not treated | A: A2 dropped for φ_col, mixed-gate proof added, limit remark added |
| 3 | A1 was worded per X_q (false); "projective dimension 1" is wrong; the check used one dense w | A: restated jointly; the reviewer's restricted check (null dimension 1, rank 6) is cited |
| 4 | "518 patterns" is wrong: there are 513 patterns and the rank is 518 | A |
| 5 | The minimal null edit changes two reports; τ was omitted | A |
| 6 | The Proposition 1 floor sentence is false (downward tilts are cheaper) | A: sentence deleted |
| 7 | The T1(b) proof is invalid; counterexample given | A: replaced by likelihood-ratio dominance (reviewer's route); Lemma B1 becomes a corollary |
| 8 | T1(c) and the "Reading" overclaim: the claim is sample-based, needs \|H\| = 1, and R2 goes 0 → negative | A: scoped, relabelled [N-sample] |
| 9 | The gain argument for P2 depends on γ_T vs γ_P | A: condition stated; fitted gains γ_T 0.93 > γ_P 0.78; pure gain predicts −0.03 vs +0.66 observed |
| 10 | T4(c) exception: F34/F23-R2 cells are gain-free | A |
| 11 | The T2 witness does not match its construction | A: witness replaced; triviality noted |
| 12 | T3(a) gauge wording | A |
| 13 | T5 labels | A: (a) elementary, (c) relabelled [D] |

### R2 — Statistical design

| # | finding | action |
|---|---|---|
| 1 | The ITT rule is ambiguous between the protocol/code and the cohort_status docstring; under the docstring reading P1(a) has p = .066 | A-dev: both readings reported; P1(a) downgraded to "fragile, not counted as support" |
| 2 | P1(a) is driven by two weak T1 parents (p = .076 without them) | A |
| 3 | Frozen P3 contrasts for B45 and D-R1 were omitted; specificity emerges at about step 500–1000 | A: full table in INDEPENDENT §2; MAIN wording fixed |
| 4 | The approximation-error inference is too strong | A: re-scoped as exploratory (TRAINING_STAGE §2) |
| 5 | The P5 rule is nearly vacuous; the R3 expectation was missed and is untestable as stated; the scope is R1 only | A |
| 6 | Development-based selection is legitimate, but replication is within one pipeline | A |
| 7 | Multiplicity | A: Bonferroni statement added |
| 8 | ANCOVA tests only the sharp null and is misspecified | A: support-restricted result reported; "consistent with" |
| 9 | P4(i) design points | A: matched means reported |
| 10 | Bootstrap and effective α | A |
| 11 | Inference conditions on one episode set | A: stated |
| 12 | Audit trail: npz untracked; bayes.py and ckpt.py not hashed; interim looks undisclosed | A: npz committed; disclosures in INDEPENDENT §4.7 |
| 13 | "Identical size" wording | A |

### R3 — Causal identification

| # | finding | action |
|---|---|---|
| 1 | "108 % / 89 %" hides an arm asymmetry (19 % vs 146 %) and depends on P84 | A: figures withdrawn |
| 2 | Site patches are off-distribution; the validity argument was circular | A |
| 3 | P-count is sensitive to norm-matched random site perturbations | A: reported; it weakens the P6b/P6c reading |
| 4 | The decoder subspace is not distinct from the edit's principal subspace | A: "P6c largely restates P6b" |
| 5 | Boundary specificity of the patch effect is not frozen; it is partial (58–69 %) | A: reported as exploratory |
| 6 | P6d is near-vacuous; a matched agreement-moving control is needed | A |
| 7 | P6e does not show non-redundancy for R1; joint ablation was not done | A |
| 8 | Collateral damage untested, yet MAIN still asserted a causal carrier | A: MAIN causal sentence weakened |
| 9 | Asymmetric site | A (see R4-1) |
| 10 | Edits are valid, but surgical and changing cross-group agreement | noted |
| 11 | P6a uses the pooled share | A |

### R4 — Transformer computation

| # | finding | action |
|---|---|---|
| 1 | Asymmetric site; a symmetric readout-token patch separates the arms 8/8 vs 8/8 by sign | A: added as an exploratory result; P6a "implementation diversity" reading withdrawn as partly an artefact |
| 2 | Partial patches are non-linear; shares are not decompositions | A |
| 3 | Circular validity sentence | A |
| 4 | WA and the final LayerNorm are inherited and trained, so curriculum effects are not confined to layers 0–1 | A: CAUSAL §0 |
| 5 | Attention/MLP split was cheap; partial split done (super-additive in P-count) | A: reported as exploratory; full per-head decomposition is an open question |
| 6 | Explanation (i) is not favoured | A: sentence withdrawn |
| 7 | Subspace dimension mismatch | A |
| 8 | No pooling in the readout | A |

### R5 — Adversarial theory and novelty

| # | finding | action |
|---|---|---|
| 1 | Frozen-feature transfer missing from the headline | A |
| 2 | "Interpreted" conflicts with the causal reading | A: "decision response"; interpretation-vs-representation listed as not claimed |
| 3 | The approximation-error argument is overstated; the R2/R3 comparison shows worst-group error | A |
| 4 | The discount is learned during transfer; no step-0 readout | A: TRAINING_STAGE §5 |
| 5 | Alternative operators (uniform "5/5 discount", response-weighted mixture) fail, which sharpens the claim | A: FITTED §4 |
| 6 | KL control is overstated (≈ 1e-3 KL) | A |
| 7 | Several results are elementary | A: labelled |
| 8 | The finding *is* approximation error (structured) | A: MAIN wording |

## Deviation notes (frozen material)

1. **ITT inconsistency.** The cohort_status.py docstring and protocol §4 / the code differ on ITT imputation. The frozen
   code was run as frozen, and both readings are reported (R2-1).
2. **Hashes not recorded at freeze.** bayes.py and ckpt.py were not hashed at freeze; they are unchanged in git since
   2026-11-08 (R2-12).
3. **Asymmetric site in P6.** The P6 site definition, inherited from Investigation IV's `eqv.site_positions`, is asymmetric
   across arms. The frozen verdicts stand as computed; their interpretation is corrected (R4-1).

## Unresolved criticisms (preserved)

- The operation that differs between curricula is not identified: frozen encoding vs a learned over-general rule (R3, R4).
- Collateral effects, joint ablation and per-head/LayerNorm decomposition are not done (R3-7, R3-8, R4-5).
- A trade-off along a shared approximation curve cannot be excluded for the converged models. No T1 snapshot reaches
  P-count's final R3 level (R5-3).
- Replication is within one pipeline (R2-6).
- T1(c) is a sample-based statement, not a proof (R1-8).
