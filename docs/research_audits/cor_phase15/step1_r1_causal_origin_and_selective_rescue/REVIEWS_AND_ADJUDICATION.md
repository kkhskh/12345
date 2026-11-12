# Reviews and adjudication

**Process.**
- Five independent reviewers examined the final pre-review package (commit 76a9d60), with full access to code, logs and
  checkpoints.
- None edited package files. Their reports are kept verbatim in `reviews/R1–R5_*.md`.
- All corrections are in the post-review commit, and each corrected document carries a "Post-review" banner. The pre-review
  text stays at 76a9d60.
- **No frozen file was changed.** One post-review development option, the generic ingredient adapter, was briefly added to the
  frozen `scripts/downstream.py`. It was then moved to `scripts/downstream_postreview.py`, and the frozen file was restored from
  git (hash re-verified; see `logs/COMPUTE_LOG.txt`). The two runs that started during that window executed the same code path.

## Verdicts

| reviewer | verdict |
|---|---|
| R1 formal mathematics | accept with major revisions. Proposition 1 and Lemma 1 are correct and Theorem 1 holds more generally than stated; Corollary 1 and the Theorem-2 reading were wrong |
| R2 Transformer architecture | accept after corrections. The code is exact; §3 of the reconstruction misplaced where the response is generated |
| R3 causal identification | the injection is a behavioural rescue, not a causal test of mechanism. E holds only as a behavioural rescue; D is not reached; F is unresolved |
| R4 design and statistics | accept with required revisions. The freezes and the primary result hold; "Bayes level" does not |
| R5 counterexamples and novelty | the restricted result stands; the framing must narrow. The core phenomenon is a known kind; the exact characterisation and the frozen confirmation are new |

## Findings and actions

**Labels.** A = accepted and corrected. A+ = accepted, and a new development run was added. P = partly accepted. N = noted.

| finding | action |
|---|---|
| R1-1 / R5-3: Corollary 1 false (a vote-matching rule gets R1 > 0 and R3 < 0 without other-group counts) | A: restated. The full Bayes pattern needs other-group *information* (counts or votes) |
| R1-2: R3 statement for Theorem 2 false; R2 provable | A: R2 proof added, R3 dropped |
| R1-3: Theorem-2 empirical comparison used the wrong boundary | A: recomputed with fitted log(w5/w4). Signs match; P80 fails through a low reliability gain |
| R1 (minor): explaining-away table caveats; minimality not proved; provenance; labels | A |
| R2-1: response originates in attention (query) and the MLP amplifies it, not "after attention" | A: FULL_LAYER2 §3 corrected |
| R2-2/3: attribution is direct-path only; D is not a pure residual | A |
| R2-4: the readout-token route magnitude depends on the baseline | A: both baselines reported; sign robust (8/8) |
| R2-5 / R3-4 / R5-1: random injection carries almost no information | A: re-described as a pathway control; a generic task-relevant adapter (injna) added post-review |
| R2-6: injection enters Q/K/V and the residual before LN₁ | A |
| R3-1 / R5-2: n already carries the cross-group information; supplying m does not prove absence | A+: re-described as a precomputed-conjunction contrast; injna (all ingredients, no conjunction) does not rescue (P80 −2.26, P81 −0.67) |
| R3-2 / R5-4/5: m is near-oracle on the deciding subset; generator-specific | A: stated in SELECTIVE_RESCUE §4 and MAIN |
| R3-3: H7 written after injection worked on 2 development parents | A: disclosed |
| R3-4 / R4-4 / R5-6: extra-training and generic baselines missing | A+: 3× training on P80 run (sign still wrong); LoRA, recalibration and a learned adapter listed as not run; F set to undetermined for budget |
| R3-5 / R4-4 / R5-8: W_bal not relation-free and untuned | A |
| R3-6 / R4-2: probes are not absence; P6 was low-risk | A |
| R3-7: P7 is a hybrid-patch quantity | A: caveat; confirmation mean −0.535 reported |
| R3-9: MAIN overclaimed "explained by Theorem 1" | A |
| R4-1 / R5-7: "Bayes level" hides overshoot; P3 counts overshoot as success | A: withdrawn; distance-based P3 would pass 5/8 |
| R4-3: no power analysis | A: post hoc power reported, and noted as a protocol deviation |
| R4-5 (minor): smoke-run wording, bootstrap width, unconstrained families, pretraining split | A |
| R5-9: Theorem 1 is elementary | A: labelled |
| R5 novelty: shortcut learning, simplicity bias, frozen-feature accessibility and conjunction difficulty are known | N: MAIN states what is new and what is not |

## Unresolved criticisms (preserved)

1. **No internal operation is identified as *the* cause** (outcome D not reached). The readout-token route is sufficient to carry
   the error but not necessary for it.
2. **The rescue supplies a hand-derived, generator-specific conjunction.** Whether a network can be led to construct it unaided
   (for example by a learned conjunction-capable adapter, or by unfreezing) is untested.
3. **Generic baselines are incomplete.** LoRA, logit recalibration, natural R1 up-weighting, a learned adapter, and extra
   training on more than one parent were not run.
4. **No T1 confirmation panel.** The T1–P accessibility and route contrasts rest on development parents.
5. **Replication is within one pipeline.**
6. **The P3 criterion is lenient about overshoot.**

## Defended restricted results

These survive all five reviews:
- Proposition 1 (closed form).
- Lemma 1, and Theorem 1 in its general form.
- Theorem 2 for R1, plus the R2 result.
- The exact explaining-away account.
- The frozen prospective confirmation on 8 independent lineages: P1, P2, P4 and P5 hold; P3 holds as frozen.
- The risky P4 prediction (supplying counts makes R1 worse), which held in 7/8 lineages.
- Selectivity of m-injection relative to every comparator actually run.
- Respect for the no-oracle rule: labels-only training, and m computed from X alone.
