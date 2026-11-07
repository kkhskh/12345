# PATH 1 — Final scientific assessment

**Verdict:** `PATH1_MANUSCRIPT_COMPLETE_BUT_SCIENTIFIC_BLOCKERS_REMAIN`

This is not `PATH1_EMPIRICAL_MANUSCRIPT_READY_FOR_HUMAN_REVIEW`, because Gate B (independent recomputation from saved logits) did not pass in this workspace. It is not `PATH1_NOVELTY_INSUFFICIENT_FOR_STANDALONE_PAPER`: the controlled kernel question is distinct from Li/Nanda/Elazar as a combination, with scope limitations. It is not `PATH1_REPRODUCIBILITY_BLOCKED` as a total failure: protocols, letters, and pasted FRESH/VNF stdout exist, and a private manuscript was written. The blocker is missing Drive arrays for Level 1.

## Gates

| Gate | Result | Reason |
|---|---|---|
| A Source integrity | CONDITIONAL | Letters and protocols are real git artifacts; pair-level logits are on Drive, not here |
| B Numerical reproducibility | FAIL | No local NPZ replay of SPEC/fixed-JS/NST; FRESH/VNF transcribed from stdout |
| C Dataset provenance | PASS | Splits, hashes, reuse, and confirmation unused are documented |
| D Honest causal interpretation | PASS | Probe-relative invariance; intervention effect ≠ sufficiency |
| E Controls | PASS | Wrong-target, random, mean-state, Editor A, oracle; unfair gradient baseline withheld |
| F Negative-result completeness | PASS | 08R2 NEGATIVE, COST INCONCLUSIVE, switching, factorial, absolute \(Y\) in main text |
| G Literature novelty | PASS with limitations | `NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS` |
| H Statistical validity | CONDITIONAL | Units and clustering described correctly; intervals not locally recomputed |
| I Manuscript coherence | PASS | Connected prose; no COR theory |
| J Reproducibility package | CONDITIONAL | Commands exist; Level 1 cannot be run here |
| K Reviewer objections | CONDITIONAL | Fatal array-verification objection unresolved |
| L Claim calibration | PASS | Abstract and conclusion match the limited claim |
| M Submission-policy | NOT CLAIMED | No venue selected |

## Q1. Strongest empirical result we can defend

For this frozen linear board probe on Othello-GPT, residual edits in the probe-contrast nullspace can change legal-move mass while the probe’s occupancy-probability vector is required to stay put, and correct-target directions beat wrong-target and random controls under a restored equal-JS protocol (letter `FIXED_JS_TARGET_SPECIFICITY_SUPPORTED`) as well as under equal L2 (`TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED`). A train-only map produces a similar paired advantage without \(h_D\) at evaluation (`DONOR_FREE_NULL_STATE_TRANSFER_SUPPORTED`), with mean \(Y\) still negative.

## Q2. Strongest result that may be new

Complete frozen-probe *probability* preservation plus target-specific kernel effects at matched output JS, together with evaluation-time donor-free construction from symbolic features, plus prespecified failures of joint switching and visible/null interaction. Not the general statement “probes are not causal.”

## Q3. Already established

Li et al. 2023: Othello-GPT has a causally relevant board. Nanda et al. 2023: it is linearly readable and steerable along the probe. Elazar et al. 2021 / Ravfogel et al. 2020 / Belrose et al. 2023: probes are not causal summaries; nullspace projection erases linear features. Activation steering exists.

## Q4. What probe-invisible means here

Invariant to this frozen probe’s 64×3 probability vector within a stated absolute tolerance, because the edit lies in \(\ker W\).

## Q5. What complete probe-probability invariance established

That output change is not explained by a change this probe can see. Letter `PROBE_LOGIT_INSUFFICIENCY_SUPPORTED` (arrays not local).

## Q6. What it does not establish

Invariance of all decoders; absence of other board information; on-manifold naturalness; a second world model.

## Q7. Strongest donor-based results and controls

SPEC (equal L2, 48 pairs) and restored fixed-JS (equal JS 0.01, same 48 pairs) versus three wrong donors and three random nulls. Original COST is not the equal-JS result.

## Q8. What donor-free transfer adds

A direction built from symbolic \(b_t\) without \(h_D\) at eval still beats wrong-target/random/mean-state/Editor A on paired \(Y\) in the NST sample. It does not add unsupervised rule induction; the legal mask is an input.

## Q9. Stage-conditioned share

STAGE_ONLY − Editor A = 0.030099 over FULL − Editor A = 0.053713, ratio \(\approx 56\%\), mean-level arithmetic, git-tracked.

## Q10. Legal-mask and board-only ablations

FULL increments over LEGAL_ONLY and BOARD_ONLY were below +0.01. Letter `LEGAL_MASK_SUFFICES_IN_TESTED_FAMILY` (exploratory, same 48 pairs). Board geometry can determine legality, so the mask is not uniquely necessary.

## Q11. Why not causally sufficient

Absolute \(Y\) negative; switching gate failed; factorial interaction unsupported; 08R2 Editor A validation negative; no interchange-intervention alignment.

## Q12. What switching and factorial rule out

Reliable joint T/W replacement on the same recipient under the 60% gate. The specific equal-L2 visible/null interaction with \(\rho=\nu=3.5\) and floor +0.005. They do not rule out all possible interactions.

## Q13. Fresh versus reused validation

SPEC/fixed-JS: reused 48. NST/SEM: other unused-val 48, SEM exploratory. FRESH 9182 and VNF 10182: later cohorts. Confirmation unused.

## Q14. Are CIs and effect sizes appropriate?

Clustering matches the unit. Effect sizes are small on a \([-1,1]\) mass scale. Intervals for 48-pair studies are not locally recomputed. Do not read 0.0201 as 2.01% accuracy.

## Q15. What a skeptical MI reviewer would reject

A new mechanism; a complete world model; novelty of nullspace projection or Othello steering; treating SPEC 0.39 as matched-strength; “donor-free” as training without activations.

## Q16. Contribution with all COR terminology removed

The empirical kernel question in Q1–Q2. That is the paper.

## Q17. Sufficiently novel for a standalone paper?

For a focused interpretability workshop or technical report, yes with limitations. For a broad ML main-track paper, not on current evidence. Label: `NOVELTY_SUPPORTED_WITH_SCOPE_LIMITATIONS`.

## Q18. Can main numerical claims be reproduced from available artifacts?

Not from local logits. FRESH/VNF can be checked against pasted stdout. SEM ratio is arithmetic. SPEC/fixed-JS/NST need Drive.

## Q19. What an independent lab still needs

Drive NPZ and manifests, checkpoint `synthetic_model.pth`, 08c probe, legal-set JSON, and the frozen runners. GPU identity is not in this package.

## Q20. Is the manuscript clear connected academic prose?

Yes, with residual “recorded” provenance tags that should become “we measured” after Level 1.

## Q21. Title and abstract free of unsupported generalizations?

Yes. Title names Othello-GPT. Abstract states negative \(Y\), 56% stage share, 13/32 switching, near-zero factorial.

## Q22. Does the conclusion match the evidence rather than the original ambition?

Yes. Last sentence: not a second world model.

## Q23. Would the paper remain truthful if no further COR experiment ever succeeded?

Yes. It does not depend on obstruction, germs, or Phase 2.

## Q24. Single most important unresolved reviewer objection

Level 1 verification of headline numbers and 64×3 probe-probability invariance from saved arrays. Without that, a reviewer can refuse to trust the kernel claim.

## Q25. Is Path 1 ready for external human scientific review?

**CONDITIONAL.**

The manuscript, novelty audit, negative-result register, and protocols are ready for a human to read as a *draft argument*. They are not ready as a *verified empirical submission*, because the logits that would support the 48-pair decimals are not in this workspace and were not recomputed here. A human reviewer of the science needs the Drive archive or a completed Level 1 script output. A human reviewer of the writing can start now.

Do not submit. Do not push a public copy. Do not treat this verdict as publication-ready.

## PDF compilation

`pdflatex` is not installed. `paper_master.tex` compiled with `tectonic` to `paper_master.pdf` (successful xdvipdfmx write, three overfull-hbox warnings on lines 49–52). The PDF was not visually inspected page by page (no poppler/`pypdf` extractor in the environment). The canonical full manuscript is `paper_master.md`; the TeX file is a condensed compilable counterpart.
