# Reviews and adjudication

**How these reviews were done.** The reviews were carried out by the same agent that ran the investigation. Each one was
a separate pass with a fixed adversarial brief, done after the confirmation results were in. No independent reviewer was
used, and that is a limitation. Each objection is listed with its adjudication and with any change it caused in the
documents.

## R1. Formal review

| objection | adjudication | change |
|---|---|---|
| Propositions A–C are elementary. Calling them "propositions" may suggest theorems about COR | Accepted. They are labelled [P, elementary], and the formal document states that none is a new COR theorem | wording in `FORMAL_RESULTS_AND_COUNTEREXAMPLES.md` |
| Prop. C is stated for R features. The attention reader with self-only attention is additive over groups only if the readout after pooling is linear | Checked: the reader mean-pools and then applies a linear map, so Prop. C applies exactly. The ablation confirms it (spread ≤ 0.06) | none |
| The representability claim (Result 7) is not proved in detail | Accepted as a sketch. It is labelled a constructive existence statement with no training claim | none |
| The m′ correction must be checked for the mandate's intended meaning, not just its formula | The mandate's text describes m_g as "the number of other perfect groups agreeing with group g's target-question report". That differs from Investigation VII's m, which counts h whose own q-pair agrees. The numerical check (m′ gives R1 0, R2 +1.94 under Bayes) shows m′ cannot be the factor in the exact posterior. The correction stands | none |

## R2. Causal review

| objection | adjudication | change |
|---|---|---|
| The noperf/noother masks use the true eligibility. A model computing a correlated relation would also be disrupted | Accepted as a limit, stated in `CAUSAL_IDENTIFICATION.md` §4. The self-only result does not depend on eligibility | none |
| An interchange share of 1.84 / 1.48 / 1.34 above 1 means the m-subspace is not a clean mediator | Accepted. It is reported as overshoot; "the subspace is the mechanism" is not claimed | wording |
| "Repair" was interpreted as fixing the P-count representation, but the adapter could be bypassing it | **Accepted and tested.** The adapter ablation puts the model at chance, and the parent-state mean-ablation leaves it unchanged. The repair is a bypass. This is the most important correction of the investigation; it is stated in the main result | `SELECTIVE_LEARNED_REPAIR.md`, `LEARNED_INTERNAL_MECHANISMS.md` §4, main result |
| Cross-model causal equivalence was not tested | Accepted. Marked "not identified" (L4) | none |

## R3. Transformer-mechanism review

| objection | adjudication | change |
|---|---|---|
| The "attention readers" are tiny set-Transformers over three hand-built group tokens, not Transformers over raw tokens | Accepted. The evidence on ordinary tokens (Condition X) is limited to 6000-step runs; see `TRAINING_AND_OPTIMIZATION.md` §X. Mechanism claims are restricted to the reader architecture | scope statements |
| Group-id features may let the reader memorise pair identities | Partly supported by the holdout (attn degrades on the held-out pair). The id-free variant was not run | `COMPOSITIONAL_GENERALIZATION.md` |
| The 1-block downstream in Condition Z sees 64-d per-token states. Its failure might be optimisation, not access | The mandate's distinction is respected: no information-theoretic claim is made. Native modifications are reported as exploratory (xP80 only) | `SELECTIVE_LEARNED_REPAIR.md` |

## R4. Statistical review

| objection | adjudication | change |
|---|---|---|
| With n = 8 lineages, P5's sign test is the only inferential test; the other criteria are thresholds | Accepted. Thresholds were frozen, and every per-lineage value is reported. The effect sizes are large: R2∪R3 1.31 → 0.15, KL 0.094 → 0.002 | none |
| The P5 threshold changed (0.15 → 0.25) after seeing xP81 | Disclosed in Part I (§I.4b). It happened before the freeze and on development data only. The confirmed mean (0.057) meets both bounds | none |
| P6 passes while radapt is worse than orig on R1 in 1–2 of 4 T1 lineages | Accepted. Stated as a caveat in Part II. The P6 claim is weakened to "no degradation elsewhere, R1 within bound" | Part II caveat |
| Composition confirmation used seeds 10–12 with the same eval generator as other parts; is there seed reuse? | Seeds 10–12 were never used in development. eval_conf8 uses a new stream | none |
| Development holdout and Condition X use single seeds | Accepted. Both are labelled development or exploratory and kept out of the confirmed claims | labels |
| Two runs were lost to environment restarts and rerun | Disclosed. The reruns used identical commands and seeds and are deterministic up to CPU numerics | log notes |

## R5. Adversarial novelty review

| objection | adjudication |
|---|---|
| "Relational architectures learn relational functions faster than MLPs" is well known (relation networks, set Transformers, graph networks) | Accepted. **Not claimed as novel.** What is specific here is the *sign-level* criterion and its proof that additive-over-groups rules cannot meet it (an elementary consequence), together with the confirmation that learned attention realises it through attention to eligible partners |
| "Linearly decodable ≠ used" is a known caution in probing work | Accepted. The contribution is a frozen, confirmed instance in which *failed* learners decode the relation *better* than successful ones (P7) |
| A "separable phase first" resembles known simplicity-bias or staged-learning results | Accepted as a plausible instance. No new theory is claimed. It is confirmed prospectively (P2b, 9/9) |
| The learned repair is just "train a better model on the raw inputs" | **Accepted after the bypass ablation.** It is reported as such |
| Does any of this bear on COR? | Only weakly. The investigation characterises when a mathematically defined relation is learned and causally used by a specific architecture. It does not supply a theorem connecting this to COR's obstruction or discrepancy operators. See main result, Q22–23 |

## Adjudication summary

- **Claims kept:** P1–P5 and P7 as stated; P3's mechanism within the reader architecture; Prop. C with the self-only
  confirmation.
- **Claims weakened:**
  - P6 (no R1 improvement on T1);
  - "learned repair of the P-count model" becomes "learned bypass that does not need m";
  - mechanism claims are limited to small set-Transformer readers.
- **Claims not made:** novelty of relational inductive biases; any information-theoretic impossibility; any COR theorem.
