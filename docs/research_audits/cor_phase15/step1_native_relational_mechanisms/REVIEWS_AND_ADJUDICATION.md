# Reviews and adjudication

**Disclosure.** All seven reviews were done by the same agent that ran the investigation. Each was a separate pass,
after the final confirmation results were in, with a fixed adversarial brief. They are **not independent external
reviews.** Disagreements are preserved.

## 1. Mathematical reviewer

| objection | adjudication |
|---|---|
| Theorem 1 is just Bayes' rule rearranged | **Accepted.** It is labelled elementary. Its value is the explicit S-only cross-group dependence, which is used for predictions |
| Theorem 2's corollary needs an "evidence-form" restriction that real readers violate (normalization absorbs x-dependent factors) | **Accepted.** The restriction is stated explicitly. The corollary is not applied to normalized readers |
| The pooled-vs-pairwise claim for adaptive echo is unproved | **Accepted.** It is labelled a hypothesis. Only the sufficiency spread is numerical fact |
| The brute-force check uses one generator implementation; could genlib and bayes share a bug? | No. They are independent codebases: genlib is new, and bayes.episode_posterior is from earlier work. They agree to 3e-15 |

## 2. Mechanistic reviewer

| objection | adjudication |
|---|---|
| Token-set patching at the downstream input cannot locate computation *inside* fine-tuned layers 0–1 | **Accepted.** The claims are limited to "which token states at the downstream input carry the credit". Head and MLP decomposition is open |
| The attention freeze is surgical, and an off-distribution pattern could have arbitrary effects | **Accepted.** It is labelled surgical. Necessity claims use natural patches only |
| Non-additive patch shares (> 1) show interactions, so "share" is not a decomposition | **Accepted.** No additive decomposition is claimed |
| The routing "collapse" could be a LayerNorm-scale artefact | Not tested. Activation-scale sensitivity is listed as not done |

## 3. Experimental-design reviewer

| objection | adjudication |
|---|---|
| Only 4 complete confirmation parents, so a 4/4 sign test is p = 0.0625 | **Accepted.** Thresholds are counts, not significance claims. The effect sizes (3–5× error reductions) are reported per parent |
| The completeness covariate was chosen after the retrospective analysis | It was fixed in Part I *before* any confirmation outcome existed, from pre.json only. The retrospective analysis was disclosed as development |
| full_c1 and C1 differ in trainable parameters, so this is not a matched-capacity comparison | **Accepted.** Full fine-tuning is a strategy comparison, not a capacity-matched one. The capacity-matched comparisons (LoRA, partial unfreezing) were not run |
| One seed per parent for native training | **Accepted.** Parents are the unit of independence. Seed-within-parent variability is unmeasured |

## 4. Representation reviewer

| objection | adjudication |
|---|---|
| Decode-then-compose with per-position probes could exploit position-specific shortcuts | The probes are trained on fresh episodes and evaluated on a new stream. Count accuracy 1.000 on held-out data is not a shortcut artefact |
| "Information is present" via linear probes doesn't mean it's in a form a Transformer block can use | **Accepted.** That is exactly the distinction drawn (CP1 vs CP3/CP4) |

## 5. Generalization reviewer

| objection | adjudication |
|---|---|
| The held-out-pair test in ingredient space is weak: identity-free readers pass by construction | **Accepted.** It is reported as an inductive-bias result, not as learned generalization |
| The native models' generalization was not tested | **Accepted.** It is open; native models are tied to G = 3 |

## 6. Scientific-novelty reviewer

**Position.** Most of the mathematics and the generalization results are known in kind. The completeness correction and
the frozen-prefix diagnosis are the solid new items. "Fine-tuning the whole model fixes it" is unsurprising.

**Adjudication.**
- Accepted that the fine-tuning result is unsurprising as a capability.
- Kept as a contribution the *confirmed causal account* of why the frozen variant fails while the information is
  present: NP3 plus CP1 plus CP3.
- Kept the documented mechanism non-recurrence.

## 7. Positive-theory reviewer

**Position.** The investigation underweights a real positive.
- Across 6 independent parents, the failure was predicted and diagnosed *before* observing them: information present,
  frozen prefix inert, routing carries the regime-blind response.
- The repair was predicted and obtained on 4/4.
- Theorems 1–3 generated four confirmed generalization predictions.

**Adjudication.** Accepted as a fair reading of CP1–CP3, NP1, NP3 and GP1–GP4.

**Disagreement preserved.**
- The novelty reviewer maintains that the prediction successes are largely consequences of known principles.
- The positive-theory reviewer maintains that prospective confirmation of a mechanistic diagnosis on independent models
  is itself the contribution.

## Revisions made after review

- "Per-group routing" was corrected to "regime-blind routing" (`scripts/mech_routing2.py` showed conditioning on other
  perfect groups).
- The Type IV claim was restricted to the development parent after NP2 failed.
- The completeness covariate was marked as a methodological correction, not a discovery about relational computation.
