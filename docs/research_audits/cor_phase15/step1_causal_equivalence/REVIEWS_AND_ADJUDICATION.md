# Reviews and adjudication (Investigation X)

**Disclosure.** All reviews were done by the same agent that ran the investigation. Each was a separate pass on the final
corrected results, with a fixed adversarial brief. They are **not independent external reviews.** Disagreements are kept.

## Mathematical reviewer

| objection | adjudication |
|---|---|
| Theorem X1's "impossible" cases rest partly on numerical subspace search, which the same document says can fail | **Accepted in part.** I2-VS and I1-VE impossibility have proof sketches (base/source mixing arguments). I3's impossibility is numerical only and is labelled [N] |
| Proposition 1 is trivial | **Accepted.** It is labelled elementary and is used only to constrain the experimental design |
| The per-group signature inference (S partial, E exact ⇒ per-group) runs against the identification caveat | **Accepted.** The claim is restricted to "consistent with per-group, not evidence against an S variable". It is in the open map |

## Mechanistic reviewer

| objection | adjudication |
|---|---|
| Total-effect component patching of an early MLP trivially gives share ≈ 1 | **Accepted.** It is stated as uninformative about the relational step |
| Head-profile similarity across parents depends on permutation maximization | **Accepted.** A null was computed (mean 0.62, q95 0.90–0.94), and the evidence is called weak |
| DAS at the downstream input does not show *where* E is computed | **Accepted.** The location claim is "where E is represented in an interchangeable form". The computation inside layers 0–1 is open |

## Experimental-design reviewer

| objection | adjudication |
|---|---|
| n = 3 complete parents, with ≥ 2/3 as the success criterion | **Accepted as a limitation.** Per-parent values are all reported. F1, F3, F5, F6 and F7 were 3/3 (and 4/4 including the incomplete parent) |
| The completeness exclusion could look outcome-driven | It was pre-specified in Part I before any confirmation run, and read from pre.json only. wP143's results are reported, and would *not* change any verdict except supporting F4's direction |
| The C1 das_eval crash and its rerun happened after other confirmation outputs were visible | The rerun used the identical command and the frozen R subspace already on disk. No choice was involved |
| A training-log line was seen during monitoring | Disclosed. It is irrelevant to all predictions |

## Representation reviewer

| objection | adjudication |
|---|---|
| A 6-dim subspace in a 448-dim space could fit counterfactuals by exploiting unrelated structure | The controls address this: random subspaces score 0.50, selectivity shows the subspace tracks eligibility-defining edits only, and S fits at the same site are partial. **But** C1's moderate IIA without selectivity shows that DAS fits *can* exploit other structure. Hence F4's failure, and the recommendation to pair DAS with selectivity |

## Generalization reviewer

| objection | adjudication |
|---|---|
| No generalization experiments were run | **Accepted.** It is stated in the synthesis and the open map |

## Novelty reviewer

**Position.** Causal-abstraction and DAS methods are known (Geiger et al.). The non-uniqueness of abstractions is known.
"Same function, different circuits" is a known theme.

**Adjudication.** Accepted as known in kind. The specific new items are:
- a constructive, task-specific demonstration (Theorem X1) that ties the intervention-support set to the factorization;
- prospective confirmation on independently pretrained models that one abstraction (E) is shared while the pathways
  differ;
- a validity check (selectivity) that caught a DAS false-positive mode;
- an algorithmic learning order detected only through discriminating edits.

## Positive-theory reviewer

**Position.** This investigation turns Investigation IX's apparent negative (NP2: no shared pathway) into a positive,
prospectively confirmed result of shared abstraction at the right level of description.

**Adjudication.** Accepted. Kept with the caveat that F4 failed, and that "shared abstraction" is shown for one variable
at one site.

**Disagreement preserved.** The novelty reviewer considers the result a careful application of known causal-abstraction
methodology. The positive-theory reviewer considers the *prospective, independent-parent* confirmation of a shared
abstraction under heterogeneous pathways a substantive foundational result.

## Revisions after review

- The per-group inference was weakened.
- The head-similarity claim was downgraded to weak evidence after the null computation.
- F4's failure prompted the explicit statement that interchange accuracy alone is not a mechanistic marker.
