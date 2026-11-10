# Is there a shared causal relational variable across the P-count and T1 implementations?

**Pre-review version.** Base commit 8c31051 (Investigation III); freeze commit 579a7a2. Grades: [P] proved, [N] exact numerics,
[C] confirmed on held-out models under a frozen criterion, [F] frozen criterion failed, [E] exploratory/post hoc.

## Direct answer

**Partly, at the level of the variable; not at the level of the computation that uses it.** Under the pre-registered rule the
investigation's outcome is **C (equivalence fails)**, because the necessity-specificity criterion IV(a) failed (narrowly, in two
of six models). Read without that rule, the evidence separates as follows.

1. **Shared:** an identifiable abstract relational state V = n^{(-q)} (leave-q agreement counts), defined from the generator,
   sufficient for the relational component of the decision [P], is causally used by both implementations at their own sites:
   centralised at the readout token in P-count models, as a consistent tuple of own-group counts on question q's six source
   tokens in T1 models. Abstract-variable interventions on V reproduce natural counterfactual decisions well above
   mismatched floors in all six confirmation models (cond adjusted 0.76–0.95 excluding T2, whose full-site ceiling is below its encoder score; single-group 0.84–1.03) [C: Prediction I passed];
   count-preserving changes act outside V in both [C: III passed]; marginalising V moves every model 68–88 % of the way to the
   own-question posterior, against 1–2 % for a matched-capacity wrong variable [mostly C; F: IV(a) failed at P4 0.677 < 0.70 and
   T11's rotated control 0.386 > 0.30].
2. **Shared only in one direction:** a change of V identified in a T1 model and transported through V into any P-count or T1
   model reproduces the recipient's natural response exactly (fidelity 0.98–1.01) [C: II(b)]. From P-count donors, transport is
   partial (0.28–0.86) and worse into other P-count seeds than into T1 models [F: II(a)], because most P-count sites hold a lossy
   image of V (per-component decodability 0.72–0.79). By Theorem C, even perfect transport identifies only that the donor's state
   determines V and that the recipient factors through V — not a shared downstream computation [P].
3. **Not shared:** (a) the decision response to changes of V beyond what Bayes requires — π-preserving count changes, to which an
   optimal model must not respond — is partly shared among T1 parents (r 0.38) and not among P-count seeds (−0.07) or across
   curricula (0.10) [C: VI-sep]; (b) the downstream route: in T1 the variable reaches the decision only through layer-2 attention
   (0 % of the effect survives freezing it), in P-count 45–91 % survives [N on these models]; (c) the Bayes-tracking of the V→decision
   map (r 0.49–0.72 P-count, 0.73–0.82 T1) [F: VI-bayes, P2 0.486].

So the two curricula **instantiate the same identifiable causal relational variable (with P-count's realisation coarser), and
use it through different, curriculum- and seed-specific downstream computations**. Their agreement on T4 decisions is agreement
on the variable plus approximate agreement with Bayes, not a shared computation of the decision from that variable. The answer
to the final directive is the second disjunct in a restricted form: shared relational information and a shared intervention
handle conceal different causal mechanisms downstream of the variable.

## What "equivalent" means here (FORMAL_EQUIVALENCE_AND_THEOREMS.md)

Four nested notions on a declared intervention family: information (E-Info), intervention / commuting square (E-Int), decision
response (E-DR), full computation relative to a mediator decomposition (E-Comp).

| notion | P-count | T1 | across |
|---|---|---|---|
| E-Info (V decodable at the site) | lossy (0.72–0.79; P0 1.00) | exact (≥ 0.998) | partial |
| E-Int (commuting square, cond/single) | yes | yes | — |
| transport through V | partial out, exact in | exact out and in | asymmetric (Theorem C explains) |
| E-DR on natural support | tracks Bayes 0.49–0.72 | 0.73–0.82 | only via Bayes |
| E-DR beyond Bayes (π-preserving) | seeds idiosyncratic | partly shared | **no** |
| E-Comp (attention vs MLP route) | in place, seed-dependent | attention only | **no** |

Theorem results that frame the answer: V is sufficient but not minimal (91 count vectors, 43 posteriors; single components lose
0.07 Bayes accuracy) [P, N]; a distributed tuple realises V iff written consistently [P]; transport factorizes through donor
decodability and recipient validity [P]; Bayes-optimality forces E-DR on natural support, so only deviations and off-support
extensions can reveal a shared computation [P]; five non-identifiability constructions (D1–D5) with learned instances where they
exist [P, N].

## Hypotheses

| | verdict on these six models |
|---|---|
| H1 shared abstract causal variable with shared downstream use | variable: supported; downstream: **rejected** (VI-sep, pathways) |
| H2 different projections needing reconstruction | rejected for T1 (the consistent tuple is V; transport out of T1 is exact); for P-count the issue is a lossy code, not a missing projection |
| H3 different computations with similar decisions | **supported for the V→decision map** |
| H4 intervention artefact | largely rejected (floors, nulls, natural Mahalanobis statistics, marginalisation toward own posterior); residual: T11's rotated control, T-model overshoot 1.14–1.37 |

## Site and objective

Moving P-count supervision from the readout to a source token did not move a causally used count variable to the source tokens;
in those models neither site is decisive (Prediction V failed; three seeds, one with incomplete pretraining). Why T1 stores its
tuple at the source tokens is not explained.

## Documents

FROZEN_PREDICTIONS.md (frozen block + outcomes) · FORMAL_EQUIVALENCE_AND_THEOREMS.md · COUNT_INFORMATION_STRUCTURE.md ·
CROSS_MODEL_CAUSAL_TRANSPORT.md · NECESSITY_AND_RESTORATION.md · TRAINING_OBJECTIVE_SITE_TEST.md ·
COUNTEREXAMPLES_AND_LIMITATIONS.md · REVIEW_ADJUDICATION.md · REPRODUCTION_README.md.
