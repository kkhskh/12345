# Original COR: failure matrix and the O/D question (mandate §13)

**Historical verdicts are unchanged.** All of these still stand:
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- the Step 1A–1E verdicts
- `STEP1_ONLY_KNOWN_NONLINEAR_MECHANISMS_SURVIVE` (deep investigation)
- `STEP1_NECESSARY_CONDITIONS_ONLY` (foundational bridge)
- the latent-reliability verdicts:
  - frozen primary: PARTIAL;
  - addendum secondary: NO (scoped);
  - replication: R0–R4 each 3/4 or better, with all criteria jointly passed in only 2/4 seeds.

**Evidence labels.**
- **[C]** confirmed prospectively.
- **[PH]** post hoc.
- **[T]** theorem.

## 1. Does any independently identified relation connect to the O/D formalism?

**Candidate object.** The best candidate "discrepancy" in this family is the per-group disagreement count M − n_g. It is an edge function on the source graph: 1 on edges whose endpoints disagree.

**Is it a coboundary?** Yes, trivially. Disagreement on the edge (s, t) of a categorical labelling is δ(label)(s, t) ≠ 0. That says nothing beyond "the two values differ".

**What the decision needs** (Theorems S, U, D):
- (i) the counts;
- (ii) the generator's role semantics (the a_r and the configuration prior) to turn counts into weights;
- (iii) the target question's own values.

The obstruction quantities O_c and D_c contain at most (i) and part of (iii). They do not contain (ii).

### Obstacles

**Obstacle 1: no sign without semantics [T, D2].**
- The same change in discrepancy (a group's count going from 4 to 5) has opposite Bayes-optimal effects on candidate margins under T1 and under T4.
- Measured: 16.6% sign agreement on boundary edits.
- So no function of discrepancy alone can carry the decision direction. This is the precise form of F2 and F3.

**Obstacle 2: same discrepancy, different decisions, both correct [T, F25].**
- Under T4, two groups with identical agreement counts can receive opposite routing, depending on the *other* groups' counts.
- Example: with counts (6, 2, 2), group 0 must be the colluder; with (6, 6, 2), either full-agreement group may be the colluder.

**Obstacle 3: no unique representation [T, I5, I6; E3–E5].**
- Any O/D quantity computed from a chosen representation, map or head inherits that choice.
- Head counts, the side (query or key) of a reliability tag, and its coordinates are not function-level invariants.
- The invariant objects are the input-level response law (E2) and, inside the Transformers, the readout attention pattern. Neither is an obstruction value.

**Obstacle 4: an exact computational equation can hold for a wrong decision.**
- Off the support (PP2), the learned computations remain internally consistent: tag, routing and evidence merge all still operate.
- Yet their decisions depart from the shifted generator's Bayes answer.
- An internal "compatibility" measure would not flag those decisions.

### Determination

The original O/D framework is **not useful for the discovered mechanism**. Its only overlap is the elementary fact that disagreement is a coboundary-type edge function. Everything decision-relevant lies in source semantics that O/D does not encode.

**What a repaired version would need:** a role model per source and a likelihood map from discrepancy to role. Adding those makes it Bayesian latent-role inference, which is H1 and not a new obstruction theory.

**Status of this conclusion.** It is a restricted result for this generator family. It is not a universal impossibility.

## 2. Failure issues F1–F7, F25, F29, F30: evidence from this step

| id | failure | evidence this step | status change |
|---|---|---|---|
| F1 | missing noncircular causal bridge | the decision-level bridge (E2) is confirmed prospectively across five implementations [C, PP1]. The relation (agreement counts) is defined without logits. The sign rule's inputs are computed from the generator. The bridge runs through Bayes semantics, not through O/D | **advanced (restricted)**: a noncircular, implementation-independent relation → decision law exists *for this family and its training support* |
| F2 | lower discrepancy can worsen a candidate | theorem-level: Theorem U and the phase diagram give the exact region where more agreement lowers reliability [T]. Learned models follow it at the boundary [C, PP1] | **characterised**: not merely "can"; when and how much |
| F3 | consistency ranking ≠ preference | D2 [T] | unchanged in substance; sharpened |
| F4 | no distinctive prospective prediction | PP1 passed, but it is a Bayes prediction. PP2 failed as frozen; the failure is informative but not a success | **partial**: a prospective prediction survived; it is not distinctive of COR |
| F5 | representation and map non-identifiability | I5 (exact head split vs a preregistered criterion), I6 (Q/K placement), E3–E5 ladder [T/C] | **confirmed, with precise scope**: the response law and the routing pattern are invariant; representations and head sets are not |
| F6 | output optimisation hidden in repairs | all subspaces from input contrasts; head sets by all-subset enumeration | procedural progress |
| F7 | outcome-dependent tuning | PP1–PP5 frozen before running; PP2 and PP5 failures kept; post hoc analyses labelled | procedural progress |
| F25 | same consistency, different decisions | Obstacle 2 [T] | sharpened |
| F29 | no advantage over existing methods | `BASELINES_AND_NOVELTY_AUDIT.md`: the mathematics is known (Dawid–Skene, Dong 2009 copy detection, Bayes) | **unchanged** |
| F30 | excessive unfrozen choices | the remaining conventions are listed (linear-subspace class, sites, k) | partial |

## 3. Which parts of the original COR ambition are possible, conditional, unsupported or unidentifiable

**Possible, and demonstrated (restricted).** Internal relations among source representations causally shape candidate decisions:
- through reliability-gated routing (6/7 seeds);
- or through direct rereading (1/7).

This holds for an independently defined, generator-grounded relation.

**Conditional.** Directional (sign) statements hold only given the source semantics (D1), and only on the training support (PP1 vs PP2).

**Unsupported.** A generator-free consistency or obstruction score that ranks candidates or fixes the response direction (D2, F2, F3).

**Unidentifiable from behaviour.**
- Where the relation is represented (I6).
- How many components implement it (I5).
- In the I2 regime, whether agreement means honesty or collusion at all.
