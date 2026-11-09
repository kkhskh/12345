# Open foundational map (continuation)

Each open problem is listed with what it depends on and what evidence would resolve it. Directions that were not examined at all are listed separately (§3). Nothing here proposes closure.

## 1. Open problems, by dependency

| id | problem | depends on | what would resolve it |
|---|---|---|---|
| OF-1 | **A useful distribution-level sign theorem.** C4 is vacuous at the observed ε (review R2). At density ratio ≥ 7.7 the learned models still get every \|Δm\| ≥ 1 sign right. Reviewer FM's multi-cell construction shows C7 is realisable for a constructed model | C1 / C4 / C7 | a bound in terms of the *distribution* of pointwise endpoint errors rather than its mean (for example, quantile-based), validated against the high-ratio family; a *learned* model that fails on a high-W family |
| OF-2b | **When do transferable features exist during T1 training?** The 1000-step-fine-tune snapshots are confounded (R3) | PP3 | snapshots every 100 steps, scored with the 2000-step protocol and with readout-token probes |
| OF-2 | **Why a seed lands in class A′, B or C.** No account links training dynamics to class | PP3 dynamics (partial, §TRANSFERRED_FEATURE) | many seeds with snapshots, plus a predictor of class from early-training quantities, frozen before the seeds finish |
| OF-3 | **Seed 12's cross-question channels.** There are at least three besides own-question routing (other sources' values, other readouts, the readout residual). Under frozen routing, 0.40 of the tag effect survives. No low-dimensional variable is localised | R1; COMPUTATIONAL_CLASSES §1 | path patching that separates the value path from LN renormalisation; a variance-matched necessity protocol; more class-C seeds |
| OF-4 | **The PP2 direction mechanism.** Revised hypothesis after the probe control: global extrapolation goes with representations carrying *graded* agreement counts, and local extrapolation with a *saturated* full-agreement detector | P-A outcome | (i) a probe-free causal test: inject graded-count features into a local Transformer and check whether it turns global; (ii) DeepSets with saturating pooled features; (iii) many Transformer seeds |
| OF-5 | **R1 beyond pattern-preserving reparametrisations.** Attribution invariance under value-equal mass moves (L1) | R1 | either an impossibility theorem for pattern-sensitive attributions under the full function-preserving group, or an attribution on a coarser invariant (for example the readout attention pattern quotiented by value-equivalence classes) |
| OF-6 | **OD1 beyond independence.** The "only if trivial semantics" claim was false (OD1′, review R2). Open: does a weight-coincidence condition exist for *dependent* sources (for example a correlated pair, which empirically gave consistency 1.000)? | OD1′ | a characterisation for product-form-violating likelihoods, or a counterexample |
| OF-7b | **Is an intermediate competitor necessary for the U-union failure?** The exact check shows 45/49 failures with one; reviewer FM's grid found none without | U-scope | a proof, or a counterexample without an intermediate-agreement role |
| OF-7 | **The multi-dimensional U.** For groups of size ≥ 3, the agreement statistic is a vector (the partition of the group's reports). Is the role posterior quasi-concave along each coordinate? | U | a proof or counterexample using `roles_engine` extended to size-3 groups |
| OF-8 | **A learned mechanism under unknown or changing groups.** The atlas PP4 models failed to learn T1-perm beyond the own-question level. The continuation adds the bound (≤ 1/3 on the full pairing under T1) but no learned mechanism | I1′; the training budget | a model that beats own-question Bayes with unknown pairing, plus a mechanism analysis. T4-perm is the identifiable case (mass 0.64 at Q = 6) and the natural first target |
| OF-9 | **Mixture-α learned behaviour.** Do models trained on a two-peak generator (U-scope counterexample) reproduce the two-peak reliability response? | U-scope | train on `roles_engine` mixture generators and run the PP1-style edit test on both peaks |
| OF-10 | **The beyond-reliability families.** GATE is solved at Bayes level, but its last-layer routing is causally unnecessary (R3). TRANS composition was not learned in a truncated run | §11 | GATE: path patching through the layer-2 MLP and readout residual, plus more seeds. TRANS: a full 6000+ step run or a σ-first curriculum, plus a τ probe |
| OF-11 | **P-A power.** Four held-out models | P-A | ≥ 10 held-out seeds per architecture, with the same frozen predictor |

## 2. Corrections now depending on other corrections

- The atlas claim "the routing pattern is the most invariant internal object" is now qualified twice:
  - by R1, which is an attribution-level invariance;
  - by L1, which shows value-equal mass moves change patterns.
- Any future "invariant mechanism" claim must state which of the two equivalence relations it uses.
- PP1's interpretation is downgraded from "implementation-independent causal law" to "agreement wherever true effects exceed the models' pointwise margin errors" (empirical coverage; C4 is vacuous). Every downstream document relying on PP1 as evidence *about mechanism* inherits that downgrade. That includes the atlas H3 row.

## 3. Directions explicitly not examined

- **Adaptive sources**, whose reliability changes within an episode in response to other sources (mandate §10 lists adaptive sources). Not implemented.
- **More than one colluding group, or colluders that copy a reliable source** (rather than an independent common value). `roles_engine` supports configurations of this kind, but no phase diagram was computed for them.
- **Learned models on any landscape generator** (biased errors, informative colluder, mixture α). Bayes level only.
- **Changing group membership across questions** (§9.3). Not implemented.
- **Architectures other than the 3-layer Transformer and DeepSets** (for example an MLP over group histograms on T4) for the PP2 direction question.
- **Larger K or Q** for any learned-model result.
