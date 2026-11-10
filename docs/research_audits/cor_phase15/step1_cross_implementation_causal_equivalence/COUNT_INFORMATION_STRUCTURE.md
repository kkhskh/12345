# Count information structure of each implementation (mandate §5)

Sources: `logs/info_theory.json` (generator, exact), `logs/{cal,conf}/info_structure.json` (natural conditional dependence,
held-out R² and decoders), `logs/{cal,conf}/results.json` (`decoder_acc_per_component`, family support statistics).
Confirmation models unless marked (cal).

## 1. Generator-defined variables and their order (exact)

| variable | definition | relation to the decision at q |
|---|---|---|
| V = n^{(-q)} ∈ {0..5}³ | leave-q agreement counts of the three groups | sufficient for the relational component (Theorem A2), not minimal |
| π = π(n) | P(R \| X_{-q}) ∝ ρ^{n_g}[n_h = 5] | minimal sufficient; 43 classes on the 91 possible count vectors |
| n_g (one component) | one group's count | insufficient: Bayes accuracy 0.669–0.673 vs 0.745 |
| two components | | insufficient: 0.733–0.739 |
| tuple (n_{g(s)})_{s ∈ 6 tokens of q} | own-group count at each source token | information-equivalent to V when consistent (Theorem B1) |
| X_q alone | | 0.580 |

So: π ≺ V ≅ consistent tuple, strictly ordered by sufficiency; single components and pairs are strictly weaker than V and
incomparable with each other. The candidate's own evidence X_q is needed in addition for the decision (Theorem A3).

## 2. Discriminating counterfactual families (all support-checked)

| family | what changes | marginal support | same world | Bayes at q |
|---|---|---|---|---|
| cond | X_{-q} resampled from P(X_{-q} \| X_q) | 100 % | 60 % | changes |
| single-group edit | one A_{k,g}, k ≠ q (n_g ± 1, other components fixed) | 100 % (rejection) | 96 % | changes |
| π-preserving edit | two components ± 1 together, colluder set fixed | 100 % | 100 % | **exactly unchanged** (≤ 2e-15) |
| relabel | every other question's reports cyclically relabelled | 100 % | 100 % | unchanged, counts unchanged |

π-preserving edits exist in 81 % of decisions. No family uses off-support episodes; abstract (activation-level) interventions
off natural support are labelled as such (`off_natural_support`, CROSS_MODEL_CAUSAL_TRANSPORT.md §3).

## 3. What each implementation carries at its site (natural episodes, held-out)

| model | site | R² of site from additive count code | per-component decoder accuracy (pooled / per-position) | n decodable at the *other* site |
|---|---|---|---|---|
| P2 | readout 36+q | 0.951 | 0.717 / 0.730 | q's source tokens 0.85 |
| P3 | readout | 0.967 | 0.756 / 0.757 | 0.62 |
| P4 | readout | 0.947 | 0.776 / 0.785 | 0.74 |
| T2 | q's source tokens | per token, own-group 0.145 | 0.998 / 0.998 | readout 0.93 |
| T10 | source | 0.152 | 0.999 / 0.999 | 0.56 |
| T11 | source | 0.247 | 1.000 / 1.000 | 0.77 |

(cal: P0 readout R² 0.989, decoder 1.000; P1 0.883, 0.776; T0, T1 decoders 0.997, 1.000.)

**P-count readout: global but lossy.** Counts explain 95–97 % of the variance of the readout state, yet the three components
are recoverable at only 73–79 % per component (two of four P seeds in cal and conf are at this level; P0 reaches 1.00). The count
code at layer-2 input is therefore a non-injective image of n in most P seeds: different count vectors map to nearby states
(the pretraining objective was solved at the final layer, not at this site). Pairwise interaction features add nothing beyond
the additive code (R² 0.951→0.951, 0.967→0.968, 0.947→0.948; matched null +0.000–0.002): within this resolution the P readout
code is **additive across groups**, i.e. a sum of separately addressable group components at one token.

**T1 source tokens: own-group, slot-bound, exactly decodable.** A source token's state depends on its own group's count;
adding the other two groups' counts raises held-out R² by 0.070 (T2), 0.000 (T10), 0.002 (T11) against capacity-matched nulls
of −0.003 to −0.004 (cross-pair counts; shuffled other-group counts). T10 and T11 are strictly own-group; T2 carries some
other-group dependence (as did T0 in cal: +0.017 against null −0.002). Encoders fitted to the wrong group's count fail
causally (adjusted cond IIA −0.58 to −1.21 in every T model). Jointly, question q's six tokens decode n at 0.998–1.000, so the
distributed tuple carries V exactly and distinguishes π-equivalent count vectors (excluding Theorem D5's "log-odds only"
alternative for T models).

**Where existing evidence cannot distinguish.** For P seeds with lossy decoders (P1–P4) D5 is not excluded: the readout may
carry a coarsened statistic between π and n. The R² measure cannot tell an injective-but-curved code from a non-injective one;
decoder capacity (logistic, linear) is the limiting assumption.

## 4. Answer to §5

P-count (most seeds) carries a **centralised, additive, lossy** code of the full leave-q count vector at the readout token; T1
carries an **exact, distributed, slot-bound tuple of own-group counts** at question q's source tokens that is
information-equivalent to the full vector only jointly (one token or one group is insufficient: Theorem A5, Prediction I(b)).
They are information-equivalent at the level of V only for T models and the best P seed; for the other P seeds, P's site
holds strictly less than V.
