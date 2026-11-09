# Baselines and novelty — Investigation III

Prior-art references carried over from the continuation audit (verified there unless marked [m] = from memory, not
re-verified here). No new web search was run for this package; claims of possible novelty are therefore weak.

| result | closest established work | classification |
|---|---|---|
| TRANS T1/T2 (factorisation; uniform factor ⇒ uniform composite; entropy of products) | elementary Bayes; entropy of convolutions on finite groups [m] | **known mathematics**, applied |
| OD2 (NSR ⇔ COND for K ≥ 3) — closes the continuation's open "only if" | weighted vs simple majority coincidence (Nitzan–Paroush 1982; Ben-Yashar–Paroush 2001; Taylor–Zwicker 1999 simple games) | **known in principle**; the exact equivalence with discrepancy order is a short proof; restricted useful theorem |
| AD2 camouflaged copier non-identifiable | atlas I2; label-switching non-identifiability | **known in principle** |
| AD1 static counts insufficient under change points | change-point / HMM filtering (standard) | **known mathematics** |
| P-count transfer replicates over 5 seeds; P-group does not (2/3 fail); P-local fails (2/2) | curriculum / transfer learning; auxiliary-task pretraining | **new learned-model observation** (prospectively confirmed, Q3-A) |
| abstract-variable interchange (generator-defined linear encoder; same procedure across models) | interchange interventions / causal abstraction (Geiger et al.); DAS-type alignment (fitted to outputs — here fitted to the generator variable instead) [m] | **methodological variant** of known causal-abstraction tests; useful, not new in principle |
| P-count vs T1: same function transferred, different site and code | underspecification / representational non-uniqueness | **newly documented learned implementation** |
| GATE: redundant backup routes, seed-dependent layers, one seed with a single necessary last-layer head | "hydra effect" / self-repair (McGrath et al. 2023) [m]; backup heads in IOI (Wang et al. 2023) [m] | **instance** of known redundancy phenomena; **causal-identification result** for this task |
| TRANS aux run: maps represented at the statistical ceiling yet not composed | compositional generalisation failures (SCAN, COGS) [m]; probing ≠ use (Ravichander et al. 2021; Elazar et al. 2021) [m] | **instance**; the specific demonstration (ceiling-level map codes, chance-level composed decisions, causal null) is a clean new observation in this programme |
| TRANS holdout run ≈ training-prior Bayes on held-out pairs | low task diversity ⇒ Bayes under the pretraining prior (Raventós et al. 2023) | **instance** of a known phenomenon |
| ADAPT curriculum model uses positional evidence (history intervention 0.88, confirmed prospectively) | in-context change detection / filtering in sequence models [m] | **new learned-model observation** in this programme |
| failed internal extrapolation predictors (P-A1; Q3-D2 see outcome) | underspecification and OOD unpredictability (D'Amour et al. 2020) | **negative result consistent with known theory** (EX1 is trivial) |

Strong baselines used: exact Bayes in every task; restricted-prior Bayes (TRANS holdout); static-count Bayes (ADAPT); "no
τ evidence" reference (TRANS); matched head / random / variance-matched subspace controls (GATE, P-count); random-network and
raw-input probe controls (TRANS); architecture label, constants and behavioural probes (Q3-D).

Net: no new mathematical theory. The genuinely informative findings are learned-model and causal-identification results:
the replicated P-count transfer with its causally identified readout code; GATE's seed-dependent but structurally shared
circuit; represented-but-uncomposed maps in TRANS; prior-fitting instead of composition; positional reliability tracking
under a curriculum.
