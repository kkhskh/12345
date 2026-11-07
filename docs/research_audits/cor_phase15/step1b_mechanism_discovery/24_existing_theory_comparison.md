# 24 — Literature comparison

Sources checked in Step 1A against publisher pages, reused here with the new hypotheses. No new DOI was invented. Elazar et al. (2021) is cited from the TACL record already in the program; the fetch of `doi.org/10.1162/tacl_a_00359` timed out in Step 1A, so no extra theorem number from that PDF is claimed.

| Hypothesis | Closest prior work | What the work establishes | What it does not | Our remainder |
|---|---|---|---|---|
| H-SUF | Geiger et al., JMLR 2025 | Faithfulness under mechanism transformation | A coboundary score | None as COR |
| H-SUF estimation | Geiger et al., CLeaR 2024, DAS | Gradient search for distributed alignment | Independence from the intervention loss | Using DAS as `P` is not a forecast of that loss |
| Vacuous maps | Sutter et al., NeurIPS 2025 | Unrestricted alignments can fit any algorithm | Shared linear `ρ` is outside that class | Shared linear `ρ` still does not sign `R` |
| Probe ≠ use | Elazar et al., TACL 2021 | Removing readable information need not remove the behavior | Kernel causal effects in Othello | Direction 1 is the complementary insertion |
| Partial abstraction | Pîslar et al., CLeaR 2025 | Combine high-level models vs interchange accuracy | A consistency energy | — |
| H-COMPOSE, H-COORD | Path patching, mediation, Hessians (inside the JMLR unification) | Path effects and interactions | New COR object | Architecture can predict `I=0` vs not; the value is curvature |
| H-CORR | Classical coding theory, not a Transformer result | Majority vote corrects Hamming-1 | That a residual MLP is a decoder | EMPIRICALLY_UNTESTED |
| N1 | Information: a function constant on a line cannot determine a nonconstant function on that line | Elementary | A named theorem in the cited MI papers | The application to every relational COR-style summary is the audit content |

A skeptical researcher would call H-SUF causal abstraction, H-COMPOSE an interaction, PATH attribution, and N1 an information obstruction. Those names are accepted.
