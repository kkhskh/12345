# Post-results reviews and adjudication (mandate §13, "AFTER confirmation")

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Reviewers.** Two independent agent reviewers, **not human peer review**. Their reports are verbatim in `reviews/`:
- `POST_REVIEW_D.md`: causal interpretation, statistics, novelty;
- `POST_REVIEW_E.md`: integrity, provenance, reproduction, completeness.

**Verdicts.**
- **D:** the main claims stand only after corrections. Five interpretations must be withdrawn or relabelled.
- **E:** approved with mandatory corrections. The integrity and determinism of the frozen confirmation were independently verified:
  - rescoring gave identical bands;
  - the two interrupted runs were replayed bit-identically;
  - four further continuations and window states were bit-identical;
  - all checkpoint hashes match;
  - the freeze was pushed 9 s before the first confirmation job.

Every mandatory item was **accepted**. The table gives the resolution and where it was applied.

| # | finding | resolution | applied in |
|---|---|---|---|
| D1 | "value-vector learning is sufficient" is backwards; the route's attention share still falls by 63–72% of the block's fall under vsg/vsgo | Reworded: *removing* value(-and-output) learning through T is sufficient; a routing-state change following from it is not excluded; a statement about a modified learning rule | MAIN §1, Q10, Q27; FORWARD B3; MAP |
| D2 | M vsg sufficiency rests on a CONTRADICTED prediction; the joint P4 rule failed on both bases | vsg sufficiency labelled [expl]. Q12 says "vsg on M: no". The M elimination rests on M_P4_vsgo_switch, with the departure from the joint rule disclosed. The central-success eliminations are X8a and X8b only | MAIN §1, §2, Q10, Q12; FORWARD B3 |
| D3 | "V+O carries most of the E effect" interprets a PARTIAL | Replaced by "not established" | MAIN §1, §2, Q10; FORWARD B3 |
| D4 | "frz nil shown to be a moment artefact" is not established; freeze arms were called natural trajectories | Reworded as "consistent with, not established (E823 both partial; no contrast on M)". Freeze arms are not natural trajectories. Q7 rewritten with the intervention categories | MAIN §2, Q5, Q7; ROUTING C2/C3; TRANSPLANTS A3 |
| D5 | the M "combination" is a sufficient set; dev results labelled [frozen]; E X7 interaction undocumented | Rewritten as sufficient layer pairs, with what is established and what is not. [dev] labels corrected. The E X7 interaction is documented | MAIN §1, Q2, Q6, §2; TRANSPLANTS; MAP |
| D6 | the functional battery does not support Claim 2; "86 of 86 switches" is misdescribed; dev SR ranges were quoted | "Causal implementation reorganisation with a precision difference". Claim 2 is not established. Counts: 86 non-hybrid models, 40 switched and 46 not, 3 units per base. Confirmation SR ranges used. Hybrid asymmetry stated | MAIN §1, Q15–Q17; FUNCTIONAL; FROZEN Stage 6; MAP |
| D7 | "general disturbance eliminated" needs scope; randnorm and pnz are functionally inert | Scoped to the tested forms. Only lbl is functionally comparable. "Not weak controls" deleted. Step-100 numbers added | MAIN §1, Q14; NON-TARGETED D2/D4; MAP |
| D8 | "27 of 35" double-counts | Family-level tally added. Item counts are disclosed as non-independent | MAIN Q22–Q24; FROZEN |
| D9 | P3 is band inclusion, not equivalence; E–M comparisons are confounded with one initialisation seed per base | Stated | MAIN scope line, Q5, Q21; FROZEN; FORWARD B3 |
| D10 | the novelty section omits the closest prior work (Kleinman, Achille & Soatto, ICLR 2024; Chen et al. 2024; Singh et al. 2024; Cloud et al. 2024; Zhang, Bengio & Singer; Frankle, Schwab & Morcos; Juneja et al.; Lubana et al.) | Q28 rewritten in the reviewer's wording. The gradient-only deficit result is a replication of Kleinman et al.'s mechanism in a transformer, not new | MAIN Q28; MAP |
| D11 | this file was cited before it existed | Written (this document) | — |
| D12 | evidence labels missing or wrong on Q5, Q7, Q10, Q12, Q16, Q17 and Q22–Q31 | Labels added or corrected | MAIN §3 |
| D13 | "stable hybrid" unsupported (no readout beyond step 1211) | Removed; limitation stated | ROUTING C3; MAIN §4 |
| D14 | the E layer-0 headline needs the tautology caveat | Added next to the headline | MAIN §1 |
| D15 | "not yet visible in step-100 function" on M ignores the attention-share difference (B 0.167–0.170, I 0.138–0.139) | Qualified as "barely visible"; the share difference is stated | MAIN Q1; TRANSPLANTS |
| E1 | the functional battery was frozen 14 min after confirmation started | Chronology given with exact times. The battery is labelled [frozen-addendum], not part of the reviewed freeze | FROZEN chronology; FUNCTIONAL; MAIN labels and §4 |
| E2 | `score_func19.py` was written after results were read, and its F3 included INCONCLUSIVE(causal_NONE) models | Disclosed. The scorer now applies the literal rule: M_F3 16/16, with the 5 excluded models reported separately as F3x. No band changed | `scripts/score_func19.py`; FROZEN; FUNCTIONAL |
| E advisory | restart time; overwritten first-attempt logs; non-reproducible list order in score output; no log of the scoring session; driver script not in the freeze | Restart time corrected (23:50:27Z). Overwritten logs and list order disclosed. Scoring-session commands are recorded in COMPUTE_LOG from this point on. `scripts/conf_lanes19.sh` is listed in the archive manifest; it is not in the freeze | FROZEN; COMPUTE_LOG |
| D advisory | various | Reviewed. Wording changes adopted where compatible with the above. The rest is kept in `reviews/POST_REVIEW_D.md` | — |

## Claims withdrawn or weakened after the final reviews (mandate Q26)

1. "Value-vector learning through the route is sufficient on M." Now: *removing* V(+O) learning suffices [vsgo frozen; vsg expl]. A routing-state mediator is not excluded.
2. "V+O learning carries most of the effect on E." Now: not established.
3. "The frz nil is a moment artefact." Now: consistent with that, not established.
4. "M: an identified combination of layer 0 with a partner." Now: sufficient layer pairs. Single-component necessity is not established.
5. "Claim 2 in the weak sense." Now: causal implementation reorganisation with a precision difference.
6. "86 of 86 switches." Now: 86 non-hybrid models (40 switched).
7. "Stable hybrid." Now: hybrid at step 1211.
8. Novelty: the gradient-only-deficit finding is a replication of Kleinman et al. (2024), not a new increment.
9. "These are not weak controls." Now restricted to lbl.

## Claims that survived both reviews

- The selection is carried by step-100 parameters, not Adam moments (full-state level).
- E: the layer-0 state is sufficient and necessary. Neither its QK part nor its OV part is necessary alone.
- M: L0+L1 and L0+L2 are each sufficient; L0 alone is not; restoring L0+L2 prevents the switch.
- The tested general-disturbance forms (X8a, X8b) are eliminated on both bases. This is the mandate §19 success condition.
- On M, removing V+O learning through T, with the forward pass and per-step routing gradient intact, is sufficient (M_P4_vsgo_switch).
- Freezing routing parameters is nil on M and partial on E.
- Content loss with ordinary gradients (vrs) is sufficient on both bases.
- All of this holds within one parent, one initialisation seed per base and 3 data orders per base.
