# Adversarial cross-examination and adjudication

## Independence of the reviews

Four reviewers ran as **separate subagents with fresh contexts**:

- B: mathematics and counterexamples
- C: causal identification and representation (mandate roles C and E)
- D: Transformer architecture
- F: novelty and prior work

Each saw only the package files, plus the mandate (C) and prior audits for context. None saw the reasoning that produced the package. Each was told to modify only its own review file. That instruction was followed: `git status` showed no other changes. Reviewers B, C, and D ran their own probe scripts outside the package, and F used web search.

Roles A (strongest positive case), G (cross-examination), and H (final adjudication) were performed by the package author in the shared context. **This adjudication is therefore not independent.** It is offered as the author's response, and every point below is checkable against the review files.

## Strongest positive case (role A) against the strongest negative case (B/C/D)

**Positive.**
- The architecture contains exactly two relation-sensitive aggregators, softmax and the final norm, and their effects are exact identities on canonical variables.
- Construction D shows that "more agreeing evidence lowers the decision" is realizable.
- RB gives a metric-free criterion for when any relation can set direction.

That is a located, identified bridge with one open premise.

**Negative.**
- The located premise is the decision arrow itself (C, FATAL-4).
- The "relation" is DLA minus DLA, gated by a salience change (C, FATAL-2; B, S2; D, SERIOUS-5).
- The norm channel's "alignment" is the norm of the sum (B, S4; C, SERIOUS-2; D).
- RB is generic (C, FATAL-1).
- The prediction in the earlier main argument was false as stated (B, S2; C, FATAL-2).

**Confrontation.** The positive case's exact identities are not disputed by any reviewer. The dispute is whether they constitute a *consistency* bridge. The test is mandate §8, last paragraph: does the relation supply directional information not already supplied by the readout plus the weight change? CI.3* settles it. Hold the weight fixed, and other sources contribute only a positive rescaling. With the weight free, the sign is `sign(Δw)·sign(DLA_j − DLA_head)`. That is weight-space sensitivity on readout-defined quantities. The negative case wins.

## Finding-by-finding adjudication

| Finding | Accepted? | Action |
|---|---|---|
| C FATAL-1: RB invariant to relabeling, so generic | **Accepted** | RB reclassified as a necessary condition (RB.2 note; main §5) |
| C FATAL-2 / B S2 / D S5: the substitution sign is not `sign(v'−M)` | **Accepted** | CI.3 scoped; CI.3* added and tested; main §8 prediction withdrawn |
| C FATAL-3: route 2.1 not justified | **Accepted** | H1 reclassified: ordinary normalization, not a reconciliation mechanism |
| C FATAL-4: A4 encodes the outcome | **Accepted** | A4 rejected as a bridge premise (main §4) |
| C FATAL-5: substitution failure | **Accepted in part.** The investigation did address the original question and found that it reduces to normalization. The earlier verdict wording ("does not fail; it is located") was a substitution in the conclusion and is withdrawn | verdict changed |
| C SERIOUS-1: sign from the readout | Accepted | main §2 |
| C SERIOUS-2 / B S4 / D S3: alignment = norm of sum | Accepted | relation (C) withdrawn |
| C SERIOUS-3: CI.3 not realizable by substitution | Accepted | as above |
| C SERIOUS-4 / B S3 / D S6: CI signs `a`, not `m`; the σ channel | Accepted | PROOFS CI note; main §4 |
| C SERIOUS-5: open ledger | Accepted | main §3; `OUTCOME_LEAKAGE_AND_FREEZING.md` §2 |
| C SERIOUS-6: aggregation over contexts breaks metric-freeness | Accepted | freezing protocol: per-context RB with a pre-stated combination rule |
| C SERIOUS-7: RB.6 predicts from the null | Accepted | main §10 arrow 4 |
| C SERIOUS-8: straw-man baseline | Accepted | main §8 |
| C SERIOUS-9: files and logs missing | At review time, yes: the review ran while the files were being written. Resolved | logs and all files present (see `logs/`) |
| B S1: decision scalar must be frozen | Accepted | RB.2 note |
| B S5: NC.3 "any path" false | **Accepted (theorem clause corrected)** | PROOFS NC.3; new test with two flips |
| B M1: ties | Accepted | `has_crossover` strengthened; test added |
| B M2: box bound | Accepted | Hadamard bound stated; designs larger than 3×3 refused |
| B M3: raw writes not invariant | Accepted | main §3 |
| B M4–M10 | Accepted | wording fixed in PROOFS. "Multi-token edits" replaced by "single-row edits propagated to all positions" |
| D SERIOUS-1: RMSNorm `κ ≡ 0`, no flips | **Accepted and proved** | PROOFS NC scope; test added |
| D SERIOUS-2: NC is within `C`; effective null space; use `P·diag(γ)·W_U` | Accepted. The code already uses `γ⊙Δu` with centering, which is the same space | PROOFS NC scope |
| D SERIOUS-4: relational reading of CI exact only at layer 0 | Accepted | PROOFS CI note |
| D SERIOUS-7: attention sinks remove dilution | Accepted | PROOFS construction D |
| D SERIOUS-8: A4 fragile | Accepted | main §4 |
| D MINOR: "real block" overclaims | Accepted | "real architecture with designed weights" |
| F: CI-5 = Rushing and Nanda 2024 | Accepted | cited |
| F: RB.6 = Greco et al. 2008; Dunn and James 2003 | Accepted | cited |
| F: FIB attribution to Geiger et al. not found | Accepted | attribution withdrawn |
| F: Loftus 2×2 restriction | Accepted | main §8 no longer makes the unrestricted claim |

**Rejected findings.** None. One partial disagreement remains. Review C judged the identified relations "not consistency relations". The author holds that source-versus-consensus *is* a consistency relation in the ordinary sense. It is simply not one with a distinctive causal role, because it enters only through attention reallocation, multiplied by readout-defined attributions. Either way, the verdict is the same.

## Final adjudication (role H, same context; see the disclosure above)

No noncircular, distinctive consistency–decision bridge survives in the investigated classes. The output is a set of exact boundaries. **Verdict: `STEP1_NECESSARY_CONDITIONS_ONLY`.** Reviewer C recommended this label as primary. No reviewer argued for a stronger verdict.
