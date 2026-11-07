# Phase 1.5 final verdict

**PHASE15_NECESSARY_CONDITIONS_AND_BOUNDARIES_ESTABLISHED**

No candidate dossier survives. The strong negative is Theorem U together with Theorem L: a non-constant edit has no finite bound from budget B0, and no Lipschitz certificate undercuts Cauchy–Schwarz. The conditional kernel equality is a real identity and a rejected successor.

`PHASE_2 = NO GO`. `OBSTRUCTION_READY = NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` is unchanged.

Operational completion, mathematical completion, and scientific readiness are separated at the end. They are not the same fact.

## Gates

These are the gates for this reconstruction. They are not the Phase 1 matrix in `docs/research_audits/cor_phase1/13_stop_go_gate_matrix.md`.

| gate | result | reference |
|---|---|---|
| A. Preserved negatives rechecked, not rewritten | PASS | `final/MATHEMATICAL_RECONSTRUCTION.md`; tests for `O≠D`, both-deletion, value-only sign, remainder, fiber, final LN |
| B. Extraction tests and manifest | PASS | `logs/clean_room.txt`, `logs/manifest_check.txt` |
| C. Deep question carried to a boundary | PASS | Theorems U, L, K, J in `workstream_1/finite_response.md` |
| D. `do` and `R` fixed before the bound | PASS | Opening of workstream 1 |
| E. Two routes that are not one function | PASS | Closed forms versus recomputed `response`; SymPy versus numeric LN |
| F. Sufficiency | CONDITIONAL | Feature 3 fails; kernel sufficiency holds only under its hypothesis and adds no non-constant edit when `QKV` is injective |
| G. Residual taxonomy | PASS | `workstream_3/compatibility.md` |
| H. Where the sign enters | PASS | Skip dot, `Dm`, or `R`. Not `O`, not `[ds]` |
| I. Competitors | PASS | Direct `R`, gradient, Hessian-scale ratio, Lipschitz, value-only path, interchange / abstraction |
| J. Dropped assumptions | PASS | `workstream_6/adversarial_audit.md` |
| K. Novelty | PASS | Matrix in `final/NOVELTY_AND_PRIOR_WORK.md`; no successor |
| L. Scope | CONDITIONAL | One pre-norm block. Stacks, post-norm, RMSNorm, rotary, and trained points are unresolved and are not claimed |

## Forty questions

**Q1.** What is the intervention and the response?  
`do(H_t <- H_t+δ)` replaces one residual token and recomputes LayerNorm, causal attention, the second LayerNorm, and the `tanh` MLP. `R` is the change in `w·H2[r]`. The derivative `Dm` is not that number.

**Q2.** Does `D` decreasing raise `L`?  
No. On the identity edge the zero cochain has `D=0` and a smaller first-coordinate readout than `s=(1,0)`, which has `D=1/2`.

**Q3.** Does `D_c < D_{c'}` rank the margins?  
No. One cochain and two readouts `w` and `-w` share `(O,D)` and reverse the margin.

**Q4.** Is `O` equal to `D`?  
Only for the special edge metric `W=(d M^{-1} d^T)^+`. Euclidean weights give `O=1` and `D=1/2` on `s=(1,0,0,0)`.

**Q5.** Does `D` rank decisions?  
No. It is a quadratic form of `ds`. The head is an extra datum.

**Q6.** Is a zero discrepancy a correct decision?  
No. `P=0` yields the zero cochain on every input. The kernel contains that zero whether the readout is positive or negative.

**Q7.** Does the kernel contain `0`?  
Yes. `d` is linear, so `d0=0`.

**Q8.** Is an exact coboundary the zero cohomology class, and does the class constrain `R`?  
`ds=d(s)` lies in `im(d)`, so the class in `C^1/im(d)` is `0`. The zero functional does not depend on `w` or on `δ` and cannot constrain `R`.

**Q9.** Can unrestricted maps manufacture agreement?  
Yes. Sutter, Minder, Hofmann, and Pimentel, Theorem 1: under their Assumptions 1–5, any algorithm is an input-restricted distributed abstraction of any network. A rank-one `ρ` from one pair is the small version: it zeros that pair by construction.

**Q10.** Is a gradient transport of the absolute state?  
No. The first-order expansion keeps the value `m(H)`. Applying `J` to `H` drops the offset. The seed-0 direction orthogonal to `g` has linear term about `0` and finite response about `-0.629` at step `0.2`.

**Q11.** Does nonlinearity imply a nonzero finite interaction?  
No. `u^2+v^2` has four-term interaction `0`. `(u+v)^2` has interaction `2uv`. Both deletions can be necessary for `b=-1.5`, `c1=c2=1` while the interaction is `0`.

**Q12.** Does changing only `M` change `O`?  
No. `O=(ds)^T W(ds)` does not read `M`. The check with `M=diag(4,4,1,1)` moves `D` and holds `O` fixed.

**Q13.** What is the Jacobian rank for `ε>0`?  
The bracket has kernel `span{1}` and norm `1/sqrt(v+ε)`. If every gain is nonzero, `rank J=d-1`. One zero gain, with the others nonzero, leaves that rank because the image of the bracket is `1^⊥`.

**Q14.** What is the rank at `ε=0`?  
For `v>0` and invertible gain, the kernel is `span{1,z}` and the rank is `d-2`. That sentence is not applied at `ε>0`.

**Q15.** Does `c1=1`, `c2=-0.5` make both deletions necessary?  
No. The sum is `0.5>0`. Deleting `c2` leaves `1`. A positive sum admits at most one necessary deletion-to-zero.

**Q16.** Can a normalization-invisible direction pass through the skip?  
Yes. A constant is invisible to both LayerNorms and is added by the residual. If the token is read, `R=a 1^T w`.

**Q17.** Do raw residual identities apply to a final-normalized logit?  
No. Replacing the readout by `w·LN(H2[r])` kills the constant the raw dot keeps. The two responses differ by `a 1^T w`.

**Q18.** What does the constant-shift identity say?  
`H2(H+E+C)=H2(H+E)+C` for a one-token constant `C`, every branch weight, every `ε≥0`. The interaction of `C` with `E` is zero. It is pre-norm plus addition. It does not determine a centered edit.

**Q19.** Does the centered part depend on the branch weights?  
Yes. On seed 7 the live block and the zero-branch block disagree on a centered edit at the read token. Theorem U makes the dependence unbounded.

**Q20.** Can a value-only linearization have the opposite sign from the total derivative?  
Yes. Seed 16: total `+0.02019538349844552`, value-only `-0.008580691798845158`.

**Q21.** Can a dropped coordinate hide opposite signs?  
Yes. Seed 16, coordinate 3 increased by `2` at token 0, intervention `(0.4,-1.2,0.3,0.5)` at token 1. The kept coordinates agree and the responses have opposite signs.

**Q22.** Does `Dm=0` imply `R=0`?  
No. The seed-0 direction orthogonal to the token-1 gradient has linear term about `0` and finite response about `-0.629` at step `0.2`.

**Q23.** Is there a non-constant family, a quantity free of `R` and `Dm`, and a bound tighter than the shift identity and Cauchy–Schwarz?  
Not on budget B0: Theorem U. Not by a Lipschitz certificate: Theorem L. On budget B1, only under a kernel hypothesis that, when `QKV` is injective and gains are full, contains no non-constant edit. The manufactured point meets the numerical comparison and is not that generic family.

**Q24.** What is the finite level set of LayerNorm at `ε>0`?  
The constants, if at most one gain is zero. A strictly larger set exists when two gains vanish: `γ=(1,1,0,0)`, the algebraic pair in the catalog.

**Q25.** Can budget B0 bound the centered response?  
Only by `+∞`, uniformly over branch weights. The scale family gives `R=α ||w||^2/T` with `w=(0.2,-0.1,0,0.3)` and `T=3`.

**Q26.** Can a Lipschitz certificate undercut Cauchy–Schwarz?  
No, when `m` is differentiable at the point and the inequality holds on a convex neighborhood. The derivative test forces `Λ≥||g||`.

**Q27.** When does a non-constant edit have an exact skip response?  
When its finite LayerNorm increments lie in the branch input kernels, Theorem K gives `R` equal to the skip dot. Another cancellation could make the read-token branch increment orthogonal to `w` without those products vanishing. The seed-0 search did not find a centered edit whose whole branch increment is zero, and that absence is not a proof. The kernel route itself is empty of non-constant edits on the injective ensemble.

**Q28.** Does Jacobian membership give the finite identity?  
No. Theorem J: the linear skip matches, and at step `0.8` the finite gap is about `0.00741`, increasing at steps `1.5`, `2.5`, and `4`.

**Q29.** Do relations among representations add information beyond the tuple?  
No. A relation is a function of the tuple. The feature-3 tuple does not separate the seed-16 responses, and neither does any function of it.

**Q30.** What separates redundancy, necessity, and synergy?  
Information redundancy is agreement of response fibers. Causal redundancy is a deletion that preserves a sign. Additive necessity can hold for both deletions with interaction zero, as with baseline `-1.5`. Synergy, here, is a nonzero four-term interaction. Softmax reweighting is compensation inside one equation. These four move independently.

**Q31.** What does a nonzero compatibility residual mean?  
One of: surrogate error, missing parents, off-manifold stitch, incomplete abstraction, genuine interaction, a changed definition, or numerics. The scalar norm of the worked pair does not separate a missing parent `-p` from a surrogate `+p`. They cannot be separated by a functional that has thrown those data away.

**Q32.** Does a gradient-built metric inherit the response?  
Yes. `(g·δ)^2` is an even function of the linear response and changes when `g` changes. Euclidean `O` on a fixed cochain does not move when an unused readout is negated, so it does not inherit the sign either.

**Q33.** Does cohomology add a causal theorem?  
No. The class is zero. It was removed from the prospective mechanism.

**Q34.** Is there a unique intrinsic decision germ?  
No. `P=0` and an invertible `P` are both legal and do not define the same state. Sufficiency, pre-registration, invariance, and uniqueness do not hold together for any examined `P` that is not the full state, the derivative, or the constant-shift symmetry.

**Q35.** Did any distinctive prediction survive?  
No. The kernel specification has a model class, an outcome, a quantity, a family, a budget, a relation, a falsifier, and competitors. It fails because on injective weights it is the shift identity, and off those weights the witness was built to satisfy the hypothesis. `O` and `D` do not predict `sign(R)`.

**Q36.** Which claims stay conditional, and which assumption was dropped?  
Theorem K stays conditional on the kernel products. Theorem F's converse needs the gain count; dropping a second gain produces the algebraic counterexample. Theorem U needs a non-constant edit; a constant edit is invariant under the scale family. The exotic zero stays a conjecture in the open-cell sense. The historical `D→L` claim is disproved, not conditional.

**Q37.** What is new relative to the primary papers?  
Theorem F is a finite-level calculation on LayerNorm, not an abstraction theorem. Theorem L is the mean-value inequality. Theorem U uses pre-norm so that a constant does not see the branch and a centered increment does. Theorem K is an unread direction plus the skip, which is Geiger et al.'s hard intervention in the special case where the mechanism ignores the direction, together with Xiong-style pre-norm. Olivieri and Pérez Rodríguez's sensitivity field is a derivative and sits in budget B2. Sutter et al., Theorem 1, is why an unrestricted map is not evidence. Pîslar et al. combine models by an input partition; this package does not.

**Q38.** Does the name COR still fit?  
It fits the original quadratics as names. It does not fit the original ranking hypothesis, it does not fit a successor, and it does not fit an empirical claim. The broader ambition reduces to symmetries and unread subspaces that do not use those quadratics.

**Q39.** What is the single verdict, and what stays unresolved?  
The verdict is `PHASE15_NECESSARY_CONDITIONS_AND_BOUNDARIES_ESTABLISHED`. Unresolved: exotic branch zeros at generic weights, stacks, post-norm, RMSNorm blocks, rotary scores, trained operating points. None of those is a premise that would turn `O` into a cause.

**Q40.** What would a later argument have to add, and was it added?  
A non-constant family on an open set of weights, a quantity in B0 or B1, and a finite bound tighter than the shift identity and the Cauchy number. Budget B0 cannot. Lipschitz certificates cannot. The kernel family can, and only off that open set, at a point built for the purpose. The argument that was asked for was not added. Phase 2 stays closed.

## Completion, stated separately

**Operational completion.** The package is the directory `overnight_reconstruction`. From its root, with no `PYTHONPATH` set, `python3 -m pytest tests -q -p no:cacheprovider` and `python3 scripts/verify_manifest.py` are the checks. The extraction log and the manifest log record those exit codes. A failure of either log supersedes this paragraph.

**Mathematical completion.** Theorems F, U, L, K, and J are proved for this block under the quantifiers in the register. The exotic large-amplitude zero is not proved absent. That open cell does not restore a generic centered bound. Completion of the examined question is the boundary, not a universal claim about every future architecture.

**Scientific readiness.** There is no successor to freeze for a later experiment, no identified germ, and no distinctive prediction. Readiness for a COR program or for Phase 2 is absent. `OBSTRUCTION_READY` stays NO.
