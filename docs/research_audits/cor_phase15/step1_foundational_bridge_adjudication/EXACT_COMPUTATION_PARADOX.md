# The exact-computation paradox

**The dilemma (mandate §4).** If a compatibility equation is the model's true structural equation, every recomputed forward pass satisfies it. Satisfaction is then vacuous, and correct and incorrect decisions both satisfy it (1A Theorem Z; 1C-R Q17). If instead the equation is a projected approximation `P_v h_v ≈ ρ(P_u h_u)`, its residual measures surrogate error (1A Theorem App; 1C Q10).

**Resolution adopted here.** Consistency is not a property of one state with respect to an equation. It is a relation between **independently manipulable evidence variables**. Its causal role is defined as a property of the **decision function on an interventional family**: nonremovable interaction, meaning failure of relation-blindness (RB). Both horns are avoided:

- It is not vacuous. Whether `m(a,b) = ψ(f(a)+g(b))` holds is a substantive fact about the computation. It fails in Construction D and holds for a linear readout of a residual sum (SS).
- It is not surrogate error. Nothing is approximated. The relation acts through operations the network actually performs: a convex combination in each head (CI) and a norm in the final LN (NC).

The escape works for identification, but it does not produce a consistency mechanism. Both operations are ordinary normalization, and the relation-dependence they create is readout-defined (see the table).

## The five versions in mandate §4

| Version | What is consistent with what | Who defines it | Does the model use it? | When it fails | Why the decision changes | Can it hold for a wrong decision? Fail for a correct one? |
|---|---|---|---|---|---|---|
| **A. Represented claims** | evidence source A against B: same candidate or conflicting | task semantics, frozen in advance | Only where they are read separately: across positions by attention, across features by LN. Not at a single cut (SS) | conflict | through the aggregating operation; the sign needs strengths (FIB.1) | yes, both (CORR) |
| **B. Causal abstraction** | low-level head against high-level "weighted mean" | architecture: CI makes it exact | yes, exactly | never, for the head | — | the abstraction is exact whatever the correctness |
| **C. Counterfactual commitments** | the decision's orderings across a factorial family | the design | the family's order structure *is* the computation's behavior | RB violation | by definition it is a property of `m` | yes, both |
| **D. Reconciliation mechanism** | source against consensus `M`, a quantity the head computes | architecture | the head computes `M`; the deviation enters only multiplied by the weight change (CI.3*) | a deviating source pulls the head only if it gains weight | CI.1; but the σ channel and downstream paths can reverse the decision | yes, both |
| **E. Partial sufficient representations** | parts of a jointly sufficient state | analyst | only via the tuple; a relation adds no information beyond the tuple (1C Q9); at one cut, only via the sum (SS) | — | — | — |

**What survives.** As a *boundary*, D: other sources matter only via reallocation. It is ordinary softmax normalization, not a consistency mechanism. A is causally relevant only through D or the final norm. C is a generic interaction property (RB). B adds nothing to D. E reduces to sufficiency and causal abstraction, and SS makes its "agreement" reading inert at a cut. **No version escapes the dilemma while keeping a consistency-specific causal role.**

**Questions 10–14 of mandate §30.** A wrong decision can be computationally consistent: unanimous wrong evidence gives a positive head margin for the wrong candidate (CI.4). A correct decision can be inconsistent: conflicting evidence where the stronger side is right. Exact compatibility is vacuous (1A Z). Approximate compatibility measures surrogate error (1A App). The surviving object is separated from surrogate error because it involves no approximation. It is separated from "genuine decision conflict" by definition: conflict is a task-semantic relation between sources. What the model *does* with conflict is the empirical question (RB plus A4).

**Correctness is not the target.** By CORR, no internal relation determines correctness. The bridge concerns the decision response, which is what mandate §1 asks for.
