# Reviewer 3 — Causal inference

Provenance: subagent [causal and representation review](b51bc858-1cf6-4f05-b506-725c1387427a), this session, shared with Reviewer 4. Not an external causal theorist.

## What held

The fiber theorem in `proofs/fiber_and_trace.md`, with both the set of states and the family of interventions quantified. The recomputed-residual vacuity, scoped to assignment of the child. The scalar-versus-tuple claim. The split between editing `M` and a coordinated congruence. No derivation of `sign(R)` from `O` or `D`.

## Defects

1. `03` marked "the tuple determines a response that is a function of the tuple" as a proved implication. That sentence is the definition of a function of the tuple.
2. The `W_Q=0` arrow skipped LayerNorm and query bias. Same defect as Reviewer 2, from the causal side: the intervention family was not the one named by the pre-norm equation.
3. `05` said simultaneous clamps recompute descendants "using the imposed values," which reads as if the clamped descendant itself is recomputed, and said an ancestor effect "double-counts" a path. Truncation screens the path. It does not double-count it.
4. `03` reversed the fiber quantifiers. Equal response implying equal `P` is not sufficiency.

## Disposition

The tuple row is now labeled a definitional identity. The `W_Q=0` arrow is scoped to the bias-free map. The simultaneous-intervention row and the following paragraph now say the clamped node is not recomputed and the screened path is absent. The fiber row now says equal projections imply equal responses.
