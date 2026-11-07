# 10 — What a projection determines

Let `P` delete feature 3 of every token. On seed 16, let `H'` be `H` with `H_{0,3}` increased by `2`. Then `P(H)=P(H')`. The intervention is token 1, `δ=(0.4,-1.2,0.3,0.5)`.

\[
R(H,1,δ) = -0.121187908, \qquad R(H',1,δ) = +0.108445402.
\]

The same reduced state, the same intervention, opposite response signs. No function of `(P(H), t, δ)` returns both signs. This is the fiber condition from Step 1C, realized on the recomputed block rather than on a two-token attention toy. The missing coordinate is a feature of an earlier token. It can enter queries of token 0, keys seen by token 1, values mixed into token 1, the first normalization of token 0, and, after the residual update, the second normalization and the MLP of every later token the mask allows. The example does not isolate which of those routes carries the sign change. It shows that deleting the coordinate is enough to make the responses disagree.

## Families

Sufficiency is a statement about a family. On the constant-shift family the response does not depend on `H` at all, so every projection is sufficient, including the projection that discards every coordinate. That fact is the symmetry in `05`. It does not make the discarded coordinates idle for other interventions. The feature-3 deletion fails on the seed-16 family: the two states agree after the deletion and their responses have opposite signs.

## Several projections

The pair `(P_3(H), H_{0,3})` separates `H` from `H'` in the example. `P_3` drops feature 3 at every token, and the two states already agreed on feature 3 except at token 0, so restoring that one coordinate separates this pair. It does not restore a coordinate that was not the one which differed. A scalar function of the pair can identify `H` with `H'` again. The example does not search over scalars. A function of a tuple cannot separate what the function collapses, and the tuple separates only what its coordinates separate.

No projection in this study was trained. Feature 3 was selected after checking that these two responses differ. The check and the reported pair are the same pair. There is no held-out response. The result is an existence proof that this coordinate deletion fails for this intervention. It is not a probe, and it is not evidence that the coordinate was fixed before `R` was seen.
