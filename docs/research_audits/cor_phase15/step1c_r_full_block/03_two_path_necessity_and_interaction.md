# 03 — Necessity, redundancy, and interaction

Deletion means the chosen contribution is set to zero and every other term in the expression is left as written. The sign in question is the sign of the resulting scalar. A contribution is **necessary** for the positive sign, under that deletion, when the deleted expression is not strictly positive. It is **redundant** for the positive sign when the deleted expression remains strictly positive. One contribution is **sufficient** when the expression with every other varied contribution deleted remains strictly positive. The baseline, if it is not one of the deleted contributions, stays.

These words are properties of an intervention and a threshold. They are not properties of a formula in isolation.

## The zero-baseline sum

Let `m = c_1 + c_2 > 0`, and let deletion of `c_i` replace that coefficient by zero.

If both deletions fail to stay positive, then `c_2 ≤ 0` and `c_1 ≤ 0`, so `c_1+c_2 ≤ 0`, contradicting the hypothesis. Therefore at most one of the two deletions can be necessary.

The Step 1C pair `c_1=1`, `c_2=-0.5` has margin `0.5`. Deleting `c_2` leaves `1>0`. Deleting `c_1` leaves `-0.5`. Only `c_1` is necessary. The negative piece is not necessary for the positive sign. Removing it increases the margin. "Both signs appear in the sum" is true of this pair and is not the deletion claim.

## A biased sum

Let `m = b + c_1 + c_2` with `b=-1.5`, `c_1=c_2=1`. Then `m=0.5`. Deleting either contribution leaves `-0.5`. Both are necessary for the positive sign under this baseline and this deletion.

The map `(u,v) ↦ b+u+v` has four-term interaction

\[
I(u,v)=(b+u+v)-(b+u)-(b+v)+b = 0
\]

at every pair. Both-necessary does not require a nonzero interaction. The baseline does the work. Changing `b` from `0` to `-1.5` changes the necessity facts and does not change `I`.

A nonlinear example can also make both deletions necessary. It is not needed for the existence claim, and it must not be inferred from the biased sum.

## The other words

| word | definition used here | additive sum `b+c_1+c_2` |
|---|---|---|
| necessary | the named deletion misses the positive sign | possible, as above; impossible if `b=0` and both deletions are required |
| redundant | the named deletion keeps the positive sign | the deletion of `c_2=-0.5` from `1+c_2` |
| sufficient | that contribution, with the others deleted and the baseline kept, is itself positive | `c_1=1` is sufficient against `b=0`; neither `c_i=1` is sufficient against `b=-1.5` |
| interaction | the four-term contrast of two increments from a stated baseline is nonzero | identically zero |
| synergy | used here only as a synonym of nonzero interaction, so that it cannot silently mean "both necessary" | zero in the biased example, where both are necessary |
| compensation | one parent lies in the structural equation of another, so deleting the parent changes the other term when the equation is recomputed | does not happen in the sum: deletion does not change the other coefficient. A recomputation can change the other term and still miss the old margin |

Redundant causal effect is not redundant information. Two vectors can carry the same inner product against `w` and be far apart as vectors. That was already a Step 1C counterexample and is not repeated as a new mechanism.

Compensation, as the word is used here, means that one parent lies inside the equation of another. Softmax renormalization can move weight onto a remaining token when a logit is removed and the row is recomputed. That changes the other term. It need not restore the margin. The additive sum has no such edge. The row constraint of softmax is not the deletion model above, and it is not evidence that a second variable repairs the sign.

Interaction is baseline-dependent as a number when the baseline of the four-term contrast moves, and the identical vanishing of `I` for an affine map is not. Necessity is threshold-dependent and baseline-dependent even for an affine map. Keeping the three notions under one word is what made `c_2=-0.5` look necessary.
