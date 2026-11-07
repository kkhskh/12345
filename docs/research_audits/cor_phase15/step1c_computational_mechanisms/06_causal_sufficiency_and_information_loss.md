# 06 — Causal sufficiency and information loss

## A1. The question

A candidate-related representation is `z=P(h)`. The question is whether `z` determines the response `R(h,\delta)` of a named intervention family. Exact sufficiency, approximate sufficiency, a restricted family, several projections together, and a relation among those projections are different questions.

## A2. Exact sufficiency is fiber constancy

There is a function `g` such that `R(h,\delta)=g(P(h),\delta)` on a set of states and a family of interventions if and only if equal projections imply equal responses on that set and that family. The proof is the definition of a function on the fibers (`proofs/fiber_and_trace.md`).

This is necessary and sufficient for exact determination. It does not say that a useful `P` exists. It does not say that an approximate equality of projections yields an approximate equality of responses. Approximate sufficiency needs a modulus of continuity for `R` in a topology that `P` controls. If `R` can differ by a fixed gap on pairs with `P(h)=P(h')`, no modulus exists, and the approximation question is already closed for that `P`.

The blind-statistic theorem is the special case in which the fiber of the statistic contains two interventions with opposite response signs. N1 states it for a linear `Q` and a linear response: if the domain contains both `\Delta` and `-\Delta` in `\ker Q`, with `L(\Delta)\neq 0`, no function of `(Q(h),Q(h+\Delta))` matches both signs. The measurement has to be the statistic the predictor is allowed to see. A kernel of some other matrix is irrelevant unless it sits inside the kernel of that statistic. That is the correction in `01`.

What can still be predicted when sign sufficiency fails:

- Nothing about the sign, on that domain, from that statistic alone.
- A magnitude bound, if a separate Lipschitz hypothesis is assumed on the full map. The bound uses the Lipschitz constant, not `Q`. If the constant is taken from the full Jacobian, the predictor is the Jacobian, and `Q` is not doing the work.
- The response on a smaller family, if the opposite-sign displacements are excluded and the exclusion is checked independently. Excluding them by requiring the sign in advance is circular.

N1 does not say that every causal question is a sign question. A bound that is silent on sign can be true and still be the ordinary Lipschitz bound.

## A3. The same projection, two attention outputs

`P` keeps coordinate 0. Queries and keys are read from coordinate 1. Values are read from coordinate 0, so the two states below have the same values.

\[
X_a=\begin{pmatrix}1&0\\ 0&0\end{pmatrix},\qquad
X_b=\begin{pmatrix}1&0\\ 0&4\end{pmatrix}.
\]

`P(X_a)=P(X_b)`. Causal attention with those maps sends token 1 to output `1/2` on `X_a`. On `X_b` the legal scores are `(0, 16)`, so the output is `1/(1+e^{16})`, about `1.125\times 10^{-7}`, not `0`. The weight on the token whose value is `0` is `e^{16}/(1+e^{16})`, not `1`. The outputs differ. The lost coordinate is a key and a query feature. The projection never recorded the coordinate that moved the pattern.

The intervention family is total recomputation of attention from `X`. It is not a value-only clamp. Under a value-only clamp the pattern is fixed, the output is linear in `V`, and a projection that determines `V` determines the output. Sufficiency depends on the intervention family. A projection that is enough for clamped-pattern value edits is not enough for residual edits that move `Q` and `K`.

## A4. Several projections, and relations

If `P_1` and `P_2` each drop a coordinate the block still reads, the pair can still fail. If the concatenation `(P_1 h, P_2 h)` separates the fibers of `R`, the concatenation is sufficient. That is the same theorem applied to a larger measurement. Step 1B's N2 is this existence fact for linear factors. The sufficient object is the tuple.

A relation `ψ(z_u,z_v)` is a function of the tuple. It is sufficient for `R` only when `R` is constant on the fibers of `ψ`. The difference `z_v-z_u` does not separate `(0,0)` from `(1,1)`. A response that equals `z_v` differs on those pairs. Connecting two representations by a formula does not add information beyond the pair. It removes information unless `ψ` is invertible in the coordinates `R` actually uses.

There is a case where a relation looks stronger than either factor: each factor is constant and the relation is not, or each factor fails to be sufficient while the pair succeeds. The information is in the pair. Reporting only the relation is a choice to discard part of the pair. That choice is sufficient only under a further invariance — for example, `R` depends on the two states through `z_v-\rho z_u` alone — and that invariance is a factorization hypothesis of the same kind Phase 1 already isolated. It has to be established independently. It is not a consequence of writing `\rho` down.

## A5. What this is

The theory of when a summary supports a causal prediction, in the families examined here, is the fiber theorem plus N1 plus the attention counterexample.

A summary determines a response exactly when the response is constant on the summary's fibers, for the intervention family named in the claim. A linear summary that is blind to a direction cannot determine a linear response that uses that direction, once both orientations are in the domain. A projection that drops a coordinate read by `Q` or `K` does not determine residual-edit responses of attention, even when it determines the values. Several summaries become sufficient only as a joint measurement. A relation computed from them is a further summary, not an additional source of information.

No row of this theory says that a small compatibility residual produces a better decision.
