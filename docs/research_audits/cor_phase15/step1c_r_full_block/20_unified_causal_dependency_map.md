# 20 — Dependency map

Arrows are one of: STRUCTURAL EQUATION, PROVED IMPLICATION, CONDITIONAL IMPLICATION, MEASUREMENT RELATION, UNESTABLISHED CONNECTION. There is no statistical hypothesis in this step. There is no arrow from a discrepancy to the sign of `m`.

```text
H --STRUCTURAL EQUATION--> LN1 --STRUCTURAL EQUATION--> X
X --STRUCTURAL EQUATION--> (Q,K,V) --STRUCTURAL EQUATION--> P --STRUCTURAL EQUATION--> A
H --STRUCTURAL EQUATION--> H1 = H + A
H1 --STRUCTURAL EQUATION--> LN2 --STRUCTURAL EQUATION--> MLP --STRUCTURAL EQUATION--> H2
H2[2] --STRUCTURAL EQUATION--> m = w · H2[2]

E_{t,δ} --PROVED IMPLICATION--> dX = J_LN1 dH          (per token)
dX --PROVED IMPLICATION--> dA                          (Q, K, V, masked softmax, W_O)
dH, dA --PROVED IMPLICATION--> dH1 = dH + dA
dH1 --PROVED IMPLICATION--> dH2 = dH1 + dM(J_LN2 dH1)
dH2 --PROVED IMPLICATION--> Dm = w · dH2[2]

δ = a 1 --PROVED IMPLICATION--> dX = 0 and A unchanged
        --PROVED IMPLICATION--> R = a 1^T w or 0       (by the read token)

value-only clamp --MEASUREMENT RELATION--> a different intervention
                 --UNESTABLISHED CONNECTION--> sign of Dm
                 (seed 16: the signs differ)

P(H) --MEASUREMENT RELATION--> a reduced state
     --UNESTABLISHED CONNECTION--> sign of R
     (seed 16, feature 3: the signs differ)

structural residual of H1 = H+A(LN(H))
    --PROVED IMPLICATION--> 0 on every recomputed trace
    --UNESTABLISHED CONNECTION--> sign of m

surrogate ρ --MEASUREMENT RELATION--> approximation error
           --UNESTABLISHED CONNECTION--> decision conflict
           (seed 1: the residual moves and m does not)
```

The only proved arrow from a restricted property of `δ` to the finite `R`, which does not evaluate `A` or `M`, is the constant-shift arrow. Its hypothesis is the architecture. Its conclusion is the response on that line. Extending the arrow to a general `δ` is the unestablished connection the rest of the map refuses.
