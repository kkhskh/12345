# 15 — Coordinates, parameters, and identification

## Three transformations

Changing the basis of `H` by an invertible `A`, and conjugating every weight that reads or writes `H`, preserves `m` if `w` is transformed as a covector. The intervention `E_{t,δ}` must be transformed as a vector in the same basis. The numerical coordinates of `δ` change. The physical edit does not.

LayerNorm is not that conjugation. It is a function inside the block. A fixed `w` applied to `H` and to `LN(H)` are different scalars. Step 1C's sign flip at `ε=0` remains a valid example of that difference. It is not an example about `ε>0` rank.

A positive rescaling of a `tanh` hidden unit is not function-preserving, because `tanh` is not positively homogeneous. The ReLU calculation from Step 1C is not repeated on this activation. Function preservation and normalization remain different operations.

## What is fixed without looking at `R`

The architecture, `ε`, the mask, and the parameter arrays are fixed by `numpy.random.default_rng` under the named seeds before a response is used as a score. NumPy 1.26.4 is the stream that produced the recorded floats. Feature 3 was not the output of a fitted probe. It was selected by exhibiting two states whose responses differ, and those responses are the ones reported. That is an existence proof. It is not a coordinate chosen independently of `R`. A reader who wants a coordinate that does not move `R` can take the all-ones direction on a non-read token: every state shift in that direction preserves `R` for constant interventions. The interesting coordinates are the ones the invariance does not kill.

`O` and `D` from the earlier audits are not functions of this block until a projection, a transport, and a quadratic form are chosen. Changing the quadratic form on the cochain still does not change a quadratic form that does not contain it. That Step 1C correction stands and is not a statement about `R`.

## Identification of a bridge

A candidate that predicts `sign(R)` must name the inputs in the budget table of `13`. If the inputs include `Dm`, the candidate is the derivative. If they include only `P(H)`, seed 16 is a counterexample. If they include the architecture and the readout index alone, the constant-shift family is predicted and the general family is not. There is no fourth input in this study, short of the weights themselves, that identifies the general sign.

Using the reported `R` to choose which coordinate to delete puts that pair in the selection. The fiber result is still a valid counterexample to sufficiency of the deleted coordinate, because the two states and the two responses are exhibited together. It is not a validation of a predictor on a new pair. No transport or metric in the tests was solved from `R`. The rank-one map was solved from one activation pair.
