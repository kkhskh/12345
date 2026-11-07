# 07 — Identifiability and leakage

The score depends on `(P, ρ, M, W, K)` and, in the repair objective, on `β`. Theorem NFL already says that even a fully frozen tuple does not determine `L`. This file records the further fact that the tuple itself is not fixed by the definitions. Freezing it on a training split removes one leak and leaves the others.

## 1. What could fix each object

| Object | Architecture fixes it? | Natural activations can estimate it? | Needs labels or outputs? | Analyst choice that remains |
|---|---|---|---|---|
| Graph `K` | Only the full computational graph, which these stalks are not | No | No | Which sites, which parents, whether edges are duplicated |
| `P` | No | A probe fit needs a target. The target may be a label, a board, or a logit | Often yes | Rank, loss, which coordinates count as the stalk |
| `ρ` | Only if it is set equal to a declared Jacobian of a declared parent set | Ridge on pairs `(s_u, s_v)` | Uses the stalk, hence whatever `P` used | Shared versus input-specific; ridge penalty; Jacobian versus regression |
| `M`, `W` | No | No | No | Any positive-definite pair |
| `β`, `τ` | No | Only by fitting responses | Yes if fit to `Δm` | The entire scale |

A train/test split on `ρ` stops `ρ` from being a function of the evaluation prompts. It does not stop `P=0` from being a legal map. It does not stop `M` from being chosen after the ranking is known. A high-capacity probe fit on the training split can memorize training labels. It does not, merely by being high-capacity, encode the answers on a disjoint evaluation split. A probe that is allowed to ignore its input can still match a constant algorithm, which is CE-18, and that vacuity is about the alignment class rather than about a shared linear residual.

## 2. Attacks

**Projection collapse (CE-08).** `P=0` gives `s=0` and `O=D=0` on every input, including wrong decisions. The definitions still hold. Assumption that would block it: `P` fixed by a predeclared probe with a predeclared rank and a nondegeneracy constraint such as `P P^T = I`. That constraint is a convention. It does not make `s` causal.

**Outcome encoding.** If the stalk is allowed to be the logit, `s` contains `L` and a consistency score on `s` is a consistency score on a quantity that already is the decision. The score has not predicted the decision from something else.

**Outer-product map (CE-09).** For one example with `s_u ≠ 0`, `ρ = s_v s_u^T / ||s_u||^2` forces `ρ s_u = s_v`. An input-specific family of such maps forces `O=0` on every evaluated example. Assumption that would block it: one `ρ` per edge, shared by all inputs, frozen before evaluation. The shared-`ρ` assumption is necessary and is still not sufficient (CE-06: the shared least-squares map can leave a large residual that is only a dropped coordinate).

**Metric reversal (CE-07).** `s_1=(1,0,0,0)`, `s_2=(0,0,1,0)`, two identity edges. `M=diag(10,10,1,1)` ranks `D(s_1)>D(s_2)`. `M=diag(1,1,10,10)` reverses the ranking. The same reversal holds for `W=diag(10,1)` versus `diag(1,10)`. Blocking assumption: one metric frozen without reference to the evaluation ranking. The frozen metric is a convention. Two analysts can freeze different conventions and get opposite candidate orderings from the same activations.

**Graph refinement (CE-12).** Duplicating a mismatched edge doubles Euclidean `O`. Blocking assumption: a frozen graph. Two frozen graphs still disagree.

**Coordinate change.** File 08. Euclidean `O` moves under a legal rescaling of a stalk unless `W` is transformed with it.

**Input-specific overfitting.** The outer-product attack above.

**Candidate-label leakage.** `P^c` may depend on `c` by notation. If the fit for `P^c` uses the correctness of `c` on the evaluation items, the stalk contains the label. Blocking assumption: fit `P` on a training split that is not the evaluation split. This is necessary and does not identify a unique `P`.

**Validation leakage.** Choosing `β` in `O - β m`, or `τ` in Update Proposition 2, on the evaluation `Δm` makes the reported relationship a fit. Blocking assumption: freeze `β` and `τ` before seeing `Δm`. A frozen `β` still defines an objective that contains `m`. Its parameter gradient uses `∇m`. The scalar `O` itself is not that gradient. The objective is a declared mixture of obstruction and margin, not an obstruction mechanism.

**Overexpressive alignment (CE-18).** A lookup that ignores the activation and emits the constant `1` matches a constant high-level algorithm on every input, while the network outputs `(0, 1)`. Sutter, Minder, Hofmann, and Pimentel (2025) prove a general version: without a restriction on the alignment class, any network can be aligned with any algorithm, including a randomly initialized network on IOI at perfect interchange-intervention accuracy. A `ρ` or `P` class that is rich enough to contain a lookup makes a zero residual uninformative. Train/test separation does not repair that. A class can generalize a vacuous alignment.

## 3. Why these blocks are not repairs of the bridge

Each blocking assumption is a pre-registration constraint. These degrees of freedom show that Definitions 2–4 do not pin down a unique score. They do not show that a fully specified intervention response is unidentified: once the network, the input, the site, and `δ` are fixed, recomputation defines `R_c`. After the conventions are frozen, the negative statement that remains is Theorem Z together with the two-head construction: the frozen residual still does not determine the sign the clamped network produces. Identifiability of a residual is not identification of a cause.

## WHERE DOES THE CAUSAL CONTENT ENTER?

It would have to enter in a constraint that ties the frozen tuple to `L` before `L` is measured. No such constraint is in Definitions 2–4. Adding `L = q(D)` adds the content by hand and reopens the leak if `q` is fit on `L`.
