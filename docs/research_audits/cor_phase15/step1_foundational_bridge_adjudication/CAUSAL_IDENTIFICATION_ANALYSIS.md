# Causal identification analysis

## Role of each consistency quantity (mandate §7)

| Quantity | (1) computed by the network | (2) proxy for an operation | (3) property of a representation | (4) predictor of response | (5) constraint on a family | (6) post hoc statistic |
|---|---|---|---|---|---|---|
| consensus `M` (per head, on `ũ`) | **yes**: the head output projected on `ũ` | — | — | — | — | — |
| deviation `v'_j − M` | not as a variable; it enters the increment of a computed variable only multiplied by the weight change (CI.3*) | ordinary softmax normalization | — | yes for the direct numerator term; equals head patching | — | — |
| "writer alignment" in `N_C` | it is `‖PΣc‖²`, the norm of the sum | — | of the sum, not of the writers | the norm channel (NC); not a relation (SS) | — | — |
| agreement indicator (semantic) | no | — | — | no: fibers mix signs (FIB.1) | — | design label |
| relation-blindness of `m` | no | — | — | yes: forced orders (RB.6) | **yes**: a property of `m` on the family | — |
| `O`, `D` | no | surrogate error | of fitted cochains | no (1A) | — | **yes** |

## Intervention semantics

`do(C = c)` is never used. Every intervention is on a structural variable: an input token, a site's value, or an addition at the final residual. A relation is never intervened on directly. Several interventions can produce the same relation value with different effects. This is exactly what SS (same-cut relations) and FIB.1 (the agreement indicator) prove. The causal claims are always of the form "an intervention on `X` changes `m` by an amount that equals a function of the relation". That is a structural identity (CI, NC), not a claim that the relation is a manipulable cause.

The claim is therefore restricted as follows. Softmax reallocation and final-norm scaling are the operations through which other sources can affect a source's effect. The relations are not causal variables with their own `do`. Neither operation is a dedicated, consistency-sensitive mechanism (route 2.1 fails), and the corresponding predictor reduces to head patching (route 2.2 adds nothing distinctive).

## Independence from outcomes

The predictors use `w`, `v`, `M` (forward pass up to the head), `ũ`, `κ` (weights), and `a`, `σ` (base-state forward pass). They do not use the response `R`. The base margin `m` enters NC only through `sign(a) = sign(m − κ)`. That is the current state, not the response. This is the same status as "the current margin" in any response analysis, and it is disclosed.

## The open premise

A4 (path dominance) would be the only premise carrying causal information about the *decision* beyond the direct term. After review it is **rejected as a bridge premise**. It asserts that the response has the sign of a computed quantity, so the decision-level prediction would be A4 restated (review C, FATAL-4). Checking it on a calibration family needs an extra transport assumption. Even in one attention-only layer, the σ channel violates it in about 15% of random cases (review B, S3).

## Statistical identifiability of RB in a real study (forward-looking, not executed)

Logits are deterministic, so ties and noise come only from the choice of prompts within a cell, if cells pool prompts. If cells are single prompts, RB is a deterministic test. If cells pool prompts, order-restricted inference is required (Iverson and Falmagne 1985, as cited in the conjoint-measurement literature). That would be a protocol obligation for any future authorized test.
