# 05 — Local-to-global topology audit

## 1. Three different ideas

| # | Idea | In COR as written? |
|---|---|---|
| 1 | Observed local-state mismatch `ρ s_u ≠ s_v` | YES. This is `ds`. |
| 2 | Distance to globally consistent assignments `D(s)` | YES, as a metric on 0-cochains. |
| 3 | Nontrivial cohomological / gluing obstruction | NO for globally defined `s`. The manuscripts themselves forbid `[ds]∈H^1`. |

A large edge residual is idea 1. Idea 2 is least-squares to `ker d`. Idea 3 would require a class that can be nonzero.

## 2. Graph catalog

Let stalks be 1-D, `ρ` identity unless noted. Homogeneous constraints `ρ g_u = g_v`.

| Graph | Can observed `s` have `O>0`? | Does a nonzero `g∈ker d` exist? | Cycle inconsistency of **prescribed** local data? | Does `D>0` imply topology? |
|---|---|---|---|---|
| One edge | YES | YES, `span{(1,1)}` if `ρ=1` | NO. Any pair is a 0-cochain; mismatch is not a missing global object | NO |
| 3-node chain | YES | YES, global constants (for `ρ=I`) | NO | NO |
| Branching tree | YES | YES | NO | NO |
| DAG | YES | YES if maps are consistent as a diagram; otherwise `ker d={0}` still contains 0 | Failure of diagram commutativity is linear algebra | NO |
| Triangle | YES | YES for identity maps (constants) | For identity sheaf, cycles do not create extra constraints beyond path equality | Not automatically |
| Multiple paths `u→v` | YES | Path-disagreement of **maps** can force `ker d={0}` | This is overconstrained linear maps, still not `H^1` of a 0-cochain | NO |
| Disconnected | YES per component | Product of kernels | NO | NO |
| Empty edges | NO (`O=0`) | Whole `C^0` | NO | `D=0` always |

Distinction:

**Observed assignment fails to be a global section** (`s ∉ ker d`).

versus

**No compatible global assignment exists for prescribed local data on a subcomplex** (relative `H^1(K,U)`).

COR’s `D(s)` is the first. Update Proposition 5 / original Proposition 6 is the second, and is not what `O_c(x)` computes on a full 0-cochain.

Verified tree/single-edge case: `test_attack_i_tree_mismatch_is_not_cohomology`.

## 3. Cohomology requirements

To claim sheaf cohomology one needs: a site or cell complex, restriction maps that compose, `d^1 d^0=0`, and a class whose vanishing is independent of metric.

MVP and both PDFs **prove** that for a globally defined 0-cochain, `[ds]=0` in `H^1`. Therefore the operational obstruction cannot be that class.

Relative cohomology for partial sections is mentioned, not implemented, not measured on Transformers, and not connected to `Δm`.

Accurate description: **linear mismatch operator on a directed graph, plus a quadratic energy.** Not topology.

## 4. Nontrivial global sections

`0 ∈ ker d` always. Nonemptiness of `ker d` is vacuous.

For identity maps on a connected graph, `ker d` is a diagonal (“constant”) subspace. A constant vector can have zero obstruction and no candidate information (`test_attack_h_zero_obstruction_without_discriminative_content`).

The theory does not require: nonzero sections, normalization, boundary conditions, semantic preservation, or an output relationship. Homogeneous constraints make “a globally consistent state exists” true in a sense that does not imply a useful decision representation.

Existence of any global section ≠ existence of a causally meaningful section.
