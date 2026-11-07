# Relation versus aggregate: matched interventions (§11, §12)

Code is in `src/matched.py` (X1–X6) and `src/program_a.py` (A-matchS). Logs are `logs/matched.txt` and `logs/program_a.txt`. All interventions are recomputed source changes, so every compared state is reachable, unless a row is marked otherwise.

## 1. Raw-sum matching is not a matched comparison (X1)

At A's reader site, the pair (1, 1) vs (3, −1) gives:

| pair | u | v | Ŝ = u+v | ‖(u, v)‖ | m_A |
|---|---|---|---|---|---|
| (1, 1) | 0.795 | 0.795 | 1.589 | 1.124 | +1.411 |
| (3, −1) | 1.318 | −0.708 | 0.609 | 1.496 | −1.661 |

The two inputs share a raw sum, but the reader receives a different aggregate `Ŝ` and a different norm. A difference in margin therefore cannot be attributed to the relation. Proposition N1 in `MLP_NORMALIZATION_THEOREMS.md` explains why LN makes this unavoidable.

## 2. Exact post-LN aggregate matching (X2, A-matchS)

For a base pair that agrees, we solve by bisection for a conflicting pair with exactly the same `Ŝ` at the reader.

**Reachability theorem (R).** Ignoring ε, let

```
Ŝ_conf_max = d/sqrt(d + ρ²)
Ŝ_agr_max  = 2d/sqrt(2d + ρ²)
```

For A, with `d = 6` and `ρ = 3`, these are 1.549 and 2.619. A conflicting partner with aggregate `Ŝ` exists if and only if `|Ŝ| < Ŝ_conf_max`.

*Proof.*
1. `x = a/σ_A` increases strictly toward `√d`.
2. In conflict, `Ŝ = (x+y)/σ_R` with `y < 0`. Its supremum is approached as `x → √d` and `y → 0⁻`.
3. In agreement, the supremum is at `x = y → √d`. ∎

**Consequence.** Strongly agreeing states (`Ŝ ≥ 1.549`) have **no** aggregate-matched conflicting counterpart under this architecture. For example, (1, 1) has `Ŝ = 1.589`. The "same aggregate, different relation" comparison does not exist there, and no experiment can supply it.

**Reachable matched pairs (X2).** Example: base (0.5, 0.5) with `Ŝ = 0.934`, matched with (2.5, −0.452).

| reader | m(agree) | m(conflict) |
|---|---|---|
| designed A | +0.607 | −0.904 |
| sum-only | +0.607 | +0.607 (identical) |

**Six-reader comparison (A-matchS).** 188 reachable random matched pairs:

| reader | mean \|Δm\| | decision flips |
|---|---|---|
| R1 designed agreement | 1.260 | 1.000 |
| R4 reversed polarity | 1.260 | 1.000 |
| R2 net evidence | **0.000** | **0.000** |
| R3 generic interaction | 0.783 | 0.261 |
| R5 magnitude | 0.689 | 0.356 |
| R6 agreement + support | 1.260 | 0.293 |

## 3. Sufficient-statistic impossibility (X5; §11, last paragraph)

**Proposition SS.** Suppose every path from the sources to the readout passes through a site whose state is a function of a statistic `T(a, b)`, so there is no bypass. Then any two inputs with equal `T` give identical outputs, whatever their relation.

**Verified.** The sum-only reader gives identical margins, to below 1e−9, on every matched-`Ŝ` pair (assertion in `matched.py`).

**The converse limit (X3).** A's reader input `(u, v)` is an injective function of `(x, y)`. With `σ² = (ρ²/d + ε)/(1 − (u²+v²)/d)`, the inversion error is 6e−15.

So once *all* reader coordinates are matched, the relation is matched too. "Same reader state, different relation" is impossible for any reader that sees both sources. A relation-versus-aggregate claim is therefore meaningful only **relative to a declared set of non-relational coordinates**, such as `Ŝ`. It is not meaningful relative to "everything else".

## 4. C′: matching the relation-blind pathway (X4)

The aggregate here is the **relation-blind margin** `m_blind`: the same circuit with block 1's MLP removed, i.e. uniform block-2 routing. Every matched pair is reachable, and both members are actual inputs.

| base (agree) | m_blind | m | matched conflict | m_blind | m | flip |
|---|---|---|---|---|---|---|
| (1, 1) | +0.822 | +1.890 | (3, −0.238) | +0.822 | −2.630 | yes |
| (1.5, 0.5) | +0.845 | +2.141 | (3, −0.200) | +0.845 | −2.627 | yes |
| (0.6, 0.6) | +0.338 | +1.307 | (2, −0.634) | +0.338 | −2.631 | yes |

Strong agreement, e.g. (2, 2), has no conflicting partner with equal `m_blind`. This is the same reachability limit as in §2.

**This is the strongest on-manifold relation result in the package.** Two real inputs reach identical relation-blind computations and opposite decisions. The only computational difference between them is the relation feature, which enters through the block-2 query.

## 5. B: matched uniform-routing margin (X6)

(1, 1) and (3.5, −0.055) have equal uniform-routing margins (+2.391). Their full margins are +5.192 and +2.405. The relation changes the magnitude, through routing toward B, but not the decision. B is a confidence modulator, not a decision switch, in these pairs.

## 6. Mediation levels (§12)

| mechanism | input association | controlled source intervention | mediation | necessity | sufficiency |
|---|---|---|---|---|---|
| A (designed) | yes | yes (recomputed I-flip: 0% sign kept) | the reader is the only path; `q` mediates 100% by construction (no bypass, X5 holds) | yes: removing the agreement units leaves only the sum statistic, so matched pairs are identical | yes, off-manifold (q is a function of `(u, v)`) |
| C′ | yes | yes (TE median −4.92 for `(a,b) → (a,−b)`) | q_R interchange carries a median **100%** of TE, q_B 0%, block-2 scores 100%, values 1.8% (`PROGRAM_C`, C2). This is **off-manifold**: q_R is a deterministic function of R's e1/e2 coordinates, so the patched state is unreachable (proof in `DYNAMIC_ATTENTION_AND_DEEP_MECHANISMS.md` §C2; grid distance 5.49 vs a 0.35 resolution control) | yes: with block 1's MLP ablated, routing is uniform and the decision flips on matched pairs | yes, off-manifold |
| C′, on-manifold | — | — | **matched relation-blind pairs (X4) give an on-manifold contrast**: no off-manifold state is used | — | — |
| B | yes | yes | K-only and V-only patches separate the routes (Program B, B4) | routing ablation removes the agreement-dependent magnitude | yes |
| redundant C′ variant | yes | yes | **not unique**: ablating either copy alone preserves 200/200 decisions | per-copy necessity fails | — |
