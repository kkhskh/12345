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

These are suprema, approached as `|a| → ∞`. Within `|a| ≤ 3`, the largest conflict value is 1.376 (R1 F12).

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

**What X4 shows, and what it does not (corrected after review: R2 FATAL-1, R5 H1).**

- X4 shows that, on reachable inputs, **the block-1 MLP pathway is necessary for the decision difference**. Two real inputs with identical relation-blind margins reach opposite decisions.
- X4 does **not** show that the pathway computes a *relation*:
  - An additive-gate twin passes X4 identically. This twin has the same weights except a per-source AND gate in block 1: same blind pathway, same flips, same C2 shares, and 100% the same decisions.
  - So does the fully relation-free twin. Its blind margin is constant, so every pair counts as matched.
  - The two inputs also differ in more than the relation feature: their values differ too.
- The reviewed claim "the only computational difference is the relation feature" is **withdrawn**.

### 4b. The mediator sign test that does separate them (added after review)

Code is in `src/mediator_sign_test.py`; the log is `logs/mediator_sign_test.txt`.

| test | C′ | additive twin |
|---|---|---|
| S1: gate feature at R, signs (++, −−, +−, −+) at magnitudes (1, 0.7) | +2.99, +2.99, −2.99, −2.99 | +1.84, **−5.52**, −1.84, −1.84 |
| S1: candidate-symmetric pattern on 500 random magnitude draws | 500/500 | 0/500 |
| S2: interchange R's block-1 output from (−a, −b) into (a, −b) [off-manifold] | routes to sources 1.000 | routes to prior 1.000 |
| S3: graded single cancellation (T-RF′), order reversals out of 435 row pairs | 357 | 0 |

S1 and S2 identify the gate feature as a candidate-symmetric relation (D1). S3 is the behavioural (graded) signature of a cross-source interaction. **X4, S1 and S2 together** make the engineered identification chain complete.

## 5. B: matched uniform-routing margin (X6)

(1, 1) and (3.5, −0.055) have equal uniform-routing margins (+2.391). Their full margins are +5.192 and +2.405. The relation changes the magnitude, through routing toward B, but not the decision. B is a confidence modulator, not a decision switch, in these pairs.

## 6. Mediation levels (§12)

| mechanism | input association | controlled source intervention | mediation | necessity | sufficiency |
|---|---|---|---|---|---|
| A (designed) | yes | yes (recomputed I-flip: 0% sign kept) | the reader is the only path; `q` mediates 100% by construction (no bypass, X5 holds) | yes: removing the agreement units leaves only the sum statistic, so matched pairs are identical | yes, off-manifold (q is a function of `(u, v)`) |
| C′ | yes | yes (TE median −4.92) | The q_R interchange alone reproduces a median **100%** of TE in all three counterfactual families. The value interchange alone reproduces 1.8% (b-flip), **90.9% (a-flip)** and 75.3% (asymmetric). q_B carries 0 to −1%. The shares are **not additive**: each patch alone can be sufficient. This is **off-manifold** (C2-OFF). | the block-1 pathway is necessary (X4) | yes, off-manifold; relation identity by S1/S2 |
| C′, on-manifold | — | — | matched relation-blind pairs (X4) are on-manifold, but show only that the pathway is necessary | — | — |
| B | yes | yes | K-only and V-only patches separate the routes (Program B, B4) | routing ablation removes the agreement-dependent magnitude | yes |
| gain-redistributed C′ variant | yes | yes | **not unique**: ablating either copy alone preserves 200/200 decisions. The variant is not exactly function-preserving (Δm ≤ 0.33, R3) | per-copy necessity fails | — |
