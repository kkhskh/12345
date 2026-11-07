# R1 — Mechanism construction and attack (A, B, C)

Reviewer role: adversarial mechanism construction. I judged the package only from its files. I share no context with the authors beyond those files. This is not independent peer review of the whole package.

**State reviewed.** Commit `2e9e30e`, `THEOREMS_AND_COUNTEREXAMPLES.md` sha256 `94cbc93f…`, `cbmath.py` `e6659e15…`. The working tree changed while I was reviewing: `experiments.py` gained `supplementary()`, and new docs were added. I read the supplementary log from the working tree.

**Reproduction.** I ran `python3 src/experiments.py` in a copy, so the package logs were not overwritten. The log matches the committed log line for line (31/33). `pytest tests -q`: 27 passed.

**My scripts.** All in `/tmp/r1/`: `attackA.py`, `attackA2.py`, `attackB.py`, `attackB2.py`, `attackB3.py`, `attackC.py`, `misc.py`. Each imports `src/cbmath.py` read-only from the copy.

No FATAL findings. The proofs of A1, B1 and C1 are correct within their stated hypotheses. The problems are missing hypotheses, scope overclaims, two mis-scored or powerless prereg tests, a repeated matching error, and coverage gaps.

---

## SERIOUS

**S1. A1 has no hypothesis linking reader coordinates to sources, and A1.3 is false with several heads.** (THEOREMS §2, A1 and A1.3)

A1 is stated in terms of reader coordinates `x = ⟨p,ζ⟩` and `y = ⟨r,ζ⟩`. Nothing in A1 requires `x` to depend on source A alone, or `y` on source B alone. The claim "sign m = sign(ab)" therefore needs a missing condition.

Evidence:
- Package supplementary run: when B is written at 60° or 30° to A, the grid error is 8.3% and 19.4%.
- My two-head construction (`attackA.py` (b)). Head 0 is the designed uniform head. Head 1 has `W_Q, W_K ≠ 0`. It attends to A when `b < 0` and writes `−9a·e1`. Then:
  - `sign(x)` depends on `b`. At `(1,−1)`, `x = −1.882`.
  - The block becomes "follow A": grid error against `sign(ab)` is **0.50**; against `sign(a)` it is **0.00**.
- So "any attention pattern that gives positive weight to both sources leaves sign m unchanged" holds only for a single head, or for heads whose OV maps all preserve sign on the source coordinates.

Correction:
- Add (A0): `x = f_A(a)` and `y = f_B(b)`, each depending on one source only and sign-preserving.
- Restrict A1.3 to that case.
- State that the "agreement" is between the reader's axes `p` and `r`, not between the source write directions.

**S2. The candidate-relativity claimed for B and C holds only for two candidates.** (THEOREMS §5; MAIN line 35; QK doc line 67)

Both mechanisms encode "supports c versus c′" as the sign of one scalar. I built a natural 3-candidate version of the QK gate (`attackB3.py`): evidence vectors in candidate space, score `λ⟨a,b⟩`, and B's value written onto the unembeddings.

| Case | Gate | Fixed routing |
|---|---|---|
| A = (1, 0.8, −1.8), B = (1, −1.8, 0.8). Both prefer c1, but ⟨a,b⟩ = −1.88. Prior +0.15 on c2. | w_B = 0.000, logits (0.598, 0.627, −1.075): **c2 wins** | c1 wins |
| A = (1, 0, −1), B = (0, 1, −1). Different top candidates, but ⟨a,b⟩ = +1 because both oppose c3. | w_B = 0.998: B's c2 vote is amplified; L2 − L1 = 1.62 | L2 − L1 = 0.10 |

So the relation is a global inner product, not agreement relative to a candidate pair. C's block 2 reads a one-dimensional `Ŝ` and has no K ≥ 3 analogue at all.

Correction: scope every "candidate-relative" claim to K = 2 with an antisymmetric scalar encoding, and report the K ≥ 3 counterexample.

**S3. B1's hypotheses do not exclude other writers to Δu.** (THEOREMS §3)

(B4) excludes only components that *read* the gated head's output. B1 then states `m = v_A + T` as a conclusion, which silently assumes nothing else writes to Δu.

Within (B1)–(B4) as written, I added a second layer-2 head with `W_Q = 0` whose value at B writes `−6ŷ·fU` (`attackB2.py`). Then:
- B1.3 (unanimity) fails in **348/450** agreeing grid cells.
- At `(1,1)`, `m = −0.492`.

Correction: add (B5), an exhaustive-path condition: the only source-dependent writers to Δu are `v_A` and the gated head. This matches C's "no other path to m".

**S4. PB6 is scored PASS but its additivity half failed.** (`experiments.py` PB6; `logs/experiments.txt` line 62)

The preregistration says: "the margin is additive … **and** the RB check is relation-blind". The code checks only relation-blindness. The additivity residual is 0.208, against a `qk_off` margin range of ±2.391 (`misc.py`).

Correction: record PB6 as FAIL on additivity and PASS on RB. The true count is **30/33**.

**S5. The PA11 "errors concentrate near the axes" test cannot fail.** (`experiments.py` PA11)

The pass criterion is: median of `min(|a|,|b|)` over error cells ≤ 0.75. Under the null, with errors spread uniformly over the grid, that median is already **0.5** (`attackA.py` (d)). At η = 0.01 there are no errors, so the criterion passes vacuously.

Correction: compare the error locations against the uniform null (for example, by permutation). Mark the "near the axes" claim as untested.

**S6. Mechanism C's matched-sum controls repeat the PA9 error.** (PC2, PC3; THEOREMS §4 says "matched-Ŝ pairs")

The pairs are matched on the raw source sum, not at the block-2 reader:
- Pair 1: Ŝ = 0.9853 vs 0.9736.
- Pair 2: Ŝ = 0.716 vs 0.647.

The reported ablated gap of 0.182 is attributed to "LN-scale coupling". That is a misdiagnosis: it comes from the unmatched Ŝ. Matching on post-LN Ŝ by root-finding (`attackC.py`):

| Model | Agreement gap at matched Ŝ |
|---|---|
| Designed | **3.941** |
| Block-1 MLP ablated | **0.0000** (exact) |
| Sum-only block 1 | 7.6e−13 |

The conclusion survives and is in fact cleaner, but the text is false as tested.

The same error recurs in the supplementary random-reader statistic, "output varies across the raw matched-sum triple 400/400". That statistic has no power: the sum-only reader also varies on that triple (1.270, 1.122, 0.830).

Correction: match at the reader's input everywhere, and drop or redo the 400/400 statistic.

**S7. "Two-block reconciliation" is a depth-2 MLP at one position, not rereading.** (`cbmath.reconciliation`)

- Block 1's attention has `W_Q = W_K = 0` (fixed uniform copy).
- Block 2 has `attn_on=False`.
- All of C's computation happens in the read residual.

Mandate §5C ("later head whose query depends on earlier combined evidence", "candidate-specific rereading") and §12.4 ("without artificially fixed routing") are not met by C. PC6, where one wide layer fits C with RMSE 0.032, is consistent with this. Only B has cross-layer attention composition.

Correction: add a C variant in which a block-2 head's query depends on `q̂` (or on the block-1 combination) and re-reads the source positions. Otherwise state plainly that deep rereading was examined only through B.

**S8. B and C have no perturbation or robustness analysis, and C's just-outside counterexamples were not built.** (mandate §12.3, §12.5)

My results:

- **B is robust** (`attackB.py`, `attackB2.py`):
  - Unanimity accuracy under weight noise: 1.000 (η = 0.03), 0.974 (η = 0.1), 0.720 (η = 0.3).
  - "B counts more when agreeing": 1.00, 0.97, 0.68 at the same η.
  - Parameter sweep, 36 configurations of λ, wU, kB: PB2 holds 36/36; PB5 has a nonempty band 36/36.
- **C1.1 is fragile** (`attackC.py`). At η = 0.03 the margin sign disagrees with `sign(a+b)` in **13.5%** of cells, with errors up to `|a+b| = 1.25`. A small leak of the candidate-blind `q` into Δu beats the weak `εŜ` term in the dead zone.
- **C's amplification is robust**: 20/20 (η = 0.03), 19/20 (η = 0.1), 9/20 (η = 0.3). Block-2 sweep: 26/27 configurations.
- **Block-3 re-reader** (ReLU on fU, threshold 0.15, gain 6): 48/156 cells violate `sign m = sign S`. At a matched-Ŝ pair with S > 0, `m_agree = −8.02` and `m_dis = −0.05`. Agreement now decides *against* net evidence.
- **Final LN with bias** (Δu·β = 1): 36 violations in C.
- **B with final LN + bias**: unanimity fails in 18/450 cells. THEOREMS reports only that LN bias produced no PB9-metric cell.

Correction: add these to the register. State C1.1's exact-pairing and Δu ⊥ e3 conditions as measure-zero conditions.

---

## MINOR

**M1. Mechanism A with moving attention was never run; it holds.** The package tested A only with clamped patterns. My single-head runs with random nonzero `W_Q, W_K` (sd 0.5, 1, 3; 50 seeds; 7200 cells each) give 0 sign errors with `|m| > 1e-9`. However, source weights fall to 8.7e−76, so the margin can vanish. The sign law then says nothing about the decision. Report this.

**M2. Labeling and reachability.**
- A1.3 cites PA3, which uses clamped patterns, but does not label it a mechanism-dissection intervention.
- PA8 and the post-hoc runs set `mid` to states such as `x = 3` that no input can reach: the supremum of `|x|` is √6 ≈ 2.449 at d = 6. Label these "off-manifold residual interventions".

**M3. B1 claim 4 assumes `s_0 = 0`.** It writes `w(0) = 1/(1+n)`; in general `w(0) = 1/(1 + n·e^{s_0})`.

**M4. A4 needs two unstated conditions:** Δu ⊥ 1 and a uniform final-LN gain. Otherwise the effective readout is `g⊙Δu`. With gain `1 + N(0, 0.3)`, grid error is 2.8% on average and 16.7% at worst (`attackA.py` (e)).

**M5. The PC2 prior was set to the midpoint of the observed margins.** The test therefore reduces to `amp_ok`. This is disclosed. Keep it labeled as an existence demonstration, not a prediction.

**M6. A with three candidates confirms it is not candidate-relative.**
- Unembeddings (e3, −e3, 0.5·e3): `(1,1)` and `(−1,−1)` give identical logits (1.411, −1.411, 0.706), so agreement on any side picks c1.
- With a +0.5 bias on c3, weak agreement `(0.3,0.3)` picks c3.

So A is a consistency classifier. Add this to the A doc.

---

## OK

- **Proofs.** A1 (each listed activation has an even part increasing on [0,∞)), A2, A3, B1 (within B1–B4) and C1 check out. They are near-tautological once the gate is assumed, which the package's own check 5 already concedes.
- **No sign of outcome tuning.** The preregistration (19:11) predates `cbmath` (19:13), but the 2-minute gap cannot rule out interactive runs. The B and C parameter sweeps above make the predictions insensitive to the chosen constants, so the point is moot.
- **PB3** is correctly labeled as dissection, and its 0.104 residual is disclosed. PB4's Q-only and K-only substitutions are correct single-site interchanges.
- **Non-candidate-relativity of A** is confirmed (PA13; M6).
- **Identifiability gauges PI1–PI5** reproduce.

---

## Coverage against mandate §5, §6 and §12 (what is missing)

1. **Higher-dimensional or vector evidence:** none for any family. This is the reason S2 went undetected.
2. **Perturbations:** A only. B and C were missing (S8).
3. **Nondegenerate routing:** missing for A (M1 now covers it) and for C (S7).
4. **Mechanism B:**
   - No strength control that holds `x̂ŷ` fixed while `|a|` and `|b|` vary.
   - No matched-state control.
   - Its only simpler baseline is `qk_off`. There is no comparison against an MLP-computed `ab` interaction feeding a fixed head.
5. **Mechanism C:** no correctly matched controls (S6), no failure construction outside its class (S8), and no rereading (S7).
6. **Several heads:** examined only for gauges (PI1), never as competing writers (S1, S3).
7. **More than two candidates:** not examined (S2, M6).

**Bearing on the verdict.** The mechanisms are explicit, hand-built gates; nothing here suggests they go beyond ordinary nonlinear interaction or QK composition. If S7 is not fixed, the deep-rereading class is materially unexamined. That puts the package at risk of `STEP1_INVESTIGATION_INCOMPLETE` under mandate §17.
