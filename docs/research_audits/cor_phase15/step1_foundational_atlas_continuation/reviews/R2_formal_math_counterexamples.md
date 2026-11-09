# R2 review: formal mathematics and counterexample construction

Reviewer perspective: are the quantifiers, assumptions and proofs right, and do the stated counterexamples refute what they claim? Scope: THEOREMS_COUNTEREXAMPLES_AND_LIMITS.md (THM), PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md (PP1), RELATIONAL_GENERATIVE_LANDSCAPE.md (LAND), REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md §1 (REP), COR_TEN_FAILURE_MATRIX.md §2 (COR).

All checks are exact-Bayes numpy runs using the package's own engines (`src/roles_engine.py`, atlas `src/gen_family.py`). Each ran in under 3 minutes, one at a time, with no training. The check scripts are in `/tmp/r2chk/*.py`, outside the package. The commands and outputs that matter are quoted below.

Summary of severities: 2 critical, 7 major, 9 minor, plus one list of results that check out (F19).

---

## Critical

### F1 (critical). The "52% flip at ρ → 1" (THM §4 l.59–63, LAND §1.5 l.84–89) is a NaN artefact
`scripts/landscape_phase.py` (5) recomputes the posterior at assumed ρ = 1.0 on episodes sampled at ρ = 0.6. In most of these, every group disagrees somewhere in the leave-q questions, so every configuration has likelihood 0. `roles_engine.posterior` then computes `loo − loo.max()` = −inf − (−inf) = NaN. `argmax` of an all-NaN row returns 0. This is the same inf − inf failure the LAND "bug note" (l.10) says was fixed. The fix moved it but did not remove it.

`/tmp/r2chk/rho1.py`, same 3,000 episodes and seed as the script:
```
frac (episode,q) with NaN posterior at rho=1: 0.78
flip 0.95->1.0 overall 0.525   among NaN 0.655   among finite 0.060
argmax at NaN rows: [0]
flip 0.95->0.99 0.031 ; 0.95->0.999 0.026 ; 0.95->0.9999 0.0235
max|dp| (finite rows) rho=0.99 vs 1: 0.0123 ; 0.999: 0.00123 ; 0.9999: 0.000123
```
Measured properly (moving toward ρ = 1 from below), the flip fraction is 2–3%, the same as the interior steps. The "52%" headline and the "Consequence" built on it are unsupported: "T4 sits at a boundary point where the posterior is maximally sensitive", and "one structural reason why behaviour at T4 constrains behaviour at ρ < 1 … so little" (LAND l.89). Remove these. If ρ = 1 is to be scored, use the continuous extension: keep only the configurations with the minimal number of colluder disagreements.

### F2 (critical). The K2 counterexample (THM l.114) is impossible, and K2 itself is tautological
The counterexample says: "Take Bayes plus a perturbation supported on uniformly shifted inputs of measure zero under T1." But T1 has **full support** on the finite input space {0,1,2}^{6×6}. Reliable reports put probability α or (1−α)/(K−1) > 0 on every value, and noise reports put 1/K on every value. So no non-empty input set has T1-measure zero. In particular, inputs where all groups agree in a question k have positive probability. It follows that E_T1[KL(p ‖ p̂)] = 0 forces p̂ = p on **every** input. THM §4 (l.67–69) proves that the T1 Bayes posterior is in 𝒦_rel, so **every model that is KL-exact on T1 is in 𝒦_rel**. That is the opposite of the claimed [X], so "membership in 𝒦_rel is … not implied by behaviour on N" is false for exact behaviour. It can only be stated for ε > 0.

Separately, K2's conclusion ("the response to uniform shifts: none") follows from membership in 𝒦_rel alone. A uniform shift leaves (X_q, relative counts) unchanged by definition, so the KL-exactness hypothesis does no work. Either drop K2, or restate it as "ε-KL-closeness on T1 does not imply approximate membership in 𝒦_rel", with an explicit ε-construction.

---

## Major

### F3 (major). The C4 bound is vacuous on PP1, so it does not "explain the 0.996+ agreement"
The C4 constants are right. Pinsker gives t > τ ⇒ KL > 2τ². Markov and (DR) give W·ε/(2τ²) per endpoint. A union over the two endpoints gives ≤ W·ε/τ² when W ≥ 1. One detail: b_τ must be read as +∞ when τ ≥ π.

Evaluating the bound on the frozen scored population (47,005 edits) at the reported ε gives (`/tmp/r2chk/c4vac.py`):
```
pi_min quantiles (base) 5/50/95: 0.023 0.108 0.492
W=1.2 eps=0.0012: profile tau -> (margin tail, W eps/tau^2):
 0.005 (0.25, 57.6)  0.01 (0.31, 14.4)  0.02 (0.60, 3.6)  0.03 (0.79, 1.6)  0.05 (0.90, 0.58)  0.08 (1.00, 0.23)
min over tau of the C4 bound: 1.016   (W=1.2, eps 0.0008..0.0019: 1.011..1.025; W=2: 1.018..1.042)
```
The bound is ≥ 1 for every τ, every reported ε and W ∈ {1.2, 2, 4}. Several statements therefore have no support from C4:
- PP1 l.133–134: "C4 reduces to natural KL ≈ 10⁻³ plus a margin tail. That explains the 0.996+ agreement *mechanistically*."
- The corrected PP1 claim (l.142–148).
- COR l.24 ("a consequence of natural KL ≈ 10⁻³ with W ≈ 1 (C4)") and COR l.81 (F1 "explained").
- THM K1 (l.101–106): every observed class is in 𝒦_0.002, where K1 implies nothing.

The cause is that Pinsker + Markov on the average KL is far too loose when π_min is small (median 0.11). The 0.996 agreement is explained by the *pointwise* certificate C1 (99% certified), not by C4. These sentences should be rewritten to say so.

A related conflation: the corrected claim's type (i) is "within its error budget", that is, C4's b_τ. The empirical statement "all sign failures are of type (i)" actually uses the C1 errors |e(x)| + |e(x′)|. These are different sets.

### F4 (major). "W ≈ 1.2" is a coarse-grained lower bound on the W in (DR), and (DR) fails for the family as defined
- The ratio is computed on sorted agreement-count patterns (343 classes; `pp1_cx_matched.py` l.25–36). The supremum over coarse classes is ≤ the pointwise sup of dE/dN, so it is a *lower* bound on W. The exact pointwise ratio for the S1-restricted single-cell family (preimages enumerated, exact log N(x); `/tmp/r2chk/wratio.py`, 400 bases): **max 1.96, p99 1.69, median 1.08**. Recomputing the pattern-level ratio (`/tmp/r2chk/wpat.py`) gives max 1.36 at pattern (2,2,6), not 1.2.
- For the family as defined, 41,430 of 135,000 endpoints have N(x′) = 0, so dE/dN = ∞ there. C4 only applies after restricting to S1 and renormalising, which multiplies W by 1/P_E(S1) ≈ 1.44.
- Scoring further conditions on |Δm| > 0.5, so the base endpoint is no longer N-distributed. Its ratio can reach 135,000/47,005 ≈ 2.9. C4 assumes W = 1 for the base.

W is still O(1), but "W ≈ 1.2" should not be presented as the constant in C4.

### F5 (major). The C6 quantifier is false as stated ("for every generator with non-degenerate margins … sign agreement far from 1")
CX1, logits s(x)·(centred Bayes log-probs) with s > 0, preserves the **full** ranking at every x. So sign m̂_{c1,c2}(x) = sign m_{c1,c2}(x) at both endpoints, and every edit that swaps the order of c1 and c2 is necessarily signed correctly. On PP1 (`/tmp/r2chk/c6.py`):
```
frac c1/c2 order-swapping among scored: 0.142
CX1 agreement: overall 0.223 | on order-swapping 1.000 | on order-keeping 0.094
```
For any edit family concentrated on decision-flipping edits, the construction gives agreement 1. C6 needs a hypothesis, for example "E puts mass ≥ μ on order-preserving edits with Δm ≠ 0". Then agreement can be pushed down to about 1 − μ. A proof is also missing for "every generator": only one construction on one generator is checked.

### F6 (major). OD1's "iff" / "each assumption's failure breaks the bridge" is contradicted by the package's own log and by small extensions
- `logs/od_bridge.json`: `correlated_pair_kappa0.7` gives consistency **1.000** (unique-min-D 1.000), even though conditional independence is violated. THM l.85 prints this, yet l.90 says "Each assumption's failure breaks the bridge", and COR l.56–58 repeats it.
- `/tmp/r2chk/od.py` (20,000 episodes × 6 questions each):
```
unequal_0.8_0.75_0.7    argmax-consistency 1.0  unique-minD 1.0  full-ranking 1.0
unequal_0.85_0.8_0.75   argmax-consistency 1.0  unique-minD 1.0  full-ranking 1.0
corr_kappa0.7_two_pairs argmax-consistency 1.0  unique-minD 1.0  full-ranking 1.0
corr_kappa0.9_all_pairs argmax-consistency 1.0  ...
corr_kappa1.0_one_pair  argmax-consistency 0.978
```
The reason: with S = 6 and K = 3, the decision-level bridge holds whenever the log-odds weights w_s lie close enough together that no integer vote difference changes sign. A correlated honest pair is worth between 1 and 2 votes. So "the bridge holds iff source semantics are trivial" (THM l.94, COR F3 l.83) is false at the argmax level. It is refuted by unequal accuracies and by correlated pairs, which are exactly the families the package tested, only at extreme values. Only the "if" direction stands.

Also:
- The OD1 statement omits the **uniform prior on Y**, which the proof needs.
- The check is argmax with Bayes-favourable tie-breaking, while the statement is about rankings.

### F7 (major). V1's claim that biased errors do not move the boundary (LAND l.53) is false
The cited `phase.json` rows (4) only test a_c ∈ {0.2, 0.5, 0.8, 0.9}, all below 0.97, so they cannot locate a boundary. In theory, with error bias b toward (y+1), the reliable pair's ratio becomes (α/e_dir)², where e_dir = (1−α)(b + (1−b)/(K−1)) when v = c+1, and (1−α)(1−b)/(K−1) when v = c+2. The threshold therefore splits. `/tmp/r2chk/v1bias.py`, bias 0.8:
```
a_c=0.90: follow|v=c+1 0.207   a_c=0.92: 0.995   (predicted threshold 0.908)
follow|v=c+2 stays 0.01–0.02 up to a_c=0.97     (predicted threshold 0.9988)
```
V1 itself is correct for symmetric errors. A fine grid (`/tmp/r2chk/v1.py`) puts the jump between a_c = 0.9695 and 0.970 against a predicted 0.9697. K = 5, α = 0.6 jumps between 0.895 and 0.905 against a predicted 0.900. At ρ = 0.9 the threshold disappears, which is outside V1's ρ = 1 scope.

### F8 (major). The R1 limit L1 "impossibility" (THM l.42, REP l.48) is a tautology, and a non-trivial invariant exists
"No attribution … can be invariant to those *and* still distinguish heads with different patterns" holds only if "pattern-sensitive" means "changes whenever the raw pattern changes". Counter-construction: define classes by the **effective pattern**, Ā_h = attention mass aggregated over the positions whose value vectors (V_h rows) are identical for that head. Then:
- moving mass between equal-value positions leaves Ā_h unchanged, and so leaves the function unchanged;
- exact splits keep Ā_h;
- gauge keeps Ā_h;
- heads with different effective routing are still distinguished.

So the quotient game over effective-pattern classes is invariant under (a), (b) and the L1 moves. L1 should be restated as a limit of *raw-pattern* classes. This is a paper construction only; it was not run.

### F9 (major). The ρ = 1 "singularity" statement mixes scales (THM l.56–58, LAND l.86–87)
- On the probability scale, w_c(ρ) = (1−ρ)^{n_c} A_c / Σ_d (1−ρ)^{n_d} A_d is a ratio of polynomials. After dividing by (1−ρ)^{n_min} it is smooth with bounded derivative up to and including ρ = 1. F1's output confirms this: |Δp| scales linearly in 1 − ρ, with a Lipschitz constant of about 1.2. Only *log-odds* have unbounded sensitivity. The claim "the answers are therefore maximally sensitive to the copy rate as ρ → 1" does not follow: answers depend on probabilities.
- "Continuous in ρ on [0,1]" holds only through the continuous extension. At ρ = 1 the posterior is undefined (0/0) for x outside the ρ = 1 support, which is 78% of (episode, q) pairs here.
- The log-odds rate is (n_c − n_d)/(1−ρ) between configurations, not n/(1−ρ), and the agreeing factors also depend on ρ.

---

## Minor

### F10 (minor). C5 is correct with c1, c2 fixed, but its gloss overreaches
With the base pair fixed, temperature gives m̂ = m/T exactly, so C5 and the TEMP rows hold. But "monotone-transform invariance: temperature or ECE does not affect signs" (THM l.33) and "Calibration … neither necessary nor sufficient" (PP1 l.77–78) go further than C5:
1. Per-probability monotone recalibrations are not of the form φ(m). An example is uniform mixing p̂ = (1−λ)p + λ/K. Analytic counterexample: x: p = (0.5, 0.25, 0.25), x′: p = (0.06, 0.01, 0.93). Then Δm = +1.10, but with λ = 0.3, Δm̂ = 0.283 − 0.492 < 0. On the PP1 population this does not occur (λ ≤ 0.3 keeps sign agreement 1.000 at ECE 0.126; `/tmp/r2chk/c5.py`), but C5 does not cover it.
2. C5 shows only "not necessary". "Not sufficient" needs a separate witness. A trivial one is the constant marginal predictor: perfectly calibrated, with Δm̂ ≡ 0.

### F11 (minor). C7 needs a bounded-KL assumption, and its "open construction" closes trivially
"Arbitrary on A has KL ≤ η·B" requires sup_A KL ≤ B, for example probabilities bounded below. That assumption should be stated.

PP1 l.134–136 calls a W ≫ 1 family "open". A natural multi-cell family closes it: break the reliable pair's agreement in n_k other questions. Against the CX2 low-density-corrupted model (natural KL 0.013; `/tmp/r2chk/c7.py`):
```
n_k=1: endpoints below natural p2 0.022, sign agreement 0.993
n_k=3: 0.124, 0.967
n_k=5: 0.464, 0.875
```
So C7 does bite outside single-cell edits.

### F12 (minor). Theorem D's applicability row (THM l.18) is misleading
Absolute continuity fails inside S1. Edits that create a new perfect group give configurations with π = 0 → π′ > 0. `/tmp/r2chk/dac.py`: **3,264 S1 edits violate AC, all in S1 ∩ S2, none in S1 \ S2.** So the row's "S1 edits only" is not the domain of the identity.

### F13 (minor). The U-scope counterexample is real; the presentation should be exact and at fixed contexts
The worry was that the averaged table might be an artefact. It is not. Averaging the exact P(rel | X) over episodes with n₀ = n equals the exact posterior P(rel | n₀) by the tower property. This posterior has W_j = prior_j, so it is itself an instance of the U setting. Exact values (`/tmp/r2chk/uscope.py`): 0.2924, 0.3076, 0.3067, 0.2447, **0.1170**, 0.1537, 0.7390. These match the Monte Carlo values within 0.004 and are not quasi-concave.

At **fixed** other counts (Theorem U's own setting), the union is non-quasi-concave in **45 of 49** contexts, for example (n₁, n₂) = (0, 4): 0.499, 0.519, 0.525, 0.474, 0.301, 0.390, 0.910. Report this, not the average.

"The failure needs an intermediate-agreement competitor" (LAND l.76) is asserted without proof. A grid search found no counterexample: competitor above both reliable types, or below both; 25 × 25 α pairs × 21 ρ × 6 priors × Q ∈ {6, 10}; 0 hits (`usearch.py`, `usearch2.py`). Label it a conjecture.

### F14 (minor). The I2 line is exact and provable; the [N] label and "only there" are loose
The atlas I2 proof covers the whole line with a_rel = α² + (1−α)²/(K−1). Only the reliable group depends on y, its marginal pair law is "agree on a uniform value w.p. a_rel, else a uniform ordered distinct pair", and this matches the colluder exactly at ρ*. Exact check (`/tmp/r2chk/i2.py`): max |Δ log L| under the swap is ≤ 1.4·10⁻¹⁴ for K = 3, 5 and α = 0.7, 0.9; it is 0.8–1.4 at ρ* + 0.05. Upgrade to [P].

"Only there" is shown at ρ* ± 0.15 only. "Whole line" rests on three α values. Separately, LAND l.84 says "13% and 11% at the steps crossing ρ*", but only one step (0.45 → 0.50) crosses ρ* = 0.49.

### F15 (minor). The V1 check does not test V1's stated setting
V1 assumes known roles and no other evidence, so its decision is deterministic. The check (`landscape_checks.py` l.104–111) uses inferred roles, Q = 6 and a noise group, so P(follow) is 0.92, not 1. The jump agrees with the threshold (F7), but the check is evidence for a corollary, not for the statement.

### F16 (minor). S-without-(O) is valid but not new; its target should be named
The 0.458 spread is at the Y-posterior level with (X_q, n^{−q}) fixed. It refutes J1, and refutes S by contrapositive through Theorem M. Atlas Theorem S already added (O) after R1-a with an example (0.1366 vs 0.0043). "(O) is *needed*, not just used" adds nothing, and the spread statistic is "max deviation from the class mean", not a range.

### F17 (minor). The I1′ proof sketch establishes less than the statement
"The noise sources re-pair in 3 equivalent ways" equalises only the terms in which the true reliable pair is called reliable. Terms in which a noise pair of the true pairing is called reliable are not symmetric for a fixed x. So the per-episode bound ≤ 1/3 needs an argument. Numerically it holds: max 0.326 over 1,500 sampled episodes; adversarial search max 0.312 (`/tmp/r2chk/i1p.py`, `i1p2.py`).

### F18 (minor). Notation and labelling slips
1. THM l.68 writes P(g reliable | counts) ∝ Π (a_rel/a_n)^{n_g}…. There is no product: the configuration g has a single rel factor. The proof is otherwise correct.
2. "C1 is never violated in 10 models [E]" (THM l.29, PP1 l.104) checks an algebraic identity (Δm̂ = Δm + e(x′) − e(x)). It carries no empirical content.
3. The R1 implementation uses correlation > τ (`class_shapley.py` l.65–73), not equality. Correlation 1 does not imply equality: (1−a)·p + a·uniform has correlation 1 with p. So τ = 0.99 classes are not the theorem's classes, even in the limit. L2 acknowledges tolerance but not this.

### F19 (no defect). What checks out
- C2 and C3, including the factor 2 and the π − t denominator.
- The C4 constants (see F3 for why the bound is still vacuous).
- R1's proof for exact splits, gauge and their compositions; it extends to value-splits with Σ W_V′W_O′ = W_V W_O.
- The T1 relative-count invariance.
- The V1 formula for any K at ρ = 1.
- The I2 line.
- PP1 support counts, and S3 = S2 for single-cell edits.

---

## Required changes (priority order)
1. Withdraw the 52% figure and the "maximally sensitive" consequence (F1, F9).
2. Withdraw the K2 counterexample and restate K2 for ε > 0 (F2).
3. Remove "C4 explains the 0.996 agreement" wherever it appears (PP1 §4–5, COR l.24 and l.81, K1), and attribute the agreement to the pointwise C1 certificate (F3, F4).
4. Add the order-preserving-mass hypothesis to C6 (F5).
5. Withdraw OD1's "only if" and iff language, and add the uniform-Y assumption (F6).
6. Correct the V1 bias sentence (F7) and L1 (F8).
