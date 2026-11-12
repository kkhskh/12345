# R5: Counterexamples and novelty (pre-review commit 76a9d60)

**Read.** MANDATE_VII §14; MAIN, SELECTIVE_RESCUE, FORMAL, R1_INFO, FROZEN_SUFFICIENCY, LEARNING_DYNAMICS, OPEN; confirmatory7.json,
oracle_reader*.json, downstream.py, checkpoints.

**Own checks.** Generator-only and run with `nice`. The scripts are in /tmp/r5check, outside the package. They use
`build_eval(19_000_000)` with 6000 episodes.

## Findings

**F1 [HIGH]: no generic-adapter control. The random injection is a null control.**
*Location:* SELECTIVE §2; P2.
*Problem.* `injrand` is a hash of X taken mod 7, so it has no learnable structure. Its zero-initialised map stays inert: KL and E
match C1 in all 8 parents, and the P2 deltas ≈ the P1 deltas. Nothing tests a task-relevant, non-hand-composed adapter, such as
per-group perfect-at-q bits, the raw agreement matrix, or a small MLP adapter on (n, a).
*Correction.* Rename P2 to "inert-feature control" and state that the generic adapter is **untested**. If an MLP adapter on (n, a)
rescues, the finding becomes "the one-block learner lacks one conjunctive step", and m loses its specificity.

**F2 [MEDIUM-HIGH]: "n separable vs m relational" mislabels the contrast.**
*Location:* SELECTIVE §2 and §4.2; FROZEN_SUFF §5; DYNAMICS §3.2.
*Problem.* The n-injection gives each readout token the one-hot counts of **all three** groups. The reconstruction
m_g ≈ Σ_{h≠g}[n_h = 5] equals the true m in **95.4 %** of decisions (my check). So n already carries m's cross-group information,
linearly. What one-hot m adds is the pre-composed conjunction: [m = 0] and [m = 2] are ANDs of other groups' bits, plus a_h.
*Correction.* Say that the injections differ in whether the cross-group conjunction is **pre-computed**, not in whether
cross-group information is present.

**F3 [MEDIUM]: Corollary 1 is overstated (counterexample).**
*Location:* FORMAL Corollary 1; R1_INFO §3; MAIN §1.
*Problem.* The claim is that Bayes R1/R3 signs require weights that depend on other groups' *counts*. Counterexample in 𝕄:
W_g = RHO^n·β^{[n=5 ∧ corr_g]}·δ^{[n=5 ∧ ¬corr_g]}, where corr_g means g's q-vote matches another group's q-vote. This uses votes
only.
- Lemma 1 gives R1 and R3 signs independently.
- Numerically, (β, δ) = (1, 0.01) gives R1 +0.040, R3 −2.258.
- R2 is +0.361, which is wrong.
*Correction.* The R1/R3 contrast needs only a non-own-count factor, which may be vote-based. Only the full pattern, which adds
R2 < 0, needs other groups' counts. Theorem 1 itself is correct.

**F4 [MEDIUM]: "inserts the answer"? It inserts the entire non-separable part.**
*Location:* SELECTIVE §1. Natural decisions, all q:

| rule | KL | acc | argmax = Bayes | R1 Δm |
|---|---|---|---|---|
| Bayes | 0 | 0.745 | 1.000 | +0.232 |
| W = m only | 0.128 | 0.693 | 0.910 | −0.001 |
| [m > 0]·RHO^n | 0.0001 | 0.745 | 0.995 | +0.300 |
| exclude5 (separable) | 0.150 | 0.714 | 0.946 | −3.62 |
| original models (conf) | 0.070 | 0.725 | — | −0.50 |

- m alone is not the answer. It is worse than the unrepaired models and has a null R1 response.
- Given m, Bayes is **log-separable per group**: log W_g = n_g log RHO + log m_g. One bit per group plus the counts gives Bayes
  to within 1e-4 nats.
- The injection therefore reduces the task to the separable form the learner already fits.
*Correction.* State this. The rescue confirms that the conjunction is the missing piece, but adds limited evidence beyond
Theorem 1 plus "learners fit separable weightings".

**F5 [MEDIUM]: m is legal but generator-tailored, and it encodes the role posterior.**
*Location:* SELECTIVE §1.
*Problem.* With ρ = 1, in the **91.4 %** of decisions with H = 1, one-hot m identifies the colluder exactly. m is the support
of the colluder posterior, and it is valid only for pc = 1, ρ = 1. It is not a test-time oracle, but "never receives the latent
roles" undersells how much inference it hands over.
*Correction.* Add that wording.

**F6 [MEDIUM]: the extra-training control is missing, and the rescue may be transient.**
*Location:* MAIN §4 F; READOUT §5 (empty).
*Problem.*
- xP80 C1 12,000 was still running at review (its log ends at step 500, R1 −0.33). Asoft-12000 was dropped.
- The oracle reader fixes R1 with budget alone (−0.277 → +0.008 at 16k).
- injm decays: conf E_R1 +0.208 at step 1000 → −0.029 at 4000, and dev P80 +0.59 → +0.23.
*Correction.* Mark outcome F as **undetermined for budget**. Report injm beyond 4000 steps.

**F7 [MEDIUM]: "Bayes level on average" hides a bimodal result.**
*Location:* MAIN §3; SELECTIVE §4.1.
*Problem.* Repaired E_R1 is +0.32/+0.43/+0.35 in three parents and −0.37/−0.13/−0.22/−0.30/−0.31 in the other five. Mean
|E_R1| goes 0.72 → 0.30, a 58 % reduction.
*Correction.* Report mean |E|.

**F8 [LOW-MEDIUM]: W_bal is one untuned point, and it is not "generic".**
*Location:* SELECTIVE §2 and §4.3.
*Problem.* Its classes use H and agree2, the same relation as m, at training time. It sits at one extreme (about 12×), and is
scored on natural KL, which reweighting penalises by design. `W_r1up` is coded but was never run.
*Correction.* Call it "one reweighting baseline". Run a weight sweep, or list it as open.

**F9 [LOW]: Theorem 1 is elementary.**
It is monotonicity of a mixture margin in one weight, under likelihood-ratio dominance. It is correct and diagnostic, but binds
only 𝕄. "Theorem" overstates its weight relative to Lemma 1.

## Novelty

These are known phenomena, precisely instantiated:
- shortcut learning and simplicity bias (Geirhos 2020; Shah 2020): separable majority-fitting features, failure on rare
  conjunctive cases (2.8 %);
- probe accessibility predicting what shallow frozen-feature heads use;
- group-reweighting trade-offs (Sagawa et al.);
- the difficulty of XOR or explaining-away conjunctions for additive learners.

The package should cite this framing and not present "separable inductive bias" as a new mechanism. What is new:
- (i) an exact characterisation of the interaction Bayes needs, plus a sign theorem for the eligibility family;
- (ii) a frozen, preregistered confirmation on 8 independent lineages, including the counter-intuitive P4 (more accessible
  counts worsen R1, 7/8);
- (iii) a route-level curriculum contrast.

## Defence against broad rejection

- The rescue is not an answer leak: m alone gives acc 0.693 and R1 ≈ 0.
- No test-time oracle is used.
- The n-injection specificity was a frozen, falsifiable prediction, and it held.
- Theorem 1 correctly explains why separable fixes (W_bal, n) move R1 and R3 together.
- The disclaimers on operation-level identification (D) and on network-level theorems are appropriate and should stay.

## Verdict

**Restricted result stands; framing must narrow.** Supported: supplying a pre-composed cross-group eligibility conjunction
selectively reduces the P-count R1 error under the original frozen one-block transfer, on independent lineages.

Not supported as stated:
- specificity versus a generic adapter (F1);
- the "separable vs relational information" contrast (F2);
- the Corollary's other-counts necessity (F3);
- outcome F for budget (F6);
- "Bayes level" (F7).

Required: corrections for F1–F3, F6 and F7. Unresolved: the generic-adapter and extra-budget controls.
