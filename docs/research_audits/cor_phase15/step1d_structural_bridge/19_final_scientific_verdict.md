# 19. Final scientific verdict

## Primary verdict

`STEP1D_STRUCTURAL_CERTIFICATES_ONLY`

Valid directional and architecture-specific bounds survive. No distinct relational causal mechanism was established. The original COR formulas are not supported. Phase 2 stays `NO GO`.

Levels: Level 1 (scope repair) and Level 2 (a proved structural certificate, with the source of its numerical gap identified) are met. Level 3 (a joint-path theorem strictly tighter than the separate aligned bound under the same budget) is not met. Level 4 is not met.

## Six separate judgments

| judgment | result |
|---|---|
| Code correctness | 13 tests passed, two forwards, analytic JVP against central differences, SymPy identities |
| Proof correctness | Theorems S, S-iso, Align, Kernel-out, Persist, the final-LN diameter, and the product-ball maximum have written proofs; the grid is a check |
| Theorem scope | U and L are restated with the quantifiers they actually have; the overnight universal B1 sentence is withdrawn |
| Causal interpretation | The surviving bounds constrain `|R|` for a raw readout. They do not identify a decision, a germ, or a compatibility obstruction |
| Novelty | The bounds reduce to the mean value inequality, Cauchy–Schwarz, and the chain rule. The new content is the scope correction and the negative joint-path result |
| Scientific readiness | Not a Phase 2 hypothesis. `OBSTRUCTION_READY = NO`. `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED` |

Operational completion: the package, the tests, and the manifest check. Mathematical completion: the scoped theorems and the product-ball ceiling for the examined joint upgrades. Scientific readiness: not ready for an authorized causal-bridge experiment.

## Gates

| gate | status | reference |
|---|---|---|
| A. Theorem U scoped | PASS | `02_theorem_U_restriction.md`. The final-LN diameter is stated with fixed `γ, β, w`. No claim that every architecture is unbounded. |
| B. Theorem L scoped | PASS | `03_theorem_L_restriction.md`. Isotropic `Λ ≥ ||∇m||`. Directional `B_dir` is a different object. |
| C. Finite versus first-order | PASS | At `α=10^{-6}`, `|∇m·δ| = 5.19e-8 < |R| = 1.33e-7`. The Cauchy number is not used as a finite bound. |
| D. Information versus access | PASS | `04`. Regime 1 determines `R` and forbids evaluating it. |
| E. Counterexample reproduced | PASS | `05`. Conditions met. Floats differ from the inaccessible audit except for the Cauchy number, which matches `||w||`. Certificate proved in `06`, not by the sample. |
| F. Nontriviality | PASS | `08` reports the full scale grid. The Cauchy comparison holds only for `rate < 1/2`. The output-kernel identity is the order-1 exact case and is not open. |
| G. Fair baseline | PASS | `07`. The relational candidates are compared with `B_align` and `B_dir`, not with `||∇m|| ||δ||`. They do not undercut `B_align`. |
| H. Relation defined independently | PASS | R1–R3 are defined in `09`–`11` without fitting `R`. R1's `Q_1` equals a piece of `∇m`, which is recorded as a failure of independence from the derivative, not hidden. |
| I. Counterexample resistance | PASS | `15` and `counterexamples/register.md`. Each witness meets the hypotheses of the claim it attacks. |
| J. Representation | PASS | Euclidean operator norm and the functional `w` are explicit in `06` and `13`. `O` and `D` are not used. Function-preserving hidden permutations and orthogonal head remixes preserve the certificate. |
| K. Literature reduction | PASS | `14`. S, Align, Kernel-out, and R3 are identified with existing calculus or with the numerical skeleton of approximate abstraction. No invented theorem numbers. |
| L. Cross-workstream synthesis | PASS | `13`. Implications are only the ones the bounds actually give. No workstream is marked successful by existing as a section. |
| M. Clean reproducibility | PASS | External extraction log in `logs/`. Tests and `scripts/verify_manifest.py` from the package root, no `PYTHONPATH`. |
| N. Honest verdict | PASS | The verdict is the structural-certificate line, not a relational bridge and not a claim that every bound failed. |

## Answers

**1. What did Theorem U prove?** On the raw readout, with `ε>0` and a nonconstant edit fixed, `|R|` can be driven to infinity by scaling a branch output aligned with the centered LayerNorm increment, while every weight-free datum stays fixed. The finite weight-free upper bound on that class is nonexistent.

**2. What did Theorem L prove?** An isotropic Euclidean Lipschitz constant of a scalar function, valid on a convex neighborhood of a differentiable point, is at least the Euclidean norm of the gradient. It is a lower bound on one number that has to serve every direction.

**3. Which overnight claims went beyond those theorems?** The executive verdict, the README, the adversarial audit's B1 sentence, the final verdict's Q23, Q26, and closing paragraph, the causal-mechanisms table, and the search-coverage list. They say, in substance, that every weight-using Lipschitz certificate is at least the Cauchy number, and that no B1 quantity on injective `Q,K,V` beats that number for a nonconstant edit. Quoted and replaced in `01`.

**4. Were those claims corrected everywhere they appeared?** They were corrected in this directory at every location `01` cites. The overnight files were not edited, so the overstatements remain visible in the historical record and are not premises of any proof here.

**5. Can a direction-specific finite certificate beat a valid isotropic bound?** Yes. `B_dir ≤ B_iso` always, under the hypotheses of Theorem S.

**6. Under what structural assumptions?** `ε>0`, the pre-norm block, Euclidean and Frobenius norms, and the Lipschitz constants of `06`. The gap equals `||w|| ||δ|| - |1_{t=r} w·δ|`. It is strict when the edit is at the readout and `δ` is not parallel to `w`. Small branch weights are not required. At unit scale the gap is `||w|| = 0.8764349701487028` and both bounds exceed `|R|` by many orders.

**7. Can the independent counterexample be reproduced?** The audit files were not available. A construction with the stated properties was computed: seed 0, output scale `10^{-6}`, unit nonconstant `δ ⊥ w`, injective stacked `Q,K,V`. `B_dir = 0.004051326443971923`, `||∇m|| = 0.8764350281115764`, `R = -1.326797428591320e-07`. The Cauchy number matches the audit's printed `0.876435` because it is `||w||` in the small-rate limit. The bound and the response do not match `6.50425e-5` and `7.58e-11`, and the weights were not adjusted to make them match.

**8. What is the mathematical source of the improvement?** Two different improvements. Against `B_iso`, the source is the exact skip inner product in place of `||w|| ||δ||`. Against the Cauchy number `||∇m|| ||δ||`, the source is that `∇m` stays near `w` while `B_dir` contains only the branch majorant, which scales like `α` near zero. Neither source is an interaction between attention and the MLP.

**9. How much comes from readout orthogonality?** On the unit orthogonal edit, the entire gap `B_iso - B_dir` equals `||w||`. At `α = 10^{-6}` that gap is `0.8764349701487028` out of an isotropic bound `0.8804862965926746`.

**10. How much comes from small branch weights?** The remaining certificate, `B_dir = 0.004051326443971923` at `α = 10^{-6}`, is the branch majorant. It is small because `L_A` and `L_M` are proportional to `α`. At `α = 1` the same term is `1.629250×10^6`. The dramatic ratio of `B_dir` to `||∇m|| ||δ||` is the orthogonality gap plus this small majorant. Remove either piece and the drama disappears: without orthogonality one pays `||w||`, and without small `α` the majorant exceeds the Cauchy number.

**11. Does the result persist on an open parameter neighborhood?** Yes, in a stated sense. `{rate < 1/2}` is open, nonempty, and on it `B_dir < ||∇m(H)|| ||δ||` for every base state and every nonzero `δ ⊥ w`. The seed-0 ray meets the set for all `α < 1.0387929173812708×10^{-4}`. Intersecting with `{σ_min(Q,K,V) > 0}` keeps a nonempty open set. The proof is continuity of `rate` and of the singular value, plus the Lipschitz bound `||∇m - w|| ≤ ||w|| rate`. Twenty nearby samples are illustrations, not the proof.

**12. What does that open-set result establish?** Local robustness of one strict comparison: a Regime 1 finite bound versus the first-order Cauchy number, uniform in `H`, on readout-orthogonal edits, including at some injective-QKV weights.

**13. What does it not establish about trained Transformers?** Anything. The set is a small-rate neighborhood. It is not a training distribution, not a density claim beyond the sublevel set itself, and not evidence about practical tightness. This package does not run trained models.

**14. What is a fair finite-response comparison against a gradient method?** A gradient method bounds `|R|` only after a remainder is proved, or it returns the exact value by a forward pass. The fair Regime 2/3 competitor is `|∇m·δ|` plus a certified remainder. At `α = 10^{-6}` the bare linear term is already below `|R|`, so it is not in the competition. Comparing `B_dir` with `||∇m|| ||δ||` is a comparison with a bound on the linear term.

**15. Which information and which operations does each method use?** Regime 1 (`B_dir`, `B_align`, `B_iso`): all weights and `δ`; spectral norms and `w·δ`; no perturbed forward and no JVP. Regime 2: those plus a JVP, returning `∇m·δ`. Regime 4: one forward, returning `R`. The inputs of Regime 1 already determine `R`. The regime forbids computing it. Costs are operation counts in `04`, not timings.

**16. When does a branch-wise bound become vacuous?** When it remains true and exceeds the scale of the responses and of the other valid bounds one would use to rank edits. On this block that happens once `L_1 ≈ 320` multiplies an order-1 attention constant. At `α = 1`, `B_dir ≈ 1.63×10^6` and `|R| ≈ 0.011`. Vacuous is not the same word as false.

**17. Can a genuine joint-path relation improve the separate bound?** Not in the three candidates, under Regime 1, against `B_align`. The product-ball theorem says no certificate that knows the increments only through radii can do better than the separate sum. The architectural coupling is real and, in every estimate obtained here, either reproduced that radius or left Regime 1.

**18. What is the mathematical definition of the relation that came closest?** `Q_1 = J_A + J_M`, the sum of the readout derivatives of the attention and MLP increments. It is also `∇m·δ - 1_{t=r} w·δ`.

**19. Is that relation independently measurable?** It is measurable by two JVPs, or by one JVP of `m` minus the skip inner product, without evaluating `R`. It is not independent of `∇m`.

**20. Does it imply a finite-response constraint?** Only after the derivative is controlled on the whole segment, or after a remainder is proved. The base-point value does not. At step `2`, `Q_1` is numerically zero and `R = -0.16901883579206622`.

**21. Is the constraint stronger than independent branch bounding?** As a bound on the linear term, `|Q_1| ≤ |J_A| + |J_M|`, and the inequality is strict on seed 0 (`0.00693` against `0.02003`). As a finite Regime 1 bound, no version undercut `B_align`.

**22. Under what conditions would an improvement persist?** A strict linear gap persists on an open set wherever `J_A` and `J_M` are nonzero and oppositely signed, because those derivatives are continuous for `ε>0`. That persistence is about the linear term. The finite improvement does not persist: the same open condition holds at the cancellation example's base point, and the step-`2` response is large.

**23. Which assumptions carry the causal information?** For Theorem S, the causal content is the decomposition `R = skip + w·ΔA + w·ΔM` together with Lipschitz bounds on `A` and `M`. The skip term is the intervention entering the residual additively. The branch terms are the intervention entering through LayerNorm and the two sublayers. Nothing in that list is an obstruction score.

**24. Do any assumptions restate the response?** `Q_1` restates the branch part of the derivative. The hypothesis `W_O w = 0` and `W2.T w = 0` does not restate `R`; it implies `R = skip` by the factorization, and the implication was checked by a separate forward. Defining `Q` as the gap between two bounds was not done.

**25. What is the strongest counterexample to the surviving candidate?** Theorem S is true on its hypotheses, so the counterexamples are to its over-readings. The strongest against "this is informative at ordinary scales" is the unit-scale row, `B_dir ≈ 1.63×10^6` against `|R| ≈ 0.011` with visible branch motion. The strongest against "this is a normalized-readout mechanism" is the zero-branch orthogonal edit, raw `R = 0`, final-LN `R = -0.32024254867419905`. The strongest against R1 as a finite theory is the cancellation witness.

**26. Does the candidate survive a final-normalized readout?** The raw directional bound does not. The raw output-kernel identity does not (`-0.13186230895395856` against a raw response of `0`). The diameter bound `2 ||w|| R_f` does survive for fixed `γ, β, w`, and it does not rank small edits. The raw constant shift also fails to transfer: a constant edit has raw response `w·1 = 1.5010589548912336` and final-LN response `0`.

**27. Does it survive function-preserving representation changes?** Signed permutations of the tanh hidden units, and orthogonal remixes of the head concatenation, preserve the map and the spectral norms, hence preserve `B_dir`. A non-orthogonal residual change of basis is not function-preserving for LayerNorm, so it is not a counterexample to invariance under equivalences. The certificate is still a Euclidean statement in the given parameterization, not an intrinsic causal scalar.

**28. Does it rely on an arbitrary metric or projection?** It relies on the Euclidean spectral norm, which is the norm in which the Lipschitz constants were proved, and on the linear functional `w`, which is the readout. It does not rely on `O`, `D`, or a projection chosen after seeing `R`. Changing the norm would change the constants. That is stated, not concealed.

**29. Does it reduce to an existing operator-norm or abstraction result?** Yes. Theorem S is a Lipschitz composition. Theorem Align is Cauchy–Schwarz on two factorizations. R1's finite form is the fundamental theorem of calculus. R3 is the approximation lemma used in causal abstraction, in the elementary scalar case. Citations are in `14`, limited to what those papers actually introduce.

**30. What contribution remains after that reduction?** The scope repair of U and L. The explicit constants and the threshold `α_*`. The proof that radius-only joint bounds cannot beat the separate sum. The demonstration that equal separate bounds can hide different finite responses, and that the statistic which sees the difference is the derivative. Those are boundary theorems.

**31. Does a projected state need to be causally sufficient?** No projected state is used. Theorem S reads the full factors and `w`. Adding a germ would be an extra representation. The question of causal sufficiency arises for the skip surrogate: it is sufficient for the raw response exactly when the branch readout gains vanish, and it is not sufficient for the final-LN response.

**32. Does the result require compatibility maps?** No. No `ρ` was defined. The structural residual of a recomputed trace is zero and does not determine the sign of `m`.

**33. Does it require the old `O`/`D` measure?** No. Changing a metric inside an obstruction score is not part of the proof. The preserved fact that `O` is not `D`, and that `D` does not rank decisions, is untouched and unused.

**34. Can it generate a prospective prediction without the target response?** Theorem S predicts an interval that must contain `R`, computed from weights and `δ` alone. On a generic orthogonal edit the interval is symmetric and does not predict a sign. Theorem Kernel-out predicts the exact value `w·δ` when both gains vanish. Both predictions are falsifiable by one forward that the predictor itself does not perform. Neither is a task-level claim about a trained model.

**35. What would falsify the proposed relation?** For Theorem S: a single edit with `|R| > B_dir` under `ε>0` and the stated norms. For Kernel-out: a nonzero branch readout contribution while `W_O w = 0` and `W2.T w = 0`. For the product-ball theorem: a pair of increments inside the radii whose weighted sum exceeds the separate sum. For the withdrawn overnight sentence: already falsified by CE-L-scope. R1's base-point slogan is already falsified by the step-2 witness.

**36. Which assumptions can be checked in a frozen Transformer without fitting evaluation outcomes?** `ε>0`, the pre-norm equations, the spectral norms of the factors, `||W_O w||`, `||W2.T w||`, and `w·δ` for an edit one chose without looking at `R`. Historical Direction 1 numbers are not among these checks and were not refit. A small gain is a small gain, not a reason to reopen Phase 2.

**37. Which parts are mathematically established?** Theorems S, S-iso, Align, Kernel-out, Persist, the final-LN diameter, the path-sum identity, the product-ball maximum, and the counterexamples listed in `15`. Each has a proof or an explicit witness and a second computational route.

**38. Which parts remain empirical conjectures?** Whether any trained model sits near `{rate < 1/2}` or near the output-kernel locus. Not conjectured here. Left unstated on purpose. Also open, and not empirical: a Regime 1 coupling estimate strictly below `B_align`. That is an unproved mathematical possibility, not a conjecture that it exists.

**39. Which broad impossibility claims should be withdrawn permanently?** "No weight-informed finite bound undercuts the Cauchy number." "Every Lipschitz certificate is at least `||∇m|| ||δ||`." "Theorem U implies unbounded responses for fixed final-LayerNorm readouts." "Injective `Q,K,V` leaves no nonconstant directional certificate." These four. They should not be replaced by "no useful bound exists anywhere in Transformer computation," which was not proved, or by "a relational COR mechanism exists," which was not proved either.

**40. What is the strongest positive conclusion we can defend?** For this raw pre-norm block, `|R|` is at most the exact skip inner product plus a sum of two branch Lipschitz terms, the constants are explicit, the bound is uniform in the base state, it is strictly tighter than the isotropic constant by the skip's Cauchy–Schwarz slack, and when `rate < 1/2` it is strictly tighter than `||∇m|| ||δ||` for every readout-orthogonal edit. When both branch outputs miss `w`, the raw response equals the skip exactly, at finite step, with the branches otherwise free to move.

**41. What is the strongest negative conclusion we can defend?** No candidate relation among the attention path and the MLP path, in the families R1–R3, produced a Regime 1 finite bound strictly below the separate aligned triangle bound. The path-derivative sum is the branch part of `∇m`. Base-point cancellation does not control the finite response. Radius-only joint packagings are at best the separate sum. The raw bounds do not govern a final LayerNorm readout. `O` and `D` remain unsupported as causes. The small-rate win against the Cauchy number is real and is not a typical-scale phenomenon on this block.

**42. What remains genuinely open?** A coupling estimate that shrinks the increment set below the Lipschitz radius and beats `B_align` in Regime 1. A proved segment-minimum replacement for `1/√ε`. Stacks, post-norm, and RMSNorm. Whether frozen trained models have small readout-aligned gains. Exotic finite branch zeros at generic weights. None of these reopens the withdrawn universal sentences.

**43. Is there enough theory to justify consideration of a separately authorized future experiment?** Not as a test of a relational causal bridge, and not as Phase 2. There is enough theory to know what a later mathematical attempt would have to prove. An inspection of `||W_O w||` and `||W2.T w||` on a frozen model would be a gain calculation, would need its own authorization, and would not be fit to evaluation responses. This package does not request that inspection.

**44. Does the original COR formulation survive?** No. The objects `O = ||ds||_W^2` and `D = min_{g ∈ ker d} ||s - g||_M^2` are not used, and the preserved counterexamples still show that they do not rank decisions. The name COR can refer to a research ambition. It does not refer to a theorem proved here. Separating the pieces: the original equations are unsupported; the original hypothesis that those equations constrain decisions is unsupported; the broader ambition of a structural constraint on causal response survives only in the weak form of Theorems S and Kernel-out; no successor mechanism was identified; no empirical validation was attempted or claimed.

**45. If the word COR were removed, would the strongest surviving theorem still be meaningful?** Yes. Theorem S is a concrete Lipschitz bound for one pre-norm block, with an explicit gap against the isotropic constant and an explicit scale on which it undercuts a first-order Cauchy number that is not itself a finite bound. Theorem Kernel-out is an exact finite identity for unread branch outputs. Both statements remain meaningful as analysis of that block. Neither needs the name COR, and neither becomes a decision theory when the name is removed.
