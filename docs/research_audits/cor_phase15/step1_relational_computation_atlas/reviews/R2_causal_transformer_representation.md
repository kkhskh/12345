# R2: Causal identification, Transformer computation, representation theory (adversarial review)

Scope: CAUSAL_IDENTIFICATION_AND_INTERVENTIONS.md, COMPUTATIONAL_EQUIVALENCE_AND_INVARIANCE.md, THEOREMS §4 (I5–I7), src/fwd.py, scripts/{mechanism_atlas, site_specificity, onsupport_matched, pp5_headsplit, pp5b_exact_split_posthoc, replicate_t4_lastH_copy}.py, logs/mech/*.json, logs/pp5*.json.
I ran five small checks on checkpoints `replication/T4_M_K3_s{10,11,12,13}_fromT1.npz` and `pp5b_s11_exact_split5.npz`. They use 600–1500 episodes, the package's own streams and the package's own code paths. The scripts sit outside the package, in /tmp/r2chk/chk{1..5}.py. Numbers marked [R2] come from these checks.

Severity counts: fatal 0, major 6, minor 8. The behavioural results (G1–G5, PP1) survive. The main internal-level claims that fail are three: "patched states are on-manifold", "G10 = natural necessity of the relation", and "Shapley share is the split-invariant replacement".

---

### 1. fwd.py ≡ nets.forward; the hooks are correct (no defect; one robustness note) — minor
- **Claim checked.** fwd.py is identical to nets.forward, and the head_out/pattern surgery is correct.
- **Evidence.**
  - The code paths match line for line: pre-LN (`ln(x,g1,b1)`) before Q/K/V, `1/sqrt(dh)` scaling, no mask, residual add, pre-LN GELU MLP, final LN, readout slice `6Q:7Q`.
  - `head_out` is `einsum(a·v, W_O)` per head, and its sum over h is exactly `attn_out`. Logged max |Δp| is 7.7e-7 to 1.2e-6.
  - The row masks broadcast correctly. For head_out `(1,H,T,1)` over `(N,H,T,d)`, the mask is on query rows t. For patterns `(1,H,T,1)` over `(N,H,T,S)`, it masks query rows and not key columns.
  - `ho_P` is applied per head before summation. A P ≠ I is only ever used with all heads selected, so per-head projection equals projecting the sum (by linearity).
- **Robustness note.** mechanism_atlas.py, site_specificity.py and onsupport_matched.py read `H = layers[0]["WQ"].shape[0]`. On the 5-head split checkpoint they would fail with a broadcast error. So none of the per-head analyses (Shapley included) was ever run on the split model; see #5.
- **Fix.** Read H from the analysed layer (as replicate_t4_lastH_copy.py already does).

### 2. "G6 patch states are on-manifold; the patch effect is not an off-manifold artefact" (§2.5) — refuted — major
- **Claim attacked.** The reachability test (Mahalanobis distance, pooled covariance, 1e-3 ridge) shows G6 states are natural. The mismatched null (IIA ≈ 0.45) is a "real, natural-state baseline".
- **Evidence.**
  - **The pooled covariance is the wrong reference.** It is fitted over all 36 source positions. Between-position and between-value variance dominate it: at a single position the top eigenvalue is 204, against a 10th eigenvalue of 0.49 [R2]. The 2-D contrast directions Pk carry 43% of the pooled variance [R2]. Any displacement along them looks "typical" by construction. The random-direction comparison (G12) is therefore not a meaningful control.
  - **A position-conditional covariance gives the opposite result** (seed 11, q = 0, fitted on half the episodes, evaluated on the other half) [R2]:

    | quantity | natural held-out | full source replacement (each state is natural) | cond Pk patch | mismatched Pk patch |
    |---|---|---|---|---|
    | fraction beyond p99, λ = 1e-3 | 2.9% | 2.8% | 61% | 78% |
    | fraction beyond p99, λ = 1e-1 | 1.9% | 1.6% | 39% | 60% |

  - **The query-side 5-D patch at the readout token was never tested.** I tested it: 61% of patched readout states lie beyond the natural p99, against 0.9% for natural held-out states [R2].
  - **The patched states are jointly unreachable.** Natural source tokens carry episode-level tags that agree across questions. The patch writes E′'s tag into question q only, while the other questions keep E's tags. The distance between q's tag coordinates and the mean over the other questions has median 2.6 for natural states, against 9.1 for the cond patch and 17.9 for the mismatched patch. 87.5% of cond-patched episodes exceed the natural p99 [R2]. A marginal per-token test cannot detect this.
- **Consequence.** G6, G9 and G11 are interventions on off-manifold hybrid states. The mismatched-null IIA of 0.45 is an off-manifold baseline, not a natural one.
- **Fix.**
  - Use a position-conditional (or position × value) density, or a kNN or classifier two-sample test.
  - Test the joint state of all tokens: cross-question tag consistency, and the readout token.
  - Report G6 as "sufficiency under an off-manifold hybrid".
  - For an on-manifold mediation test, patch all questions' tags jointly from E′ (episode-level interchange), not q's tokens alone.

### 3. G10 resample ablation is not "natural necessity" of the relational computation — major
- **Claim attacked.** G10 measures natural necessity, and the heads it ranks are the relation heads. §2.4 then infers agreement between mediation and necessity.
- **Evidence.**
  - Replacing all last-layer head outputs at readout tokens with those of an independent episode drops accuracy to 0.338–0.375 (seed 12: 0.49). That is chance (1/3), well below the own-question Bayes level of 0.58.
  - So the ablation does more than remove cross-question information. It removes the X_q evidence, which reaches the readout only through these heads, and it injects another episode's candidate evidence.
  - A conditional resample isolates the relational part: take the head outputs from a cond partner, with X_q fixed and X_{−q} drawn from P(·|X_q). In seed 10 this gives accuracy 0.537, against clean 0.747 and resample 0.337 [R2]. The relation-attributable necessity is about 0.21, not 0.41. Even that 0.21 includes an injected-wrong-context component (0.537 < 0.582).
- **Consequence.** The per-head necessity ranking mostly ranks *evidence transport*. The §2.4 agreement between mediation shares and natural-necessity shares is close to automatic, because the readout heads carry everything. It is weak evidence that the patched mediator is the natural variable.
- **Fix.**
  - Use conditional resampling, i.e. the cond partner with the same X_q, as "relational necessity". Report it against the own-question-Bayes floor.
  - Add mean or conditional-mean ablation, as §4 already concedes for the tag.

### 4. Super-additive necessity "is what a thresholded readout over summed evidence produces" — supported, but under-evidenced and mis-worded — minor
- **Evidence.**
  - Seed 10 resample-ablation margin drops (correct-class logit minus the best other logit) are 0.89, 0.78 and 0.72 for single heads, and 2.61 for all three [R2]. So the margin is near-additive (sum 2.39) while accuracy is super-additive. This supports the threshold reading.
  - "There is no downstream layer to compensate" is inaccurate. The last-layer MLP and the final LN follow the heads at the readout token, and both are nonlinear.
- **Fix.**
  - Report the margin and logit additivity, which is the actual evidence.
  - Reword the claim as "no attention layer downstream". A final MLP and LN are present.

### 5. "Shapley shares are conserved per pattern class" under the exact split (I5 invariant replacement) — false as stated — major
- **Claim attacked.** Shapley value aggregated over identical-pattern heads is the function-invariant replacement for head counts.
- **Evidence.** I recomputed all 16 subsets for the original seed 11 and all 32 subsets for the 5-head exact split, on the same episodes [R2]:

  | Shapley share | original | split |
  |---|---|---|
  | path-IIA, h0 | 0.277 | 0.308 |
  | path-IIA, h2 class (h2 + h4) | 0.441 | 0.211 + 0.211 = 0.422 |
  | path-IIA, h3 | 0.184 | 0.173 |
  | resample-drop, h2 class | 0.197 | 0.170 (−14%) |
  | resample-drop, h0 | 0.112 | 0.137 (+22%) |

- **Why it fails.** The game changes. Coalitions containing half of head 2 now exist, and the value functions (IIA, accuracy) are nonlinear. Only for additive games is the Shapley share of a split player conserved. The approximate conservation is real, but the shift is of the same order as the head differences used to rank heads.
- **Fix.**
  - Use a value function that is linear in the head outputs: a first-order logit or margin attribution, which is exactly split-invariant. Alternatively, define "players" as pattern classes before computing Shapley.
  - State that Shapley over nonlinear metrics is only approximately split-invariant.

### 6. Shapley on IIA and accuracy under interactions; mediation Shapley not logged — minor
- **Evidence.**
  - Path-IIA interactions are moderate but present. In seed 11: v{0} 0.267, v{2} 0.439, v{0,2} 0.781 (+0.075). In seed 0: v{0} + v{1} = 0.97, against v{0,1} 0.936.
  - §2.4 cites per-head "mediation shares". The script computes Shapley only for path, nec_out and nec_pat, not for med_iia. My recomputation from the logged tables (seed 0: [0.535, 0.181, 0.0, 0.01]) agrees qualitatively, but the number is not in any log.
- **Fix.** Log mediation Shapley and pairwise Shapley-interaction indices. Use KL-recovered or margin as the value function, alongside argmax-IIA.

### 7. "Path patching" (G7) is activation patching of head outputs at one position — minor
- **Evidence.** Only the readout-q rows of the last-layer `head_out` are replaced. Downstream, the effect runs through both the direct residual path and the last MLP and final LN at that token. These are not separated.
- **Assessment.** Because this is the last layer, no cross-token path is confounded. The claim "these heads carry the effect into the decision" is sound. The name and the "path" semantics (Wang et al.) are not.
- **Fix.** Rename it. Optionally split the effect into direct and MLP-mediated parts by freezing the last MLP's input at base.

### 8. The last-layer attention output is a "reliability-weighted candidate-evidence variable E_q" — the conclusion "not relational" stands, but "evidence" versus "answer" is unidentified — major
- **Evidence.**
  - I fitted 2-D contrast bases at attn_out (readout q) from cond pairs and from target pairs [R2].
  - In seeds 10, 11 and 13 the two bases coincide: principal-angle cosines are (1.00, 1.00), (0.995, 0.995) and (0.996, 0.993). With K = 3, a 2-D code is exactly a candidate-score (logit) code.
  - The site also transfers countpres changes, which Bayes says should not matter, at IIA 0.72–0.80.
  - So in these seeds the site is a decision-sufficient candidate-score code. Any decision-sufficient site aligns with E_q, so the "✓" in the E_q row of the abstraction table is vacuous. It adds nothing beyond E1 and E2.
  - Seed 12 differs: cosines (0.994, 0.018), so one dimension is target-specific. That is consistent with its lower target transfer.
- **Fix.**
  - Call it a "decision-sufficient candidate-score site" unless a compositional test separates E = Σ W_g·evidence_g from the answer code. One such test: compose W from E′ (via the pattern) with evidence from E″ (via the values) and check that attn_out predicts the composed decision.
  - Remove E_q as an independent alignment level.

### 9. The readout pattern is the "relation-specific operation" (target transfer ≤ 0.05) — the specificity test cannot fail — minor/major
- **Evidence.**
  - A pattern patch keeps the base values. A target pair's answer is a value shift, (X_q + sh) mod K, that the base values do not support. Low target IIA (0.011–0.049) is therefore nearly structural.
  - The patched target patterns *do* change decisions: change rates are 0.07–0.15 (spec_s*.json). So the patterns do depend on X_q values.
- **Fix.** Measure specificity on the pattern itself, comparing target edits with cond edits. Use the total-variation distance from the base pattern, and the group-mass shift. Reserve "relation-specific" for the case where target edits leave the pattern nearly unchanged.

### 10. "Readout attention pattern is the most invariant internal object" — true relative to the listed exact symmetries; overstated as function-level — minor
- **What holds.**
  - Patterns are invariant under the Q/K gauge (W_Q A, W_K A^{−T}): scores are unchanged.
  - They are invariant under V/O gauge and under adding a row-constant to the scores.
  - Under the exact split (I5) they are duplicated, not changed.
  - Softmax temperature is not a symmetry: rescaling the scores changes the function. So it cannot be gauged away, and the pattern is not temperature-ambiguous.
- **What fails.**
  1. Patterns are not identified by the function. Attention mass can be moved among positions whose OV images are equal or near-null, such as sink tokens or near-duplicate value vectors, without changing the output. So "basis-free" does not mean "function-level".
  2. "Aggregated over heads by Shapley share" mixes a positional object with a scalar attribution.
  3. The residual-stream gauge is understated. §3 says LN is preserved only by "rotations fixing 1 combined with uniform gains". But every LN output here feeds a linear map (W_Q/W_K/W_V, W1, W_A/W_C/W_R), so g and b can be folded into those maps. b·W_Q needs a bias term, which can be carried by re-defining b′ = b·diag(g)^{−1}R. The exact group is the whole of O(d) ∩ Stab(**1**) ≅ O(d−1), not a small subgroup.
- **Fix.**
  - State invariance as "invariant under the exact symmetry group G, which is enumerated".
  - Add the OV-null-space caveat.
  - Correct the LN-gauge statement.

### 11. I6 (query/key placement not identifiable) is not shown to be constructible in this pre-LN architecture — major
- **Claim attacked.** "Network B writes π into the readout token … q·k_s = β t_{g(s)} … scores and therefore all outputs are identical for every input."
- **Evidence.**
  1. **The query side is normalised.** q = LN(x_rd)·W_Q. Writing π into x_rd changes σ(x_rd), so q is π/σ(π, …): a nonlinear function of π, not the bilinear form the construction needs.
  2. **The key side is normalised.** k_s = LN(x_s)·W_K. The value and question embeddings change σ(x_s), so the group-identity feature is rescaled per token unless all embedding combinations are norm-equalised and mean-zero.
  3. **The readout output changes.** π stays in the readout residual and passes through the final LN, which rescales the answer logits. So "all outputs identical" also needs π to be norm-compensated or cancelled before the final LN.
- **Why it is probably repairable.**
  - The input domain is finite. Norm-compensating encodings, such as [π, √(c − ‖π‖²)] mirrored to zero mean, need upstream maps to be exact only on finitely many points.
  - With sufficient width, a GELU MLP can interpolate finitely many points exactly.
  - So exact constructibility is plausible but unproven.
- **Fix.**
  - Implement A and B numerically: a small hand-built network on a reduced (K, Q), with the max |Δ logit| reported.
  - Alternatively, downgrade I6 to "exact for a LN-free head, approximate under pre-LN", and restate the consequences as approximate.

### 12. G5 `pairswap` — on-support and essentially in-distribution; the agreement summary is inflated — minor
- **What I checked.**
  - Swapping the true reliable and noise pairs in questions ≠ q preserves every histogram, and Bayes is defined everywhere.
  - Hybrid episode log-likelihoods match natural ones: 0.8% of hybrids fall below the natural 1st percentile [R2].
  - Using the latent R to choose the swap does not bias the model-versus-Bayes comparison, because both are evaluated on the same inputs.
  - One caveat: the edit family is narrow (reliable ↔ noise only, colluder fixed). It is not a sample of "relation changes".
- **Over-reading.**
  - The 0.85–0.93 agreement on *whether* the decision changes includes a 0.62 chance agreement, because the base rate of no change is high. For seed 11, Cohen's κ = 0.79 [R2].
  - The doc says the model moves "in the direction Bayes does", but the script never compares directions. When both change, the model picks Bayes's new argmax in 81% of cases (chance 50%) [R2].
  - The model changes its argmax in 8.4% of the cases where Bayes does not (seed 11) [R2]. This confirms the over-reaction.
- **Fix.** Report κ, P(model changes | Bayes changes) and P(model changes | Bayes does not), plus the directional agreement.

### 13. I5 exact split (pp5b) — correct; the R3 flip is real but mechanically trivial — minor
- **Evidence.**
  - The 5-head construction is exact: max |Δp| is 2.4e-7, argmax agreement is 100%, and nets.forward handles per-layer head counts (dh scaling is per layer).
  - The flip happens because R3's top-2-by-R² rule picks the two duplicates, heads 4 and 2, whose R² values are identical (0.8797). So R3 freezes only one functional head.
  - This is a valid counterexample to head-count criteria. It does not support any positive claim about Shapley (see #5).
- **Fix.** Note why R3 flips.

### 14. Mismatched-source null and IIA reporting — minor
- **Evidence.**
  - The cond IIA for the tag patch (0.75–0.86) is only about 0.3–0.4 above a mismatched null of about 0.45. That null is itself off-manifold (#2).
  - IIA is computed only over pairs where the model's argmax differs between E and E′, which is a model-output-based selection. This is standard, but n_inf varies by kind: 232–369 for countpres.
- **Fix.** Report IIA minus null, with bootstrap CIs. For countpres-based IIA, the small n_inf calls for wide CIs.

---

**Survives.**
- fwd.py and hook correctness.
- PP1 and G5 behavioural relation sensitivity, which is on support and in distribution.
- The localisation of the relation's entry at the last-layer readout routing in seeds 10, 11 and 13 (pattern-patch IIA about 0.84). This rests on positive transfer, not on the target-null.
- Margin-additive, accuracy-super-additive redundancy in seed 10.
- I5's negative result: head counts are not invariant.

**Does not survive as written.**
- Reachability and "not an artefact" (#2).
- G10 as relational necessity (#3).
- Shapley as the split-invariant (#5).
- E_q as a distinct alignment level (#8).
- I6 exactness under pre-LN (#11).
