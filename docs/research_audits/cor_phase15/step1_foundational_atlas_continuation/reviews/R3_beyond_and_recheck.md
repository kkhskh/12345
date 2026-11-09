# R3 review: BEYOND_LATENT_RELIABILITY (causal identification / architecture) and re-check of the R2 corrections

Reviewer scope: (1) GATE and TRANS in `BEYOND_LATENT_RELIABILITY.md` (code `src/tasks_ext.py`, `scripts/beyond_analysis.py`, `scripts/gate_mechanism.py`, `../step1_relational_computation_atlas/src/fwd.py`; checkpoints `gate_s0`, `trans_s0`); (2) whether the R2 corrections are stated accurately and whether retracted claims survive anywhere in the directory, including the new high-ratio analysis (`scripts/pp1_highratio.py`, `logs/pp1_highratio.json`).

Reviewer checks were run from `/tmp/r3/` (`trans_check.py`, `gate_check.py`, plus a relabelled re-run of the latter). They are **reviewer-reported** and not part of the package. They used the package checkpoints and the same streams as the package scripts (GATE 9,250,000 with 4,000 episodes, giving the same 4,377 scored cases; TRANS 9,300,000 with 6,000 episodes). No training was done. The re-implemented forward pass matches `fwd.forward` to a max |Δp| of 4.8·10⁻⁷. The package's `resid_pre2|readout` value (0.973) is reproduced exactly.

Severity counts: 1 critical (F1), 7 major (F2–F5, F9–F11), 7 minor (F6–F8, F12–F15).

---

## Part 1: BEYOND_LATENT_RELIABILITY

### F1 (critical). The last layer's attention, including head-0 "routing", is causally unnecessary in GATE. Stage 3 of the reconstruction and the "query-side routing" reading are refuted.

The document claims a three-stage path "supported by interventions at each stage", with stage 3 "Head 0 routes the readout's attention to the informative pair". It also claims the readout token carries the gate status "on the *query* side". Stage 3 rests only on attention-mass correlations (§2 routing table). No stage-3 intervention was run.

Reviewer interventions (4,377 scored cases with a unique Bayes argmax, same selection as `gate_mechanism.py`):

| intervention | fraction taking the flipped Bayes answer |
|---|---|
| readout-q residual at the layer-2 input, full patch (package value) | 0.973 |
| same, but patched **only into the layer-2 query** computation | **0.003** |
| same, patched into the residual stream, keys and values but **not the query** | 0.933 |
| full readout patch with **all layer-2 attention heads zeroed** | **1.000** |
| flipped input with all layer-2 attention zeroed | 1.000 |
| base run with the layer-2 attention pattern replaced by the flipped run's (all heads / head 0 only) | 0.037 / 0.027 |

On the base run with **all four layer-2 heads zeroed**, the model still agrees with the Bayes argmax in **1.000** of the 19,648 unique-argmax cases, and accuracy is 0.8995 (Bayes 0.899). Ablating head 0 alone also leaves agreement at 1.000.

**Consequences.**
- The decision is fully formed at the readout token by the layer-2 input. The rest is done by its skip path and the layer-2 MLP. The head-0 attention pattern (0.94/0.95) is a correlate. Transplanting it does nothing (0.027), and removing all attention does nothing.
- The gate variable at the readout does **not** act through the query (0.003). "Query-side" is therefore wrong as a mechanism claim.
- The §2 "comparison with latent reliability" (*same operation*: constraint-gated last-layer routing) and the §4 row "last-layer relation-gated routing: yes (head 0, query-side variable)" are unsupported.
- The same claims are repeated in `MAIN_FOUNDATIONAL_RESULT.md` l.31 ("the same last-layer relation-gated routing appears, through a three-stage path localised by intervention"), l.61 (Q18: "recurs across two relation families … variable on different sides (key vs query)") and l.62 (Q19: "GATE: yes, the same routing operation"). All must be withdrawn or rewritten.

**Supported version.** Layer 0: the evidence tokens of q pick up the gate status. Reviewer layer-0 attention from q's evidence tokens to q's gate tokens is 0.63 for head 1 and 0.41 for head 2, which agrees with the 0.85 patch. Layer 1: the readout token aggregates the *already gated* evidence. At the layer-2 input the readout residual carries the decision, and the last layer's attention is redundant.

The mechanism is therefore *not* an instance of the latent-reliability "last-layer relation-gated routing". If anything, it is a counterexample to inferring routing from attention mass.

### F2 (major). The flip "intervention" adds no information beyond Bayes-agreement on unique-argmax inputs.

- The flipped episode is on support, as stated, and the tie exclusion in `gate_mechanism.py` is correct (margin > 10⁻⁶ at both endpoints).
- But the same log shows `base_model_agrees_bayes = 1.0` on the scored set. The reviewer found agreement of 1.000 on *all* 19,648 unique-argmax base cases.
- So "the model follows the Bayes switch in 1.000" is implied by "the model equals the Bayes argmax wherever that argmax is unique". It is an in-distribution evaluation on a selected subset, not evidence that the model represents the gate as a cause.
- It should not be called a "decision-level intervention" result distinct from behavioural fit.
- Minor related point: the overall `agree_bayes` of 0.897 is lower than accuracy only because of argmax ties. The text should say so, since 0.897 looks inconsistent with KL 2·10⁻⁴.

### F3 (major). GATE's relation is equality up to a fixed known recoding, so "not an agreement relation of any kind" is overstated.

- x₀ + x₁ ≡ 0 (mod K) holds iff x₀ = π(x₁), with π(x) = −x mod K a fixed, known bijection.
- With per-source alphabets, which O/D-style truth-discovery models routinely allow, the gate is an agreement relation between sources 0 and π(source 1).
- What actually defeats O/D is different: the relation's sources carry no answer information, and the relation selects *which* answer sources to count. That is an "agreement among non-answer sources gates the weights" structure.
- "A second, independent counterexample to … O/D" should be restated: O/D in its *answer-discrepancy* form cannot express gating by non-answer sources. A generalised agreement statistic with known recodings can express GATE's relation.

### F4 (major). The TRANS one-hop reference is a "no-τ-evidence" reference. "Learns σ but not the composition" cannot be told apart from "never represents τ".

- `bayes_trans_onehop` masks the C reports of BC questions everywhere, including the BC question being scored. So it is Bayes without any τ evidence.
- In this task, τ has **no non-compositional use**: a C report informs Y only through (τσ)⁻¹, in BC and C-only questions alike. So the reference is a fair "no composition" baseline for the *output*.
- But the document's mechanistic reading ("it infers σ … the second hop is not composed") assumes τ is represented and left uncomposed. That is untested. Reviewer checks:
  - masking C in all BC questions moves model BC accuracy only from 0.629 to 0.626 (CE 0.669 → 0.671), against Bayes 0.653 → 0.627. The model uses C reports essentially nowhere;
  - model BC cross-entropy (0.669) is *worse* than the one-hop reference (0.635), and Bayes is 0.596. The accuracy match 0.629 vs 0.627 hides that the model does not even reach the σ-only reference in calibration.
- **Suggested wording:** "the model uses σ (one hop from A–B co-occurrence) and makes no use of C reports anywhere; whether τ is represented was not tested (a probe for τ at the episode or readout tokens would decide it)."

### F5 (major). "Neural failure, not a statistical one" overstates what was shown, and the training curve suggests the run was stopped mid-transition.

- The C-only Bayes value is correct. A reviewer brute-force enumeration over 36 (σ, τ) pairs on 798 C-only questions matches `bayes_trans` to 9·10⁻¹⁶. Accuracy is 0.487 (0.481 with random tie-breaking), against model 0.336. The information is present *per episode*.
- But "not statistical" conflates per-episode identifiability, given the known generator, with learnability from samples. The Bayes reference knows the generative family. The network has to learn it.
- `checkpoints/trans_s0.json`: the training loss sits at 0.79–0.81 until step 2,400, then drops to 0.59–0.66 between steps 2,800 and 4,000. This happens under a cosine schedule that reaches lr ≈ 0 at step 4,000, after the budget was cut from 6,000 steps.
  - The σ hop was learned late, in an abrupt transition.
  - The final model CE (0.640 overall) is still 0.11 nats above Bayes (0.530) and 0.03 above the one-hop reference (0.610).
- The result is "not learned in a truncated 4,000-step cosine run that was still in a learning transition". It is not evidence of a neural (architectural or optimisation) limit. State the learning curve and the cut explicitly next to the claim.

### F6 (minor). The "identical to the one-hop reference" equality (0.33648…) is a coincidence.

- The one-hop C-only posterior is near uniform. Its argmax accuracy is a tie-break artefact (0.327 with random tie-breaking).
- The model predicts all three classes (counts 3,699 / 4,320 / 3,940), with mean |p − 1/3| = 0.007.
- Equal hit counts to 17 digits do not indicate a shared computation. Report "at chance (1/3)" and drop "identical".

### F7 (minor). The TRANS attention evidence is close to the uniform baseline.

- "All four heads attend 0.75–0.87 to other questions" measures layer-2 mass on the 30 other-question source tokens.
- Uniform attention over 43 tokens would already put 30/43 = 0.70 there.
- So 0.75–0.80 on C-only questions is barely above uniform. It does not show "cross-question reading happens" for C-only questions.

### F8 (minor). Other remaining points in BEYOND.

- The stage-1 claim ("evidence tokens read the gate tokens") is supported only indirectly: by the 0.85 patch, plus the reviewer's layer-0 attention values (F1). The readout token also carries 0.15 at the layer-1 input, so stage 1 is not the only route.
- §2 does not state the stream and size for the mechanism run (9,250,000, 4,000 episodes, 24,000 question instances, 4,377 scored).
- `logs/beyond_analysis.json` still contains the untied 0.643 and a `patch_gate_tokens_IIA_on_changed` of 0.032. These are superseded but not labelled as such.
- `OPEN_FOUNDATIONAL_MAP.md` OF-10 says the analysis is "behavioural plus coarse routing", while BEYOND claims a localisation. Both need updating after F1.

---

## Part 2: re-check of the R2 corrections

### Verified as correctly stated

- **52% flip / singularity (FM-F1, ST-F1).** Retracted in LANDSCAPE §1.5 and THEOREMS §4.
  - `phase.json` (`argmax_sensitivity_rho_guarded`) gives 21.9% of cells defined at ρ = 1, a flip rate on those cells of 0.062 (from 0.95), and 0.128 / 0.114 at ρ = 0.5 / 0.55.
  - `review_checks.json` gives 0.024 / 0.034 / 0.025 for the refined steps and 0.048 at 1.0.
  - The text matches. "Interior 1–5%" slightly understates ρ = 0.6 (0.054), which is negligible.
  - No surviving "52%" or "maximal sensitivity" claim exists outside retraction notes.
- **C4 vacuous (ST-F2, FM-F3, NV-A2).** Correct in PP1 §2/§5, THEOREMS C4/K1, COR_TEN §1 and F1, and MAIN l.45 and l.90.
- **K2.** Withdrawn in THEOREMS §6 and COMPUTATIONAL_CLASSES §2.
- **OD1 "iff trivial semantics".** Withdrawn in THEOREMS, COR_TEN §2 and F3, BASELINES, OPEN OF-6 and MAIN.
- **Biased-error V1.** The `review_checks.json` jumps match the text: 0.226 → 0.991 at 0.905 → 0.91, and 0.055 → 0.846 at 0.995 → 0.999.
- **C6 existence form, CA-F5/F6/F7 rewordings, the 0.968–1.000 correction, and the S1 \ S2 hypothesis withdrawal.** All are correctly stated in PP1, THEOREMS and COMPUTATIONAL_CLASSES.
- **The T4-plug-in 0.587 reference.** Correctly stated in COR_TEN §1.
- **High-ratio family (`pp1_highratio.json`).** Reported correctly:
  - unfiltered 0.963–0.980;
  - |Δm| > 0.5: 0.985–0.999;
  - |Δm| ≥ 1: 1.000 for all ten models;
  - bin sizes 372 / 814 / 1,043 / 8,425;
  - P(A_q) = 0.130, so W ≥ 7.7 is a valid event-level lower bound;
  - CX4 KL 0.82 and accuracy 0.52.
  - Dm = 0 edits (1,346 of 12,000) are excluded from "all S1". This is fine, but should be said.

### F9 (major). A retracted PP1 interpretation survives in OPEN_FOUNDATIONAL_MAP §2 and BASELINES.

- `OPEN_FOUNDATIONAL_MAP.md` l.28: "PP1's interpretation is downgraded … to 'a consequence of Bayes-closeness under W ≈ 1'". This is exactly the claim R2 found false: C4 is vacuous, and W is not ≈ 1 (≥ 1.96 pointwise on S1, ≥ 7.7 on NP).
- `BASELINES_AND_NOVELTY.md` l.28 lists "PP1 sign agreement is a consequence of KL-closeness, not of mechanism" as **known**. This contradicts PP1 §5 and THEOREMS K1 ("a property of these models' pointwise accuracy, not a consequence of ε").
- **Fix:** "a consequence of pointwise margin accuracy at the edit endpoints (empirical; not implied by natural KL)".

### F10 (major). The corrected PP1 headline (§5) states an identity as an empirical fact, and claims coverage it did not measure for the second family.

- "Signs agree with Bayes whenever |Δm| exceeds the models' pointwise margin errors" is C1, which PP1 itself calls an identity. Its only empirical content is the coverage.
- §5 says "for … both edit families tested", but C1 coverage was computed only on the filtered frozen PP1 population (98.6–99.7%). `pp1_highratio.py` computes no coverage.
- For the NP family, the empirical statement available is the |Δm|-binned agreement.
- **Rewrite:** "(i) on the filtered PP1 edits, the C1 condition holds for 98.6–99.7% of edits; (ii) on NP, every sign with |Δm| ≥ 1 is correct, and agreement degrades toward chance as |Δm| → 0. C1 coverage on NP was not computed." The same applies to MAIN l.18 and COR_TEN F1 ("holds wherever true effects exceed the pointwise margin errors").

### F11 (major). The OD1′ "iff" is mis-stated for unequal weights: ranking equality fails at D-ties.

- THEOREMS l.76 and COR_TEN §2 say the D-ranking *equals* the Bayes ranking on every report pattern iff w(P) < w(R) for |P| < |R|.
- With unequal weights, equal-size disjoint report sets give a D-tie that Bayes breaks strictly. Example: two sources of accuracy 0.9 and 0.85, one reporting c and one c′. So full ranking equality fails whenever any two weights differ, inside the band too.
- What the condition characterises is "no strict reversal", or "the Bayes argmax lies in the arg-min-D set". The latter is what `od_bridge.json` actually checks.
- The "only if" direction also needs a pattern in which the remaining S − |P| − |R| sources do not make a third value the D-argmin. That part was not verified (the text says the enumeration was not checked).
- Restate it as "no strict reversal" and keep the [P] status as reviewer-proposed and unverified.

### F12 (minor). Residual "colluder code" and "natural necessity" wording.

- `BASELINES_AND_NOVELTY.md` l.30: "the colluder code picks the argmax-agreement group off support". This is the CA-F11-retracted reading.
- `EXTRAPOLATION_AND_INTERNAL_STRUCTURE.md` §3.1 (title and l.55, "keep the colluder code … then reassign it") is not covered by the §3.3 revision.
- `REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md` l.58: the section heading is still "natural necessity versus patchability", although the table was renamed.

### F13 (minor). The K2 replacement is invoked as an explanation without its constant.

- COMPUTATIONAL_CLASSES §2 says KL-closeness "already bounds the shifted-input KL by the density ratio times ε". W_s was never computed for the uniform-shift edit.
- The shift maps no-agreement questions to all-agree questions, so W_s may be large.
- The bound also controls KL(Bayes ‖ model) on shifted inputs, not the model's own shift response. Converting it needs a TV/Pinsker step at both endpoints, with the same vacuity risk as C4.
- Say that the bound is qualitative here, or compute W_s.

### F14 (minor). Small numeric points.

- The adjudication's "clustered CIs ±0.003" is ±0.004 for tf s2 ([0.959, 0.968]).
- PP1 §4b says |Δm| < 0.1 agreement is "at chance (0.49–0.66)". The upper end is not chance.
- The THEOREMS phrase "flip rates near ρ = 1 are 2.4–6.2%, like interior steps" compares a 21.9%-defined, selected subset with full interior cells. It is not like-for-like.

### F15 (minor). `MAIN_FOUNDATIONAL_RESULT.md` was not among the six documents listed, but it carries the GATE claims refuted in F1 (l.31, l.61, l.62, l.76 "localised mechanism"). It needs the same correction.

---

## Summary of required actions

1. Withdraw head-0 routing, "query-side" and "same operation as latent reliability" for GATE, in BEYOND §2 and §4 and in MAIN (F1, F15). Re-describe GATE as gating computed in layers 0–1, with redundant last-layer attention.
2. Re-label the GATE flip result as Bayes-agreement on unique-argmax inputs (F2). Soften the O/D implication (F3).
3. TRANS: describe the result as "uses σ; uses no C evidence; τ representation untested"; report the CE gap and the late loss transition with the cut; drop "neural, not statistical" and "identical" (F4–F7).
4. Fix the surviving PP1 interpretations in OPEN and BASELINES, and the PP1 §5 headline (F9, F10).
5. Restate OD1′ as "no strict reversal" (F11). Fix the residual wording (F12–F14).
