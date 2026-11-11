# R3 — Causal identification review (pre-review commit 9389496)

CEUM = CAUSAL_EVIDENCE_UPDATE_MECHANISM.md. Reviewer checks: `reviews/checks/r3_subspace_controls.py`, `r3_normmatched.py`,
and recomputation from logs/mech_conf. They use stream 16_000_000 and reproduce the frozen values (e.g. P84 site R1 −2.088).

## Findings

**1. [major] The P6b/P6c "108 % / 89 % of the difference" figures hide arm-specific failure and overshoot.** Location: CEUM §1
and §3.2, MAIN_RESULT §3. All 16 models use the same 251 R1 pairs, so Δm_B(R1) = +0.248 for every model. The −Δm_B "floor"
therefore cancels in T − P: B_patch(T) − B_patch(P) is exactly the difference in raw patched Δm. The floor does not bias the
contrast. It does show what the contrast is made of:
- the site patch recovers **19 %** of the T1 R1 response (mean +0.040 vs +0.209);
- it recovers **146 %** of the P-count response (−0.716 vs −0.490).

Sub: 24 % vs 117 %. The "108 %" comes from two errors that partly cancel. Dropping P84 lowers it to 80 % (site) and 60 %
(sub). The median difference for sub is 0.37 (53 %). In T1, R1 is additive (interaction 0.003) and mostly
**non-site** (82 % recovered). P-count is non-additive (−0.34). Non-site alone carries 41 % of the difference. *Correction:* report per-arm recovery (or pairwise IIA). Replace "the
site-only patch carries the difference" with "the site patch reproduces the P-count discount; the T1 credit flows largely
elsewhere."

**2. [major] Site patches are off-distribution, and the validity argument is circular.** Location: CEUM "Intervention site".
Share 1.00 for the all-position patch is guaranteed by construction: adding A_after − A_before everywhere *is* the after
run. It says nothing about partial patches. A site-only patch pairs after-episode site states with before-episode context
(including the unedited token at k). No input produces that combination. The evidence that this matters:
- P84: site R1 −2.09 vs full −0.61 (share 2.54);
- P86: −0.68 vs −0.33;
- F34 R1, P84: site −1.63 vs full **+0.44** (sign reversal); P87: −0.14 vs +0.19.

In F34 the site patch creates a T−P difference of 0.33, where the true F34 difference is 0.085. The patching procedure can
therefore manufacture curriculum differences. *Correction:* withdraw the validity sentence. Call these interchange patches
with unverified support, and report the F34 artefact.

**3. [major] P-count site sensitivity is partly generic perturbation sensitivity.** I tested a random direction at the site,
matched per episode to ‖ΔA_site‖ of F45:

| model | random patch mean \|Δm\| | random share | F45 full \|Δm\| |
|---|---|---|---|
| P84 | 1.37 | 0.57 | 1.72 |
| P83 | 0.43 | 0.18 | — |
| T70 | 0.31 | 0.07 | — |

In P84, a random readout-token perturbation reproduces most of the response, and P84 dominates the P-arm means. *Correction:* add norm-matched random and shuffled-delta controls. Report P6b/c with P84
excluded, or robust (median) versions alongside the frozen values.

**4. [major] The "decoder count subspace" is not a distinguished variable.** Location: P6c, CEUM §3.2, MAIN_RESULT ("mostly
within a count-decodable subspace"). The rank-6 decoder row space captures 45–75 % of ‖ΔA_site‖² in 5 of 6 models checked. In
P-count it is rank 6 in a 64-d single-token site (9 % of dimensions). As a control I took the top-6 PCs of the *F34* site
deltas, with no count labels and a different family. They capture 77–97 % of the F45 delta and reproduce site-level R1
effects as well as or better than "sub":

| model | sub R1 | PCA R1 |
|---|---|---|
| P84 | −1.96 | −2.01 |
| P83 | −0.65 | −0.57 |
| P82 | −0.64 | −0.61 |
| P86 | −0.39 | −0.63 |
| T73 | +0.05 | +0.14 |

Rank-matched random subspaces do almost nothing (≤ 11 % of the norm). P6c therefore mostly restates P6b: the edit's site
delta is low-rank, and any subspace aligned with it works. Count-labelling adds no identified content. A logistic row space
can also hold anything correlated with n_g (latent configuration, other counts, token content at k). The other-group subspace is weaker (P84 −0.80,
P83 −0.31), so some group specificity exists. Decoders with accuracy 0.52–0.67 define poorly identified subspaces. *Correction:* describe P6c as "the dominant low-rank direction of the edit's site
delta". Require PCA, rank- and norm-matched controls before attaching a "count" label.

**5. [major] The boundary specificity of the mechanism (mandate §9.4) is not tested by any frozen criterion.** The behavioural
claim is about D-R1 (45 − 34). Patch analogue from the logs:
- site D-R1: T−P = 0.42 (p = 7.8e-5; the arms separate) vs full D-R1 0.61, i.e. about 69 %;
- sub D-R1: 0.36 (58 %), p = 4.7e-4.

These numbers support a partial and more defensible claim than P6b/c, but they are exploratory. *Correction:* report them
as exploratory and state that P6 does not test specificity to the boundary.

**6. [major] P6d is close to vacuous.** It compares F0 *site* |Δm| with F45 *full* |Δm|, pooled over arms and regimes. The full
F0 responses are themselves about 0 in R1 (|Δm| ≤ 0.025), so any F0 patch is bounded small. The test measures the models'
insensitivity to F0, not the specificity of the patch. Patching only the site deltas also leaves the activation-change
magnitude unmatched. F0 is also a weak control. It produces no new agreement event: it changes b to an absent third value
and alters cross-group relations differently from F45. A count-preserving edit that *moves* an agreement (e.g.
`eqv.pi_preserving_edit`) is the matched control §5/§9.6 asks for. T72/T74 show full F0 |Δm| ≈ 0.22 (23–35 % of F45). *Correction:* reword P6d as "the models ignore F0". Add the controls from finding 3 and an
agreement-moving F0.

**7. [major] The redundancy logic (P6e) does not meet T5(b)/(c)4.** The ablation is a single subspace interchange at the site.
It is not a joint ablation with the non-site route, which T5(b) itself requires. For the effect of interest (R1) the median
remaining share is **0.58 (P-count)** and 0.71 (T1); the pooled 0.42 mixes regimes. "Largely non-redundant" is not
supported for the curriculum-specific R1 effect. In T72/T74 the remaining share is 1.00 (no effect). *Correction:* report
the R1 remaining share. Call the subspace one of at least two routes. Run joint site-subspace + non-site ablation.

**8. [major] Collateral damage is untested (§9.5, §13).** CEUM §3.4 acknowledges this, but MAIN_RESULT still says
"Causally, the difference is carried by…". *Correction:* measure accuracy/KL and F34/F23 under the ablation, or remove that
headline sentence.

**9. [minor] The site is defined asymmetrically across arms.** P-count uses one readout token (64-d); T1 uses six source
tokens (384-d). The T−P contrasts in P6b/c therefore mix the model difference with intervention dimensionality.
*Correction:* state this, and add rank-normalised comparisons.

**10. [minor] Source-level edits.** F45/F34/F23 are valid same-world, supported, X_q-fixed single-report edits, and π depends on
X_{−q} only through n (verified, Lemma B2). They are surgical do(x_kb := x_ka) edits, not draws from a structural
counterfactual over the noise. They also change cross-group agreement at k (Bayes-irrelevant,
model-visible; §5.7). *Correction:* label them "surgical report edit, same latent world" and note the side effect.

**11. [minor] P6a framing.** P6a uses the pooled F45 share; R1-specific T1 site shares are 0.04–0.62 (median about 0.37).
"Implementation diversity" is one reading; off-site routing in the weakest models (T72/T74) is another.

## Verdict

The source-level counterfactuals are sound. Supported: an interchange at layer 2's input reproduces the P-count R1 discount,
partly boundary-specifically (exploratory D-R1 about 58–69 %). Not supported:
- that the curriculum difference is "carried by" the site change (T1 is 19 % site-recovered);
- that a *count* subspace is the variable (a label-free PCA control does as well);
- that the patches are on-distribution;
- R1 non-redundancy;
- preservation of baseline decisions.

Denying Outcome B is correct. MAIN_RESULT's causal sentence and "P6b/P6c confirmed … carries the difference" overclaim.
Downgrade them to "localised interchange evidence in the P-count arm; support and specificity unresolved".
**Verdict: major revisions to the causal sections; behavioural results and outcome classification unaffected.**
