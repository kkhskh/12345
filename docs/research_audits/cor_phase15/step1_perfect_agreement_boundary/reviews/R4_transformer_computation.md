# R4 — Transformer-computation review (pre-review commit 9389496)

**Scope.** I reviewed CAUSAL_EVIDENCE_UPDATE_MECHANISM.md (CEUM), MAIN_RESULT.md §3, mechanism.py, cohort6_train.py, and the
forward pass in fwd3/nets/train_tasks/eqv.Model. New checks are in `reviews/checks/R4_path_split.py`. They use stream 16_000_000
and the same F45 pairs and R1 definition as the package (251 R1 pairs); outputs are in `R4_*.json`. No package file was edited.

**Architecture.** The model has 43 tokens: 36 source tokens (6k+s), 6 readout tokens (36+q) and 1 episode token. It is pre-LN with
3 layers, 4 heads, d = 64 and a GELU MLP of width 256. The answer for question q is WA·LN_f(resid_post2[36+q]): one token, linear,
no pooling. Transfer freezes layers 0–1 and the embeddings and re-initialises layer 2. **WA, gf and bf are inherited from
pretraining and trained** during transfer.

## Findings

**1. HIGH — The site is asymmetric, and it hides where T1's R1 credit lives** (CEUM §1–2; MAIN §3; eqv.site_positions).
- *The sites differ in kind.* The P-count site is the readout token 36+q, which reaches the logits directly through the residual
  stream. The T1 site is q's source tokens, which reach the logits only through layer-2 attention.
- *A symmetric site separates the arms.* I patched token 36+q in both arms. The R1 Δm is positive in 8/8 T1 parents (+0.05 to
  +0.14) and negative in 8/8 P-count parents (−0.34 to −2.09). The T − P difference is ≈ 0.82, against a full difference of 0.70.
- *The T1 site carries the pooled response, not the R1 credit.* q's source tokens carry T1's pooled response (share ≈ 1.0 in 4
  parents). Their R1 share has median ≈ 0.37, and R1 Δm ≤ 0.04 in 5/8 parents. Parent T74's non-site response comes mostly from
  other questions' source tokens, including the edited token (share 0.71).
- *What this does to P6a/P6b.* P6b's "108 %" mainly reflects that the T1 site excludes T1's credit. P6a's "implementation
  diversity" is partly an artefact of the site definition.
- *Correction.*
  - State the asymmetry.
  - Add the readout-token patch as an exploratory result.
  - Rewrite CEUM §1 ("Why P6a fails") and §2. The frozen verdicts stand; their interpretation changes.

**2. HIGH — The partial patches are non-linear, so the shares are not decompositions** (CEUM §1, §3.2; MAIN "108 %/89 %/42 %").
- *Why it should be non-linear.* A delta at resid_pre2 passes through LN1, softmax, LN2, GELU and LN_f.
- *Half-scale patches.* A ×0.5 site patch gives 0.38× (P80) and 0.42× (P84) of the full-patch response, and the sign flips in T70.
- *Additivity fails.* In P84, site (−2.09) + non-site (−0.47) compares with full −0.60: the site patch overshoots 3.5× and is
  compensated elsewhere.
- *Shares exceed 1.* R1 site shares are above 1 in 3/8 P-count parents.
- *Means are fragile.* P84 alone moves the P-arm mean in P6b by ≈ 0.19.
- *Correction.*
  - Call the shares responses to hybrid inputs.
  - Drop "x % of the difference", or flag it as non-additive.
  - Report medians alongside means.

**3. MEDIUM — The validity sentence is a tautology** (CEUM l. 13–14).
- Layers 0–1 are frozen. Adding the full after-minus-before delta at every position therefore yields exactly the after-episode, so a
  share of 1.00 checks the code, not validity.
- The site, sub and subc patches are off-manifold hybrids (finding 2).
- *Correction.* Restrict "activation the model actually produces" to the all-positions patch.

**4. MEDIUM — Curriculum effects are not confined to layers 0–1** (CEUM §3.3; MAIN §3).
- *The readout starts in different places.* WA, gf and bf start from curriculum-specific values. In P-count pretraining WA gets no
  task gradient (the loss uses WX on LN_f(36+q)), only weight decay. In T1 pretraining WA is trained to answer.
- *The direct path matters.* With layer-2 attention and MLP frozen to their before-values, the residual → LN_f → WA path alone still
  carries 9 % (P80) and 40 % (P84) of the R1 response, against 0–4 % in T1.
- *Correction.*
  - Name three loci: the frozen layers and embeddings; the inherited-then-tuned WA/LN_f; and the layer-2 solution reached from both.
  - "The T4 objective is identical" therefore does not favour explanation (i).

**5. MEDIUM — The attention/MLP split was cheap and is now partly done** (CEUM §3.4; MAIN "Not tested").
`run_frozen` already path-patches attn_out2 and mlp_out2, and fwd3 exposes head_out2, pattern2 and mlp_hidden2 for per-head
Q/K/V/O work. Each model took about 30 s on one thread.

R1 mean Δm:

| model | full | site | direct only | direct+MLP | direct+attn |
|---|---|---|---|---|---|
| P80 | −0.70 | −0.56 | −0.06 | −0.23 | −0.15 |
| P84 | −0.61 | −2.09 | −0.33 | −0.70 | −0.44 |
| T70 (all pos.) | +0.12 | — | 0.00 | −0.03 | +0.09 |
| T74 (all pos.) | +0.38 | — | 0.00 | +0.06 | −0.05 |

- *P-count.* The discount needs layer-2 attention and the MLP at 36+q jointly; their effect is super-additive.
- *T1.* T70's credit runs mainly through attention. T74 depends on both paths jointly.
- *Correction.*
  - Say "not decomposed, although the hooks exist".
  - Cite this split as exploratory (4 models, unfrozen).
  - The mandate (§11.III) asks for exactly these interventions where the forward pass supports them, and it does here.

**6. MEDIUM — Explanation (i) is weakly supported** (CEUM §3.3).
- *A conditional readout is available in one layer.* Layer 2 can attend from 36+q to q's source tokens, which hold the other groups'
  votes and are unchanged by the edit in P-count (q-source patch ≈ 0 in 7/8). The 256-wide MLP at 36+q can then form the
  conjunction. T1's layer 2 does exactly this.
- *The discount is active computation.* Finding 5 shows the P-count discount is not a passive linear readout of a "5" feature. It is
  layer-2 attention × MLP computation, which fits (ii), a learned over-general rule, at least as well as (i).
- *How to test (i) directly.*
  - Train a fresh layer 2, or a probe from frozen resid_pre2 at all positions, to produce the exact-Bayes R1 margin.
  - Test whether the count code at 36+q is categorical or graded. P-count pretraining used 6-way categorical targets there, which
    favours a saturated "5" class.
- *Correction.* Present (i) and (ii) as undistinguished, and delete "which makes (i) more plausible".

**7. LOW — The P6c subspace is not matched in dimension** (mechanism.py `Pg`).
- The rank is ≤ 6 per group. That is ≤ 6/64 of the P-count site but ≤ 6/384 of the T1 site.
- The projection is taken in raw space before LN.
- *Correction.* Note the mismatch, or add a dimension-matched random-subspace control.

**8. LOW — Readout pooling (mandate §9) does not apply.** The architecture has no pooling. The analogous question is how much of
token 36+q's state arrives through the residual stream versus layer-2 attention. Findings 1 and 5 address it in part.

## Verdict

The data reproduce and I found no code error. The interpretation in CEUM §1–3 and MAIN §3 needs revision on four points:
1. the site definition is asymmetric (a symmetric site separates the arms 8/8 vs 8/8);
2. the shares come from non-linear hybrid patches and are not decompositions;
3. the readout parameters are inherited and trainable;
4. explanation (i) is not favoured.

Outcome A is unaffected, and "B not reached" stands. Reword the causal summary to: "carried at the readout token's layer-2 input in
both arms, through non-additive attention × MLP computation in layer 2; operation unidentified."

**Verdict: revise — major on interpretation, none on data.**
