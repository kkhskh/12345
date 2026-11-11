# Causal evidence on the boundary-update mechanism

**Protocol.** P6_MECHANISM_PROTOCOL.md. It was frozen in commit 357a98b, after the behavioural result and before any
mechanism data on the cohort.

**Data**
- Units: the 16 cohort parents at step 4000.
- Episodes: independent pair episodes from stream 16_000_000 (3000 decisions). The decoder was fitted on stream 16_200_000.
- Code: `scripts/mechanism.py` and `scripts/p6_analysis.py`. Results: `logs/mech_conf/` and `logs/p6.json`.

**Intervention site.** `resid_pre2` is the input to the transfer-trained layer 2, which is the output of the frozen,
curriculum-pretrained layers 0–1. Every intervention adds, to the *before* episode of a pair, part of the activation change
(A_after − A_before) that the source edit causes. Patching all positions reproduces the response exactly (share 1.00 in every
model), so the patches are valid in the sense of moving the site to an activation the model actually produces.

## 1. Frozen results

| id | criterion | result | verdict |
|---|---|---|---|
| P6a | median site share ≥ 0.8 in each arm | T1 0.85, P-count **0.77** | **not confirmed** |
| P6b | B_site-R1, T1 − P > 0 | +0.757, p = 7.8e-5, CI [0.49, 1.18]; 108 % of the full difference | **confirmed** |
| P6c | B_sub-R1 (decoder count subspace of the edited group at the site), T1 − P > 0 | +0.621, p = 7.8e-5, CI [0.33, 1.05]; 89 % of the full difference | **confirmed** |
| P6d | F0 site \|Δm\| < 0.2 × F45 \|Δm\| | ratio 0.041 | **confirmed** |
| P6e | median remaining share after ablating the site-subspace change < 0.5 | 0.42 (T1 0.26, P 0.46) | **confirmed** (P-count is close to the bound) |

**Why P6a fails.** In two T1 parents (72 and 74, the two weakest T1 models: acc .696 and .713) the decision-site positions
(question q's six source tokens) carry almost none of the response: site share 0.04 and 0.01, non-site share 0.94 and 0.99.
Their count decoders at the site are poor (0.60 and 0.58), so these parents route the boundary information through other
positions. In the P-count arm the site share ranges over 0.63–1.26, with median 0.77.

The site definition is therefore *not* a universal locus. That is an implementation-diversity finding: different parents use
different positions.

## 2. A caveat on P6b and P6c

B_patch = mean(Δm_patch − Δm_B). If a patch had no effect at all, B_patch would equal −Δm_B ≈ −0.23 in R1. Several T1 values
(−0.10 to −0.30) are near that floor, including parents 72 and 74, where the site carries nothing. The contrast is therefore
partly "the P-count patch produces a strongly negative update; the T1 patch produces little".

The raw patched responses are clearer:

| | site-patched Δm in R1 | full Δm_M in R1 |
|---|---|---|
| P-count | negative in **8/8** parents (−0.34 to −2.09) | −0.23 to −0.70 |
| T1 | ≥ 0 in 6/8 parents (−0.05 to +0.15) | +0.12 to +0.38 |

Excluding parents 72 and 74 does not change the conclusion: the remaining T1 B_site-R1 values are −0.30 to −0.10 against
P-count values of −2.34 to −0.58.

## 3. What is identified

1. **Representation.**
   - The 4→5 change is present in the frozen layer-0–1 output at the decision site in most parents.
   - Count decoding at the site after the edit is 0.94–1.0 in 6/8 T1 and 2/8 P-count parents; the other P-count parents
     range 0.57–0.89.
   - The exploratory T1-vs-P comparison of decoding is not significant (p = .12).
   - By T2, decodability is not the difference.
2. **Use, localised.**
   - In P-count models, the curriculum-specific *negative* R1 update is carried by the site change.
   - Its projection on the decoder's count subspace for the edited group carries more than half the F45 response (by
     regression share) in 4/8 P-count parents (shares 0.28–1.09) and 4/8 T1 parents (0.00–0.88).
   - The subspace-patched R1 excess separates the arms completely: every P-count value (−2.21 to −0.45) is below every
     T1 value (−0.25 to −0.15).
   - It is specific: same-token, count-preserving F0 patches give 4 % of the F45 response.
   - It is largely non-redundant: removing that subspace change from the after-episode leaves a median 46 % of the response
     in P-count models and 26 % in T1.
3. **Not identified: which operation differs.** Layer 2 is trained from scratch in both arms on top of *different* frozen
   representations. A cross-curriculum transplant (T5(c), condition 3) is not possible because the residual spaces are not
   aligned. The data therefore cannot separate two explanations:
   - **(i)** P-count pretraining encodes "count = 5" in a form that layer 2 cannot easily read conditionally on the other
     groups' votes;
   - **(ii)** the same information is available, and layer 2 learns an over-general "5/5 → discount" rule from it.

   The T4 transfer objective is identical in both arms, which makes (i) more plausible. That is an inference, not a
   demonstration.
4. **Q/K/V/O pathway, LayerNorm, MLP.** Not decomposed in this investigation. The interventions are at the residual stream
   only. Collateral damage of the subspace ablation on non-boundary decisions was not measured. The mandate's "while
   preserving relevant baseline decisions" (Outcome B) is therefore not established.

## 4. Verdict for this document

The evidence localises the curriculum difference to the change that the 4→5 edit makes at layer 2's input, mostly within a
count-decodable subspace, in most parents. It does **not** identify a distinct computation, and it fails the frozen
localisation criterion in the P-count arm. Outcome B is **not** reached.
