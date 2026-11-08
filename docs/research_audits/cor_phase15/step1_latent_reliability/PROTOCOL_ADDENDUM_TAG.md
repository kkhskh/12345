# Protocol addendum: source-token reliability tag and readout routing

**Status.** This addendum was written **after** the frozen `PROTOCOL.md` and after the dev-split pipeline check of the M/T1 pilot model (seed 100). It was written **before**:

- any confirmation-split data were generated;
- any dev or confirmation analysis of a seed 0–2 model;
- any analysis by `scripts/analyze_tag.py` on any model (its criteria are fixed here, before its first run).

The addendum **adds** analyses. It changes no frozen definition, threshold, site rule or verdict rule in `PROTOCOL.md`. The frozen primary answer is reported exactly as `PROTOCOL.md` §6 defines it. Verdicts from this addendum are reported as **secondary, post-pilot-registered** results.

## Why (what the pilot dev check showed)

The M/T1 pilot is at Bayes level on the dev split (acc 0.7455 vs Bayes 0.7447). Four observations from its dev analysis:

1. **Decodable but inert at the readout.** The reliability posterior is linearly decodable at the final readout residual (R² 0.77), yet patching that probe subspace has no effect (IIA 0.02). The output-free contrast subspace at the final site is the answer-readout direction: it changes the output under target-only pairs (change rate 0.92) and its content is the target question's own reports. The frozen site rule therefore picks a readout-equivalent site. The frozen criteria correctly reject this (P2 and P4).
2. **Routing at the last layer.** Last-layer heads 1 and 3 put readout-q attention on the groups of question q in proportion to P(group reliable | X) (R² 0.90 and 0.74). Patching only the readout-q attention row of the last layer (all heads) from cond pairs gives IIA 0.78 and kl_recovered 0.97.
3. **Where the information arrives.** The full residual of question q's own tokens (6 source tokens plus readout) at the input to the last layer carries the cross-question effect (IIA 0.87). The readout token alone at that site does not (IIA 0.07).
4. **Working hypothesis.** This suggests a two-stage computation. Earlier layers write a reliability tag onto each source token. The last-layer readout attends by that tag.

The frozen protocol does not test (i) a tag at the source tokens, or (ii) whether the routing is driven by that tag. This addendum adds both. It applies the same thresholds to the same models and splits.

## Definitions (fixed)

- **Candidate sites for the tag:** resid_pre1 … resid_pre{L−1}. These are the inputs to layers 1 … L−1, excluding the embedding and the final output.
- **s_tag:** the candidate site with the highest dev-split probe R² for the tag target. It is written to the dev json and reused unchanged on the confirmation split.
- **Tag target for the source token (q, s)**, where g is the group of s:
  - T1–T3: half the log odds that group g is reliable, under π_{−q}. 1-dimensional, k_tag = 1.
  - T4: the centred log marginal probabilities of g being reliable, colluding or noise under π_{−q}, keeping the first two. k_tag = 2.
  - T0 and T5: the leave-one-out agreement count of g. k_tag = 1.
- **U_tag:** the k_tag-dimensional row space of a ridge probe (λ = 0.01·N) fitted on the source tokens of all questions.
  - **U_tag_q012:** the same probe fitted on questions 0–2 only.
  - **U_D2:** the top-k_tag PCA directions of source-token activation differences under on-manifold cond pairs. Output-free.
- **Patch location:** the 6 source tokens of question q at s_tag. Pairs use the frozen constructions (cond / perm / splice / target).
- **Data streams:** dev base 1,049,000; confirmation base 827,000. These are disjoint from `analyze_model.py`.

## Tests and thresholds (confirmation split)

| id | test | pass if |
|---|---|---|
| K1 | tag probe R² at s_tag (leave-one-out target) | ≥ 0.6 and ≥ random-init R² + 0.3 |
| K2 | cond interchange with U_tag | IIA ≥ 0.7, ≥ random mean + 0.3, kl_recovered ≥ 0.5 |
| K3 | perm (matched aggregate) with U_tag | IIA ≥ 0.6 |
| K4 | target pairs with U_tag | change_rate ≤ 0.10 |
| K5 | resample-ablate U_tag at all 36 source tokens | drop_frac ≥ 0.5 and random-mean drop_frac ≤ 0.2 |
| K6 | U_tag_q012 on questions 3–5 (cond) | IIA ≥ 0.7 × K2 IIA |
| K7 | content of output-free U_D2 coordinates | R²(tag target) ≥ R²(other-question histograms) + 0.2 **and** ≥ R²(own reported value) + 0.2 |
| R1 | routing mediation: fraction of the E→E′ shift in last-layer readout group attention reproduced by the U_tag patch alone | ≥ 0.5 |

The frozen Type B criterion of `PROTOCOL.md` §6 (last-layer pattern patch IIA ≥ 0.7 and kl_recovered ≥ 0.5; some head with group-mass R² ≥ 0.5) is reused as it stands.

**Reported only:**

- pattern-patch transfer under target pairs;
- the confidence-head response (T2);
- U_tag vs U_D2 overlap.

**Secondary verdicts per (task, size):**

- **Tag-and-route mechanism:** K1–K7, R1 and the frozen Type B criterion all pass in ≥ 2/3 of the seeds.
- **Tag only:** K1–K7 pass, but R1 or Type B fails.
- **Route only:** Type B passes, but some K fails.
