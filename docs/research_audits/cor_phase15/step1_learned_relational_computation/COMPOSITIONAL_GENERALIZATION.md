# Compositional generalization (Problem F, empirical part)

**Status: development only, one seed per reader. Not part of the frozen confirmation.**

## Design (`scripts/gen_holdout.py`)

- **Training distribution.** The T4 generator, with every episode rejected in which groups 0 **and** 1 both agree on all six
  questions. This is rejection sampling of the natural generator: every training episode is on support, but the
  distribution is shifted. In such episodes, decisions where the two perfect groups are exactly {0, 1} never occur in
  training. The conjunction "perfect and agreeing" is therefore seen for the pairs {0,2} and {1,2}, never for {0,1}.
- **Test.** Development F45 pairs, split by the pair {edited group, existing perfect group}: *held-out* {0,1} vs *seen*.
- **Readers.**
  - Condition R attn, relnet and mlp, trained on labels for 16k steps, with the group-id one-hot included in the features.
  - A *lookup* baseline: a Laplace-smoothed table over the exact R feature key, built from 4000 × 256 training episodes.
- **Invariances**, on all development natural decisions:
  - group-slot permutation (group 0 ↔ 1, a symmetry of the generator);
  - candidate relabelling (y → y+1 mod 3).

  Both are reported as mean |logit change| after undoing the transformation. The typical logit scale is 1.25–1.28.

## Results

E is the mean error, then the mean |E| (absE) and n.

| reader | R1 held-out {0,1} | R1 seen | R2 held-out | R2 seen | R3 held-out | R3 seen | group-perm \|Δ\| | relabel \|Δ\| |
|---|---|---|---|---|---|---|---|---|
| attn | −0.42 / 0.42 (157) | −0.13 / 0.16 (312) | −0.16 / 0.16 (29) | −0.05 / 0.09 (69) | −0.29 / 0.32 (253) | −0.14 / 0.16 (540) | 0.016 | 0.043 |
| relnet | **−0.14 / 0.26** | +0.04 / 0.22 | −0.03 / 0.14 | −0.01 / 0.11 | −0.25 / 0.28 | −0.17 / 0.21 | 0.024 | 0.063 |
| mlp | −2.75 / 2.75 | −0.50 / 0.50 | −0.65 / 0.65 | −0.09 / 0.17 | −1.38 / 1.38 | −0.33 / 0.34 | 0.080 | 0.085 |
| lookup | −3.92 / 3.92 | −0.06 / 0.25 | −1.71 / 1.71 | −0.01 / 0.18 | −1.56 / 1.56 | −0.01 / 0.20 | 0.171 | 0.174 |

## Reading

1. **The lookup memorises the seen pairs and collapses on the unseen one.** Its seen-pair R1 error is −0.06 and its
   held-out R1 error is −3.92. This calibrates the test: the held-out pair genuinely requires generalization.
2. **The relational readers generalize partially.**
   - The pairwise relnet keeps the R1 sign on the held-out pair: E −0.14, so dm_R1 ≈ +0.09 with Bayes at +0.23.
   - Attention loses most of the R1 credit: E −0.42, so dm ≈ −0.19, the wrong sign.
   - The MLP generalizes far worse: −2.75.

   The ordering follows how directly the architecture shares parameters across group pairs. relnet applies one φ to
   every ordered pair; attention shares weights across groups but uses group-id features in its keys and queries; the
   MLP has separate weights per slot.
3. **A likely source of the degradation (untested) is group identity.** These readers see the group-id one-hot. A reader that keys on group
   identity can learn "pair {0,2} and pair {1,2}" without learning "any two groups". The variant without group ids
   (`Rnoid`, implemented in `readers.features(drop_id=True)`) was **not** run in the holdout. It is the natural next test:
   with ids dropped, relnet and attention are exactly permutation-equivariant, and the held-out pair is no longer
   distinguishable from seen pairs.
4. **Invariances.** The relational readers are close to group-permutation invariant (0.016–0.024, against a logit scale of
   about 1.28) and to candidate relabelling (0.04–0.06). The MLP and lookup are not (0.08–0.17). These invariances are
   learned, not built in, because group ids are present.

## Generator variations

None were run. K, Q and the group structure were fixed (`OPEN_FOUNDATIONAL_QUESTIONS.md` item 7).
