# Independent verification of E1 mediator measurements

Code: `reviews/independent_verification/verify.py` (pure numpy, float64). Raw output: `results.json`.
It does not import `src/mediator.py` or `src/run_protocol.py`. `src/model.py` / `src/ckpt.py` are imported only in `check_forward()`, for the one logit-equality check.

## Forward-pass check
I compared my logits with `model.forward` on 256 random inputs per model. The largest |diff| was 1.0e-6 for T1_TF1_s4, T1_TF2_s2 and T3_TF1_s2, and 9.3e-6 for T4_TF1_s2. That is float32 versus float64 rounding.

## Method (my own choices)
- **Pairs.** I wrote my own generator. Agree member: a = t*z with t ~ U(0.01, 0.99). Conflict member: one source pushed past z by u ~ U(0.01, 3-|z|). |z| ~ U(0.2, 2.5) with a random sign. For T3, p is shared within each pair. RNG seeds are 90001-90004. Both the sampling and the seeds differ from the package's.
- **S_rel.** Uncentered SVD of the site differences over 800 pairs. k is the smallest number of components reaching 90% of the squared singular-value energy, capped at 8.
- **Evaluation.** 500 fresh pairs, restricted to pairs where the model's decisions differ.
- **IIA.** The patch is h_base - P h_base + P h_src at the last position, then the rest of the network runs. IIA is the share of pairs where the patched decision equals the model's decision on the source, averaged over the two directions.
- **Random baseline.** 10 random k-dim subspaces (QR of a Gaussian). The package uses 20.
- **Necessity.** Set the S component of both members to the pair mean, then count pairs whose decisions still differ.
- **Sufficiency.** Patch P h_base + (I-P) h_partner and check the base decision is kept, averaged over both directions.
- **Additive R².** Two-way ANOVA main effects of the S coordinates on a 61x61 grid. For k > 1, R² is pooled over coordinates, which makes it rotation-invariant. For T3, I fit p = +1 and p = -1 separately and pooled the sums of squares.

## Results

| model / site | quantity | package | mine | agree? |
|---|---|---|---|---|
| REL-XNOR TF1 s4, resid_mid0@R | n model-differs | 500 | 500 | yes |
| | k | 2 | 2 | yes |
| | IIA full site | 1.000 | 1.000 | yes |
| | IIA S_rel | 1.000 | 1.000 (1.0 / 1.0 by direction) | yes |
| | IIA S_rand mean | 0.038 | 0.000 | yes (diff 0.038 < 0.05) |
| | necessity flip_after | 0.000 | 0.000 | yes |
| | sufficiency | 1.000 | 1.000 | yes |
| | additive R² | 0.789 | 0.789 | yes |
| REL-XNOR TF2 s2, resid_mid1@R | n model-differs | 500 | 500 | yes |
| | k | 1 | 1 | yes |
| | IIA full site | 1.000 | 1.000 | yes |
| | IIA S_rel | 1.000 | 1.000 | yes |
| | IIA S_rand mean | 0.000 | 0.000 | yes |
| | necessity flip_after | 0.000 | 0.000 | yes |
| | sufficiency | 1.000 | 1.000 | yes |
| | additive R² | 0.037 | 0.039 | yes |
| REL-MIS TF1 s2, resid_mid0@R | n model-differs | 500 | 500 | yes |
| | k | 1 | 1 | yes |
| | IIA full site | 1.000 | 1.000 | yes |
| | IIA S_rel | 1.000 | 1.000 | yes |
| | IIA S_rand mean | 0.012 | 0.003 | yes |
| | necessity flip_after | 0.110 | 0.096 | yes |
| | sufficiency | 1.000 | 1.000 | yes |
| | additive R² | 0.686 | 0.686 | yes |
| REL-GATE TF1 s2, resid_mid0@R | n model-differs | 243 | 254 | yes (about half, as expected) |
| | k | 1 | 1 | yes |
| | IIA full site | 1.000 | 1.000 | yes |
| | IIA S_rel | 1.000 | 1.000 | yes |
| | IIA S_rand mean | 0.000 | 0.000 | yes |
| | necessity flip_after | 0.012 | 0.004 | yes |
| | sufficiency | 1.000 | 1.000 | yes |
| | additive R² | 0.9994 | 0.9994 pooled (p=+1: 0.9992, p=-1: 0.9996) | yes |

The singular-energy spectra also agree closely:

| model | package | mine |
|---|---|---|
| XNOR TF1 s4 | [0.658, 0.342, ~5e-7] | [0.650, 0.350, ~7e-7] |
| XNOR TF2 s2 | — | [0.949, 0.050] (k=1 either way) |
| MIS TF1 s2 | — | [0.988, 0.012] |
| GATE TF1 s2 | [0.99957, ...] | [0.99958, ...] |

## Discrepancies and likely causes
No quantity is outside tolerance. The small differences that remain have identifiable causes:

- **S_rand for XNOR TF1 s4.** The package reports mean 0.038 and max 0.268. I get 0 for all 10 of my draws. With k = 2 in d = 16, a random plane only occasionally overlaps S_rel enough to transfer the decision. The package draws 20 subspaces and I draw 10, so its mean includes the occasional lucky draw. Both are far below the 0.2 gate.
- **Necessity on REL-MIS and REL-GATE.** My values are 0.096 and 0.004; the package's are 0.110 and 0.012. Different pair samples and seeds explain gaps of this size.
- **Count of model-differs pairs for REL-GATE.** I get 254, the package 243. Decisions can differ only when p ≠ sign(z), so about half the pairs qualify in both cases.

## Conclusion
I reproduced the package's E1 numbers for these four model/site combinations with independent code and independently generated pairs. That covers k, the cut, the S_rel interchange, the random baseline, necessity, sufficiency and the additive R² of the S coordinates. The headline contrast also reproduces:

- XNOR TF2 s2 has a strongly non-additive mediator (R² ≈ 0.04).
- XNOR TF1 s4 and MIS TF1 s2 are partly additive (≈ 0.79 and ≈ 0.69).
- GATE TF1 s2 is essentially additive (≈ 0.999).
