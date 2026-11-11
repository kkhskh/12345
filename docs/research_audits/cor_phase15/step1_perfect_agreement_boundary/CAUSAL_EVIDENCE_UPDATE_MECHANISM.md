# Causal evidence on the boundary-update mechanism

> **Post-review [R3, R4].** This document was rewritten after the causal-identification and Transformer-computation
> reviews. The frozen P6 verdicts are unchanged; their interpretation is substantially weakened. Pre-review text: commit
> 9389496. Reviewer checks are in `reviews/checks/` and are exploratory and unfrozen.

**Protocol.** P6_MECHANISM_PROTOCOL.md, frozen in commit 357a98b after the behavioural result and before any cohort
mechanism data.

**Data**
- Units: the 16 parents at step 4000.
- Episodes: independent pairs from stream 16_000_000. There are 251 F45-R1 pairs, with Δm_B = +0.248 on average, shared by
  all models. The decoder was fitted on stream 16_200_000.
- Code: `scripts/mechanism.py` and `scripts/p6_analysis.py`. Results: `logs/mech_conf/` and `logs/p6.json`.

## 0. Architecture facts that constrain the interpretation [R4]

**Layout and readout.**
- There are 43 tokens: 36 source tokens, 6 readout tokens (36+q) and 1 episode token.
- The model is pre-LN, with 3 layers, 4 heads, d = 64 and a GELU MLP of width 256.
- The answer for question q is WA · LN_f(resid_post2[36+q]). That is a linear readout of one token; there is **no pooling**,
  so the mandate's "readout pooling" item does not apply.

**What transfer trains.**
- Transfer freezes layers 0–1 and the embeddings and re-initialises layer 2.
- The readout WA and the final LayerNorm parameters are **inherited from pretraining and keep training**.
- In P-count pretraining, WA received no task gradient (the pretraining loss used a different head), only weight decay.

The curriculum can therefore act through three places:
1. the frozen layers and embeddings;
2. the inherited-then-tuned readout;
3. the layer-2 solution reached from each starting point.

**The frozen "site" differs in kind between arms.**

| arm | site | how it reaches the logits | dimension |
|---|---|---|---|
| P-count | the readout token 36+q | directly | 64 |
| T1 | question q's six source tokens | only through layer-2 attention | 384 |

The frozen P6 criteria inherit this asymmetry.

## 1. Frozen results (verdicts unchanged)

| id | criterion | result | verdict |
|---|---|---|---|
| P6a | median site share ≥ 0.8 in each arm | T1 0.85, P-count **0.77** | **not confirmed** |
| P6b | B_site-R1, T1 − P > 0 | +0.757, p = 7.8e-5, CI [0.49, 1.18] | **confirmed** |
| P6c | B_sub-R1 (decoder count subspace at the site), T1 − P > 0 | +0.621, p = 7.8e-5, CI [0.33, 1.05] | **confirmed** |
| P6d | F0 site \|Δm\| < 0.2 × F45 \|Δm\| | ratio 0.041 | **confirmed**, but near-vacuous (§3) |
| P6e | median remaining share after ablating the site-subspace change < 0.5 | 0.42 (T1 0.26, P 0.46) | **confirmed** on pooled pairs; not for R1 (§3) |

## 2. What the frozen results do and do not show

**Which responses are measured.** The baseline −Δm_B is common to all models, so it cancels in the T1 − P contrast [R3-1].
The contrast is real. It is not, however, a "share of the difference" in any additive sense:
- The site patch recovers **19 %** of T1's R1 response and **146 %** of P-count's.
- The subspace patch recovers 24 % and 117 %.
- T1's R1 credit flows mostly through non-site positions.
- The pre-review figures "108 %" and "89 % of the difference" are withdrawn. Without parent P84 they fall to 80 % and 60 %.

**Partial patches are non-linear hybrids, not decompositions [R4-2, R3-2].**
- A delta added at resid_pre2 passes through LayerNorm, softmax, GELU and the final LayerNorm.
- A half-scale site patch gives about 0.4× the full-scale response, and the sign flips in T70.
- In P84, site (−2.09) plus non-site (−0.47) compares with a full response of −0.60.
- In P84 at F34-R1, the site patch gives −1.63, while the model's true response is +0.44.
- The pre-review sentence calling these patches "activations the model actually produces" was circular. It is true only of
  the all-position patch, which reproduces the after-episode by construction because layers 0–1 are frozen [R4-3].

**P-count models are sensitive to almost any change at the site [R3-3].** A random direction with the same norm as the edit's
change, added at the site, gives:

| parent | mean \|Δm\| | regression share |
|---|---|---|
| P84 | 1.37 (true F45 response 1.72) | 0.57 |
| P83 | | 0.43 |
| T70 | | 0.31 |

**The "count subspace" is not a distinct variable [R3-4].**
- The top six principal directions of the F34 edit changes, built without any count labels, hold 77–97 % of the F45 site
  change.
- They reproduce the "sub" effects as well or better.
- Random subspaces of the same rank do almost nothing.
- The decoder subspace therefore works because it lies where the edit's change lies. P6c largely restates P6b.
- The subspace is also not dimension-matched across arms: ≤ 6/64 dimensions in P-count against ≤ 6/384 in T1 [R4-7].

## 3. Controls

- **Specificity [R3-6].** P6d is near-vacuous, because models barely respond to F0 at all. F0 also creates no new
  agreement. A matched control that moves an agreement while preserving the count was not run.
- **Redundancy [R3-7].** P6e ablates one route only.
  - For R1 pairs the median remaining share is 0.58 (P-count) and 0.71 (T1), so non-redundancy is **not** shown for the
    pairs that carry the effect.
  - Joint ablation, which T5(b) requires, was not done.
- **Boundary specificity of the patch effect [R3-5, exploratory].** The patch version of D-R1 (F45 minus F34) is 0.42 for
  the site patch and 0.36 for the subspace patch, against 0.61 in full. That is partial, and no frozen criterion covers it.
- **Collateral damage** to non-boundary decisions was **not measured** [R3-8].

## 4. Exploratory reviewer analyses (unfrozen; `reviews/checks/`)

**A symmetric site, the readout token 36+q in both arms [R4-1].**
- The R1 patched Δm is **positive in 8/8 T1 parents** (+0.05 to +0.14).
- It is **negative in 8/8 P-count parents** (−0.34 to −2.09).
- At the shared readout position's layer-2 input, the arms therefore separate by sign.
- This is the cleanest localisation in the package. It was not preregistered.

**Path split (4 models) [R4-5].** Layer-2 attention and the MLP were frozen to their before-episode values.

| model | full | direct only | direct + MLP | direct + attention |
|---|---|---|---|---|
| P80 | −0.70 | −0.06 | −0.23 | −0.15 |
| P84 | −0.61 | −0.33 | −0.70 | −0.44 |
| T70 | +0.12 | 0.00 | −0.03 | +0.09 |
| T74 | +0.38 | 0.00 | +0.06 | −0.05 |

- The P-count discount needs layer-2 attention and the MLP *jointly* (super-additive).
- The direct residual-to-readout path carries 9–40 % of it in P-count and 0–4 % in T1.
- T70's credit runs mainly through attention; T74 needs both paths.
- Per-head Q/K/V/O and LayerNorm analyses were not done, although the hooks exist.

## 5. What is and is not identified

**Representation.**
- The 4→5 change is present in the frozen layer-0–1 output.
- Count decoding after the edit is 0.94–1.0 in 6/8 T1 and 2/8 P-count parents; the other P-count parents range 0.57–0.89.
- The arm difference in decoding is not significant (p = .12, exploratory).
- By T2, decodability is not the difference.

**Localisation.**
- Exploratory and symmetric: the arms separate in sign at the readout token's layer-2 input.
- Frozen: the site-patch contrast holds, but it rests on non-linear hybrid inputs at asymmetric sites in a P-count arm that
  responds to arbitrary site perturbations.

**Operation: not identified.**
- In P-count models the discount is active, non-additive layer-2 attention × MLP computation (4-model exploratory split), not
  a passive readout.
- Two explanations remain undistinguished:
  - **(i)** the frozen representation encodes "5/5" in a form that is hard to read conditionally;
  - **(ii)** layer 2 learns an over-general "5/5 → discount" rule.
- The pre-review sentence "which makes (i) more plausible" is **withdrawn** [R4-6]. A single layer can do the conditional
  read, since T1 does it, and the inherited readout differs between arms (§0).

**Not tested.** Collateral effects, joint ablation, per-head and LayerNorm decomposition, and cross-curriculum transplant.

## 6. Verdict

The behavioural difference has a localisable correlate: opposite-signed effects of the edit at the readout token's
layer-2 input, carried through non-additive attention × MLP computation in layer 2. The frozen P6 evidence for a specific
count-subspace mechanism is weak, for four reasons:
- hybrid non-linear patches;
- a non-specific subspace;
- a near-vacuous specificity control;
- no non-redundancy for R1.

**Outcome B is not reached.**
