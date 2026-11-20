# Functional against causal organisation (mandate Workstream E)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Evidence labels.** [proof] / [num] / [dev] / [frozen] / [interp].

## E1. Three claims

1. **Route reorganisation.** The same computation is carried by different components.
2. **Different functional computation.** A different input–output rule, or a different evidence-combination function.
3. **Different internal algorithm.**

This report keeps them apart.

## E2. Behavioural tests (route-free)

### Battery (`scripts/func_battery19.py`)

Natural T4 episodes from the exact generator, binned by their *latent* structure at the target question q. The model is scored against exact Bayes (`genlib.posterior_closed`).

Cells:
- the reliable pair agrees at q, and the colluders agree on a different answer, or on the same one;
- the reliable pair splits at q;
- role evidence is weak or strong.

All cells are on-support. No off-support inputs were used.

### Frozen rule (commit `8ca9db7`, frozen-addendum)

The battery and its calibration were produced while the confirmation was running. They were frozen 14 minutes after the first confirmation job started, before any confirmation output was read: self-attestation, consistent with file access times (post-review E1). The rule is not part of the reviewed design freeze `478a038`.

SR = KL(split) / KL(consensus):
- FE if SR ≥ 2.2;
- FM if SR ≤ 1.6;
- FAMBIG otherwise.

### Calibration and validation [dev]

- **12 XVIII reference models:**
  - EARLY 2.60–3.78;
  - MIXTURE 0.98–1.27.
- **52 held-out models**, including other parents:
  - competent pure EARLY 2.55–4.02;
  - competent pure MIXTURE 1.03–1.33, with one exception at 1.74, which is ambiguous rather than wrong;
  - no crossings.

### Frozen results (`logs/conf/func_score_c19.json`, stream 93 300 000)

| check | E | M |
|---|---|---|
| F1: anchors' SR class equals their causal class | 6/6 SUPPORTED | 6/6 SUPPORTED |
| F2: causal-switch arms take the new class | 16/16 SUPPORTED | 18/18 SUPPORTED |
| F3: causal-none arms keep the B class | 19/19 SUPPORTED | 16/16 SUPPORTED under the literal frozen rule. The 5 INCONCLUSIVE(causal_NONE) M models, scored by the first scorer version and not in the rule, also match (post-review E2) |
| F4: hybrids (descriptive) | 7 of 7 mixture-like (SR 1.19–1.47) | 5 of 6 mixture-like; M823 I←B L0 early-like (2.40) |

So all 86 non-hybrid confirmation models have the error structure of their causal class: 40 switched and 46 not, from 3 independent data orders per base. Confirmation SR ranges: EARLY 2.33–3.50, MIXTURE 1.03–1.52.

### What this shows, and its limits

[frozen] The two organisations make systematically different errors on valid relational inputs:
- the EARLY organisation errs mostly when the reliable pair disagrees at the target question;
- the MIXTURE organisation errs roughly uniformly.

Among **non-hybrid** models, every intervention that switched the causal route also switched this behavioural signature, and every one that did not, did not.

The hybrids are asymmetric. All 7 E hybrids (φ 0.50–0.69) are already fully mixture-like. M823 I←B L0 (φ 0.75) is early-like.

The signature is route-free, which is a real gain over the RPC score. But [expl, post-review D6] it correlates 0.97 with early-route necessity and 0.94 with KL, and a KL cut at 0.02 reproduces every classification. It is therefore a behavioural re-expression of which route dominates, together with the precision level.

**Limit [num, dev; fixed in advance].** Simple degradations of exact Bayes give:
- uniform lapse: SR ≈ 0.2;
- temperature: SR 0.4–0.5;
- unstructured logit noise: SR ≈ 3.

So the EARLY signature is what "Bayes plus unstructured logit noise" would produce: errors concentrate where the Bayes margin is small.

[interp] The EARLY organisation is a ≈ 2× less precise approximation to the same Bayes rule. This is **causal implementation reorganisation with a precision difference** (mandate E4 wording). It does **not** establish a different functional rule (Claim 2) or a different algorithm (Claim 3). The earlier phrase "Claim 2 in the weak sense" is withdrawn (post-review D6).

The XIII-B power-mean exponent and its lapse-aware refit (XVIII B1) point the same way. The difference between the organisations is in precision, not in a fitted operator. Ledger N20 stands.

## E3. Test-time causal tests

- **Primary measure.** Route necessity: blocking own0 at layer 0 (nec_E) or own2 at layer 2 (nec_M) at test time. These are the **same edge sets** as the training-time block.
- **Measurement dependence.** The functional battery above is the independent check required by mandate E3.
- **Agreement.** Necessity is stable across three independent streams for 59 of 60 validation models. Causal and functional classes agree on every non-hybrid confirmation model.

## E4. Competence

- Every one of the 99 confirmation models is competent: KL ≤ 0.035 and accuracy ≥ 0.806 on stream 94 700 000.
- The causal-route classification, the RPC class and the SR class are reported per model in `logs/conf/score_c19.json` and `logs/conf/func_score_c19.json`.
- Hybrids are not forced into the binary classes: 13 confirmation models are HYBRID.

## Conclusion

The intervention-induced switch is a change of causal route with an accompanying precision and error-structure difference: **causal implementation reorganisation with a precision difference**.

No different functional rule is established, and no different algorithm. Durability is shown to step 1211 only.
