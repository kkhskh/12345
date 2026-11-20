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

### Frozen rule (commit `8ca9db7`, written before any confirmation output was read)

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
| F3: causal-none arms keep the B class | 19/19 SUPPORTED | 21/21 SUPPORTED |
| F4: hybrids (descriptive) | 7 of 7 mixture-like (SR 1.19–1.47) | 5 of 6 mixture-like; M823 I←B L0 early-like (2.40) |

So 86 of 86 non-hybrid confirmation models have the error structure of their causal class.

### What this shows, and its limits

[frozen] The two organisations make systematically different errors on valid relational inputs:
- the EARLY organisation errs mostly when the reliable pair disagrees at the target question;
- the MIXTURE organisation errs roughly uniformly.

Every intervention that switched the causal route also switched this behavioural signature. Every intervention that did not, did not.

The signature is route-free, unlike the RPC score of pre-review C3, so it is independent evidence that the switch is more than a relabelling of which edges are necessary.

**Limit [num, dev; fixed in advance].** Simple degradations of exact Bayes give:
- uniform lapse: SR ≈ 0.2;
- temperature: SR 0.4–0.5;
- unstructured logit noise: SR ≈ 3.

So the EARLY signature is what "Bayes plus unstructured logit noise" would produce: errors concentrate where the Bayes margin is small. The two organisations are therefore shown to be **different approximations to the same Bayes target**, with different, input-dependent error structure. That supports **Claim 2 only in the weak sense**. No distinct evidence-combination operator is identified, and nothing here supports Claim 3.

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

The intervention-induced switch is a change of causal route *together with* a change in input-dependent error structure. That is more than route reorganisation of an identical function (Claim 1 alone), and less than a demonstrated new algorithm (Claim 3).

The precise statement is: **a durable switch between two competent approximations to the same Bayes computation, which differ in which route carries the computation and in where they err.**
