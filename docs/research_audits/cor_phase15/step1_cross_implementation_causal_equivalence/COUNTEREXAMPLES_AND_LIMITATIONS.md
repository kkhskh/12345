# Counterexamples and limitations

## Counterexamples established here

1. **Transport through a shared abstract variable does not identify shared computation** (Theorem C1–C2, confirmed: every exactly
   decodable donor transports at fidelity 0.98–1.01 into every recipient, whatever its curriculum). Any future claim of
   "implementation-independent causal abstraction" from transport alone is refuted by construction.
2. **Shared information and shared intervention behaviour with non-shared decision responses**: T1 and P-count models both satisfy
   intervention equivalence for V (Prediction I, cond/single adjusted IIA 0.76–1.03), yet their responses to π-preserving count
   changes share r ≈ 0.10 across curricula vs 0.38 within T1 (Prediction VI).
3. **Same decoded variable, different causal role** (Theorem D1): P's source tokens decode n at 0.62–0.85 but carry 5–27 % of
   the causal effect; T2's readout decodes n at 0.93 and carries 59 % (wrong-site marginalisation).
4. **Distributed state not replaceable by one local variable** (Theorem D3; Prediction I(b)): one token of a T pair reproduces
   −0.10 to 0.55 of the effect that the consistent pair reproduces.
5. **High interchange from a coarser statistic** (Theorem D5): additive one-hot count encoders fit any site storing only
   configuration log-odds; excluded empirically only where decoders are exact (T models, P0).
6. **Operator dependence of necessity**: span ablation restores cleanly in P (0.97–0.98, except P4 0.79) and not in T (0.35–0.68),
   because the T encoder span overlaps own-report content; the on-support marginalisation operator does not have this problem.
7. **Supervision site does not determine computation site** (TRAINING_OBJECTIVE_SITE_TEST.md): moving the count supervision
   to a source token did not produce a source-site count variable.

## Limitations

- **Models.** Three confirmation models per curriculum (P2–P4; T2, T10, T11) plus two calibration models each. All conclusions
  are about these models. T2 had been touched once in Investigation III. The P-count seeds are heterogeneous (decoders 0.72–1.00
  across P0–P4), so "P-count implementation" is not one thing.
- **Encoders and decoders** are linear in one-hot count features and multinomial-logistic respectively. A nonlinear decoder
  might recover more of P's count state; P's "lossy code" is relative to this class.
- **Floor-adjusted statistics** are unstable when ceiling ≈ floor (S6, P2/P4 source sites; T2 ceiling below the encoder). Raw IIA,
  unconditional agreement and Pearson r are reported throughout for this reason.
- **Interventions are activation additions** within encoder images; they are not natural interventions even when the patched
  states have natural Mahalanobis statistics. Natural references (input-level families) are always reported beside them.
- **Single site layer.** Only the layer-2 input was examined (the output of the frozen transferred layers). Earlier layers of the
  computation that produces V were not traced.
- **Pathway decomposition** uses one declared mediator split (attention vs MLP at the readout, layer 2); E-Comp is relative to it.
- **The site experiment** is three seeds with one pretraining failure and degenerate floors; it does not answer why T1 places
  its tuple at source tokens.
- **Frozen-threshold failures were narrow** (P4 0.677 vs 0.70; T11 rotated 0.386 vs 0.30; P4 restoration 0.785 vs 0.80;
  P2 Bayes-tracking 0.486 vs 0.50). They are reported as failures; the thresholds were set from calibration without margin.
- **Post-hoc analyses** (per-position decoders) are labelled; no frozen criterion was changed.
