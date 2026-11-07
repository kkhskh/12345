# Open research obligations (within Phase 1.5 Step 1)

Ordered by priority. None of these is Step 2 or Phase 2 work.

1. **A preregistered confirmation of T-RF's empirical consequences.** Use a new task family and new seeds. Predict *in advance* that:
   - learned models do not compute a causally used relation for candidate-directed reconciliation rules;
   - they do compute one for candidate-symmetric rules.

   The present B-add and additivity analyses were exploratory (post hoc), so they cannot serve as that confirmation.
2. **Tasks where a candidate-directed rule is *not* threshold-representable.** For example, rules whose outcome depends on agreement **strength** at a fixed support pattern, such as "follow the sources only if both exceed a magnitude ratio". Determine whether a relation feature then becomes necessary, and whether learned models develop it.
3. **Continuous T-RF.**
   - Prove or refute that T4's additive sign-representability holds with continuous (not only grid) functions.
   - Characterise exactly which continuous-evidence rules are additively representable outside dead bands.
4. **Residual-basis gauge in learned models.** Retrain with LN bias disabled, or with bias-carrying Q/K/V maps, so that residual rotations can be conjugated through LN. Then test the invariance of probe directions as subspaces.
5. **A unit criterion that implies use.** The preregistered `I_agree`/`D_sum` criterion flags unused units in T3. Develop a criterion that combines activation statistics with a *generator-only* interventional test, so that units are selected by use without touching the margin.
6. **The K ≥ 3 learned mechanism.** TF2-AO reaches 0.991 on T5b despite T-B-IMP. Identify how two attention layers implement candidate-relative agreement. Candidates are multi-position softmax normalization and LN centring. Then test whether the result is a relation computation or the additive solution.
7. **Redundancy-aware mediation.** Generalise the C3 redundancy result into a necessity test that ablates all copies. Use subspace interchange on the union of identified sites, and report *set*-necessity.
8. **Reachability-aware matched contrasts for learned models.** Extend T-R and X4 to learned models by solving for input pairs that match the model's own relation-blind pathway. The candidate pathway is the model with its relation-unit population mean-ablated.
9. **Transfer (§16).** Report and interpret the exploratory width, head and MLP-width transfer runs (`logs/transfer.txt`). Then extend to:
   - different normalisation (no final LN; RMSNorm);
   - additional bypass paths in learned models.
10. **A fresh adversarial review** of any mechanism added after the current review round, as §19 requires.
