# Open scientific problems (this research line only)

1. **Operation-level identification.**
   - *Known.* In the trained P-count models the R1 discount runs through one head's query on the readout token, plus a
     direct/MLP path.
   - *Gaps.* Heads are not aligned across parents. The route is not necessary for the error: replacing the readout-token input
     still yields a wrong R1.
   - *Needed.* A characterisation of the class of one-block solutions reachable from P-count states. Do they all lack a
     usable m-like feature, and what invariant do they share?
2. **Does the network ever construct m?**
   - *Known.* The repair supplies m.
   - *Open.* Whether layer 2 of a T1 model computes an m-like variable is untested. A causal test would be: (i) a frozen
     decoder for m_g at the source-key level; (ii) interchange of that variable between R1-before and R1-after states;
     (iii) a matched control.
3. **The unfreezing control** (mandate §6.3) was not run. Does unfreezing layer 1 let P-count models build the relation, or
   do they keep the separable route?
4. **Objective vs pipeline.** A P-count objective variant could add a cross-group target, for example "is another group
   perfect?". It would test whether the representation difference is caused by the objective. Not run.
5. **The sign of the repaired R1 is not robust.** It is slightly negative in 3/8 new lineages, and repaired development models
   over-credit and then drift. Why does the learner partly un-learn the injected relation late in training?
6. **Exploratory correlation.** The repair gain anti-correlates with MLP-probe accessibility of H2 (r = −0.80, n = 8). Does
   partial accessibility create a competing route? This is post hoc and needs a frozen test.
7. **Mathematics.**
   - Extend Theorem 1 beyond mixture-form rules, for example to monotone additive-logit approximations of the posterior.
   - State a learnability result for the eligibility interaction under SGD on a separable feature basis. The oracle-reader
     budget effect (4000 vs 16,000 steps) calls for one.
   - The R3 analogue of Theorem 2 is false in general. Characterise when it holds.
8. **Generality.** Only the 4→5 boundary of one generator was studied. Is the separable-discount failure generic for
   explaining-away structures (copying, collusion) in other relational tasks?

**Implication for the broader COR foundations.** The investigation isolates a concrete, generator-defined relational variable,
m_g, and proves that it is necessary for a family of candidate-response signs. It shows that representations differ in its
accessibility, and that supplying it causally and selectively corrects decisions on independent models. It does not supply a
general decision theory, and it does not revive the O/D framework.
