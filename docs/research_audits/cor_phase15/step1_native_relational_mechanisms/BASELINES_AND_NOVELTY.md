# Baselines and novelty

## Baselines, with the information each receives

| baseline | input | role | result (complete confirmation parents unless stated) |
|---|---|---|---|
| exact Bayes | full episode | reference | 0 on every metric |
| decode-then-compose | native states → linear probes → exact rule | information ceiling of native states | exact (KL ≤ 0.0001) |
| original P-count transfer C1 | raw tokens; frozen pretrained 0–1 | the failing system | absE_R1 0.81–1.06, KL 0.03–0.08 |
| keep3 | raw tokens; frozen 0–2 | "give it the clean counts" | worse R1 (dev) |
| native-state MLP reader | frozen final native states | Type II, matched to the native information | absE_R1 0.44–0.52 |
| full fine-tuning full_c1 | raw tokens; all layers trainable | Type III native learning | absE_R1 0.21–0.35, KL 0.002–0.007 |
| Investigation VIII radapt | raw tokens **plus engineered counts and reports** | Type I bypass (more information than native methods) | absE_R1 ≈ 0.18 (Inv. VIII). **Not comparable**: different information |
| identity-free deep-sets / attention readers | engineered primitive per-group evidence | relational inductive bias | Bayes-level; generalize to the held-out pair |
| concatenation MLP | the same | memorization baseline | fails on the held-out pair (≈ 70×) |
| lookup / simpler statistical models | — | Investigation VIII lookup collapses on the held-out pair; not rerun | — |
| T1 parents, matched generic adapters, from-scratch raw-token Transformer | — | **not rerun** in Investigation IX | — |

## Known vs new

| item | status |
|---|---|
| Bayes posterior, mixture over latent roles | **known** (Bayesian model averaging) |
| sum decomposition of symmetric functions; finite-domain universality | **known** [Zaheer et al. 2017; Wagstaff et al. 2019] |
| ANOVA interaction order; additive models cannot represent interactions | **known** |
| relational inductive biases give permutation invariance and better held-out generalization | **known** (relation networks, deep sets, graph networks) |
| decodable ≠ used | **known** caution in probing work |
| explicit pooled form p ∝ C(S,C)·A − C(S−1,C−1)·B for the role model, with S as the single cross-group scalar | **new, elementary** |
| exact interaction order 1 + C, which separates latent reliability (order 1) from colluder eligibility (order 2) | **new, elementary**; used to classify COR families |
| adaptive-echo sources need cross-group joint statistics (within-group counts insufficient, spread 0.95) | **new, elementary, numerical** |
| pretraining completeness: a bimodal outcome of P-count pretraining that separates the original transfer's KL perfectly and does not explain R1 (confirmed prospectively) | **new methodological correction** to Investigations VI–VIII, which had mixed two failure types |
| P-count R1 failure is not an information failure: exact Bayes is linearly decodable from complete parents' states (confirmed) | **new implementation finding** |
| the original block's failure is a regime-blind routing reallocation, and the frozen prefix carries no cross-group information (confirmed) | **new implementation finding** (partly falsified in its quantitative form, CP5) |
| full fine-tuning gives native Type III repair on 4/4 independent parents | **new implementation finding**, modest: "fine-tuning fixes it" is unsurprising in general; what is specific is that the frozen-prefix limitation is identified causally |
| native implementations of the repair differ across parents; a causal pathway found on one parent did not recur | **new negative finding** |
| exact representability does not give learned size extrapolation; partial copying breaks learned hard gates (confirmed) | **expected** given known extrapolation failures; specific confirmed instances |

**Net.**
- *Mathematics:* elementary but useful classification tools.
- *Mechanistic contribution:* moderate. It gives a clean separation of information availability, frozen-prefix limits
  and downstream shortcut, with prospective confirmation, plus a documented case of mechanism non-recurrence.
- *Methodology:* one significant correction (completeness).
- **No claim** of a new theory of relational computation or of COR.
