# Outcome leakage, circularity, and specification freezing

## 1. Two kinds of circularity

**Conceptual circularity.** A premise already states the conclusion. Examples: "`L` is monotone in `D`"; "`∇O` is anti-aligned with `∇m`"; `S = −ΔO + βΔm` with `β` large. A held-out split cannot rescue these (mandate §7).

**Statistical leakage.** A parameter is chosen using the responses it is later said to predict.

Audit of the surviving object:

| Item | Conceptual circularity? | Leakage? |
|---|---|---|
| SS, NC, CI, RB, MIX, CORR | No. The premises are the architecture, the LN formula, the weighted-mean algebra, and the order structure. None mentions `R` | No parameters |
| `ũ` (candidate functional) | No. It is the frozen readout, a weight, not a response. It plays the same role as defining "the margin". Using it to *define* `v_j` is like direct logit attribution: it says which direction is "for `c`". It does not say how `m` responds to an intervention | No |
| A4 (path dominance) | **Yes, in effect** (review C, FATAL-4). It asserts that the response has the sign of a computed direct term, so assuming it states the prediction. Rejected as a premise | yes, if checked on test cells; a transport assumption otherwise |
| Construction D weights | No | Weights solved exactly from the token design (salience, evidence), asserted to `1e-9`. No response was used in the design |
| Repair objectives | Rejected (MIX) | — |

## 2. Choice ledger: surviving object versus original COR

| Choice | Original COR | Surviving object | Justification, fixed before evaluation |
|---|---|---|---|
| candidate definitions | answer pair; latent `y` | vocabulary indices `(c,c')` or a set `C` | task |
| representation dimensions | ranks `r_v ∈ {4,8,16}` searched | none (canonical scalars) | — |
| projector construction | ridge, PCA, probe | none | — |
| graph construction | analyst `K_x` | none | — |
| transport operators | ridge or Jacobian `ρ` | none | — |
| metrics | `M`, `W` | none (ordinal, RB.2) | — |
| normalization | per-score | none | — |
| intervention class | activation additions at chosen sites and scales | token substitution at frozen evidence slots; for NC, edits in `N_C` | design |
| outcome | `Δm`, JS, accuracy, mixed | the margin order on factorial cells | — |
| coefficients | `α, β, γ` | none | — |
| model class | any | LN-final residual Transformer | architecture |
| selection data | train-clean activations (and evaluation-tuned `β`) | none for the object. Calibration family for A4 only | — |
| allowed revisions | unrestricted | none after freezing. A failed prediction is recorded, not repaired | — |
| frozen prediction | none | RB-forced orders, plus the dilution sign `sign(v'_j − M)` | design |

That leaves zero fitted parameters, but about a dozen discrete design choices (review C, SERIOUS-5): candidate pair and competitor; read position; evidence slots; level token lists, including the neutral token, whose salience can decide the sign; contexts; layer and head set; number of levels; tie tolerance; the train and held-out split; the rule for combining tables across contexts; and the decision scalar. Because random tables are relation-sensitive about 86% of the time, every one of them must be frozen.

## 3. Freezing procedure (for any future, separately authorized test)

**Compartment 1: discovery.** Hypotheses come from this document and from the architecture. Historical Othello, IOI, factual, and 2B/2D outcomes may motivate a hypothesis but are not data for it.

**Compartment 2: predictor definition.** Frozen and hashed before anything is evaluated:
- the model checkpoint;
- the read position rule;
- the candidate pair;
- the evidence slots and the level-to-token lists (neutral, weak support, strong support, oppose);
- the factorial design;
- the training comparisons `𝒯`;
- the held-out pairs and their RB-forced orders, computed by `rb_forces` from `𝒯` alone;
- for the consensus law: the rule that classifies a cell as "dilution", using `w`, `v`, `M` computed from a forward pass that stops at the head. These are activations, not the margin.

**Compartment 3: nuisance calibration (A4).** On a disjoint calibration family, measure direct versus total effect signs by path patching. Report the agreement rate. Do not tune anything on it.

**Compartment 4: test.** Evaluate the frozen decision scalar on the held-out cells only, one RB test per context, combined by the pre-stated rule. A metric-free claim cannot average margins across contexts (review C, SERIOUS-6). A forced-order violation falsifies relation-blindness for that model and family. As a result about consistency, it carries no information beyond that (review C, FATAL-1).

## 4. What cannot be rescued by a split

A sign that is assumed rather than derived. The surviving theorems derive their signs: CI from algebra, NC from monotonicity of `1/σ`, RB.3 from strict monotonicity of `ψ`. The total-effect sign is not derived. It is the open arrow, and it is labeled as open, not assumed.
