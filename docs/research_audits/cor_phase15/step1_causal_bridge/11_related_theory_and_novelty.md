# 11 — Related theory

Citations were checked against the publisher pages listed in the assignment. Claims below use those abstracts and the theorems proved in this directory. Where a paper's internal theorem numbering was not re-read from the PDF in this step, the comparison is limited to the statement the publisher page actually makes, and the limitation is explicit.

## 1. Causal abstraction

Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard (JMLR 2025, 26(83):1–64) define causal abstraction as a relation between a low-level model and a high-level algorithm under transformations of mechanisms, not only hard interventions. The abstract states that the framework unifies activation patching, path patching, mediation, scrubbing, causal tracing, circuit analysis, concept erasure, interchange, distributed alignment search, and steering.

What COR would need to add is a prediction that this framework does not already make. Faithfulness in that framework is agreement between intervened high-level and low-level mechanisms. Family A's commutation identity is weaker: CE-13 satisfies commutation and the high-level stalk does not mediate the logit. Family E's path effect is the object the JMLR paper already places under path patching and mediation. A consistency residual that vanishes on exact recomputed traces (Theorem Z) is not an additional faithfulness score. It is zero in the case where the abstraction matches the computation.

Classification of Family A against this paper: not an instance of the faithfulness theorem. Commutation and interchange invariance are incomparable on the examples in file 05. Zero linear residual is neither necessary nor sufficient for an exact transformation of the algorithm that produces the logit.

## 2. Learned alignments

Geiger, Wu, Potts, Icard, and Goodman (CLeaR 2024, PMLR 236:160–187) estimate distributed alignments by gradient descent on an intervention objective. The alignment is tied to interchange behavior by the training loss. A `ρ` fit by ridge regression on clean cochains is not that estimator. Optimizing an alignment against intervention outcomes and then reporting those outcomes is the circularity in file 07. This step does not propose a new alignment algorithm.

Classification: Family A is not DAS. Using DAS as the definition of `ρ` would make the score an interchange fit, which is the existing method.

## 3. Unrestricted alignments are vacuous

Sutter, Minder, Hofmann, and Pimentel (NeurIPS 2025) prove that, without a restriction on the alignment class, a network can be mapped to an arbitrary algorithm, and they report perfect interchange-intervention accuracy for alignments of randomly initialized language models on IOI. CE-18 is a two-point illustration of the same mechanism, not a replacement of their theorem.

Classification of any bridge that allows `P` or `ρ` to be an arbitrary function of the input: too weak to assess, and already shown to be uninformative by this paper. A single linear `ρ` shared across inputs sits outside that unrestricted class. Being outside it does not put the map under Theorem Z. Theorem Z applies only when the matrix intertwines `f`. Otherwise the fit residual is Theorem App's approximation error, which still does not determine the sign of `R_c`. An input-dependent attention weight is an expressive map of the kind the vacuity result is about. It is not a route back into Theorem Z.

## 4. Readout is not use

Elazar, Ravfogel, Jacovi, and Goldberg (TACL 2021) remove linearly recoverable information and measure behavior. The result that matters here is the separation between a probe and a causal effect. CE-13 is that separation inside a commuting diagram: the stalk is a perfect probe of coordinate 1 and the logit uses coordinate 2. Nothing in this step strengthens their separation into a consistency-to-decision theorem.

## 5. Partial abstractions

Pîslar, Magliacane, and Geiger (CLeaR 2025, PMLR 275:114–138) combine several high-level models and trade the number of inputs a hypothesis covers against interchange-intervention accuracy. The relevant precedent is that partial faithfulness is already a defined, intervention-based quantity. A sum of local residuals is not that quantity. This step does not claim a new partial-faithfulness theorem.

## 6. Classification of each family

| Family | Relation to the cited work |
|---|---|
| A | Weaker than causal-abstraction faithfulness. Not new once commutation is distinguished from interchange |
| B | Standard Lipschitz bound on distance to a subspace |
| C | Sufficient statistic through `Π`. Under the assumption, `D` is idle. Not a new sufficiency theorem |
| D | An energy assigned by the analyst. Not an implementation theorem |
| E | Jacobian-vector products and path patching, already inside the JMLR unification |
| F | Margin-versus-Jacobian stability. Update Proposition 2 adds an unused slack term |
| G | Graph specification error, or Theorem Z when the graph is right |
| H | Interaction / Hessian. Not Definition 3 |

No family is a genuine extension that predicts an intervention response the cited frameworks do not already define. The negative theorems in files 02, 03, and 09 are audit results about COR's objects. They are not claimed as new mathematical theories of neural networks.

## 7. What would have counted as an extension

A theorem of the following shape would have been an extension: from `(P, ρ, M, W)` frozen without `∇L`, derive the sign of `R_c` on a stated intervention class, in a model that does not assume that sign, and show a system where attribution's ranking differs. No such theorem is available. File 12 records the comparisons that block the obvious attempts.
