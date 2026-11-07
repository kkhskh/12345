# Primary sources actually read

Quotations and definition numbers below were taken from the texts fetched for this reconstruction. Theorem numbers that were not in those texts are not used.

## Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, and Icard

Journal of Machine Learning Research 26(83):1–64, 2025. https://www.jmlr.org/papers/v26/23-0058.html

The paper's own abstract lists three contributions: interventionals (maps from mechanisms to mechanisms) beyond hard and soft interventions; formalizations of polysemantic neurons, the linear representation hypothesis, modular features, and graded faithfulness; and a common language for activation and path patching, mediation, scrubbing, tracing, circuits, erasure, sparse autoencoders, masking, distributed alignment search, and steering.

Definitions used, from the body that was read:

- Definition 9: a hard intervention replaces mechanisms by constants.
- Definition 11: an interventional edits mechanisms. Theorems 20 and 21 are cited in the paper as the modularity results for intervention algebras.
- Definition 25: exact transformation, generalized from hard interventions to intervention algebras.
- Definition 33: constructive causal abstraction, a lossy exact transformation.
- Definition 44: simple interchange intervention. Definition 46: distributed interchange intervention.

Our kernel hypothesis is a hard intervention on the branch in the special case where the branch mechanism does not read the supplied direction. Exactness of `R = w·δ` is then the residual equation, not a new abstraction theorem. We do not claim an interchange-intervention accuracy, and we do not define a high-level algorithm.

## Sutter, Minder, Hofmann, and Pimentel

NeurIPS 2025, "The Non-Linear Representation Dilemma." Proceedings text and arXiv:2507.08802.

Theorem 1, as stated in section 4 of the paper: given any algorithm `A` and any network `N` such that Assumptions 1–5 hold, `A` is an input-restricted distributed abstraction of `N`. The assumptions, in their wording, are a countable input space, input-injectivity of the layers, strict output-surjectivity, matchable partial orders, and that the network solves the task. They note that perfect interchange-intervention accuracy is impossible when the network does not solve the task, while an alignment can still score well on intervened inputs.

This is the precise form of "unrestricted maps manufacture agreement" used in this reconstruction. Our factor-kernel test is a linear restriction of the kind their dilemma says causal abstraction needs. The restriction is the attention and MLP multiplication, which is already the block. It does not create a new representation theorem. Their IOI experiment on random Pythia models is not refit here.

## Pîslar, Magliacane, and Geiger

CLeaR 2025, PMLR 275:114–138. https://proceedings.mlr.press/v275/pislar25a.html

Definition 1 combines high-level models by an input-space partition. Definition 2 measures strength by the proportion of inputs not assigned to a trivial model. Faithfulness is interchange-intervention accuracy. The paper's result is a tradeoff between that strength and that accuracy on GPT-2-small fine-tuned on arithmetic and Boolean tasks.

We do not learn a partition and we do not report an accuracy. The constant-shift quotient (response depends on the read-token mean and not on the rest of a constant edit) is an architectural identity, not their combine operation.

## Olivieri and Pérez Rodríguez

arXiv:2605.25225, preprint, submitted 24 May 2026, revised 11 June 2026. Title on the arXiv abstract: "Transformer Field Theory: A Response-Theoretic Approach to Mechanistic Interpretability." Treated as a preprint, not as a refereed theorem list.

The response formula in the HTML text is first-order: a sensitivity field `a(t,x)` predicts a patch, and a Green function is the propagator of the linearized depth equation. The adjoint field is identified with that sensitivity. Those objects are derivatives of the response. They sit in budget B2 of this reconstruction. They are competitors, not a quantity computed without `Dm`. Their GPT-2 measurements were not repeated. No theorem number is cited because the preprint's HTML develops the linear-response formulas without a numbered theorem matching our block.

## Architecture sources, used as definitions

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin, 2017. Scaled dot-product attention `softmax(QK^T / sqrt(d_k)) V`, multi-head concatenation, residual addition around sublayers. Our attention implementation is that formula with a causal mask.
- Ba, Kiros, and Hinton, 2016. Layer normalization. The map in this package is `γ ⊙ (h-μ)/sqrt(v+ε) + β`. The `ε` split, the rank, and Theorem F are calculations on that formula.
- Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, and Liu, ICML 2020. Pre-norm places the normalization inside the residual branch, `x + F(LN(x))`. The shift identity is that arrangement plus addition. It is not claimed for post-norm.
- Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, and Olah, 2021, "A Mathematical Framework for Transformer Circuits." The residual stream is the shared channel; with the pattern held fixed, the OV path is linear in `V`. Our value-only derivative is that frozen pattern. Seed 16 is what the sign does when the pattern is not frozen. Elhage, Lasenby, and Olah, 2023, is the privileged-basis note cited by Sutter; we use it only as that citation, not as a basis theorem of our own.

Path patching, as used by the IOI study of Wang, Variengien, Conmy, Shlegeris, and Steinhardt, 2023, and by Goldowsky-Dill, Heimersheim, Ortega, Bloom, Biderman, and collaborators in the open-problems discussion, replaces a path and compares outputs. A path effect here is either the value-only derivative or the full recomputed `R`. Neither is `O`.

Distributed alignment search is Geiger, Wu, Potts, Icard, and Goodman, the paper Sutter labels 2024b, "Finding alignments between interpretable causal variables and distributed neural representations." We do not run DAS.

## What this reconstruction does not borrow

No claim that a sensitivity field, an interchange accuracy, a combined causal model, or a Green operator improves on `Dm` for this block. The novelty matrix in `final/NOVELTY_AND_PRIOR_WORK.md` is restricted to the statements that were proved here.
