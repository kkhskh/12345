# Citation verification

Every bibliography entry below was checked against a publisher page, ACL Anthology, PMLR, JMLR, or arXiv abstract on 2026-11-05. Unresolved items are marked.

| Key | Title | Authors | Year | Venue | Stable ID | Supports which claim | Status |
|---|---|---|---|---|---|---|---|
| li2023emergent | Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task | Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg | 2023 | ICLR | https://mlanthology.org/iclr/2023/li2023iclr-emergent/ ; arXiv 2210.13382 | Othello-GPT; nonlinear probes; interventional board edits | VERIFIED |
| nanda2023emergent | Emergent Linear Representations in World Models of Self-Supervised Sequence Models | Neel Nanda, Andrew Lee, Martin Wattenberg | 2023 | BlackboxNLP | DOI 10.18653/v1/2023.blackboxnlp-1.2 ; https://aclanthology.org/2023.blackboxnlp-1.2/ | Linear player-relative probes; vector-arithmetic steering | VERIFIED |
| elazar2021amnesic | Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals | Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg | 2021 | TACL 9:160–175 | DOI 10.1162/tacl_a_00359 ; https://aclanthology.org/2021.tacl-1.10/ | Probe accuracy ≠ causal use; INLP erasure | VERIFIED |
| ravfogel2020null | Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection | Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg | 2020 | ACL | DOI 10.18653/v1/2020.acl-main.647 | Nullspace projection as erasure | VERIFIED |
| belrose2023leace | LEACE: Perfect Linear Concept Erasure in Closed Form | Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman | 2023 | NeurIPS | https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html | Closed-form linear erasure | VERIFIED |
| geiger2024das | Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations | Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, Noah Goodman | 2024 | CLeaR / PMLR 236:160–187 | https://proceedings.mlr.press/v236/geiger24a.html | DAS; distributed alignment; not our experiment | VERIFIED |
| geiger2025abstraction | Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability | Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, Thomas Icard | 2025 | JMLR 26(83):1–64 | https://jmlr.org/papers/v26/23-0058.html | Abstraction vs steering/erasure | VERIFIED |
| belinkov2022probing | Probing Classifiers: Promises, Shortcomings, and Advances | Yonatan Belinkov | 2022 | Computational Linguistics 48(1):207–219 | DOI 10.1162/coli_a_00422 (standard); confirm PDF if citing page numbers in camera-ready | Survey of probe limitations | VERIFIED_TITLE_YEAR; page numbers should be rechecked on MIT Press at submission |
| turner2023actadd | Activation Addition: Steering Language Models Without Optimization | Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid | 2023 | arXiv 2308.10248 | https://arxiv.org/abs/2308.10248 | Activation steering without a probe kernel constraint | VERIFIED as preprint; peer-reviewed venue NOT VERIFIED in this audit |
| rimsky2024caa | Steering Llama 2 via Contrastive Activation Addition | Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Turner | 2024 | ACL | DOI 10.18653/v1/2024.acl-long.828 | Contrastive steering | VERIFIED |
| zou2023repeng | Representation Engineering: A Top-Down Approach to AI Transparency | Andy Zou, Long Phan, Sarah Chen, et al. | 2023 | arXiv 2310.01405 | https://arxiv.org/abs/2310.01405 | Reading/steering vectors | VERIFIED as preprint; author list must be copied in full from arXiv at camera-ready |
| hewitt2019structural | Designing and Interpreting Probes with Control Tasks | John Hewitt, Percy Liang | 2019 | EMNLP | DOI 10.18653/v1/D19-1275 | Control tasks for probes | VERIFIED_STANDARD; included only if the related-work paragraph cites control tasks |
| meng2022rome | Locating and Editing Factual Associations in GPT | Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov | 2022 | NeurIPS | DOI 10.52202/06xxxx — use NeurIPS 2022 proceedings page; arXiv 2202.05262 | Causal tracing / rank-one editing; different task | VERIFIED as NeurIPS 2022 paper via arXiv 2202.05262; DOI to be copied from NeurIPS page at submission |

## Deliberately not cited as Othello evidence

Internal COR theory PDFs, Phase 1 math audit, and unfinished obstruction notes are not bibliographic support for the empirical claims. They are project history.

## Unresolved

- Exact DOI landing page for Belinkov 2022 page numbers: mark **recheck**.
- Turner et al. ActAdd peer-reviewed version: **NOT VERIFIED**; cite as arXiv unless a later venue is confirmed.
- Zou et al. 2023 full author list: **copy from arXiv** before any public version.
- Hardware, licenses of the Nanda redistribution, and dataset licenses: `MODEL_SPEC['license_note']` says research checkpoints redistributed by Neel Nanda from Li et al. ICLR 2023. Redistribution permissions for a public paper package remain **NOT VERIFIED** legally.

No BibTeX entry in `references.bib` was invented from a title-only snippet without a publisher or arXiv page.
