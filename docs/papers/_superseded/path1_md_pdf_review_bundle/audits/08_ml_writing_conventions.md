# ML writing conventions (venue requirements versus editorial practice)

Also filed to satisfy the Part 2 filename as `writing_conventions_research.md` (copy of this document).

Sources were consulted on 2026-11-05. Venue rules change. This file is not a submission checklist for a selected conference.

## A. Actual venue requirements (starting sources)

### NeurIPS 2026

- Main Track Handbook: https://neurips.cc/Conferences/2026/MainTrackHandbook
- Reviewer guidelines: https://neurips.cc/Conferences/2026/ReviewerGuidelines
- Paper checklist: https://neurips.cc/public/guides/PaperChecklist

Recorded from the 2026 handbook (V2026.3) as fetched:

- Double-blind review. OpenReview profiles required.
- Contribution types include empirical work and **Negative Results**.
- Mandatory paper checklist.
- Code of ethics and academic integrity policies apply.
- LLM/AI assistance: the handbook and checklist require disclosure of LLM use; unsanctioned LLMs must not be used by reviewers on confidential submissions.
- Formatting, page limits, and anonymity rules are conference-year specific and must be re-checked at submission time.
- These are **requirements**, not style advice.

### ICML 2026

- Author instructions: https://icml.cc/Conferences/2026/AuthorInstructions

Requirements typically include anonymization, page limits, a reproducibility statement, and a style template. Exact 2026 page counts and extra-page rules must be read on that page at submission time. Do not assume NeurIPS rules transfer.

### ICLR 2026

- Author guide: https://iclr.cc/Conferences/2026/AuthorGuide

ICLR is open review with public comments. Anonymization, dual-submission, and LLM policies can differ from NeurIPS. Verify before any submission. No venue has been selected.

### Nature formatting (style, not a target)

- https://www.nature.com/nature/for-authors/formatting-guide
- https://www.nature.com/articles/nmeth.4532 (Kwok, “How to write a first-class paper”, *Nature* methods/career content)

These are editorial recommendations: short titles, abstracts that state the result, figures that carry a claim, limitations stated plainly. They are not NeurIPS formatting rules.

## B. How strong empirical ML papers handle standard sections

The following is **editorial practice** observed in Li et al. (ICLR 2023), Nanda et al. (BlackboxNLP 2023), Elazar et al. (TACL 2021), Geiger et al. (CLeaR 2024 / JMLR 2025), not a requirement.

| Section | Practice | Not a requirement |
|---|---|---|
| Title | Technical, searchable, scoped to the setting | Need not contain “novel” |
| Abstract | Problem, setting, measurement, headline number, limit, conclusion | Need not open with a field-wide revolution |
| Introduction | Starts at the scientific uncertainty, not at “Transformers have changed AI” | Contribution bullets are optional |
| Related work | Synthesizes conflicts (probe vs causal; erasure vs steering) | Annotated bibliography is poor practice, not a rules violation |
| Methods | Defines the intervention, site, estimand, and unit | Equations are not required for prestige |
| Results | Hypothesis → design → number → interpretation → limit | Tables without prose are common and weak |
| Negative results | NeurIPS 2026 explicitly lists Negative Results as a contribution type | Burying them in a footnote is a scientific problem, not a template issue |
| Uncertainty | Intervals, *n*, unit | “Significant” without a test is informal misuse |
| Reproducibility | Code, seeds, data; checklist at NeurIPS | Claiming three reproducibility levels because tables compile is false |
| Ethics / AI | Venue checklist | Listing an AI system as an author is forbidden |
| Appendices | Extra ablations, proofs, hyperparameters | Main-text claims must still stand without the appendix |

## C. Notes from the closest empirical papers (reasoning style, not wording)

**Li et al., ICLR 2023.** Opens with a concrete question: does a sequence model trained only on move tokens form an internal board, or only surface statistics? Moves from nonlinear probes to interventional board edits. Causal claims are tied to those edits. Limitations include the synthetic game.

**Nanda, Lee, and Wattenberg, BlackboxNLP 2023.** Opens by tightening Li et al.: the board representation is linearly readable in a player-relative basis, and vector arithmetic on those probe directions changes play. Causal evidence is steering along the **probe-visible** direction. Negative or incomplete mechanistic stories (how the world model is computed) are discussed without converting them into a general Transformer theory.

**Elazar et al., TACL 2021.** Opens with the gap between encoding and use. Amnesic probing **removes** linearly readable information and measures behavior. They report that probe accuracy does not track behavioral importance.

**Geiger et al., 2024/2025.** Causal abstraction and DAS ask when an interpretable high-level variable aligns with a distributed representation under interchange interventions. Faithfulness is graded. Steering and concept erasure are placed in that language; they are not automatically alignments.

What this manuscript should copy is the **discipline**: one question, an intervention that matches the question, numbers with a unit, and a conclusion no broader than the model and probe. What it must not copy is distinctive phrasing.
