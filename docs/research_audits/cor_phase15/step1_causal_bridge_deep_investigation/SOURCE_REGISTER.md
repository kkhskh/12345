# Source register

## Repository and commits

- **Repository:** `kkhskh/12345`.
- **Work branch:** `cor-phase15-step1-deep-investigation`, created from `cor-phase15-step1-foundational-bridge` at `49c1ec6`.
- **Ancestry:** `49c1ec6` descends from `cursor/overnight-reconstruction-b68c` at `680c5fd`. `main` (`b9516a7`) holds no audits.
- **Branch heads checked on 2026-11-07:**

| Branch | Head |
|---|---|
| `08r1-real-model-repair` | `e06bb5d` |
| `08r2-implementation-repair` | `3211f0d` (adds archive zips of the overnight, 1D and 1E audits; no new code) |
| `cor-phase15-step1-foundational-bridge` | `49c1ec6` |
| `cursor/overnight-reconstruction-b68c` | `680c5fd` |
| `main` | `b9516a7` |

## Read for this step

| Source | Use |
|---|---|
| `docs/notes/Causal_Obstruction_Response_theory_Update.pdf`, §1–3 (text extracted) | central thesis: "distributed local evidence … must become sufficiently compatible to support a stable global commitment"; "stable decisions tend to exhibit low candidate-specific incompatibility" |
| `docs/research_audits/cor_phase1/01, 02, 04, 06, 09, 10, 15` | definitions, gauge, causal bridge, claim register, counterexamples, verdict |
| Step 1A–1E and overnight verdict files | read in the previous step; summarized in `../step1_foundational_bridge_adjudication/SOURCE_REGISTER.md` |
| `../step1_foundational_bridge_adjudication/` (whole package) | the submission audited and corrected here |
| `sources/independent_audit_handoff/` | external independent audit (a ChatGPT session; delivered by the user as `add75bfd-COR_Claude_independent_audit_handoff.zip`, SHA-256 `c457794c…`) |
| `src/model.py` | the frozen-model architecture: GPT-2 small through TransformerLens, `fold_ln=False`. GPT-2's MLP uses tanh-GELU, the activation covered by A1 |
| `src/toy_othello.py`, `src/toy_modular.py` | in-repo toy models use GELU and ReLU respectively, both covered by A1 |

**How the audit's claims were checked.** Its archive hash `239924ce…` matches the previous package's zip. Its two scripts were rerun here and reproduce exactly. Its three findings were verified independently before adoption.

## External literature

- Verified by search in the previous step: Stolfo et al. 2024; Rushing and Nanda 2024; Luce and Tukey 1964; Scott 1964; Loftus 1978; Wagenmakers et al. 2012; Geiger et al. 2025; Sutter et al. 2025; Elhage et al. 2021.
- Added this step by reviewer R4, with URLs in `reviews/R4_baselines_novelty.md`: Brunner et al. 2020; Méloux et al. 2025; Yarotsky 2017; McDougall et al. 2023; McGrath et al. 2023; Goldowsky-Dill et al. 2023; Godfrey et al. 2022; Ai and Norton 2003. These are cited through R4 and were not re-fetched here. R4 itself flags one detail as unconfirmed (whether Elhage et al. explicitly define `W_QK`).

## Missing

- None of the files the mandate names is missing.
- `INDEPENDENT_FOUNDATIONAL_BRIDGE_AUDIT.md` was not in the repository. It arrived through the user's upload.
- Reviewer scratch scripts were in `/tmp` and are not preserved.

## Not used

Pretrained weights, sealed confirmation data, and historical evaluation outcomes as data.
