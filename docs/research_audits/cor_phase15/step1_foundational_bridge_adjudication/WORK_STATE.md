# Work state

Branch `cor-phase15-step1-foundational-bridge`, based on `origin/cursor/overnight-reconstruction-b68c` at `680c5fd2b2f33f9fb7388bff206b4d892f97aacd`.

## Done

1. Sources read: Phase 1 audit (`cor_phase1/02,04,06,09,10,15`), the final verdicts of Steps 1A, 1B, 1C, 1C-R, 1D, and 1E, and the overnight reconstruction. See `SOURCE_REGISTER.md`.
2. Strategy fixed: build on canonical variables (input tokens, per-position residual, per-component writes, attention weights, frozen unembedding), not learned `P` and `ρ`.
3. `proofs/bridge_math.py` and `tests/test_bridge.py`: 21 tests pass (log in `logs/`).
4. Prior work verified by web search: Stolfo et al. 2024; Rushing and Nanda 2024; Luce and Tukey 1964; Scott 1964; Krantz et al. 1971; Loftus 1978; Wagenmakers et al. 2012.

## Theorem set (status in `THEOREM_REGISTER.md`)

SS (sum screening), NC (norm channel, single crossing), CI (softmax consensus identity), RB (ordinal relation-blindness: LP criterion, invariance, independence ⇒ no direction dependence, prospective forcing), D (dilution crossover in a full block), LN (norm-coupling crossover), MIX (mixed objective split), CORR (correctness is not internally identifiable).

## Next

`PROOFS.md`, then the main argument, dossiers, matrix, and verdict. Then independent subagent reviews, adjudication, clean-room verification, and push.

## Resume instructions

Run `python -m pytest tests -q` from this directory. Read `RESEARCH_JOURNAL.md` for the reasoning trail.
