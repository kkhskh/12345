# 01 — Prior errors and what was not edited

Repository commit inspected: `f95d134fee8bfd870561e21bc57ada18c46248be`. The Phase 1, Step 1A, Step 1B, and Step 1C directories are untracked in the working tree. They were read and not modified. This directory is the correction.

Preserved status, not reopened: `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`, `PHASE_2 = NO GO`, `OBSTRUCTION_READY = NO`.

## Error A — epsilon-zero LayerNorm used too broadly

Step 1C, `04_transformer_computational_equations.md`, states that LayerNorm is invariant under `h ↦ a h + b 1` for `a>0`, and that the Jacobian has rank at most `d-2` wherever it exists. The proof note `proofs/layernorm_readout.md` derives the rank bound for `eps=0`. The summary sentence in `04` does not carry that restriction.

The corrected split is in `02_layernorm_epsilon_theory.md`.

- Adding a multiple of the all-ones vector leaves `LN_ε` unchanged for every `ε≥0`.
- Positive scaling leaves it unchanged only for `ε=0` and a positive scale.
- For `ε>0` and every gain entry nonzero, the Jacobian has rank `d-1`, not `d-2`. The centered direction is not in the kernel. Its singular value is small when `ε` is small, so a loose numerical cutoff reports the epsilon-zero rank by mistake.
- A direction killed by the normalization is not killed by `y = h + A(LN(h))`. The skip carries it.

## Error B — a deletion that is not necessary was labeled necessary

Step 1C, `07_multi_path_coordination.md`, has a bullet titled "Both are necessary" whose example is `w·p_1=1`, `w·p_2=-0.5`, margin `0.5`. Deleting the second contribution leaves margin `1`. Only the first deletion reverses the sign. The following sentence, that both signs of the pieces appear in the sum, is a different claim and does not repair the title.

`03_two_path_necessity_and_interaction.md` proves that a zero-baseline sum cannot make both deletions necessary, and gives a biased sum in which both are necessary while the four-term interaction remains zero.

## Packaging

The Step 1C README tells the reader to run

`python -m pytest docs/research_audits/cor_phase15/step1c_computational_mechanisms/tests/test_step1c_math.py -q`

That path exists in the repository and does not exist inside the released zip, whose root is `cor_phase15/step1c_computational_mechanisms/`. This package documents commands from its own root. The manifest lists only files in the archive. Cache files are not assets.

## What this round adds beyond the repair

One pre-norm block, with positive `ε`, causal multi-head attention, a residual intervention, and a tanh MLP, is differentiated and compared with finite differences. The question after that calculation is whether any relation among the paths constrains the response without already being the derivative of the response. The shift invariance below is exact and uses fewer weights than the Jacobian. It is still the definition of pre-norm LayerNorm plus addition. It is not a new obstruction.
