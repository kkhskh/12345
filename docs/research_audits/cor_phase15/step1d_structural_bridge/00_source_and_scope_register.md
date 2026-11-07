# 0. Sources, scope, and what was not read

## Repository materials that were read

- `docs/research_audits/cor_phase15/overnight_reconstruction/`, including `proofs/block.py`, `proofs/response_bounds.py`, `workstream_1/finite_response.md`, `final/PHASE15_FINAL_VERDICT.md`, `final/ADVERSARIAL_MATHEMATICAL_AUDIT.md`, `EXECUTIVE_SCIENTIFIC_VERDICT.md`, `README.md`, `final/CAUSAL_MECHANISMS.md`, and `literature/PRIMARY_NOTES.md`.
- The block specification fixed there: `T=3`, `d=4`, two heads, head width 2, tanh MLP, `ε=1e-5`, causal mask, raw readout on the last token. `make_params` was reimplemented against NumPy 1.26.4 `default_rng`. Seed 0 returns margin `0.4644721976523223`, matching the overnight record.

Phase 1, Step 1A, Step 1B, Step 1C, and Step 1C-R were not rewritten. Their preserved corrections are used as lemmas: `O` is not `D`; a zero discrepancy is not a correct decision; `ε>0` LayerNorm is not scale-invariant; the constant-shift identity is pre-norm plus addition; a value-only derivative can disagree in sign with the total derivative; `Dm = 0` does not imply finite `R = 0`.

## Materials that were not accessible

`INDEPENDENT_OVERNIGHT_AUDIT.md` and `independent_boundary_checks.py` are not in the repository. They were not read. Every example below is reconstructed from the mathematical specification in the Step 1D mandate. Where the recomputed floats differ from `6.50425e-5`, `0.876435`, and `7.58e-11`, the difference is reported and the weights were not edited after seeing `R`.

The untracked local zip `docs/research_audits/cor_phase15_overnight_reconstruction.zip` was not used.

## Environment

NumPy 1.26.4, pytest 9.2.0, SymPy 1.14.0. PyTorch is not installed. No second route uses it. No pretrained Transformer was run. No timing benchmark was recorded. Costs in `04_information_vs_computational_access.md` are operation counts.

## Scope of the model

One pre-norm block

```text
X = LN_ε(H),   H1 = H + Attn(X),   U = LN_ε(H1),   H2 = H1 + MLP(U),
m(H) = w · H2[r],   r = T - 1.
```

```text
do(H_t ← H_t + δ) replaces token t and recomputes the block.
R(H, t, δ) = m(do(H_t ← H_t + δ)) - m(H).
```

A final LayerNorm readout `w · LN_final(H2[r])` is studied only as a different functional. It is not silently substituted for `m`.

## Reviews

No isolated reviewer agent was available in this session. The prosecution pass, the defense pass, and the cross-examination are separate critical passes written in this same context. They are not external referees.

## Official status, unchanged

`COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`. `OBSTRUCTION_READY = NO`. `PHASE_2 = NO GO`.
