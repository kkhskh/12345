# Deviations and post-freeze code changes (chronological)

1. **analyze_model.py memory fix** (before any analysis of a seed 0–2 model).
   - The attention patterns of pair episodes are now cached only for `cond` pairs, the only kind that uses them.
   - The first run of the pilot dev analysis ran out of memory (OOM-killed).
   - No definition or threshold changed.
2. **Addendum 1 and addendum 2** (`PROTOCOL_ADDENDUM_TAG.md`).
   - These add analyses only. Both were registered before any confirmation data; see that file for exact timing.
3. **Known flaw in the frozen "distributed but localized" verdict** (noticed while writing the aggregator, before any confirmation result was seen).
   - At the final site resid_post{L−1}, the full residual of the readout token *is* the model's output representation. Patching it from E′ is therefore trivially complete (IIA = 1).
   - So the literal criterion holds for every model and is uninformative.
   - `aggregate.py` reports two versions:
     - the literal one (`Distributed_literal`);
     - one that excludes the final site (`Distributed_nonfinal`).
   - The frozen primary-answer rule mentions "distributed evidence". It is applied literally, as frozen, and the flaw is flagged wherever the answer depends on it.
4. nets.py/train_run.py: optional auxiliary R supervision flag (arch suffix _auxR), default off; used only for the post hoc T4 run in checkpoints/posthoc_auxR. ckpt.load_base added to read baseline checkpoints (object arrays).
5. posthoc.py content surrogates (linear in raw reports; all-group counts) were token-invariant, guaranteeing R^2 ~ 0 (review R_math_causal #5). Fixed by interacting with source / group identity; post hoc reruns use the fixed version.
6. Review fixes after confirmation results (reported **alongside**, never replacing, the frozen/registered verdicts):
   - `P6_gated` and `K6'_gated`: the reuse criteria literally pass when the base effect is absent (0.7 × ~0 threshold). Gated versions require P2 / K2′ respectively.
7. Post hoc experiments added after the reviews (all labelled POST HOC):
   - T1→T4 curriculum: fine-tune each T1 M seed on T4 for 3000 steps (`scripts/train_curriculum.py`).
   - Per-group-histogram MLP baseline `mlp_grouphist`, which the mandate's §10 "group-specific aggregation" asks for and which was missing.
8. Clean-extraction verification passes (scripts/verify_extraction.sh): VERDICTS and verdicts.json reproduced, E1 example reproduced, 38 checkpoints loadable.
9. Replication (REPLICATION_PROTOCOL_T4.md): no post-freeze code changes to replicate_t4.py after the first new-seed run. Post hoc diagnostics (posthoc_layers/attn_targets/freeze.py) were run only after all four confirmatory evaluations.
