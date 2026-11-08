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
