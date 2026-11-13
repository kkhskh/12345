# Stage 0: confirmation cohort, frozen before any Investigation IX hypothesis was tested

The P-count lineages **P130–P135** (6 independent pretrainings) were started on 2026-11-13. They use
`scripts/cohort9_pretrain.py`, which is the Investigation VIII Stage-0 script unchanged except for the seeds. This
happened before any Investigation IX experiment had run. Only `pretrain.log` and `pre.json` (pretraining loss and the
P-count task evaluation) are read before the frozen protocol. No T4 transfer, probe or evaluation touches these
lineages until `PROSPECTIVE_PREDICTIONS.md` Part I is committed with hashes.

These lineages are genuinely independent pretrained parents: new seeds for initialization and the pretraining data
stream. They are distinct from the Investigation VI development parents xP80/xP81 and from the Investigation VIII
confirmation lineages zP100–107. Those older lineages are used **only for development** in Investigation IX, because
their T4 outcomes are already known.
