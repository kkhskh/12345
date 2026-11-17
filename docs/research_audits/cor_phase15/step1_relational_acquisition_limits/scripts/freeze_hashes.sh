#!/bin/bash
# XVI: write logs/FREEZE_XVI_HASHES.txt (paths relative to the package root): generator, diagnostics, trainer, mathematics, scorer,
# design, queue, reference caches, math verification, every imported sibling module, and the 6 XV checkpoints that the continuations start from.
cd "$(dirname "$0")/.."
F="src/xvitask.py src/xvidiag.py scripts/xvi_train.py scripts/xvi_math_core.py scripts/xvi_math.py scripts/xvi_math_verify.py scripts/score_XVI.py scripts/xvi_design.py
logs/XVI_design.json logs/XVI_conditions.json logs/queue_train.txt logs/xvi_math_verified.json $(ls logs/refs_*.npz)
../step1_latent_reliability/src/bayes.py ../step1_latent_reliability/src/ckpt.py ../step1_latent_reliability/src/nets.py
../step1_relational_computation_atlas/src/fwd.py ../step1_relational_computation_atlas/src/gen_family.py ../step1_relational_mechanism_deepening/src/fwd3.py
../step1_cross_family_relational_foundations/src/gatelib.py ../step1_foundational_atlas_continuation/src/tasks_ext.py ../step1_foundational_atlas_continuation/src/train_tasks.py
../step1_relational_learning_order_laws/src/xvtask.py
$(for r in C_s201 C_s202 C_s203 D_s201 D_s202 D_s204; do echo ../step1_relational_learning_order_laws/confirm_outputs/$r/step4000.npz; done)"
for f in $F; do echo "$(sha256sum "$f" | cut -d' ' -f1) $f"; done > logs/FREEZE_XVI_HASHES.txt
wc -l logs/FREEZE_XVI_HASHES.txt
