#!/bin/bash
# XVII: logs/FREEZE_XVII_HASHES.txt (paths relative to the package root)
cd "$(dirname "$0")/.."
F="src/xvitask.py src/xvidiag.py src/xviislot.py scripts/xvii_train.py scripts/xvii_confirm_analysis.py scripts/xvii_slot_train.py scripts/xvii_learning_arm.py
scripts/score_XVII.py scripts/xvii_design.py scripts/xvi_math_core.py scripts/when_file.sh logs/XVII_design.json logs/queue_xvii.txt $(ls logs/refs_*.npz)
../step1_latent_reliability/src/bayes.py ../step1_latent_reliability/src/ckpt.py ../step1_latent_reliability/src/nets.py
../step1_relational_computation_atlas/src/fwd.py ../step1_relational_computation_atlas/src/gen_family.py ../step1_relational_mechanism_deepening/src/fwd3.py
../step1_cross_family_relational_foundations/src/gatelib.py ../step1_foundational_atlas_continuation/src/tasks_ext.py ../step1_foundational_atlas_continuation/src/train_tasks.py"
for f in $F; do echo "$(sha256sum "$f" | cut -d' ' -f1) $f"; done > logs/FREEZE_XVII_HASHES.txt; wc -l logs/FREEZE_XVII_HASHES.txt
