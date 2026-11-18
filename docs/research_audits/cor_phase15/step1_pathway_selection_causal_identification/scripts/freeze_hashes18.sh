#!/bin/bash
# XVIII freeze: SHA-256 of every frozen file (package and sibling dependencies listed in logs/DEPENDENCIES.json) -> logs/FREEZE_XVIII_HASHES.txt
cd "$(dirname "$0")/.."
F="scripts/xviii_train.py src/xviiim.py scripts/measure18.py scripts/score_XVIII.py scripts/preflight_XVIII.py scripts/xviii_design.py logs/XVIII_design.json logs/queue_c18.txt
   scripts/h3_immediate18.py scripts/continue18.py scripts/xviii_parents.py scripts/score_parents18.py logs/queue_scr.txt scripts/lane2.sh scripts/when_file.sh scripts/run_all18.sh scripts/score_cont18.py scripts/deps_inventory18.py scripts/unit_sha_far.py scripts/unit_modes.py scripts/unit_modes2.py scripts/unit_modes3.py
   PRE_EXPERIMENT_IDENTIFIABILITY_REVIEW.md ORIGINAL_COR_OD_RELEVANCE.md logs/preflight_XVIII.json logs/unit_modes.json logs/unit_modes2.json logs/unit_modes3.json logs/unit_sha_far.json"
{ sha256sum $F; python3 -c "
import json
d = json.load(open('logs/DEPENDENCIES.json'))
for m in d['external_python_modules'].values(): print(m['sha256'] + '  ../' + m['path'])"; } > logs/FREEZE_XVIII_HASHES.txt
sha256sum -c --quiet logs/FREEZE_XVIII_HASHES.txt && echo "freeze hashes OK ($(wc -l < logs/FREEZE_XVIII_HASHES.txt) files)"
