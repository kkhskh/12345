#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs/analysis/curriculum
w() { while pgrep -f "python3 scripts/(analyze_|posthoc)" > /dev/null; do sleep 30; done; }
for s in 0 1 2; do
  b=T4_M_K3_s${s}_fromT1; ck=checkpoints/posthoc_curriculum/$b.npz; o=logs/analysis/curriculum/$b
  while [ ! -f checkpoints/posthoc_curriculum/$b.json ]; do sleep 60; done
  [ -f ${o}_tag_conf.json ] && continue
  w; python3 scripts/analyze_model.py $ck T4 dev ${o}_dev.json > ${o}_dev.out 2>&1
  w; python3 scripts/analyze_model.py $ck T4 conf ${o}_conf.json ${o}_dev.json > ${o}_conf.out 2>&1
  w; python3 scripts/analyze_tag.py $ck T4 dev ${o}_tag_dev.json > ${o}_tag_dev.out 2>&1
  w; python3 scripts/analyze_tag.py $ck T4 conf ${o}_tag_conf.json ${o}_tag_dev.json > ${o}_tag_conf.out 2>&1
  w; python3 scripts/posthoc.py $ck T4 ${o}_tag_dev.json logs/posthoc/$b.json > logs/posthoc/$b.out 2>&1
  w; python3 scripts/posthoc_joint.py $ck T4 logs/posthoc/joint_$b.json > logs/posthoc/joint_$b.out 2>&1
done
