#!/bin/bash
# Repeatedly analyse every finished Transformer checkpoint in checkpoints/main (dev then conf, model then tag).
cd "$(dirname "$0")/.."
mkdir -p logs/analysis/main
while true; do
  did=0
  for js in $(ls -tr checkpoints/main/*.json 2>/dev/null); do
    b=$(basename $js .json)
    case $b in *_mlp_*|*_deepsets) continue;; esac
    t=${b%%_*}; ck=checkpoints/main/$b.npz; o=logs/analysis/main/$b
    [ -f ${o}_dev.json ] || { python3 scripts/analyze_model.py $ck $t dev ${o}_dev.json > ${o}_dev.out 2>&1; did=1; }
    [ -f ${o}_conf.json ] || { python3 scripts/analyze_model.py $ck $t conf ${o}_conf.json ${o}_dev.json > ${o}_conf.out 2>&1; did=1; }
    [ -f ${o}_tag_dev.json ] || { python3 scripts/analyze_tag.py $ck $t dev ${o}_tag_dev.json > ${o}_tag_dev.out 2>&1; did=1; }
    [ -f ${o}_tag_conf.json ] || { python3 scripts/analyze_tag.py $ck $t conf ${o}_tag_conf.json ${o}_tag_dev.json > ${o}_tag_conf.out 2>&1; did=1; }
  done
  [ -f logs/analysis/STOP ] && exit 0
  [ $did = 0 ] && sleep 60
done
