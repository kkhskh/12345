#!/bin/bash
# XIII-C: block until every listed ft_factorial run directory has final.json, then exec the command after "--".
cd "$(dirname "$0")/.."; dirs=(); while [ "$1" != "--" ]; do dirs+=("$1"); shift; done; shift
for d in "${dirs[@]}"; do until [ -f "checkpoints/ft/$d/final.json" ]; do sleep 30; done; done
exec bash -c "$*"
