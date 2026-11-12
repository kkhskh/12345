#!/bin/bash
# wait until no process matches $1, then run lane $2 on queue $3
cd "$(dirname "$0")/.."; while pgrep -f "$1" > /dev/null; do sleep 30; done; QUEUE=$3 scripts/lane.sh $2
