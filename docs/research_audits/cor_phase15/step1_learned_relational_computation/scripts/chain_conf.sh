#!/bin/bash
# Wait until pgrep pattern $1 is gone AND the protocol freeze file exists, then run lane $2 over queues $3 $4 ... in order.
cd "$(dirname "$0")/.."; pat="$1"; name="$2"; shift 2
while pgrep -f "$pat" > /dev/null || [ ! -f logs/PROTOCOL_HASHES.txt ]; do sleep 30; done
for q in "$@"; do QUEUE=$q bash scripts/lane.sh $name; done
