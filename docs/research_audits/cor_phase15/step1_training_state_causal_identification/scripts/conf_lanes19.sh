#!/bin/bash
# confirmation driver: lane $1 works through queue_c1 (step-100 window states), then queue_c2 (continuations + measurement)
cd "$(dirname "$0")/.."; QUEUE=logs/conf/queue_c1.txt bash scripts/lane2.sh $1; QUEUE=logs/conf/queue_c2.txt bash scripts/lane2.sh $1
