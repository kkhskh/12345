#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "scripts/train_run.py" > /dev/null; do sleep 20; done
scripts/queue.sh logs/train/jobs_main.txt
