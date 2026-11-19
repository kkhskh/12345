#!/bin/bash
# XIII-D: block until every listed file (path relative to the package root) exists and has been unchanged for 60 s, then exec the command after "--".
cd "$(dirname "$0")/.."; fs=(); while [ "$1" != "--" ]; do fs+=("$1"); shift; done; shift
for f in "${fs[@]}"; do until [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 60 ]; do sleep 30; done; done
exec bash -c "$*"
