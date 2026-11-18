#!/bin/bash
# Step 18: reconstruct the full archive from its parts and check per-part and whole-archive SHA-256.  Usage: verify_parts17.sh PARTS_DIR
set -u; D=$1; cd $D || exit 1
echo "== multipart reconstruction $(date -u +%FT%TZ)"
sha256sum -c cor_inv17_relational_operator_reuse.parts.sha256 && echo "per-part hashes OK"
cat cor_inv17_relational_operator_reuse.zip.part_* > /tmp/x17_rebuilt.zip
W=$(grep 'WHOLE ' cor_inv17_relational_operator_reuse.whole.sha256 | awk '{print $1}'); G=$(sha256sum /tmp/x17_rebuilt.zip | awk '{print $1}')
[ "$W" = "$G" ] && echo "whole-archive hash OK $G" || echo "WHOLE HASH MISMATCH"
unzip -tq /tmp/x17_rebuilt.zip && echo "zip integrity OK"; rm -f /tmp/x17_rebuilt.zip
