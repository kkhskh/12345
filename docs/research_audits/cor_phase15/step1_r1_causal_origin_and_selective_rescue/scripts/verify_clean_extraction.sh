#!/bin/bash
# Clean-extraction check (Investigation VII): git-archive the committed tree, then
#  [1] MANIFEST.sha256;  [2] forward-pass reproduction: re-evaluate the committed final checkpoints of yP90 C1 and yP90 C1_injm on the
#  committed confirmation evaluation set and compare with the committed step-4000 evaluation (tol 1e-5);  [3] replay of the frozen
#  confirmatory analysis (confirm7_analysis.py) and comparison with committed logs/confirmatory7.json.
set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); REL=docs/research_audits/cor_phase15
OUT=${1:-/tmp/cor_inv7_clean}; rm -rf "$OUT"; mkdir -p "$OUT"; git -C "$REPO" archive HEAD $REL | tar -x -C "$OUT"
C="$OUT/$REL/step1_r1_causal_origin_and_selective_rescue"; cd "$C"; export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
echo "[1] manifest"; sha256sum -c --quiet MANIFEST.sha256 && echo "    manifest OK"
echo "[2] forward-pass reproduction"; python3 - <<'PY'
import json, sys, os, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import jax, jax.numpy as jnp, nnlib as NL, evalrun as ER, ckpt as C
N = NL.N; ev = dict(np.load("logs/eval_conf.npz", allow_pickle=True))
for tag in ("yP90_C1_r0", "yP90_C1_injm_r0"):
    committed = json.load(open(f"checkpoints/conf/{tag}.json"))["evals"]["4000"]
    t = jax.tree_util.tree_map(jnp.asarray, C.load(f"checkpoints/conf/{tag}.npz")); pf = NL.load("checkpoints/yP90/pre.npz")
    if "injm" in tag:
        import importlib.util; spec = importlib.util.spec_from_file_location("ds", "scripts/downstream.py")
        sys.argv = ["x", "yP90", "C1_injm", "0", "1"]   # not executed: build features with the frozen definition below
        import r1lib as L
        def feats(X):
            n = L.E.loo(X); V = np.stack([L.votes(X, qq) for qq in range(6)], 1); perf = (n == 5) & (V >= 0); m = perf.sum(-1, keepdims=True) - perf
            return np.eye(3)[np.minimum(m, 2)].reshape(*n.shape[:2], 9).astype(np.float32)
    def logits(X):
        z = NL.frozen_z(pf, jnp.asarray(X))
        if "injm" in tag: z = z.at[:, 36:42].add(jnp.asarray(feats(X)) @ t["Wi"])
        x = NL.block(t["l2"], z); return np.asarray(N.ln(x, t["gf"], t["bf"])[:, 36:42] @ t["WA"])
    fn = lambda X: np.concatenate([logits(X[i:i + 2000]) for i in range(0, len(X), 2000)])
    r = ER.run(fn, ev); a, b = r["families"]["F45"]["R1"]["E"], committed["families"]["F45"]["R1"]["E"]; k1, k2 = r["quality"]["KL"], committed["quality"]["KL"]
    print("    %-18s E_R1 %.6f vs %.6f  KL %.6f vs %.6f  %s" % (tag, a, b, k1, k2, "OK" if abs(a - b) < 1e-5 and abs(k1 - k2) < 1e-5 else "DIFF"))
PY
echo "[3] replay of frozen confirmatory analysis"; cp logs/confirmatory7.json /tmp/conf7_committed.json; python3 scripts/confirm7_analysis.py > /dev/null 2>&1
python3 -c "import json;a=json.load(open('logs/confirmatory7.json'));b=json.load(open('/tmp/conf7_committed.json'));print('    OK' if json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True) else '    DIFF')"
echo done
