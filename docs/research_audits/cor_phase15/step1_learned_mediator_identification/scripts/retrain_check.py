"""A: regenerate a subset of historical checkpoints with the vendored code and compare."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model as M, ckpt
out = {}
for task, arch, seed in [("T1", "TF1", 0), ("T1", "TF2AO", 1), ("T3", "TF2", 0), ("T4", "TF1", 3)]:
    t = time.time()
    p, l, dt = M.train(task, arch, seed)
    h = ckpt.load(ROOT / "checkpoints/historical" / f"{task}_{arch}_s{seed}.npz")
    fa, fb = ckpt.flatten(p), ckpt.flatten(h)
    maxdiff = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa)
    val = M.sample(task, 20000, np.random.default_rng(1000))
    out[f"{task}/{arch}/s{seed}"] = dict(max_param_abs_diff=maxdiff, acc_retrained=M.accuracy(p, task, val),
                                        acc_historical=M.accuracy(h, task, val), bitwise_identical=maxdiff == 0.0)
    print(task, arch, seed, out[f"{task}/{arch}/s{seed}"], flush=True)
(ROOT / "logs" / "retrain_check.json").write_text(json.dumps(out, indent=1))
