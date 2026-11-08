"""Usage: train_run.py TASK SIZE SEED STEPS OUTDIR [K]
Trains one episode Transformer, saves checkpoint (.npz) and a json with loss history, compute and eval metrics."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nets as N, ckpt as C
task, size, seed, steps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
K = int(sys.argv[6]) if len(sys.argv) > 6 else 3
arch = sys.argv[7] if len(sys.argv) > 7 else "tf"
aux_r = arch.endswith("_auxR"); arch = arch.replace("_auxR", "")
os.makedirs(out, exist_ok=True)
tag = f"{task}_{size}_K{K}_s{seed}" + ("" if arch == "tf" else f"_{arch}") + ("_auxR" if aux_r else "")
logf = open(os.path.join(out, tag + ".log"), "w")
def log(s):
    logf.write(s + "\n"); logf.flush()
p, hist, dt = N.train(task, size, seed, steps, K=K, log=log, arch=arch, aux_r=aux_r)
C.save(os.path.join(out, tag + ".npz"), p)
nparam = sum(int(v.size) for v in __import__("jax").tree_util.tree_leaves(p))
ev = N.evaluate(p, task, K=K, seed=999, arch=arch)          # dev evaluation stream (seed 999)
rec = dict(arch=arch, task=task, size=size, seed=seed, steps=steps, K=K, batch=256, train_seconds=dt, n_params=nparam,
           examples_seen=steps * 256, hist=hist, dev_eval=ev)
json.dump(rec, open(os.path.join(out, tag + ".json"), "w"), indent=1)
log(json.dumps(rec["dev_eval"]))
