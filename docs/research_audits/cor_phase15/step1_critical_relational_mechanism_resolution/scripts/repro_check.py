"""XIII-C A1: verify that ft_factorial (L=W=D=s) reproduces XIII-B native_train full_c1 seed s BIT-FOR-BIT.
Runs ft_factorial for the first K steps (FX_STOP=K; the 4000-step LR schedule is unchanged) and compares every parameter leaf with XIII-B's
saved step<K>.npz of the seed-s run.  Usage: repro_check.py PARENT SEED [K=50]  -> logs/repro_<PARENT>_<SEED>_<K>.json"""
import json, os, sys, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import native as NV
parent, seed = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 50
env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1", FX_STOP=str(K))
subprocess.run([sys.executable, os.path.join(HERE, "ft_factorial.py"), parent, str(seed), str(seed), str(seed)], check=True, env=env, stdout=subprocess.DEVNULL)
out = os.path.join(HERE, "..", "checkpoints", "ft", f"{parent}_fx_L{seed}_W{seed}_D{seed}_stop{K}")
ref = os.path.join(NV.X13B, "checkpoints", "native_train", f"{parent}_full_c1_s4000_seed{seed}_eval_conf13", f"step{K}.npz")
a = np.load(os.path.join(out, f"step{K}.npz")); b = np.load(ref)
diffs = [float(np.abs(a[f"arr_{i}"] - b[f"arr_{i}"]).max()) for i in range(len(a.files))]
res = dict(parent=parent, seed=seed, K=K, n_leaves=len(diffs), n_ref=len(b.files), max_abs_diff=max(diffs), bitwise=all(d == 0 for d in diffs))
shutil.rmtree(out); json.dump(res, open(os.path.join(HERE, "..", "logs", f"repro_{parent}_{seed}_{K}.json"), "w"), indent=1); print(res)
