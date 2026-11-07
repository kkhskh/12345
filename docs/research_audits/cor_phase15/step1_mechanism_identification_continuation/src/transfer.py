"""EXPLORATORY transfer (§16) for the learned models: residual width, head count, MLP width, depth.
Seeds 5, 6 (never used elsewhere). Tasks: T1 (relation required) and T2 (relation not required).
Reports: val accuracy, D_sum, number of relation units (preregistered criterion), effect of ablating them.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned as Lm
import learned_analysis as LA

def main():
    out = {}; lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    T0 = time.time()
    variants = [("d8", dict(D_MODEL=8, DH=4)), ("d32", dict(D_MODEL=32, DH=16)), ("heads1", dict(HEADS=1, DH=16)),
                ("heads4", dict(HEADS=4, DH=4)), ("mlp8", dict(MLPW=8)), ("mlp64", dict(MLPW=64))]
    base = dict(D_MODEL=Lm.D_MODEL, HEADS=Lm.HEADS, DH=Lm.DH, MLPW=Lm.MLPW)
    for name, kw in variants:
        for k, v in {**base, **kw}.items():
            setattr(Lm, k, v)
        for task in ["T1", "T2"]:
            for seed in [5, 6]:
                for arch in ["TF1", "TF2"]:
                    P, l, dt = Lm.train(task, arch, seed)
                    val = Lm.sample(task, 20000, np.random.default_rng(1000))
                    acc = Lm.accuracy(P, task, val)
                    r = LA.analyse_binary(task, arch, seed, [0], P=P)
                    us = r["units_summary"]
                    row = dict(acc=acc, D_sum=r["grid"]["all"]["D_sum"], flip=r["grid"]["all"]["flip"],
                               n_rel=us["n_rel"], acc_ablated=us["ablated"]["acc"], flip_ablated=us["ablated"]["flip"])
                    out[f"{name}/{task}/{arch}/s{seed}"] = row
                    log(f"{name} {task} {arch} s{seed}: {json.dumps(row)}")
    for k, v in base.items():
        setattr(Lm, k, v)
    log(f"wall {time.time() - T0:.0f}s")
    (HERE.parent / "logs" / "transfer.txt").write_text("\n".join(lines) + "\n")
    (HERE.parent / "logs" / "transfer.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
