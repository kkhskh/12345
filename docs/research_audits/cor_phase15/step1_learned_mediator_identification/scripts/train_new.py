"""Train NEW models (identical training procedure to the historical preregistration):
   AGG-NL (internal code T6) seeds 0-4; held-out confirmation seeds 7-9 for REL-XNOR (T1), REL-GATE (T3),
   REL-MIS (T4), K3-GATE (T5b). Saves npz checkpoints + a training record."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model as M, ckpt
jobs = [("T6", s) for s in range(5)] + [(t, s) for t in ["T1", "T3", "T4", "T5b"] for s in (7, 8, 9)]
rec = {}
T0 = time.time()
for task, seed in jobs:
    val = M.sample(task, 20000, np.random.default_rng(1000))
    for arch in M.ARCH:
        p, l, dt = M.train(task, arch, seed)
        ckpt.save(ROOT / "checkpoints/new" / f"{task}_{arch}_s{seed}.npz", p)
        acc = M.accuracy(p, task, val)
        rec[f"{task}/{arch}/s{seed}"] = dict(val_acc=acc, final_loss=l, sec=dt)
        print(task, arch, seed, round(acc, 4), round(l, 4), round(dt, 1), flush=True)
rec["_config"] = dict(D_MODEL=M.D_MODEL, HEADS=M.HEADS, DH=M.DH, MLPW=M.MLPW, STEPS=M.STEPS, BATCH=M.BATCH, LR=M.LR,
                      optimizer="adam + cosine decay", init="attn 0.02, mlp/input 0.1", wall=time.time() - T0)
(ROOT / "logs" / "train_new.json").write_text(json.dumps(rec, indent=1))
