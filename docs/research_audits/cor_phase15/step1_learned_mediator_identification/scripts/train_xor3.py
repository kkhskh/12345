import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model as M, ckpt
rec = {}
for seed in range(7):
    val = M.sample("T7", 20000, np.random.default_rng(1000))
    for arch in M.ARCH:
        p, l, dt = M.train("T7", arch, seed)
        ckpt.save(ROOT / "checkpoints/new" / f"T7_{arch}_s{seed}.npz", p)
        rec[f"T7/{arch}/s{seed}"] = dict(val_acc=M.accuracy(p, "T7", val), final_loss=l, sec=dt)
        print(arch, seed, rec[f"T7/{arch}/s{seed}"], flush=True)
(ROOT / "logs" / "train_xor3.json").write_text(json.dumps(rec, indent=1))
