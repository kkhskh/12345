"""Evaluate the 16 Investigation VI parents (step 4000, development) on the Investigation VII dev evaluation set."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, nnlib as NL, evalrun as ER
ev = dict(np.load(os.path.join(HERE, "..", "logs", "eval_dev.npz"), allow_pickle=True)); out = {}
for m in [f"xT{s}" for s in range(70, 78)] + [f"xP{s}" for s in range(80, 88)]:
    r = ER.run(NL.full_model_logits(NL.load(NL.dev_ckpt(m))), ev); out[m] = r; h = ER.headline(r)
    print(m, " ".join(f"{k} {v:+.3f}" for k, v in h.items()), flush=True)
json.dump(out, open(os.path.join(HERE, "..", "logs", "dev_parents_eval.json"), "w"), indent=1)
