"""Build generator-side evaluation sets.  Usage: build_eval.py NAME STREAM   (dev: 17_100_000; confirmation set fixed in the protocol)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, r1lib as L
name, stream = sys.argv[1], int(sys.argv[2]); ev = L.build_eval(stream)
np.savez_compressed(os.path.join(HERE, "..", "logs", f"eval_{name}.npz"), **ev)
lg = {k: ev["cB_" + k if k != "base" else "cB_base"] for k in L.eval_names()}
res = L.evaluate_logits(ev, lg)   # exact Bayes evaluated as a 'model': E must be 0 everywhere
summ = {f: {r: dict(n=v["n"], dmB=round(v["dmB"], 3)) for r, v in d.items()} for f, d in res.items()}
json.dump(summ, open(os.path.join(HERE, "..", "logs", f"eval_{name}_summary.json"), "w"), indent=1); print(json.dumps(summ))
print("max |E| for Bayes-as-model:", max(abs(v["E"]) for d in res.values() for v in d.values()))
