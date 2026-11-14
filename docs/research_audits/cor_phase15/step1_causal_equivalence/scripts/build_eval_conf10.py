"""Build the Investigation X confirmation evaluation set (stream 20_900_000; never used in development).
Uses Investigation VII's r1lib.build_eval unchanged.  Output: logs/eval_conf10.npz"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "step1_learned_relational_computation", "src"))
import numpy as np, v8lib as V
L = V.L if hasattr(V, "L") else __import__("r1lib")
ev = L.build_eval(20_900_000)
np.savez_compressed(os.path.join(HERE, "..", "logs", "eval_conf10.npz"), **ev)
lg = {k: ev["cB_" + k if k != "base" else "cB_base"] for k in L.eval_names()}
res = L.evaluate_logits(ev, lg)
summ = {f: {r: dict(n=v["n"], dmB=round(v["dmB"], 3)) for r, v in d.items()} for f, d in res.items()}
json.dump(summ, open(os.path.join(HERE, "..", "logs", "eval_conf10_summary.json"), "w"), indent=1); print(json.dumps(summ))
print("max |E| for Bayes-as-model:", max(abs(v["E"]) for d in res.values() for v in d.values()))
