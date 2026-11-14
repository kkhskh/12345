"""Workstream B4: permutation-aware comparison of WHICH layer-1 components write the conjunction (E-subspace interaction shares from
logs/factorial<TAG>.json).  Profile per model: shares of h1.0..h1.3 and mlp1.  Heads are only identifiable up to permutation across parents
(different pretraining runs); within a parent (seed 0 vs seed 1 fine-tunes) head indices are inherited.  For each model pair report:
  raw   : L1 distance between profiles with head identity kept
  perm  : min over the 24 head permutations (mlp1 fixed)
  sorted: distance between sorted head profiles (permutation-invariant summary)
Usage: perm_compare.py TAG -> logs/perm_compare<TAG>.json"""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else "_dev"
d = json.load(open(os.path.join(HERE, "..", "logs", f"factorial{TAG}.json"))); full = {k: v for k, v in d.items() if k != "sanity_dS" and "full_c1" in k}
prof = {k: np.array([v["TRUE"]["share_E"][f"h1.{h}"] for h in range(4)] + [v["TRUE"]["share_E"]["mlp1"]]) for k, v in full.items()}
parent = lambda k: k.split(":")[0]
rows = []
for a, b in itertools.combinations(prof, 2):
    pa, pb = prof[a], prof[b]; raw = float(np.abs(pa - pb).sum())
    perm = min(float(np.abs(pa[list(p) + [4]] - pb).sum()) for p in itertools.permutations(range(4)))
    srt = float(np.abs(np.sort(pa[:4]) - np.sort(pb[:4])).sum() + abs(pa[4] - pb[4]))
    rows.append(dict(a=a, b=b, same_parent=parent(a) == parent(b), raw=raw, perm=perm, sorted=srt))
out = dict(profiles={k: v.round(3).tolist() for k, v in prof.items()}, pairs=rows,
           mean_same_parent={m: float(np.mean([r[m] for r in rows if r["same_parent"]])) for m in ("raw", "perm", "sorted")},
           mean_cross_parent={m: float(np.mean([r[m] for r in rows if not r["same_parent"]])) for m in ("raw", "perm", "sorted")})
json.dump(out, open(os.path.join(HERE, "..", "logs", f"perm_compare{TAG}.json"), "w"), indent=1)
for k, v in out["profiles"].items(): print(f"{k[:28]:28s}", v)
for r in rows:
    if r["same_parent"]: print("same-parent", r["a"][:14], r["b"][:14], {m: round(r[m], 2) for m in ("raw", "perm", "sorted")})
print("mean same", {k: round(v, 2) for k, v in out["mean_same_parent"].items()}, "mean cross", {k: round(v, 2) for k, v in out["mean_cross_parent"].items()})
