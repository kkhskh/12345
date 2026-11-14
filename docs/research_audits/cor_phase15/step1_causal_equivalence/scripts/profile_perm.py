"""Head-permutation-invariant comparison of component effect profiles (head indices are arbitrary across independently pretrained parents).
For each pair of models and family: max over per-layer head permutations (layers 0,1; 24 x 24) of the cosine of the 8-dim layer-0/1 head
effect vectors.  Also the identity-permutation cosine (meaningful only within a parent).  -> logs/head_profile_perm.json"""
import json, os, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); d = json.load(open(os.path.join(HERE, "..", "logs", "head_profile.json"))); ms = list(d["models"])
P = list(itertools.permutations(range(4))); out = {}
for f in ("R1", "D1", "D2"):
    V = {m: np.array(d["models"][m][f]["effects"][:8]) for m in ms}; res = {}
    for a, b in itertools.combinations(ms, 2):
        best = max(float(V[a] @ np.concatenate([V[b][:4][list(p0)], V[b][4:8][list(p1)]]) / (np.linalg.norm(V[a]) * np.linalg.norm(V[b]) + 1e-12)) for p0 in P for p1 in P)
        ident = float(V[a] @ V[b] / (np.linalg.norm(V[a]) * np.linalg.norm(V[b]) + 1e-12)); res[f"{a}|{b}"] = dict(perm_max=best, identity=ident)
    out[f] = res
json.dump(out, open(os.path.join(HERE, "..", "logs", "head_profile_perm.json"), "w"), indent=1)
for f, r in out.items():
    print(f); [print("  ", k.replace(":full_c1", "").replace(":_eval_conf9", ""), round(v["perm_max"], 2), round(v["identity"], 2)) for k, v in r.items() if "c1:0" not in k.split("|")[1] or True]
