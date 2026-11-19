"""Step-100 state distance of intervention/control states from the matched baseline B, per compose19 parameter group and in total, and
relative to the targeted block I (the predeclared matching quantity for non-targeted controls).  Usage: distance19.py BDIR IDIR X1 X2 ... > json"""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); names = json.load(open(os.path.join(H, "..", "logs", "param_names19.json")))
exec(open(os.path.join(H, "compose19.py")).read().split("def sel")[0].split("names = json.load")[1].split("\n", 1)[1])   # group()
ld = lambda d: [np.load(os.path.join(d, "step100.npz"))[f"arr_{k}"] for k in range(len(names))]
B, I = ld(sys.argv[1]), ld(sys.argv[2]); G = sorted({group(n) for n in names})
def dist(X): 
    per = {g: float(np.sqrt(sum(((X[k] - B[k]) ** 2).sum() for k in range(len(names)) if group(names[k]) == g))) for g in G}
    return dict(total=float(np.sqrt(sum(v ** 2 for v in per.values()))), per=per)
dI = dist(I); out = dict(B=os.path.basename(sys.argv[1]), I=os.path.basename(sys.argv[2]), I_dist=dI, controls={})
for x in sys.argv[3:]:
    dx = dist(ld(x)); out["controls"][os.path.basename(x)] = dict(dist=dx, ratio_total=dx["total"] / dI["total"], ratio_per={g: dx["per"][g] / dI["per"][g] for g in G if dI["per"][g] > 0})
print(json.dumps(out, indent=1))
