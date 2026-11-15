"""XIII-D Workstream B5: does an early mechanistic measure predict the final organisation better than behavioural/optimisation baselines?
Population: runs listed in the metrics file (UM file, steps 50/111/246) whose final organisation is known (FINAL file(s) or known_outcomes.json).
Target: final lik_RO1 (continuous, early-route use) and the class (early vs not).  Predictors at step k:
  mechanistic: lik_RO1(k), mix_mass(k), route_Q(k)
  baselines  : acc(k), KL(k) (behaviour) and training loss at the nearest logged step <= max(k,100) (optimisation; ft runs only)
Scores per predictor: Spearman rho with the final lik_RO1, leave-one-out R^2 of a 1-d linear fit, and AUC for the early class (final lik >= 0.6).
Usage: B5_UM=logs/unit_metrics_b5dev.jsonl B5_FINAL=logs/unit_metrics_f2.jsonl,... B5_OUT=logs/b5_transition_dev.json python3 scripts/b5_transition.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.stats import spearmanr
import xc
UMF = os.environ.get("B5_UM", os.path.join(HERE, "..", "logs", "unit_metrics_b5dev.jsonl")); OUT = os.environ.get("B5_OUT", os.path.join(HERE, "..", "logs", "b5_transition_dev.json"))
EARLY = {}
for l in open(UMF): r = json.loads(l); EARLY.setdefault(r["unit"], {})[r["step"]] = r
FIN = {}
KO = json.load(open(os.path.join(HERE, "..", "logs", "known_outcomes.json")))
for sp in EARLY:
    f = xc.factors(sp)
    if f[4] == "" and f"{f[0]}|{f[1]}|{f[2]}|{f[3]}" in KO: FIN[sp] = KO[f"{f[0]}|{f[1]}|{f[2]}|{f[3]}"]["lik_RO1"]
for fn in filter(None, os.environ.get("B5_FINAL", "").split(",")):
    for l in open(fn):
        r = json.loads(l)
        if r["step"] == 4000 and r["unit"] in EARLY: FIN[r["unit"]] = r["lik_RO1"]
def loss_at(sp, k):
    try: c = json.load(open(os.path.join(xc.unit_dir(sp), "curve.json")))
    except Exception: return None
    h = [v for s, v in c["hist"] if s <= max(k, 100)]; return h[-1] if h else None
def auc(x, y):
    p, n = x[y], x[~y]
    if len(p) == 0 or len(n) == 0: return None
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]))
def loo_r2(x, y):
    if len(x) < 4: return None
    pr = np.empty(len(x))
    for i in range(len(x)):
        m = np.arange(len(x)) != i; A = np.c_[x[m], np.ones(m.sum())]; c = np.linalg.lstsq(A, y[m], rcond=None)[0]; pr[i] = c[0] * x[i] + c[1]
    return float(1 - np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2))
res = dict(n_runs=len(FIN), runs=sorted(FIN), by_step={})
for k in (111, 246):
    us = [u for u in sorted(FIN) if k in EARLY[u]]; y = np.array([FIN[u] for u in us]); cls = y >= 0.6; R = {}
    feats = {n: [EARLY[u][k][n] for u in us] for n in ("lik_RO1", "mix_mass", "route_Q", "acc", "KL")}
    feats["loss"] = [loss_at(u, k) for u in us]
    for n, v in feats.items():
        m = np.array([a is not None for a in v]); x = np.array([a if a is not None else np.nan for a in v], float)
        if m.sum() < 4: R[n] = dict(n=int(m.sum())); continue
        xs, ys = x[m], y[m]; sign = 1.0 if spearmanr(xs, ys)[0] >= 0 else -1.0
        R[n] = dict(n=int(m.sum()), spearman=float(spearmanr(xs, ys)[0]), loo_R2=loo_r2(xs, ys), auc_early=auc(sign * xs, cls[m]))
    res["by_step"][k] = dict(n=len(us), n_early=int(cls.sum()), predictors=R)
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["by_step"].items():
    print(k, v["n"], v["n_early"], {n: (round(r.get("spearman", np.nan), 2), None if r.get("loo_R2") is None else round(r["loo_R2"], 2), None if r.get("auc_early") is None else round(r["auc_early"], 2)) for n, r in v["predictors"].items()})
