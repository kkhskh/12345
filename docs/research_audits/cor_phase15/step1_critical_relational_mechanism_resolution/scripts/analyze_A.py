"""XIII-C Workstream A analysis: factorial effects (per parent) and trajectory event times, from logs/unit_metrics*.jsonl.
Factorial (2 x 2 per parent; init level = (L = W) seed, data level = D seed):  for a measure f,
  init main effect  = mean_D [f(a, D) - f(b, D)],  data main effect = mean_L [f(L, c) - f(L, d)],  interaction = f(a,c) - f(a,d) - f(b,c) + f(b,d)
  (the four cells are four fine-tuning runs of ONE pretrained parent: n = 1 per cell, no replicate error; reported descriptively, not as tests).
Trajectory events per run: t_mix = first checkpoint with mix_mass >= 0.5; t_ro = first with lik_RO1 >= 0.3; t_beh = first with slope_e >= 0.5;
  peak_ro = max lik_RO1 before t_mix; final values.
Usage: analyze_A.py OUT_TAG  (reads every logs/unit_metrics*.jsonl) -> logs/analyze_A<OUT_TAG>.json"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
rows = {}
for f in glob.glob(os.path.join(HERE, "..", "logs", "unit_metrics*.jsonl")):
    for l in open(f):
        r = json.loads(l); rows.setdefault(r["unit"].replace(":fx:", ":fx:"), {})[r["step"]] = r
def canon(u):
    p = u.split(":")
    return (p[0], int(p[2]), int(p[2]), int(p[2]), "") if p[1] == "s" else (p[0], int(p[2]), int(p[3]), int(p[4]), p[5] if len(p) > 5 else "")
units = {}
for u, st in rows.items():
    k = canon(u)
    if k in units: units[k].update(st)
    else: units[k] = dict(st)
MEAS = ("lik_RO1", "mix_mass", "rho", "route_Q", "route_R", "l2_attn", "l2_mlp", "KL", "slope_e")
def final(k): s = units[k]; return s[max(s)] if s else None
out = dict(factorial={}, events={}, finals={})
for k, s in units.items():
    f = final(k); out["finals"]["|".join(map(str, k))] = {m: f.get(m) for m in MEAS if m in f} | dict(step=max(s))
    steps = sorted(s)
    def first(m, thr):
        for t in steps:
            if s[t].get(m) is not None and s[t][m] >= thr: return t
        return None
    tm = first("mix_mass", 0.5)
    out["events"]["|".join(map(str, k))] = dict(t_mix=tm, t_ro=first("lik_RO1", 0.3), t_beh=first("slope_e", 0.5),
        peak_ro_before_mix=max([s[t]["lik_RO1"] for t in steps if tm is None or t <= tm] or [None]) if steps else None, n_ckpt=len(steps))
for parent in sorted(set(k[0] for k in units)):
    cells = {(k[1], k[3]): k for k in units if k[0] == parent and k[1] == k[2] and k[4] == "" and max(units[k]) == 4000}
    inits = sorted(set(a for a, _ in cells)); datas = sorted(set(b for _, b in cells))
    for a in inits:
        for b in inits:
            if a >= b: continue
            for c in datas:
                for d in datas:
                    if c >= d: continue
                    need = [(a, c), (a, d), (b, c), (b, d)]
                    if not all(x in cells for x in need): continue
                    F_ = {x: final(cells[x]) for x in need}; res = {}
                    for m in MEAS:
                        v = {x: F_[x][m] for x in need}
                        res[m] = dict(cells={f"L{x[0]}D{x[1]}": v[x] for x in need}, init_effect=0.5 * ((v[(a, c)] - v[(b, c)]) + (v[(a, d)] - v[(b, d)])),
                                      data_effect=0.5 * ((v[(a, c)] - v[(a, d)]) + (v[(b, c)] - v[(b, d)])), interaction=v[(a, c)] - v[(a, d)] - v[(b, c)] + v[(b, d)])
                    out["factorial"][f"{parent}|init{a}v{b}|data{c}v{d}"] = res
json.dump(out, open(os.path.join(HERE, "..", "logs", "analyze_A" + (sys.argv[1] if len(sys.argv) > 1 else "") + ".json"), "w"), indent=1)
for k, v in out["factorial"].items():
    print(k, {m: (round(x["init_effect"], 3), round(x["data_effect"], 3), round(x["interaction"], 3)) for m, x in v.items() if m in ("lik_RO1", "mix_mass", "rho", "route_Q")})
for k, v in sorted(out["events"].items()): print(k, v, {m: (round(out["finals"][k][m], 3) if out["finals"][k].get(m) is not None else None) for m in ("lik_RO1", "mix_mass", "rho")})
