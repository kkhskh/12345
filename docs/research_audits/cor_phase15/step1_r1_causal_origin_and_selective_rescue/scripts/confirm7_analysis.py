"""FROZEN confirmatory analysis of Investigation VII (FROZEN_CONFIRMATION_PROTOCOL.md).  -> logs/confirmatory7.json"""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "checkpoints", "conf"); S = list(range(90, 98))
SPECS = ("C1", "C1_injm", "C1_injn", "C1_injrand", "W_bal")
def ld(s, spec):
    f = os.path.join(C, f"yP{s}_{spec}_r0.json")
    if not os.path.exists(f): return None
    j = json.load(open(f)); r = j["evals"].get("4000") or j["evals"].get(4000); return r
def stats(r):
    F, Q = r["families"], r["quality"]; e = lambda f, rg: F[f][rg]["E"]
    rr_n = F["F45"]["R2"]["n"] + F["F45"]["R3"]["n"]
    rr_abs = (F["F45"]["R2"]["absE"] * F["F45"]["R2"]["n"] + F["F45"]["R3"]["absE"] * F["F45"]["R3"]["n"]) / rr_n
    return dict(E_R1=e("F45", "R1"), dm_R1=F["F45"]["R1"]["dmM"], E_R2=e("F45", "R2"), E_R3=e("F45", "R3"), absE_R23=rr_abs, E_F34R1=F["F34"]["R1"]["E"],
                F0abs=F["F0"]["all"]["abs_dmM"], KL=Q["KL"], acc=Q["acc"], E_S45R1=F["S45"]["R1"]["E"])
T = {s: {sp: (stats(ld(s, sp)) if ld(s, sp) else None) for sp in SPECS} for s in S}
def signflip(d):
    d = np.asarray(d, float); obs = d.mean(); c = 0; tot = 0
    for sg in itertools.product((1, -1), repeat=len(d)): c += (np.array(sg) * np.abs(d)).mean() >= obs - 1e-12; tot += 1
    return float(obs), c / tot
def boot(d, B=20000):
    r = np.random.default_rng(7); d = np.asarray(d); m = [r.choice(d, len(d)).mean() for _ in range(B)]; return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]
def delta(spec, key="E_R1", base="C1", worst=-10.0):
    out = []
    for s in S:
        a, b = T[s][spec], T[s][base]
        out.append(worst if (a is None or b is None) else a[key] - b[key])      # ITT: a missing run counts as a large failure
    return out
res = dict(table=T)
d = delta("C1_injm"); o, p = signflip(d); res["P1"] = dict(deltas=d, mean=o, p_one_sided=p, ci95=boot(d), confirmed=bool(p < 0.05 and boot(d)[0] > 0))
d2 = [a - b for a, b in zip(delta("C1_injm"), delta("C1_injrand", worst=10.0))]; o, p = signflip(d2); res["P2"] = dict(deltas=d2, mean=o, p_one_sided=p, ci95=boot(d2), confirmed=bool(p < 0.05 and boot(d2)[0] > 0))
succ = []
for s in S:
    a, b = T[s]["C1_injm"], T[s]["C1"]
    if a is None or b is None: succ.append(False); continue
    ok = (a["E_R1"] - b["E_R1"] >= 0.2 and a["absE_R23"] - b["absE_R23"] <= 0.15 and a["KL"] - b["KL"] <= 0.01 and b["acc"] - a["acc"] <= 0.005
          and abs(a["E_F34R1"]) - abs(b["E_F34R1"]) <= 0.10 and a["F0abs"] - b["F0abs"] <= 0.05)
    succ.append(bool(ok))
res["P3"] = dict(success=succ, n_success=int(sum(succ)), confirmed=bool(sum(succ) >= 6))
d4 = [-x for x in delta("C1_injn", worst=10.0)]; o, p = signflip(d4); res["P4"] = dict(deltas_neg=d4, mean_worsening=o, p_one_sided=p, confirmed=bool(p < 0.05))
neg = [T[s]["C1"] is not None and T[s]["C1"]["dm_R1"] < 0 for s in S]; res["P5"] = dict(neg=neg, n=int(sum(neg)), confirmed=bool(sum(neg) >= 7))
pr = os.path.join(HERE, "..", "logs", "conf_probes.json")
if os.path.exists(pr):
    P = json.load(open(pr)); acc = [P.get(f"yP{s}", {}).get("qsources", {}).get("H2", {}).get("linear", 1.0) for s in S]
    res["P6"] = dict(H2_linear_qsources=acc, n_below=int(sum(a < 0.96 for a in acc)), confirmed=bool(sum(a < 0.96 for a in acc) >= 6))
mc = os.path.join(HERE, "..", "logs", "conf_mech.json")
if os.path.exists(mc):
    M = json.load(open(mc)); z = [(M.get(f"yP{s}") or {}).get("readout_token_only", 1.0) for s in S]
    res["P7"] = dict(readout_token_only=z, n_neg=int(sum(x < 0 for x in z)), confirmed=bool(sum(x < 0 for x in z) >= 6))
res["W_bal_descriptive"] = dict(deltas=delta("W_bal"), mean=float(np.mean(delta("W_bal"))))
res["injm_minus_Wbal_descriptive"] = [a - b for a, b in zip(delta("C1_injm"), delta("W_bal", worst=10.0))]
json.dump(res, open(os.path.join(HERE, "..", "logs", "confirmatory7.json"), "w"), indent=1, default=float)
for k in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
    if k in res: print(k, {x: y for x, y in res[k].items() if x not in ("deltas",)})
print("W_bal", res["W_bal_descriptive"])
