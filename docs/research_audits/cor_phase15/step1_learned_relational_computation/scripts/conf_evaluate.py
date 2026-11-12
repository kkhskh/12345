"""Evaluate frozen predictions P1-P7 on confirmation outputs (implements FROZEN_PROTOCOL_AND_CONFIRMATION.md Part I and addendum A1).
Conventions: |E_R1| = absolute value of the model's mean E on F45 R1; 'absE' = decision-level mean |E|; |E| R2uR3 = n-weighted absE over
R2, R3; preservation metrics use absE (F34 R1, F54o all, F0/FLOC/FID all) and KL.  Missing runs are reported as missing (ITT).
Output: logs/confirmation_results.json + printed report."""
import json, os, glob
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
def last(d): return d["evals"][str(max(int(k) for k in d["evals"]))]
def at(d, s): return d["evals"].get(str(s))
def stats(e):
    F = e["families"]; r2, r3 = F["F45"]["R2"], F["F45"]["R3"]
    return dict(KL=e["quality"]["KL"], E_R1=F["F45"]["R1"]["E"], absE_R1=F["F45"]["R1"]["absE"], dm_R1=F["F45"]["R1"]["dmM"],
                E_R2=r2["E"], E_R3=r3["E"], absE_R23=(r2["absE"] * r2["n"] + r3["absE"] * r3["n"]) / (r2["n"] + r3["n"]),
                F34=F["F34"]["R1"]["absE"], F54o=F["F54o"]["all"]["absE"], F0=F["F0"]["all"]["absE"], FLOC=F["FLOC"]["all"]["absE"], FID=F["FID"]["all"]["absE"])
def sign_p(k, n): return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
out = {}; rep = []
# ---- C-A composition
C = {}
for r in ("attn", "relnet", "mlp"):
    for s in (10, 11, 12):
        d = load(f"{R}/checkpoints/compose_conf/{r}_R_label_s16000_seed{s}.json"); C[(r, s)] = d
dO = load(f"{R}/checkpoints/compose_conf/attn_O_label_s16000_seed10.json")
rel = [(k, stats(last(d))) for k, d in C.items() if k[0] in ("attn", "relnet") and d]
ok1 = [k for k, x in rel if x["KL"] <= 0.002 and x["dm_R1"] > 0 and abs(x["E_R1"]) <= 0.25 and x["absE_R23"] <= 0.25]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
mkl = mean([x["KL"] for _, x in rel]); mdm = mean([x["dm_R1"] for _, x in rel]); mE = mean([abs(x["E_R1"]) for _, x in rel]); m23 = mean([x["absE_R23"] for _, x in rel])
P1 = len(ok1) >= 5 and mkl <= 0.002 and mdm > 0 and mE <= 0.25 and m23 <= 0.25
out["P1"] = dict(n_models=len(rel), n_meeting=len(ok1), mean_KL=mkl, mean_dm_R1=mdm, mean_absmeanE_R1=mE, mean_absE_R23=m23, per_model={f"{a}{b}": x for (a, b), x in rel}, confirmed=P1)
mlp = [(s, stats(last(C[("mlp", s)]))) for s in (10, 11, 12) if C[("mlp", s)]]
p2a = sum(x["E_R1"] <= -0.20 for _, x in mlp) >= 2
ph = [(k, at(d, 1000)["families"]["F45"]["R1"]["E"]) for k, d in C.items() if d]
p2b = sum(e <= -0.50 for _, e in ph) >= 8
eO = at(dO, 1000)["families"]["F45"]["R1"]["E"] if dO else None
p2c = eO is not None and eO >= -0.20
out["P2"] = dict(mlp16k={s: x for s, x in mlp}, a=p2a, phase_1k={f"{a}{b}": e for (a, b), e in ph}, b=p2b, attnO_1k=eO, c=p2c, confirmed=p2a and p2b and p2c)
# ---- C-M mechanism
M = load(f"{R}/logs/mech_reader_conf.json"); p3 = []
if M:
    for tag, v in M.items():
        b, no, so = v["base"], v["noperf"], v["selfonly"]
        a = no["F45_R1"]["E"] <= -1.0 and no["F54o"]["dmM"] >= 1.0 and abs(no["F34_R1"]["dmM"] - b["F34_R1"]["dmM"]) < 0.30
        bb = so["F45_R2"]["dmM"] >= -0.5 and so["F45_R3"]["dmM"] >= -0.5
        ic = v["interchange_R1"]; c = ic["m_subspace"]["share"] >= ic["random_subspace"]["share"] + 0.5
        p3.append(dict(tag=tag, a=a, b=bb, c=c, all=a and bb and c, noperf_E_R1=no["F45_R1"]["E"], noperf_F54o=no["F54o"]["dmM"],
                       dF34=no["F34_R1"]["dmM"] - b["F34_R1"]["dmM"], selfonly_R2=so["F45_R2"]["dmM"], selfonly_R3=so["F45_R3"]["dmM"],
                       share_m=ic["m_subspace"]["share"], share_rand=ic["random_subspace"]["share"], probes=v.get("probes")))
out["P3"] = dict(models=p3, confirmed=sum(x["all"] for x in p3) >= 2 if p3 else None)
# ---- C-Z repairs
def rep_(parent, spec):
    d = load(f"{R}/checkpoints/conf/{parent}_{spec}.json"); return stats(last(d)) if d else None
P = [f"zP{p}" for p in range(100, 108)]
orig = {p: rep_(p, "orig") for p in P}; rad = {p: rep_(p, "radapt") for p in P}; gad = {p: rep_(p, "gadapt") for p in P[:4]}; inj = {p: rep_(p, "injm") for p in P[:2]}
o = [x for x in orig.values() if x]
p4 = sum(x["E_R1"] <= -0.20 for x in o) >= 6 and mean([x["E_R1"] for x in o]) <= -0.20
out["P4"] = dict(orig=orig, n_le=sum(x["E_R1"] <= -0.20 for x in o), mean_E_R1=mean([x["E_R1"] for x in o]), confirmed=p4)
pairs = [p for p in P if orig[p] and rad[p]]
wins = sum(abs(rad[p]["E_R1"]) < abs(orig[p]["E_R1"]) for p in pairs)
mr = mean([abs(rad[p]["E_R1"]) for p in pairs]); ratio = mean([rad[p]["absE_R1"] for p in pairs]) / mean([orig[p]["absE_R1"] for p in pairs])
p5a = wins >= 7 and mr <= 0.25 and ratio <= 0.6
mO = {k: mean([orig[p][k] for p in pairs]) for k in ("absE_R23", "F34", "KL", "F0", "FLOC", "FID", "F54o")}
mR = {k: mean([rad[p][k] for p in pairs]) for k in mO}
p5b = mR["absE_R23"] <= mO["absE_R23"] + 0.05 and mR["F34"] <= mO["F34"] + 0.05 and mR["KL"] <= mO["KL"] and all(mR[k] <= mO[k] + 0.05 for k in ("F0", "FLOC", "FID"))
g = [x for x in gad.values() if x]; mrg = mean([abs(rad[p]["E_R1"]) for p in P[:4] if rad[p]])
p5c = mean([abs(x["E_R1"]) for x in g]) >= mrg + 0.20
out["P5"] = dict(radapt=rad, gadapt=gad, injm=inj, wins=wins, n_pairs=len(pairs), sign_p=sign_p(wins, len(pairs)) if pairs else None,
                 mean_absmeanE_R1_radapt=mr, absE_ratio=ratio, a=p5a, preservation_orig=mO, preservation_radapt=mR, b=p5b,
                 gadapt_mean_absE=mean([abs(x["E_R1"]) for x in g]), radapt_mean_absE_P100_103=mrg, c=p5c, confirmed=p5a and p5b and p5c)
T = [f"zT1{p}" for p in range(110, 114)]
to = {p: rep_(p, "orig") for p in T}; tr = {p: rep_(p, "radapt") for p in T}; tp = [p for p in T if to[p] and tr[p]]
p6a = sum(abs(tr[p]["E_R1"]) <= 0.25 for p in tp) >= 3
tO = {k: mean([to[p][k] for p in tp]) for k in ("absE_R23", "F34", "KL", "F0", "FLOC", "FID")}; tR = {k: mean([tr[p][k] for p in tp]) for k in tO}
p6b = tR["absE_R23"] <= tO["absE_R23"] + 0.05 and tR["F34"] <= tO["F34"] + 0.05 and tR["KL"] <= tO["KL"] and all(tR[k] <= tO[k] + 0.05 for k in ("F0", "FLOC", "FID"))
out["P6"] = dict(orig=to, radapt=tr, a=p6a, b=p6b, pres_orig=tO, pres_radapt=tR, confirmed=p6a and p6b)
# ---- P7
PP = load(f"{R}/logs/predictor_probe_conf.json")
if PP:
    fm = [v for k, v in PP.items() if k.startswith("mlp") and v["E_R1"] <= -0.20]; sa = [v for k, v in PP.items() if k.startswith("attn") and abs(v["E_R1"]) <= 0.25 and v["dm_R1"] > 0]
    p7 = None if not fm else (all(v["probe_acc"] >= 0.90 for v in fm) and mean([v["probe_acc"] for v in fm]) >= mean([v["probe_acc"] for v in sa]))
    out["P7"] = dict(probe=PP, n_failed_mlp=len(fm), confirmed=p7)
json.dump(out, open(f"{R}/logs/confirmation_results.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, "confirmed =", v.get("confirmed"))
