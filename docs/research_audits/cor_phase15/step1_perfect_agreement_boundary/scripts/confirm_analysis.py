"""Frozen confirmatory analysis (Investigation VI).  Usage: confirm_analysis.py [PHASE=conf]
Reads logs/PHASE/model_x{T,P}{seed}_{step}.json for T seeds 70-77 and P seeds 80-87 at snapshots 250,500,1000,2000,4000 and
logs/cohort_status.json (success/failure per parent).  Missing models (failed training) enter the ITT analysis as described in
FROZEN_PROSPECTIVE_PROTOCOL.md §6 (worst-case imputation in the direction against the hypothesis).  Writes logs/confirmatory.json."""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PH = sys.argv[1] if len(sys.argv) > 1 else "conf"
L = os.path.join(HERE, "..", "logs", PH); TS, PS, SN = range(70, 78), range(80, 88), (250, 500, 1000, 2000, 4000)
R1 = "R1_with_colluder_candidate"; REG = (R1, "R2_with_third_group", "R3_with_no_group")
def load(arm, s, st):
    f = os.path.join(L, f"model_x{arm}{s}_{st}.json"); return json.load(open(f)) if os.path.exists(f) else None
def stat(j, key):
    if j is None: return np.nan
    F = j["families"]
    if key == "B45": return F["F45"]["B_logratio"]
    if key == "B45R1": return F["F45"]["by_regime"][R1]["B_logratio"]
    if key == "D_R1": return F["F45"]["by_regime"][R1]["B_logratio"] - F["F34"]["by_regime"][R1]["B_logratio"]
    if key == "D_pooled": return F["F45"]["B_logratio"] - F["F34"]["B_logratio"]
    if key == "B45R1_prob": return np.nan  # computed from npz below
    if key.startswith("B45_"): return F["F45"]["by_regime"].get(key[4:], {}).get("B_logratio", np.nan)
    if key == "B0": return F["F0"]["B_logratio"]
    if key == "KL": return j["quality"]["KL_bayes"]
    return j["quality"].get(key, np.nan)
def prob_R1(arm, s, st):
    f = os.path.join(L, f"model_x{arm}{s}_{st}.npz"); g = np.load(os.path.join(L, "generator.npz"), allow_pickle=True)
    if not os.path.exists(f): return np.nan
    m = np.load(f); sel = (g["regime_F45"] == R1); return float(np.nanmean(m["Ep_F45"][sel]))
def perm(t, p, alt="greater"):
    """exact two-sample permutation of model labels, statistic mean(T)-mean(P); one-sided T>P"""
    x = np.r_[t, p]; n = len(t); obs = t.mean() - p.mean(); cnt = tot = 0
    for c in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(c)] = True; d = x[m].mean() - x[~m].mean(); cnt += d >= obs - 1e-12; tot += 1
    return obs, cnt / tot
def boot(t, p, B=20000, seed=6):
    r = np.random.default_rng(seed); d = [r.choice(t, len(t)).mean() - r.choice(p, len(p)).mean() for _ in range(B)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.quantile(d, 0.05))]
def arr(arm, seeds, st, key, impute=None):
    v = np.array([prob_R1(arm, s, st) if key == "B45R1_prob" else stat(load(arm, s, st), key) for s in seeds])
    if impute is not None: v = np.where(np.isnan(v), impute, v)
    return v
status = json.load(open(os.path.join(HERE, "..", "logs", "cohort_status.json")))
out = {"status": status}
def contrast(key, st, itt=True):
    t, p = arr("T", TS, st, key), arr("P", PS, st, key)
    if itt:  # worst case against H: missing T -> min of observed T and P; missing P -> max
        allv = np.r_[t, p]; lo, hi = np.nanmin(allv), np.nanmax(allv); t = np.where(np.isnan(t), lo, t); p = np.where(np.isnan(p), hi, p)
    else:
        ok = lambda arm, s: status.get(f"x{arm}{s}", {}).get("success", False)
        t = np.array([v for s, v in zip(TS, t) if ok("T", s) and not np.isnan(v)]); p = np.array([v for s, v in zip(PS, p) if ok("P", s) and not np.isnan(v)])
    if len(t) < 2 or len(p) < 2: return None
    obs, pv = perm(t, p); ci = boot(t, p)
    return dict(diff=float(obs), p_one_sided=float(pv), ci95=ci[:2], ci90_lower=ci[2], T=t.tolist(), P=p.tolist(), nT=len(t), nP=len(p),
                sd_pooled=float(np.sqrt((t.var(ddof=1) + p.var(ddof=1)) / 2)))
# P1 (co-primary, Holm over two): B45 pooled and B45 regime R1, ITT, step 4000
P1 = {k: contrast(k, 4000) for k in ("B45", "B45R1")}
ps = sorted([(P1[k]["p_one_sided"], k) for k in P1]); holm = {}
for i, (pv, k) in enumerate(ps): holm[k] = min(1.0, max(pv * (2 - i), holm.get(ps[i - 1][1], 0) if i else 0))
for k in P1: P1[k]["p_holm"] = holm[k]; P1[k]["confirmed"] = bool(holm[k] < 0.05 and P1[k]["ci95"][0] > 0)
out["P1"] = P1; out["P1_successful_only"] = {k: contrast(k, 4000, itt=False) for k in ("B45", "B45R1")}
out["P1_probability_space_R1"] = contrast("B45R1_prob", 4000)
# P2 specificity
out["P2"] = {k: contrast(k, 4000) for k in ("D_R1", "D_pooled")}
for k in out["P2"]: out["P2"][k]["confirmed"] = bool(out["P2"][k]["p_one_sided"] < 0.05 and out["P2"][k]["ci95"][0] > 0)
# P3 stages
out["P3"] = {str(st): {k: contrast(k, st) for k in ("B45R1", "B45", "D_R1")} for st in SN}
pos = sum(out["P3"][str(st)]["B45R1"]["diff"] > 0 for st in (500, 1000, 2000, 4000))
out["P3_rule"] = dict(n_positive_of_4=int(pos), confirmed=bool(pos >= 3 and P1["B45R1"]["confirmed"]))
# P4 quality: KL-matched comparison and quality ranges
qual = {f"x{a}{s}@{st}": {q: stat(load(a, s, st), q) for q in ("KL", "acc", "ECE", "T_star", "margin_scale", "NLL")} | {"B45R1": stat(load(a, s, st), "B45R1")}
        for a, S in (("T", TS), ("P", PS)) for s in S for st in SN}
out["quality_table"] = qual
klP = np.array([qual[f"xP{s}@4000"]["KL"] for s in PS]); klP = klP[~np.isnan(klP)]; target = float(np.median(klP)) if len(klP) else np.nan
match = {}
for s in TS:
    c = [(abs(qual[f"xT{s}@{st}"]["KL"] - target), st) for st in SN if not np.isnan(qual[f"xT{s}@{st}"]["KL"])]
    if c: st = min(c)[1]; match[f"xT{s}"] = dict(step=st, KL=qual[f"xT{s}@{st}"]["KL"], B45R1=qual[f"xT{s}@{st}"]["B45R1"])
tm = np.array([m["B45R1"] for m in match.values()]); pm = np.array([qual[f"xP{s}@4000"]["B45R1"] for s in PS]); pm = pm[~np.isnan(pm)]
klT = np.array([m["KL"] for m in match.values()])
out["P4"] = dict(target_KL=target, T_matched=match, KL_range_T_matched=[float(klT.min()), float(klT.max())] if len(klT) else None,
                 KL_range_P=[float(klP.min()), float(klP.max())] if len(klP) else None,
                 overlap=bool(len(klT) and len(klP) and klT.min() <= klP.max() and klP.min() <= klT.max()))
if len(tm) > 1 and len(pm) > 1:
    o, pv = perm(tm, pm); out["P4"].update(diff=float(o), p_one_sided=float(pv), ci95=boot(tm, pm)[:2]); out["P4"]["confirmed"] = bool(pv < 0.05 and out["P4"]["ci95"][0] > 0 and out["P4"]["overlap"])
# ANCOVA across all snapshots: B45R1 ~ arm + KL, arm effect tested by permuting parent labels (snapshots move with their parent)
rows = [(a == "T", s, qual[f"x{a}{s}@{st}"]["KL"], qual[f"x{a}{s}@{st}"]["B45R1"]) for a, S in (("T", TS), ("P", PS)) for s in S for st in SN if not np.isnan(qual[f"x{a}{s}@{st}"]["B45R1"])]
if rows:
    A = np.array(rows, float); parents = sorted(set(zip(A[:, 0], A[:, 1])))
    def beta(lab):
        X = np.c_[np.ones(len(A)), lab, A[:, 2]]; return np.linalg.lstsq(X, A[:, 3], rcond=None)[0][1]
    b0 = beta(A[:, 0]); r = np.random.default_rng(7); cnt = 0; NP = 20000
    pid = {p: i for i, p in enumerate(parents)}; idx = np.array([pid[(a, s)] for a, s in zip(A[:, 0], A[:, 1])]); nT = int(sum(p[0] for p in parents))
    for _ in range(NP):
        lab = np.zeros(len(parents)); lab[r.choice(len(parents), nT, replace=False)] = 1; cnt += beta(lab[idx]) >= b0
    out["P4_ancova"] = dict(arm_coef=float(b0), p_one_sided=float((cnt + 1) / (NP + 1)))
# P5 contexts: sign of T-P in each regime of F45 (log-ratio), and in probability space for R1
out["P5"] = {reg: contrast("B45_" + reg, 4000) for reg in REG}
out["P5_rule"] = dict(R1_confirmed=P1["B45R1"]["confirmed"], prob_R1_positive=bool(out["P1_probability_space_R1"]["diff"] > 0))
# negative control F0
b0 = np.r_[arr("T", TS, 4000, "B0"), arr("P", PS, 4000, "B0")]; out["F0_control"] = dict(max_abs=float(np.nanmax(np.abs(b0))), passes=bool(np.nanmax(np.abs(b0)) < 0.02))
json.dump(out, open(os.path.join(HERE, "..", "logs", f"confirmatory{'' if PH == 'conf' else '_' + PH}.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("P3_rule", "F0_control")}, indent=1))
for k, v in P1.items(): print("P1", k, {x: v[x] for x in ("diff", "p_one_sided", "p_holm", "ci95", "confirmed")})
for k, v in out["P2"].items(): print("P2", k, {x: v[x] for x in ("diff", "p_one_sided", "ci95", "confirmed")})
print("P4", {k: out["P4"].get(k) for k in ("target_KL", "KL_range_T_matched", "KL_range_P", "overlap", "diff", "p_one_sided", "confirmed")}, out.get("P4_ancova"))
