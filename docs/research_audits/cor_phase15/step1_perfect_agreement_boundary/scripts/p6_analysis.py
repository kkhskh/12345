"""Frozen P6 analysis (see P6_MECHANISM_PROTOCOL.md).  Reads logs/mech_conf/x{T,P}{seed}_4000.json -> logs/p6.json."""
import json, os, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs", "mech_conf")
TS, PS = range(70, 78), range(80, 88)
def ld(a, s):
    f = os.path.join(L, f"x{a}{s}_4000.json"); return json.load(open(f)) if os.path.exists(f) else None
def perm(t, p):
    x = np.r_[t, p]; n = len(t); obs = t.mean() - p.mean(); c = tot = 0
    for cc in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(cc)] = True; c += x[m].mean() - x[~m].mean() >= obs - 1e-12; tot += 1
    return float(obs), c / tot
def boot(t, p, B=20000):
    r = np.random.default_rng(6); d = [r.choice(t, len(t)).mean() - r.choice(p, len(p)).mean() for _ in range(B)]; return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
J = {a: [ld(a, s) for s in S] for a, S in (("T", TS), ("P", PS))}
get = lambda a, f: np.array([f(j) if j is not None else np.nan for j in J[a]])
out = {}
site = {a: get(a, lambda j: j["F45"]["site"]["share"]) for a in "TP"}
out["P6a_site_share_median"] = {a: float(np.nanmedian(site[a])) for a in "TP"}; out["P6a"] = all(v >= 0.8 for v in out["P6a_site_share_median"].values())
for key, nm in (("site", "P6b"), ("sub", "P6c")):
    t, p = get("T", lambda j: j["F45"][key]["R1"]["B_patch"]), get("P", lambda j: j["F45"][key]["R1"]["B_patch"])
    t, p = t[~np.isnan(t)], p[~np.isnan(p)]; o, pv = perm(t, p); ci = boot(t, p)
    out[nm] = dict(diff=o, p_one_sided=pv, ci95=ci, T=t.tolist(), P=p.tolist(), confirmed=bool(pv < 0.05 and ci[0] > 0))
    full_t, full_p = get("T", lambda j: j["F45"][key]["R1"]["B_full"]), get("P", lambda j: j["F45"][key]["R1"]["B_full"])
    out[nm]["fraction_of_full_difference"] = float(o / (np.nanmean(full_t) - np.nanmean(full_p)))
f0 = np.r_[get("T", lambda j: j["F0"]["site"]["mean_abs"]), get("P", lambda j: j["F0"]["site"]["mean_abs"])]
f45 = np.r_[get("T", lambda j: j["F45"]["mean_abs_full"]), get("P", lambda j: j["F45"]["mean_abs_full"])]
out["P6d"] = dict(ratio_mean=float(np.nanmean(f0) / np.nanmean(f45)), confirmed=bool(np.nanmean(f0) < 0.2 * np.nanmean(f45)))
rem = np.r_[get("T", lambda j: j["F45"]["ablate_sub_after"]["remaining_share"]), get("P", lambda j: j["F45"]["ablate_sub_after"]["remaining_share"])]
out["P6e"] = dict(median_remaining=float(np.nanmedian(rem)), by_arm={a: float(np.nanmedian(get(a, lambda j: j["F45"]["ablate_sub_after"]["remaining_share"]))) for a in "TP"}, confirmed=bool(np.nanmedian(rem) < 0.5))
dec = {a: get(a, lambda j: j["decode_acc_counts"][1]) for a in "TP"}; o, pv = perm(dec["T"][~np.isnan(dec["T"])], dec["P"][~np.isnan(dec["P"])])
out["P6g_exploratory_decode_after"] = dict(T=dec["T"].tolist(), P=dec["P"].tolist(), diff=o, p_one_sided=pv)
for nm in ("nonsite", "subc"):
    out[f"share_{nm}"] = {a: get(a, lambda j: j["F45"][nm]["share"]).tolist() for a in "TP"}
json.dump(out, open(os.path.join(HERE, "..", "logs", "p6.json"), "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x not in ("T", "P")}) for k, v in out.items() if not k.startswith("share")}, indent=1))
