"""Scores the FROZEN predictions P1-P6 (FROZEN_PREDICTIONS_AND_CONFIRMATION.md, Part I) on the held-out models wP140-wP143.
Reads logs/*_conf.json, logs/cross_conf/, logs/decoy_conf/ and Investigation X's logs/traj_alg_conf.json.  Writes logs/score_conf.json.
Thresholds below are the frozen ones; do not edit after the freeze commit."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs"); X10 = os.path.join(HERE, "..", "..", "step1_causal_equivalence", "logs")
P = ["wP140", "wP141", "wP142", "wP143"]; S = "_eval_conf10"; full = {p: f"{p}:full_c1:0:{S}" for p in P}; c1 = {p: f"{p}:c1:0:{S}" for p in P}
J = lambda f: json.load(open(os.path.join(L, f)))
out = {}; NEED = 3                                                    # pass in >= 3 of 4 parents for every arm
def verdict(arms): return "confirmed" if all(sum(v.values()) >= NEED for v in arms.values()) else "falsified"
# P1
fa = J("factorial_conf.json"); a = {}
for p in P:
    T, C_ = fa[full[p]]["TRUE"], fa[full[p]]["CTRL"]; L0 = sum(T["share_E"][f"h0.{h}"] for h in range(4)) + T["share_E"]["mlp0"]
    a.setdefault("full_ratio>=2", {})[p] = T["E_interaction_norm"] / C_["E_interaction_norm"] >= 2.0
    a.setdefault("full_mid0_relI<=0.05_and_|L0share|<=0.10", {})[p] = (T["stages"]["mid0"]["rel_I"] <= 0.05) and (abs(L0) <= 0.10)
    a.setdefault("c1_ratio<=1.75", {})[p] = fa[c1[p]]["TRUE"]["E_interaction_norm"] / fa[c1[p]]["CTRL"]["E_interaction_norm"] <= 1.75
out["P1"] = dict(arms=a, verdict=verdict(a))
# P2
ab = J("factorial_ablate_conf.json"); a = {}
for p in P:
    r = ab[full[p]]
    a.setdefault("full_L1_remaining<=0.05", {})[p] = r["L1"]["E_interaction_remaining"] <= 0.05
    a.setdefault("full_L1_KLe0<KLbayes_and_pref_e0>=0.5", {})[p] = (r["L1"]["KL_e0"] < r["L1"]["KL_bayes"]) and (r["L1"]["pref_e0"] >= 0.5)
    a.setdefault("full_L0_remaining>=0.95_and_dKL<=0.01", {})[p] = (r["L0"]["E_interaction_remaining"] >= 0.95) and (abs(r["L0"]["KL_bayes"] - r["none"]["KL_bayes"]) <= 0.01)
    a.setdefault("c1_L1_dKLbayes<=0.02", {})[p] = abs(ab[c1[p]]["L1"]["KL_bayes"] - ab[c1[p]]["none"]["KL_bayes"]) <= 0.02
out["P2"] = dict(arms=a, verdict=verdict(a))
# P3
def cross(A, B):
    f = glob.glob(os.path.join(L, "cross_conf", f"{A}*__{B}*.json")); return json.load(open(f[0])) if f else None
def gain(d, k): return (d["mean"]["KL"] - d[k]["KL"]) / (d["mean"]["KL"] - d["within"]["KL"])
a = {}; pairs = [("wP140", "wP141"), ("wP141", "wP140"), ("wP142", "wP143"), ("wP143", "wP142")]
for A_, B_ in pairs:
    d = cross(f"{A_}_full_c1", f"{B_}_full_c1"); k = f"{A_}->{B_}"
    a.setdefault("mapped_gain>=0.85", {})[k] = gain(d, "mapped") >= 0.85
    a.setdefault("shuffled_gain<=0.10_and_randsub_IIA<=0.60", {})[k] = gain(d, "shuffled") <= 0.10 and d["rand_sub"]["IIA"] <= 0.60
    a.setdefault("mean_patch_IIAdisc>=base+0.40", {})[k] = d["mean"]["IIA_disc"] >= d["base"]["IIA_disc"] + 0.40
c1map = {p: cross(f"{p}_c1", f"{p}_full_c1") for p in ("wP140", "wP141")}
out["P3_c1_to_full_gain(<=0.85 predicted; 2 parents, both required)"] = {p: gain(d, "mapped") for p, d in c1map.items()}
a3 = dict(a); a3_c1 = all(gain(d, "mapped") <= 0.85 for d in c1map.values())
out["P3"] = dict(arms=a, c1_to_full_ok=a3_c1, verdict="confirmed" if verdict(a) == "confirmed" and a3_c1 else "falsified")
# P4
def dec(p, spec, rule):
    f = os.path.join(L, "decoy_conf", f"{p}_{spec}_s4000_seed0{S}_{rule}.json"); return json.load(open(f))["trained"]
a = {}
for p in P:
    E, A1, PE = dec(p, "full_c1", "E"), dec(p, "full_c1", "A1"), dec(p, "full_c1", "PERM")
    a.setdefault("full_A1decoy_IIA>=0.85(false positive)", {})[p] = A1["IIA"] >= 0.85
    a.setdefault("full_E_presKL<=0.03_and_decoys>=3x", {})[p] = E["preserve_KL"] <= 0.03 and min(A1["preserve_KL"], PE["preserve_KL"]) >= 3 * E["preserve_KL"]
    a.setdefault("c1_E_presKL>=0.05", {})[p] = dec(p, "c1", "E")["preserve_KL"] >= 0.05
out["P4"] = dict(arms=a, verdict=verdict(a))
# P5
ge = J("generalize_conf.json"); a = {}
for p in P:
    r = ge[full[p]]
    a.setdefault("C2_KL_C1B<=0.05_and_KL_TRUE>=1.0", {})[p] = r["C2"]["S>0"]["C1B"]["KL"] <= 0.05 and r["C2"]["S>0"]["TRUE"]["KL"] >= 1.0
    a.setdefault("C2_acc<=0.40_and_e_readout>=0.95", {})[p] = r["C2"]["acc"] <= 0.40 and r["C2"]["e_readout_acc"] >= 0.95
    a.setdefault("b07_S>0_KL_C1B<=0.2*KL_TRUE", {})[p] = r["b07"]["S>0"]["C1B"]["KL"] <= 0.2 * r["b07"]["S>0"]["TRUE"]["KL"]
    a.setdefault("b07_S=0_min_beta_KL>=0.10", {})[p] = min(r["b07"]["beta_eff"]["S=0"]["kl"].values()) >= 0.10
out["P5"] = dict(arms=a, verdict=verdict(a))
# P6
dy = J("dyn_factorial_conf.json"); tr = json.load(open(os.path.join(X10, "traj_alg_conf.json"))); a = {}
for p in P:
    sw = next(r["step"] for r in tr[full[p]] if r["D2"]["closest"] == "B"); em = dy[full[p]]["interaction_emergence_step"]
    fin = dy[full[p]]["traj"][str(max(int(s) for s in dy[full[p]]["traj"]))]["ratio_resid2"]
    out.setdefault("P6_steps", {})[p] = dict(first_D2_B=sw, interaction_emergence=em, final_ratio=fin)
    a.setdefault("emergence<first_B_step", {})[p] = em is not None and em < sw
    a.setdefault("final_ratio>=2", {})[p] = fin >= 2.0
out["P6"] = dict(arms=a, verdict=verdict(a))
json.dump(out, open(os.path.join(L, "score_conf.json"), "w"), indent=1)
for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
    print(k, out[k]["verdict"], {arm: f"{sum(v.values())}/{len(v)}" for arm, v in out[k]["arms"].items()})
