"""Evaluate frozen predictions CP1-CP5, GP1-GP4 (PROSPECTIVE_PREDICTIONS.md Part I).  Missing runs reported as missing (ITT).
-> logs/confirmation_results.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
def ld(p): return json.load(open(p)) if os.path.exists(p) else None
COMPLETE = ["yP130", "yP131", "yP132", "yP134"]; INCOMPLETE = ["yP133", "yP135"]; ALL = ["yP130", "yP131", "yP132", "yP133", "yP134", "yP135"]
out = {}
dc = {p: ld(f"{R}/logs/decode_compose_{p}_u3_eval_conf9.json") for p in ALL}
dc2 = {p: ld(f"{R}/logs/decode_compose_{p}_u2_eval_conf9.json") for p in ALL}
c1 = {p: ld(f"{R}/checkpoints/native_train/{p}_c1_s4000_seed0_eval_conf9/final.json") for p in ALL}
mq = {p: ld(f"{R}/checkpoints/native/{p}_mlp_q_s8000_seed0_u3_eval_conf9.json") for p in ALL}
mn = {p: ld(f"{R}/logs/mech_native_{p}_c1_s4000_seed0_eval_conf9_eval_conf9.json") or ld(f"{R}/logs/mech_native_{p}_c1_s4000_seed0_eval_conf9.json") for p in ALL}
S = lambda d: d["summary"] if d and "summary" in d else (d["evals"][str(max(int(k) for k in d["evals"]))]["summary"] if d and "evals" in d else None)
# CP1
cp1c = [p for p in COMPLETE if dc[p] and dc[p]["composed"]["KL"] <= 0.005 and dc[p]["composed"]["absE_R1"] <= 0.05]
cp1i = [p for p in INCOMPLETE if dc[p] and dc[p]["composed"]["KL"] >= 0.03]
out["CP1"] = dict(per_parent={p: dict(decode=dc[p]["decode"], KL=dc[p]["composed"]["KL"], absE_R1=dc[p]["composed"]["absE_R1"], absE_R23=dc[p]["composed"]["absE_R23"]) if dc[p] else None for p in ALL},
                  u2={p: dict(KL=dc2[p]["composed"]["KL"], absE_R1=dc2[p]["composed"]["absE_R1"], decode=dc2[p]["decode"]) if dc2[p] else None for p in ALL},
                  complete_meeting=cp1c, incomplete_meeting=cp1i, confirmed=len(cp1c) == 4 and len(cp1i) == 2)
# CP2
kl = {p: S(c1[p])["KL"] for p in ALL if c1[p]}
out["CP2"] = dict(C1_KL=kl, confirmed=(len(kl) == 6 and max(kl[p] for p in COMPLETE) < min(kl[p] for p in INCOMPLETE)))
# CP3
c1s = {p: S(c1[p]) for p in ALL if c1[p]}
out["CP3"] = dict(C1={p: {k: c1s[p][k] for k in ("KL", "absE_R1", "E_R1", "absE_R23", "absE_F34", "acc")} for p in c1s},
                  n_fail_complete=sum(c1s[p]["E_R1"] <= -0.20 for p in COMPLETE if p in c1s), confirmed=sum(c1s[p]["E_R1"] <= -0.20 for p in COMPLETE if p in c1s) >= 3)
# CP4
mqs = {p: S(mq[p]) for p in ALL if mq[p]}
a = sum(abs(mqs[p]["E_R1"]) <= 0.15 for p in COMPLETE if p in mqs); b = sum(mqs[p]["absE_R1"] < c1s[p]["absE_R1"] for p in COMPLETE if p in mqs and p in c1s)
out["CP4"] = dict(mlp_q={p: {k: mqs[p][k] for k in ("KL", "absE_R1", "E_R1", "absE_R23", "absE_F34")} for p in mqs}, n_meanE_ok=a, n_paired_better=b, confirmed=a >= 3 and b == 4)
# CP5
cp5 = {}
for p in ALL:
    m = mn[p]
    if not m: continue
    rt = m["attention_mass_by_own_count"]; ratio = rt["5"]["mass_all_q"] / rt["4"]["mass_all_q"] if "5" in rt and "4" in rt else None
    fz = m["attention_freeze"]; nR2, nR3 = fz["R2"]["n"], fz["R3"]["n"]
    full23 = (fz["R2"]["full"] * nR2 + fz["R3"]["full"] * nR3) / (nR2 + nR3); frz23 = (fz["R2"]["attention_frozen"] * nR2 + fz["R3"]["attention_frozen"] * nR3) / (nR2 + nR3)
    red = 1 - abs(frz23) / abs(full23) if abs(full23) > 1e-9 else None
    cp5[p] = dict(mass_ratio_5_vs_4=ratio, a=ratio is not None and ratio < 0.6, R23_reduction=red, b=red is not None and red >= 0.5,
                  R1_full=fz["R1"]["full"], R1_frozen=fz["R1"]["attention_frozen"], c=fz["R1"]["attention_frozen"] > fz["R1"]["full"], routing=rt)
    cp5[p]["all"] = cp5[p]["a"] and cp5[p]["b"] and cp5[p]["c"]
out["CP5"] = dict(per_parent=cp5, n_all=sum(v["all"] for v in cp5.values()), confirmed=sum(v["all"] for v in cp5.values()) >= 5)
# GP
G = {}
for r in ("ds_sum", "attn", "mlp"):
    for s in (10, 11, 12):
        d = ld(f"{R}/checkpoints/gen_conf/{r}_id0_hold1_C1_G3_s12000_seed{s}.json")
        if d: G[(r, s)] = d["results"]
def gv(r, s, k, a): return G[(r, s)][k][a] if (r, s) in G and k in G[(r, s)] and G[(r, s)][k].get(a) is not None else None
gp1 = {f"{r}{s}": gv(r, s, "indist_bigN", "KL_pair01") for r in ("ds_sum", "attn") for s in (10, 11, 12)}
out["GP1"] = dict(KL_pair01=gp1, confirmed=all(v is not None and v <= 0.005 for v in gp1.values()))
gp2 = {f"mlp{s}": (gv("mlp", s, "indist_bigN", "KL_pair01"), gv("mlp", s, "indist_bigN", "KL_S2_not01")) for s in (10, 11, 12)}
out["GP2"] = dict(pair01_vs_not01=gp2, confirmed=all(a_ is not None and b_ is not None and a_ >= 5 * b_ for a_, b_ in gp2.values()))
gp3 = {f"{r}{s}": gv(r, s, "G4", "KL") for r in ("ds_sum", "attn") for s in (10, 11, 12)}
out["GP3"] = dict(KL_G4=gp3, also_G5_G6={f"{r}{s}": (gv(r, s, "G5", "KL"), gv(r, s, "G6", "KL")) for r in ("ds_sum", "attn") for s in (10, 11, 12)}, confirmed=all(v is not None and v >= 0.01 for v in gp3.values()))
gp4 = {f"{r}{s}": gv(r, s, "beta0.7", "KL") for r in ("ds_sum", "attn", "mlp") for s in (10, 11, 12)}
out["GP4"] = dict(KL_beta07=gp4, confirmed=all(v is not None and v >= 0.10 for v in gp4.values()))
out["GEN_other"] = {f"{r}{s}": dict(indist=G[(r, s)]["indist_bigN"], perm=G[(r, s)]["perm01"].get("mean_abs_logit_change")) for (r, s) in G}
json.dump(out, open(f"{R}/logs/confirmation_results.json", "w"), indent=1, default=float)
for k, v in out.items():
    if isinstance(v, dict) and "confirmed" in v: print(k, "confirmed =", v["confirmed"])
