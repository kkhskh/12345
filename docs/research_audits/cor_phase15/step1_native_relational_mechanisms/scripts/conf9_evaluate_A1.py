"""Evaluate Addendum A1 predictions NP1-NP4 (native fine-tuning) -> logs/confirmation_results_A1.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
P = ["yP130", "yP131", "yP132", "yP134"]; ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
fc = {p: ld(f"{R}/checkpoints/native_train/{p}_full_c1_s4000_seed0_eval_conf9/final.json") for p in P}
cv = {p: ld(f"{R}/checkpoints/native_train/{p}_full_c1_s4000_seed0_eval_conf9/curve.json") for p in P}
c1 = {p: ld(f"{R}/checkpoints/native_train/{p}_c1_s4000_seed0_eval_conf9/final.json") for p in P}
mf = {p: ld(f"{R}/logs/mech_native_{p}_full_c1_s4000_seed0_eval_conf9_eval_conf9.json") for p in P}
mc = {p: ld(f"{R}/logs/mech_native_{p}_c1_s4000_seed0_eval_conf9_eval_conf9.json") for p in P}
out = {}
s = {p: fc[p]["summary"] for p in P}; s1 = {p: c1[p]["summary"] for p in P}
np1 = dict(paired=sum(s[p]["absE_R1"] < s1[p]["absE_R1"] for p in P), meanE=sum(abs(s[p]["E_R1"]) <= 0.20 for p in P), kl=sum(s[p]["KL"] <= 0.005 for p in P))
out["NP1"] = dict(full_c1={p: {k: s[p][k] for k in ("KL", "acc", "absE_R1", "E_R1", "absE_R23", "absE_F34")} for p in P}, c1={p: {k: s1[p][k] for k in ("KL", "absE_R1", "E_R1", "absE_R23", "absE_F34")} for p in P},
                  counts=np1, confirmed=np1["paired"] == 4 and np1["meanE"] >= 3 and np1["kl"] >= 3)
np2 = {}
for p in P:
    pt = mf[p]["patching"]; o = pt["others_src_q"]["R1"]["dmM"]; c = pt["all_but_others_src_q"]["R1"]["dmM"]; full = pt["all"]["R1"]["dmM"]
    a2, a3 = pt["all_but_others_src_q"]["R2"], pt["all_but_others_src_q"]["R3"]
    share23 = (a2["dmM"] * a2["n"] + a3["dmM"] * a3["n"]) / (pt["all"]["R2"]["dmM"] * a2["n"] + pt["all"]["R3"]["dmM"] * a3["n"])
    np2[p] = dict(others_q_R1=o, complement_R1=c, full_R1=full, complement_R23_share=share23, ok=(o > 0) and (c < 0) and (c < full) and share23 >= 0.5)
out["NP2"] = dict(per_parent=np2, n=sum(v["ok"] for v in np2.values()), confirmed=sum(v["ok"] for v in np2.values()) >= 3)
np3 = {p: mc[p]["patching"]["others_src"]["R1"]["share_of_full"] for p in P}
out["NP3"] = dict(c1_others_src_R1_share=np3, n=sum(abs(v) < 0.25 for v in np3.values()), confirmed=sum(abs(v) < 0.25 for v in np3.values()) >= 3)
np4 = {}
for p in P:
    cr = cv[p]["curve"]; mn = min(cr, key=lambda c: c["E_R1"]); fin = cr[-1]["E_R1"]
    np4[p] = dict(min_E_R1=mn["E_R1"], at_step=mn["step"], final=fin, ok=mn["E_R1"] <= -0.30 and mn["step"] <= 1000 and fin >= mn["E_R1"] + 0.20,
                  traj=[(c["step"], round(c["E_R1"], 3), round(c["KL"], 4)) for c in cr])
out["NP4"] = dict(per_parent=np4, n=sum(v["ok"] for v in np4.values()), confirmed=sum(v["ok"] for v in np4.values()) >= 3)
json.dump(out, open(f"{R}/logs/confirmation_results_A1.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, "confirmed =", v["confirmed"])
