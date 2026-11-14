"""Evaluate frozen predictions F1-F7 (Part I, Addenda A1, A2) on the confirmation parents.  Completeness (pre.json aux_acc >= 0.99) defines
the denominator n (Part I).  Missing outputs are reported as missing.  -> logs/confirmation_results.json"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
P = ["wP140", "wP141", "wP142", "wP143"]
comp = {p: (ld(f"{R}/checkpoints/z{p[1:]}/pre.json") or {}).get("eval", {}).get("aux_acc") for p in P}
C = [p for p in P if comp[p] is not None and comp[p] >= 0.99]; n = len(C); need = n - 1
out = dict(completeness=comp, complete=C, n=n, threshold=need)
fin = lambda p, s: (ld(f"{R}/checkpoints/native_train/{p}_{s}_s4000_seed0_eval_conf10/final.json") or {}).get("summary")
out["behaviour"] = {p: {s: fin(p, s) for s in ("full_c1", "c1")} for p in P}
# F1
f1 = {}
for p in P:
    d = ld(f"{R}/logs/algdisc_eval_conf10_{p}.json")
    if not d: f1[p] = None; continue
    mf, mc = d["models"][f"{p}:full_c1:0:_eval_conf10"], d["models"][f"{p}:c1:0:_eval_conf10"]
    a = all(mf[f]["closest"] == "B" for f in ("D1", "D2", "F45_R2", "F45_R3")) and mf["D1"]["B"] <= 0.40 and mf["D2"]["B"] <= 0.40
    b = mc["D2"]["closest"] != "B"
    f1[p] = dict(full={f: {k: mf[f][k] for k in ("closest", "B", "A1", "A2", "A3", "A0")} for f in mf}, c1={f: {k: mc[f][k] for k in ("closest", "B", "A1", "A2", "A3", "A0")} for f in mc}, full_ok=a, c1_ok=b)
out["F1"] = dict(per_parent=f1, n_full=sum(bool(f1[p] and f1[p]["full_ok"]) for p in C), n_c1=sum(bool(f1[p] and f1[p]["c1_ok"]) for p in C))
out["F1"]["confirmed"] = out["F1"]["n_full"] >= need and out["F1"]["n_c1"] >= need
das = lambda p, s, v, tok, k: ld(f"{R}/logs/das/{p}_{s}_s4000_seed0_eval_conf10_{v}_L2_{tok}_k{k}_ev62500000.json")
# F2, F3, F5
f2, f3, f5 = {}, {}, {}
for p in P:
    E = das(p, "full_c1", "E", "q7", 6); S = das(p, "full_c1", "S", "q7", 2); ro = das(p, "full_c1", "E", "ro", 6); s6 = das(p, "full_c1", "E", "src6", 6)
    if E: f2[p] = dict(IIA=E["trained"]["IIA_argmax"], KLcf=E["trained"]["KLcf"], rand=E["rand"]["IIA_argmax"], KLbase=E["trained"]["KLbase"],
                       ok=E["trained"]["IIA_argmax"] >= 0.85 and E["trained"]["KLcf"] <= 0.15 and E["rand"]["IIA_argmax"] <= 0.60)
    if E and S: f3[p] = dict(S_IIA=S["trained"]["IIA_argmax"], E_IIA=E["trained"]["IIA_argmax"], S_rand=S["rand"]["IIA_argmax"], ok=S["trained"]["IIA_argmax"] <= E["trained"]["IIA_argmax"] - 0.20)
    if E and ro and s6: f5[p] = dict(ro=ro["trained"]["IIA_argmax"], src6=s6["trained"]["IIA_argmax"], q7=E["trained"]["IIA_argmax"],
                                     ok=ro["trained"]["IIA_argmax"] <= E["trained"]["IIA_argmax"] - 0.05 and s6["trained"]["IIA_argmax"] <= E["trained"]["IIA_argmax"] - 0.05)
for k, d in (("F2", f2), ("F3", f3), ("F5", f5)):
    out[k] = dict(per_parent=d, n_ok=sum(bool(d.get(p, {}).get("ok")) for p in C)); out[k]["confirmed"] = out[k]["n_ok"] >= need
# F4
f4 = {}
for p in P:
    a = ld(f"{R}/logs/das/{p}_full_c1_s4000_seed0_eval_conf10_E_L2_q7_k6_ev62500000_disc.json"); b = ld(f"{R}/logs/das/{p}_c1_s4000_seed0_eval_conf10_E_L2_q7_k6_ev62500000_disc.json")
    if a and b:
        f4[p] = dict(full=a["disc"], c1=b["disc"], rand_full=a["rand_disc"], ok=a["disc"]["IIA_B"] >= 0.90 and b["disc"]["IIA_B"] <= a["disc"]["IIA_B"] - 0.10 and b["disc"]["KL_B"] >= 2 * a["disc"]["KL_B"])
out["F4"] = dict(per_parent=f4, n_ok=sum(bool(f4.get(p, {}).get("ok")) for p in C)); out["F4"]["confirmed"] = out["F4"]["n_ok"] >= need
# F6
f6 = {}
for p in P:
    d = ld(f"{R}/logs/subspace_shift_conf_{p}.json")
    if d:
        a, b = d[f"{p}:full_c1:0:_eval_conf10"], d[f"{p}:c1:0:_eval_conf10"]
        fa = a["ratio_D2c_over_D2"] <= 0.50 and a["ratio_CNTc_over_CNT"] <= 0.60 and 0.70 <= a["ratio_D2_over_CNT"] <= 1.40; fb = b["ratio_D2c_over_D2"] >= 0.60
        f6[p] = dict(full={k: a[k] for k in ("ratio_D2c_over_D2", "ratio_CNTc_over_CNT", "ratio_D2_over_CNT")}, c1={k: b[k] for k in ("ratio_D2c_over_D2", "ratio_CNTc_over_CNT", "ratio_D2_over_CNT")}, a=fa, b=fb)
out["F6"] = dict(per_parent=f6, n_a=sum(bool(f6.get(p, {}).get("a")) for p in C), n_b=sum(bool(f6.get(p, {}).get("b")) for p in C))
out["F6"]["confirmed"] = out["F6"]["n_a"] >= need and out["F6"]["n_b"] >= need
# F7
t = ld(f"{R}/logs/traj_alg_conf.json"); f7 = {}
if t:
    for p in P:
        rows = t.get(f"{p}:full_c1:0:_eval_conf10")
        if not rows: continue
        by = {r["step"]: r for r in rows}; first_B = next((r["step"] for r in rows if r["D2"]["closest"] == "B"), None)
        a = all(by.get(s, {}).get("D2", {}).get("closest") == "A1" for s in (50, 74, 111)); b = by[4000]["D2"]["closest"] == "B"; c = first_B is not None and first_B >= 246
        f7[p] = dict(seq=[(r["step"], r["D2"]["closest"], round(r["D2"]["B"], 3), round(r["D2"]["A1"], 3)) for r in rows], first_B=first_B, a=a, b=b, c=c, ok=a and b and c)
out["F7"] = dict(per_parent=f7, n_ok=sum(bool(f7.get(p, {}).get("ok")) for p in C)); out["F7"]["confirmed"] = out["F7"]["n_ok"] >= need
json.dump(out, open(f"{R}/logs/confirmation_results.json", "w"), indent=1, default=float)
for k in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"): print(k, "confirmed =", out[k]["confirmed"], {a: b for a, b in out[k].items() if a.startswith("n_")})
