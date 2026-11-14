"""Supplementary scorer for P6 AS WRITTEN in the frozen text of FRESH_CONFIRMATION_AND_STATISTICS.md Part I.
Freeze defect (found during a pre-scoring dry run, before the final merge): the frozen scorer score_xii.py implements an EARLIER draft of P6
  (code) a: IIA_E >= 0.70 and IIA_A1 <= 0.30 on all discriminating pairs;  b: VS linear-vs-projection;  no C1 contrast
whereas the frozen text specifies
  (text) a: IIA_E (informative subset) >= 0.70 and IIA_A1 <= base_A1 + 0.10;  b: C1 contrast IIA_A1 >= base_A1 + 0.15;  c: VS linear-vs-projection.
Both are reported; neither is preferred post hoc.  Same pass rule (>= ceil(0.75 n) complete parents).  -> logs/score_p6_text.json"""
import json, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs"); CK = os.path.join(HERE, "..", "checkpoints")
S = "_eval_conf12"; PARENTS = [p for p in ("P150", "P151", "P152", "P153", "P154", "P155", "P156") if os.path.exists(os.path.join(CK, f"z{p}", "pre.json"))]
COMP = [p for p in PARENTS if json.load(open(os.path.join(CK, f"z{p}", "pre.json")))["eval"]["aux_acc"] >= 0.99]
import glob
di = json.load(open(os.path.join(L, "disc_ident_conf.json")))
for f in sorted(glob.glob(os.path.join(L, "disc_ident_c1conf_*.json"))): di.update({k: v for k, v in json.load(open(f)).items() if k != "n_pairs"})   # C1 runs added post hoc (freeze defect 2)
vs = json.load(open(os.path.join(L, "vs_linear_conf.json")))
out = {}
for grp, units in (("complete", COMP), ("all", PARENTS)):
    a = {"a_IIA_E_inf>=0.70_IIA_A1<=base+0.10": {}, "b_C1_IIA_A1>=base+0.15": {}, "c_VS_LIN<=0.7xproj_eig>=1.2": {}}
    for p in units:
        f, c = di.get(f"v{p}:full_c1:10:{S}"), di.get(f"v{p}:c1:10:{S}")
        a["a_IIA_E_inf>=0.70_IIA_A1<=base+0.10"][p] = bool(f and f["IIA_E_inf"] >= 0.70 and f["IIA_A1"] <= f["base_A1"] + 0.10)
        a["b_C1_IIA_A1>=base+0.15"][p] = bool(c and c["IIA_A1"] >= c["base_A1"] + 0.15)
        r = vs.get(f"v{p}:full_c1:10:{S}", {}).get("VS")
        a["c_VS_LIN<=0.7xproj_eig>=1.2"][p] = bool(r and r["LIN"]["KL"] <= 0.7 * min(r[k]["KL"] for k in ("PROJ_1", "PROJ_2", "PROJ_3", "ID")) and max(r["LIN_eigs"]) >= 1.2)
    need = math.ceil(0.75 * len(units))
    out[grp] = {k: dict(k=sum(v.values()), n=len(v), passed=sum(v.values()) >= need, units=v) for k, v in a.items()}
    out[grp + "_verdict"] = "confirmed" if all(x["passed"] for x in out[grp].values()) else "falsified"
json.dump(out, open(os.path.join(L, "score_p6_text.json"), "w"), indent=1)
for g in ("complete", "all"): print("P6 (text)", g, out[g + "_verdict"], {k: f"{v['k']}/{v['n']}" for k, v in out[g].items()})
