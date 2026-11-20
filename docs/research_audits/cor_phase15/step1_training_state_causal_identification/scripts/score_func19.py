"""(Written after the frozen confirmation results were read; it implements the rule of logs/FUNCTIONAL_BATTERY_FREEZE19.md literally.  Group
F3x = INCONCLUSIVE(causal_NONE) models, which the frozen F3 rule does not include, is reported separately and is not part of F3.)
Stage-6 scoring per logs/FUNCTIONAL_BATTERY_FREEZE19.md: split ratio SR -> FE (>= 2.2) / FM (<= 1.6) / FAMBIG; F1 anchors, F2 causal-switch
arms follow the new class, F3 causal-none arms keep the B class, F4 hybrids descriptive.  Per base; SUPPORTED if every eligible model matches,
CONTRADICTED if a strict majority have the opposite class, else PARTIAL.  Usage: score_func19.py SCORE.json BATTERY.json OUT.json"""
import json, sys
S, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
fc = lambda sr: "EARLY" if sr >= 2.2 else ("MIXTURE" if sr <= 1.6 else "FAMBIG")
D = json.load(open("logs/conf/design19.json")); pre = {p["name"]: f"uP160_fx_L{p['L']}_W{p['W']}_D{p['D']}_stop1211_br100_{p['name']}_" for p in D["pairs"]}
rows = {}; out = dict(models={}, verdicts={})
for pn, P in S["pairs"].items():
    F = P["families"]["T"]
    for an, a in F["arms"].items():
        k = pre[pn] + an + "@1211"; r = B[k]; sr = r["split"]["KL"] / r["agree_consensus"]["KL"]; cls = fc(sr)
        cat = a["category"]; grp = "F1" if an in ("fullB", "fullI") else ("F2" if (cat in ("BOTH", "CAUSAL_ONLY") or cat.startswith("INCONCLUSIVE(causal_SWITCH")) else ("F3" if cat in ("NEITHER", "FUNCTIONAL_ONLY") else ("F3x" if cat.startswith("INCONCLUSIVE(causal_NONE") else "F4")))
        want = (F["anchor_I"] if (grp == "F2" or an == "fullI") else F["anchor_B"]) if grp not in ("F4",) else None   # F3x: INCONCLUSIVE(causal_NONE), outside the frozen F3 rule (post-review E2), reported separately
        out["models"][f"{pn}|{an}"] = dict(SR=sr, SR_class=cls, causal_category=cat, group=grp, expected=want, match=(cls == want) if want else None)
for grp in ("F1", "F2", "F3", "F3x"):
    for base in "EM":
        m = [v for k, v in out["models"].items() if v["group"] == grp and k.startswith(base)]
        if not m: continue
        n_ok = sum(v["match"] for v in m); n_opp = sum(v["SR_class"] not in (v["expected"], "FAMBIG") for v in m); n = len(m)
        out["verdicts"][f"{base}_{grp}"] = dict(band="SUPPORTED" if n_ok == n else ("CONTRADICTED" if n_opp > n / 2 else "PARTIAL"), n=n, matching=n_ok, opposite=n_opp, ambiguous=n - n_ok - n_opp)
hyb = {k: (round(v["SR"], 2), v["SR_class"]) for k, v in out["models"].items() if v["group"] == "F4"}; out["F4_hybrids"] = hyb
json.dump(out, open(sys.argv[3], "w"), indent=1)
for k, v in out["verdicts"].items(): print(k, v)
print("F4 hybrids:", hyb)
