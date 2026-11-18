"""XVIII fail-closed scorer for the independent-parent arm (design v2 §10.5).  Reads logs/XVIII_parents.json and logs/measure_p18.jsonl
(stream 95_600_000) and writes logs/score_parents18.json.  Same phi_nec, bands and competence as score_XVIII.py, with paired anchors within
each parent x base (one data order, D = 812).  Refuses (exit 2) if any designed run of a bistable parent lacks a measurement."""
import json, os, sys, math
H = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(H, "..")
def refuse(m): print("REFUSED:", m); sys.exit(2)
PJ = json.load(open(os.path.join(P, "logs", "XVIII_parents.json"))); recs = {}
mf = os.path.join(P, "logs", "measure_p18.jsonl")
if not os.path.exists(mf): refuse("no measurements")
for l in open(mf):
    r = json.loads(l)
    if r["run"] in recs: refuse("duplicate " + r["run"])
    recs[r["run"]] = r
ell = lambda x: math.log(max(x, 0.0) + 0.01); OPP = {"EARLY": "MIXTURE", "MIXTURE": "EARLY"}
def phi(a, b, k): return 0.5 * sum((ell(a[f"nec_{r}"]) - ell(b[f"nec_{r}"])) / (ell(k[f"nec_{r}"]) - ell(b[f"nec_{r}"])) for r in ("E", "M"))
def band(ph, c, bc): return "F" if (ph >= 0.8 and c == OPP.get(bc)) else ("N" if (ph <= 0.2 and c == bc) else "P")
comp = lambda a, b: bool(a["KL"] <= 0.045 and a["acc"] >= b["acc"] - 0.03)
out = dict(parents={}, P4={})
for p, v in PJ["parents"].items():
    if not v["bistable"]: out["parents"][p] = dict(bistable=False, screen=v["screen"]); continue
    res = dict(bistable=True, bases={})
    for b in ("E", "M"):
        g = lambda arm: recs.get(v["runs"][f"{b}|{arm}"]) or refuse(f"missing {p} {b} {arm}")
        B, K = g("baseline")["final"]["1211"], g("block")["final"]["1211"]
        ev = bool(comp(B, B) and B["cls2"] in OPP and comp(K, B))
        blk = band(1.0, K["cls2"], B["cls2"]) if ev else "NONEVAL"
        arms = {"block": blk}
        for arm in ("vrg", "gsv", "sha_hi_rolled", "sha_lo_real"):
            A = g(arm)["final"]["1211"]
            arms[arm] = ("MISSING" if not comp(A, B) else band(phi(A, B, K), A["cls2"], B["cls2"])) if (ev and blk == "F") else "NONEVAL"
            arms[arm + "_phi"] = phi(A, B, K) if ev else None
        res["bases"][b] = dict(evaluable=ev, baseline_cls=B["cls2"], block_cls=K["cls2"], arms=arms)
    bb = [res["bases"][b]["arms"]["block"] for b in ("E", "M")]
    out["P4"][p] = "effect present" if bb == ["F", "F"] else ("counterexample" if "F" not in bb else "partial")
    res["arm_summary"] = {arm: (res["bases"]["E"]["arms"][arm] if res["bases"]["E"]["arms"][arm] == res["bases"]["M"]["arms"][arm] else "MIXED") for arm in ("block", "vrg", "gsv", "sha_hi_rolled", "sha_lo_real")}
    out["parents"][p] = res
k = sum(v == "effect present" for v in out["P4"].values()); n = len(out["P4"]); out["P4_summary"] = dict(effect_present=k, bistable_parents=n, candidates=5)
json.dump(out, open(os.path.join(P, "logs", "score_parents18.json"), "w"), indent=1, default=float); print(json.dumps(out["P4"]), json.dumps({p: v.get("arm_summary") for p, v in out["parents"].items()}))
