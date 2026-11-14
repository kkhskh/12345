"""FROZEN scorer for Investigation XIII (predictions defined in logs/predictions_xiii.json, which ALSO generates the prediction table in
FROZEN_PROTOCOL_AND_CONFIRMATION.md via scripts/render_predictions.py -- one source of truth, to avoid XII's text/code divergence).
Units: complete fresh parents (gate.json complete == True).  'Mixture-class' parents (frozen rule in the JSON) are the unit for mechanism arms that
need a mixture head; if fewer than 2 complete parents are mixture-class those arms are INCONCLUSIVE (not passed).
Pass rule per arm: holds in >= ceil(0.75 n) units.  Every arm also has a NULL version (the matched control substituted for the test quantity):
an arm counts as DIAGNOSTIC only if its null version FAILS on every unit; a passing arm whose null also passes is reported as NON-DIAGNOSTIC.
Usage: score_xiii.py [--dev SPEC_LIST_FILE]  -> logs/score_xiii.json (or logs/score_xiii_dev.json)"""
import json, os, sys, math, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, stats_xii as ST
L = os.path.join(HERE, "..", "logs"); CK = os.path.join(HERE, "..", "checkpoints")
DEV = "--dev" in sys.argv; SPEC = json.load(open(os.environ.get("SCORE_SPEC", os.path.join(L, "predictions_xiii.json"))))
sfx = os.environ.get("SCORE_SFX", "_dev" if DEV else "_conf")
def J(name):
    f = os.path.join(L, f"{name}{sfx}.json"); return json.load(open(f)) if os.path.exists(f) else {}
if DEV:
    UNITS = [l.strip() for l in open(sys.argv[sys.argv.index("--dev") + 1]) if l.strip()]; COMPLETE = UNITS; INCOMPLETE = []
else:
    UNITS, COMPLETE, INCOMPLETE = [], [], []
    for p in ("P160", "P161", "P162", "P163", "P164", "P165"):
        g = os.path.join(CK, f"z{p}", "gate.json")
        if not os.path.exists(g): continue
        sp = f"u{p}:full_c1:20:_eval_conf13"; UNITS.append(sp); (COMPLETE if json.load(open(g))["complete"] else INCOMPLETE).append(sp)
D = {k: J(k) for k in ("l2_scan", "l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix", "decoy_disc")}
def adapt(sp):
    tag = sp.split(":")[0] + "_" + sp.split(":")[1] + f"_s4000_seed{sp.split(':')[2]}" + sp.split(":")[3]
    d = os.path.join(L, "adapt_sel" + sfx); f = os.path.join(d, tag + ".json"); return json.load(open(f)) if os.path.exists(f) else None
def mixture_class(sp):
    r = D["l2_scan"].get(sp)
    if not r: return None
    th = SPEC["mixture_class_rule"]
    return any(r["heads"][h]["mass_q_groups"] >= th["mass"] and r["logit_var_share"][f"h2.{h}"] >= th["share"] for h in r["heads"])
def top_mix(sp):
    """Derived: the mixture head (frozen rule) with the largest direct-logit share; its correlations with the responsibility variables."""
    r = D["l2_scan"].get(sp); th = SPEC["mixture_class_rule"]
    if not r: return None
    hs = [h for h in r["heads"] if r["heads"][h]["mass_q_groups"] >= th["mass"] and r["logit_var_share"][f"h2.{h}"] >= th["share"]]
    if not hs: return None
    h = max(hs, key=lambda h: r["logit_var_share"][f"h2.{h}"]); return dict(head=h, **{f"corr_{k}": v for k, v in r["heads"][h]["corr"].items()})
def get(sp, path):
    """path like 'l2_mix_causal:donor.bayes_resp.R2' or 'adapt:full.acc' or 'expr:...'"""
    src, keys = path.split(":", 1)
    if src == "cls": return mixture_class(sp)
    if src == "mix": obj = top_mix(sp)
    else: obj = adapt(sp) if src == "adapt" else D[src].get(sp)
    for k in keys.split("."):
        if obj is None: return None
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj
def evaluate_expr(sp, expr):
    import re
    vals = {}
    for i, m in enumerate(re.findall(r"\{([^}]+)\}", expr)):
        v = get(sp, m)
        if v is None: return None
        vals[f"v{i}"] = v; expr = expr.replace("{" + m + "}", f"v{i}", 1)
    return bool(eval(expr, {"abs": abs, "min": min, "max": max}, vals))
out = dict(units=UNITS, complete=COMPLETE, incomplete=INCOMPLETE, mixture_class={sp: mixture_class(sp) for sp in UNITS}, predictions={}, checks=[])
for P in SPEC["predictions"]:
    pr = dict(arms={}); units_all = COMPLETE if P["unit"] != "incomplete" else INCOMPLETE
    units = [u for u in units_all if (not P.get("mixture_only")) or mixture_class(u)]
    for A in P["arms"]:
        res = {u: evaluate_expr(u, A["test"]) for u in units}; nul = {u: evaluate_expr(u, A["null"]) for u in units} if A.get("null") else None
        n = len(units); k = sum(bool(v) for v in res.values()); missing = sum(v is None for v in res.values())
        if n < P.get("min_units", 2) or missing: status = "inconclusive"
        else: status = "pass" if k >= math.ceil(0.75 * n) else "fail"
        diag = None if nul is None else not any(bool(v) for v in nul.values())
        pr["arms"][A["id"]] = dict(text=A["text"], status=status, k=k, n=n, null_passes_on=[u for u, v in (nul or {}).items() if v], diagnostic=diag, units=res)
        for u, v in res.items(): out["checks"].append(dict(prediction=P["id"], arm=A["id"], unit=u, passed=v))
    st = [a["status"] for a in pr["arms"].values()]
    assert st, f"{P['id']} has no arms"
    pr["verdict"] = "inconclusive" if "inconclusive" in st else ("confirmed" if all(s == "pass" for s in st) else "falsified")
    pr["non_diagnostic_arms"] = [a for a, x in pr["arms"].items() if x["status"] == "pass" and x["diagnostic"] is False]
    out["predictions"][P["id"]] = pr
cnt = dict(meets_thresholds=sum(p["verdict"] == "confirmed" for p in out["predictions"].values()),
           discriminating=sum(p["verdict"] == "confirmed" and not p["non_diagnostic_arms"] for p in out["predictions"].values()),
           non_diagnostic_pass=sum(p["verdict"] == "confirmed" and bool(p["non_diagnostic_arms"]) for p in out["predictions"].values()),
           falsified=sum(p["verdict"] == "falsified" for p in out["predictions"].values()), inconclusive=sum(p["verdict"] == "inconclusive" for p in out["predictions"].values()))
out["counts"] = cnt
json.dump(out, open(os.path.join(L, "score_xiii" + (sfx if DEV else "") + ".json"), "w"), indent=1, default=str)
for pid, p in out["predictions"].items():
    print(pid, p["verdict"], {a: f"{x['status']} {x['k']}/{x['n']}" + ("" if x["diagnostic"] in (None, True) else " NON-DIAG") for a, x in p["arms"].items()})
print("counts", cnt, "| complete", len(COMPLETE), "incomplete", len(INCOMPLETE), "| mixture-class", out["mixture_class"])
