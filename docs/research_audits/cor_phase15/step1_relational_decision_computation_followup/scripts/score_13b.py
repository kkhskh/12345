"""FROZEN scorer for Investigation XIII-B (predictions in logs/predictions_13b.json, which also renders the document table via
scripts/render_predictions_13b.py -- one source of truth).
Units: the seed-21 fine-tunes uP160/161/163/164/165 (FINE-TUNING replicates of already-pretrained parents: they test fine-tuning variability,
NOT independent pretraining).  Each prediction may restrict units by a frozen 'unit_filter' expression evaluated on the unit's own measured
data (e.g. its class features) or by the frozen parent list 'parents'.  Pass rule per arm: holds on >= ceil(0.75 n) of its units, n >= min_units
(else INCONCLUSIVE).  Prediction CONFIRMED iff every arm passes.  Every arm with a null is DIAGNOSTIC iff the null was computed on every unit
and fails on every unit; otherwise a passing arm is NON-DIAGNOSTIC.  Missing values are reported, never treated as passes or fails.
Usage: score_13b.py [--units FILE] [--sfx SFX] -> logs/score_13b<SFX>.json"""
import json, os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
SPEC = json.load(open(os.environ.get("SCORE_SPEC", os.path.join(L, "predictions_13b.json"))))
SFX = sys.argv[sys.argv.index("--sfx") + 1] if "--sfx" in sys.argv else "_conf13b"
UNITS = [l.strip() for l in open(sys.argv[sys.argv.index("--units") + 1])] if "--units" in sys.argv else [f"uP{p}:full_c1:21:_eval_conf13" for p in (160, 161, 163, 164, 165)]
UNITS = [u for u in UNITS if u]
tag = lambda sp: (lambda a: f"{a[0]}_{a[1]}_s4000_seed{a[2]}{a[3]}")(sp.split(":"))
def jl(p):
    p = os.path.join(L, p); return json.load(open(p)) if os.path.exists(p) else None
def src(sp, name):
    """source objects per unit (all written by the frozen confirmation queue under the SFX tag)"""
    par = sp.split(":")[0][1:]
    if name == "cf": return (jl(f"class_features{SFX}_{par}.json") or {}).get(sp)
    if name == "ff": return (jl(f"fit_family{SFX}_{par}.json") or {}).get(sp)
    if name == "ad":
        d = jl(f"algdisc{SFX}.json"); return None if d is None else d["models"].get(sp)
    if name == "lc": return (jl(f"l2_components{SFX}_{par}.json") or {}).get(sp)
    if name == "db": return jl(os.path.join(f"das_bal{SFX}", tag(sp) + ".json"))
    if name == "a2": return jl(os.path.join(f"adapt2{SFX}", tag(sp) + "_b0.6.json"))
    if name == "exp": return SPEC["expected_class"].get(par)
    raise KeyError(name)
def get(sp, path):
    s, keys = path.split(":", 1); obj = src(sp, s)
    if s == "exp": return obj
    for k in keys.split("."):
        if obj is None: return None
        obj = obj.get(k) if isinstance(obj, dict) else (obj[int(k)] if isinstance(obj, list) and k.isdigit() and int(k) < len(obj) else None)
    return obj
def ev(sp, expr):
    vals = {}
    for i, m in enumerate(re.findall(r"\{([^}]+)\}", expr)):
        v = get(sp, m)
        if v is None: return None
        vals[f"v{i}"] = v; expr = expr.replace("{" + m + "}", f"v{i}", 1)
    return bool(eval(expr, {"abs": abs, "min": min, "max": max, "len": len}, vals))
out = dict(units=UNITS, sfx=SFX, predictions={}, checks=[])
for P in SPEC["predictions"]:
    units = [u for u in UNITS if (not P.get("parents") or u.split(":")[0][1:] in P["parents"])]
    filt = {u: (ev(u, P["unit_filter"]) if P.get("unit_filter") else True) for u in units}
    units_in = [u for u in units if filt[u]]; pr = dict(units=units_in, filter_values=filt, arms={})
    for A in P["arms"]:
        au = [u for u in units_in if (not A.get("unit_filter") or ev(u, A["unit_filter"]))]
        res = {u: ev(u, A["test"]) for u in au}; nul = {u: ev(u, A["null"]) for u in au} if A.get("null") else None
        n = len(au); k = sum(v is True for v in res.values()); miss = [u for u, v in res.items() if v is None]
        status = "inconclusive" if (n < P.get("min_units", 2) or miss) else ("pass" if k >= math.ceil(0.75 * n) else "fail")
        diag = None if nul is None else (all(v is not None for v in nul.values()) and not any(v for v in nul.values()))
        pr["arms"][A["id"]] = dict(text=A["text"], status=status, k=k, n=n, units=res, missing=miss, null_values=nul, diagnostic=diag,
                                   failed_units=[u for u, v in res.items() if v is False])
        for u, v in res.items(): out["checks"].append(dict(prediction=P["id"], arm=A["id"], unit=u, passed=v))
    st = [a["status"] for a in pr["arms"].values()]
    pr["verdict"] = "inconclusive" if "inconclusive" in st else ("confirmed" if all(s == "pass" for s in st) else "falsified")
    pr["failed_arms"] = [a for a, x in pr["arms"].items() if x["status"] == "fail"]
    pr["non_diagnostic_arms"] = [a for a, x in pr["arms"].items() if x["status"] == "pass" and x["diagnostic"] is False]
    out["predictions"][P["id"]] = pr
V = {p: x["verdict"] for p, x in out["predictions"].items()}
out["counts"] = dict(confirmed=sum(v == "confirmed" for v in V.values()), confirmed_all_arms_diagnostic=sum(V[p] == "confirmed" and not x["non_diagnostic_arms"] for p, x in out["predictions"].items()),
                     confirmed_with_non_diagnostic_arm=sum(V[p] == "confirmed" and bool(x["non_diagnostic_arms"]) for p, x in out["predictions"].items()),
                     falsified=sum(v == "falsified" for v in V.values()), inconclusive=sum(v == "inconclusive" for v in V.values()))
json.dump(out, open(os.path.join(L, f"score_13b{SFX}.json"), "w"), indent=1, default=str)
for p, x in out["predictions"].items():
    print(p, x["verdict"], {a: f"{y['status']} {y['k']}/{y['n']}" + (" NON-DIAG" if y["diagnostic"] is False and y["status"] == "pass" else "") + (f" missing={len(y['missing'])}" if y["missing"] else "") for a, y in x["arms"].items()})
print("counts", out["counts"])
