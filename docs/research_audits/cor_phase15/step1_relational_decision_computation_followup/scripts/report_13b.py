"""POST-FREEZE reporting layer for XIII-B (rules RB1-RB4, logged in COMPUTE_LOG before any confirmation output was opened).  The frozen
verdicts of score_13b.py are reported verbatim; this adds:
  RB1 'not confirmed (arm X falsified)' whenever any arm failed;  RB2 units dropped by a unit_filter with missing inputs;
  RB3 strict diagnostic flag: null computed and failed on every unit AND the arm passed on no preflight null-scenario unit; Q1a forced non-diagnostic;
  RB4 Q4c sensitivity with PERM / MEAN decoy restorations included.
Also: crash scan of confirmation run logs; addendum AQ1 evaluation (rule in logs/ADDENDUM_A1.md).  -> logs/report_13b.json"""
import json, os, re, glob, math, sys
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs"); sys.path.insert(0, HERE)
import numpy as np
S = json.load(open(os.path.join(L, "score_13b_conf13b.json"))); SPEC = json.load(open(os.path.join(L, "predictions_13b.json"))); PF = json.load(open(os.path.join(L, "preflight_13b.json")))
nullpass = {(p, a) for p, a, u in PF["arm_passes_on_null_units"]}
rep = dict(frozen_counts=S["counts"], predictions={})
# crash scan of the K lanes
keys = [re.search(r"\[(\w+)\]", l).group(1) for l in open(os.path.join(L, "COMPUTE_LOG.txt")) if re.search(r"lane K\d start", l)]
rep["crashed_runs"] = [k for k in keys if os.path.exists(os.path.join(L, "run", k + ".out")) and re.search(r"Traceback|Killed|MemoryError", open(os.path.join(L, "run", k + ".out"), errors="ignore").read())]
rep["nonzero_rc"] = [l.strip() for l in open(os.path.join(L, "COMPUTE_LOG.txt")) if re.search(r"lane K\d end .* rc=(?!0\b)", l)]
src = open(os.path.join(HERE, "score_13b.py")).read(); src = src[:src.index("out = dict(units=UNITS")]
g = {"__name__": "s13b", "__file__": os.path.join(HERE, "score_13b.py")}; sys.argv = ["score_13b.py"]; exec(compile(src, "score13b[defs]", "exec"), g)
for P in SPEC["predictions"]:
    pr = S["predictions"][P["id"]]; units_all = [u for u in S["units"] if (not P.get("parents") or u.split(":")[0][1:] in P["parents"])]
    dropped_missing = [u for u, v in pr["filter_values"].items() if v is None]
    arms = {}
    for aid, a in pr["arms"].items():
        strict = (a["diagnostic"] is True) and ((P["id"], aid) not in nullpass) and not (P["id"] == "Q1" and aid == "a")
        arms[aid] = dict(frozen_status=a["status"], k=a["k"], n=a["n"], failed_units=a["failed_units"], missing=a["missing"], diagnostic_strict=strict if a["status"] == "pass" else None)
    failed = [aid for aid, a in arms.items() if a["frozen_status"] == "fail"]
    verdict = ("not confirmed (arm(s) " + ",".join(failed) + " falsified)") if failed else ("inconclusive (missing filter data)" if dropped_missing else pr["verdict"])
    rep["predictions"][P["id"]] = dict(frozen_verdict=pr["verdict"], reported_verdict=verdict, units=pr["units"], dropped_missing_filter=dropped_missing, arms=arms,
                                       all_passing_arms_strictly_diagnostic=all(a["diagnostic_strict"] for a in arms.values() if a["frozen_status"] == "pass"))
# RB4: Q4c with PERM / MEAN included
q4 = {}
for u in rep["predictions"]["Q4"]["units"]:
    r = g["src"](u, "lc")
    if not r: continue
    R = r["EA"]["restoration"]; mx = max(R[k]["proj"] for k in ("RESTORE_LOW", "RESTORE_TOP", "RESTORE_MAG", "RESTORE_PERM", "RESTORE_MEAN"))
    q4[u] = dict(c_with_PERM_MEAN=bool(R["removal_fraction_removed"] >= 0.4 and R["RESTORE_MIX"]["proj"] >= 0.85 and R["RESTORE_MIX"]["fid_E"]["R2"] >= 0.85 and mx <= R["REMOVED_only"]["proj"] + 0.15),
                 fid_E_restore={k: R[k]["fid_E"]["R2"] for k in ("RESTORE_MIX", "RESTORE_PERM", "RESTORE_MEAN")}, proj_restore={k: R[k]["proj"] for k in ("REMOVED_only", "RESTORE_MIX", "RESTORE_PERM", "RESTORE_MEAN")})
rep["Q4_sensitivity_PERM_MEAN"] = q4
# addendum AQ1
fp = os.path.join(L, "fit_powermean_conf13b_addendum.json")
if os.path.exists(fp):
    pm = json.load(open(fp)); rows = []
    for u in S["units"]:
        cf = g["src"](u, "cf"); rho = pm.get(u, {}).get("PM", {}).get("rho")
        rows.append((u, None if cf is None else cf["lik_RO1"], rho))
    miss = [u for u, l, r in rows if l is None or r is None]; early = [(u, r) for u, l, r in rows if l is not None and l >= 0.7]; late = [(u, r) for u, l, r in rows if l is not None and l <= 0.2]
    ok = [(l, r) for u, l, r in rows if l is not None and r is not None]; c = float(np.corrcoef([x[0] for x in ok], [x[1] for x in ok])[0, 1]) if len(ok) >= 3 else None
    holds = (not miss) and early and late and all(r <= 0.70 for _, r in early) and all(r >= 0.85 for _, r in late) and c is not None and c <= -0.7
    rep["AQ1"] = dict(rows=rows, corr=c, verdict="inconclusive" if miss else ("holds" if holds else "fails"))
json.dump(rep, open(os.path.join(L, "report_13b.json"), "w"), indent=1, default=str)
print("crashes", rep["crashed_runs"], "nonzero rc", rep["nonzero_rc"])
for p, x in rep["predictions"].items(): print(p, "frozen:", x["frozen_verdict"], "| reported:", x["reported_verdict"], "| strict-diag:", x["all_passing_arms_strictly_diagnostic"], "| dropped:", x["dropped_missing_filter"])
print("Q4 sensitivity", {u.split(':')[0]: v["c_with_PERM_MEAN"] for u, v in q4.items()}); print("AQ1", rep.get("AQ1"))
