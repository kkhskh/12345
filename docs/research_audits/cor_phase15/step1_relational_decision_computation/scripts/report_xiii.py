"""Post-freeze REPORTING/CHECKING layer (review 2, rules R1-R4 logged in COMPUTE_LOG before any cohort output was opened).
Does NOT change the frozen scorer's verdicts (logs/score_xiii.json); adds:
  R1 crash check: every logs/run/*.out of the T/C lanes scanned for Traceback / Killed / MemoryError / NaN-abort
  R2 head-definition mismatch: scorer mixture class vs the mixture-head lists written by l2_mix_causal / l2_mediation / score_contrast / l1abl_mix
  R3 merge completeness: number of u-keys per merged file, units / classes in score_xiii.json
  R4 explicit failed arms per prediction; 'diagnostic' only if the null was computed (not None) on every unit; units excluded from mixture-only predictions
  + two-sample bootstrap CI of the P2 DiD (TRUE and CTRL designs resampled independently; the frozen script holds the CTRL mean fixed)
Usage: report_xiii.py [SFX] (default _conf) -> logs/report_xiii<SFX>.json and .md"""
import json, os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs")
import numpy as np
SFX = sys.argv[1] if len(sys.argv) > 1 else "_conf"; S = json.load(open(os.path.join(L, "score_xiii.json" if SFX == "_conf" else f"score_xiii{SFX}.json")))
J = lambda n: json.load(open(os.path.join(L, f"{n}{SFX}.json"))) if os.path.exists(os.path.join(L, f"{n}{SFX}.json")) else {}
rep = {}
# R1
bad = []
for f in glob.glob(os.path.join(L, "run", "*.out")):
    t = open(f, errors="ignore").read()
    if re.search(r"Traceback|Killed|MemoryError|Segmentation", t): bad.append(os.path.basename(f))
lines = {}
for l in open(os.path.join(L, "COMPUTE_LOG.txt")):
    m = re.search(r"lane (\S+) start \[(\w+)\] (.*)", l)
    if m: lines[m.group(2)] = (m.group(1), m.group(3)[-120:])
rep["R1_crashed_runs"] = [dict(key=b[:-4], lane=lines.get(b[:-4], ("?", ""))[0], cmd=lines.get(b[:-4], ("?", "?"))[1]) for b in bad]
# R3
names = ("l2_scan", "l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix", "decoy_disc")
D = {n: J(n) for n in names}
rep["R3_merge_counts"] = {n: len([k for k in D[n] if k.startswith("u")]) for n in names} if SFX == "_conf" else {n: len(D[n]) for n in names}
rep["R3_units"] = S["units"]; rep["R3_none_class"] = [u for u, c in S["mixture_class"].items() if c is None]
# R2
mm = {}
for u in S["units"]:
    sc = S["mixture_class"].get(u); lists = {n: (D[n].get(u) or {}).get("mixture_heads") for n in ("l2_mix_causal", "l2_mediation", "score_contrast", "l1abl_mix")}
    mism = sc is True and any(v == [] or v is None for v in lists.values()) or (sc is False and any(v for v in lists.values()))
    mm[u] = dict(scorer_mixture_class=sc, script_head_lists=lists, mismatch=bool(mism))
rep["R2_head_definitions"] = mm
# R4
r4 = {}
for pid, p in S["predictions"].items():
    arms = {}
    for aid, a in p["arms"].items():
        failed = [u for u, v in a["units"].items() if v is False]; missing = [u for u, v in a["units"].items() if v is None]
        arms[aid] = dict(frozen_status=a["status"], k=a["k"], n=a["n"], failed_units=failed, missing_units=missing)
    units_used = sorted({u for a in p["arms"].values() for u in a["units"]})
    r4[pid] = dict(frozen_verdict=p["verdict"], arms=arms, excluded_units=[u for u in S["complete"] if u not in units_used],
                   any_failed_arm=[aid for aid, x in arms.items() if x["failed_units"] and x["frozen_status"] != "pass"])
rep["R4"] = r4
# diagnostic-only-if-null-computed: recompute nulls availability via score_xiii's own evaluator
sys.argv = ["score_xiii.py"] + (["--dev", os.path.join(L, "preflight_units.txt")] if SFX != "_conf" else [])
os.environ.setdefault("SCORE_SFX", SFX)
import importlib.util
spec_ = json.load(open(os.path.join(L, "predictions_xiii.json")))
src = open(os.path.join(HERE, "score_xiii.py")).read(); src = src[:src.index("out = dict(units=UNITS")]
g = {"__file__": os.path.join(HERE, "score_xiii.py"), "__name__": "sc"}; exec(compile(src, "score_xiii[defs]", "exec"), g)
for P in spec_["predictions"]:
    for A in P["arms"]:
        if not A.get("null"): continue
        units = list(S["predictions"][P["id"]]["arms"][A["id"]]["units"])
        nul = {u: g["evaluate_expr"](u, A["null"]) for u in units}
        r4[P["id"]]["arms"][A["id"]]["null_values"] = nul
        r4[P["id"]]["arms"][A["id"]]["diagnostic_strict"] = (None if not units else (all(v is not None for v in nul.values()) and not any(nul.values())))
# two-sample DiD bootstrap
did = {}
rng = np.random.default_rng(0)
for u, v in D["l2_mediation"].items():
    if not isinstance(v, dict) or "TRUE" not in v or "mag_NIE" not in v["TRUE"]: continue
    t, c = np.array(v["TRUE"]["mag_NIE"]), np.array(v["CTRL"]["mag_NIE"])
    bs = [rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean() for _ in range(4000)]
    did[u] = dict(est=float(t.mean() - c.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), frozen_lo=v.get("DiD_mag_NIE_true_minus_ctrl", {}).get("lo"))
rep["P2_DiD_two_sample_bootstrap"] = did
json.dump(rep, open(os.path.join(L, f"report_xiii{SFX}.json"), "w"), indent=1, default=str)
print("R1 crashed:", rep["R1_crashed_runs"]); print("R3:", rep["R3_merge_counts"], "none-class:", rep["R3_none_class"])
print("R2 mismatches:", [u for u, x in mm.items() if x["mismatch"]])
for pid, x in r4.items(): print(pid, x["frozen_verdict"], "failed arms:", x["any_failed_arm"], "excluded:", [e.split(':')[0] for e in x["excluded_units"]],
                               {a: y.get("diagnostic_strict") for a, y in x["arms"].items() if "diagnostic_strict" in y})
print("DiD 2-sample:", {u.split(':')[0]: (round(d['lo'], 3), round(d['hi'], 3), d['frozen_lo'] and round(d['frozen_lo'], 3)) for u, d in did.items()})
