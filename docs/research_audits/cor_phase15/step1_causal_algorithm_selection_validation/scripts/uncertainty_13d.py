"""XIII-D statistics: (1) threshold sensitivity of every frozen verdict: each numeric threshold of each prediction is scaled by 0.8 and 1.25
(one at a time, in the direction that makes the arm EASIER and HARDER) and the frozen scorer is re-run on the frozen inputs, on a temporary
copy (frozen files are never modified); (2) unit-level summaries over CONFIRMATION units only with percentile bootstrap intervals (unit resampling, 10 000 draws, seed 0)
for the principal effect sizes; (3) parent-level summary of the readout-map factorial.  -> logs/uncertainty_13d.json"""
import json, os, sys, shutil, subprocess, tempfile, copy, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); LG = os.path.join(HERE, "..", "logs")
import numpy as np
out = dict(sensitivity={}, effects={}, factorial={})
LGC = tempfile.mkdtemp()   # review-3 fix: perturbed scorers read a COPY of logs/ and write there; the frozen score files are never touched
for f in os.listdir(LG):
    src = os.path.join(LG, f)
    if os.path.isfile(src) and (f.endswith(".json") or f.endswith(".jsonl")): shutil.copy(src, LGC)
    elif os.path.isdir(src) and f.startswith("adapt4_f3a"): shutil.copytree(src, os.path.join(LGC, f))
def rescore(freeze, pred_file, scorer, mutate):
    d = tempfile.mkdtemp(); shutil.copy(os.path.join(HERE, scorer), d); P = json.load(open(os.path.join(HERE, pred_file))); mutate(P)
    json.dump(P, open(os.path.join(d, pred_file), "w")); r = subprocess.run([sys.executable, os.path.join(d, scorer), LGC], capture_output=True, text=True)
    return json.load(open(os.path.join(LGC, f"score_{freeze}.json")))["summary"]
def numeric_paths(o, path=()):
    if isinstance(o, bool): return
    if isinstance(o, (int, float)) and not isinstance(o, bool): yield path
    elif isinstance(o, dict):
        for k, v in o.items():
            if k in ("refs", "units", "w_only_cells", "elig_units", "row_units"): continue
            yield from numeric_paths(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from numeric_paths(v, path + (i,))
def setp(o, p, f):
    for k in p[:-1]: o = o[k]
    o[p[-1]] = o[p[-1]] * f
for freeze, pf, sc, key in (("f1", "predictions_13d_f1.json", "score_f1.py", "thresholds"), ("f2", "predictions_13d_f2.json", "score_f2.py", "thresholds"),
                            ("f3a", "predictions_13d_f3a.json", "score_f3a.py", "rules")):
    if not os.path.exists(os.path.join(LG, f"score_{freeze}.json")): out["sensitivity"][freeze] = "not scored"; continue
    base = json.load(open(os.path.join(LG, f"score_{freeze}.json")))["summary"]; S = {}
    P0 = json.load(open(os.path.join(HERE, pf)))
    for i, p in enumerate(P0["predictions"]):
        if key not in p: continue
        for path in numeric_paths(p[key]):
            for f in (0.8, 1.25):
                s = rescore(freeze, pf, sc, lambda P, i=i, path=path, f=f: setp(P["predictions"][i][key], path, f))
                S[f"{p['id']}:{'/'.join(map(str, path))}x{f}"] = s[p["id"]]
    flips = {k: v for k, v in S.items() if v != base[k.split(":")[0]]}
    out["sensitivity"][freeze] = dict(base=base, n_perturbations=len(S), flips=flips)
def boot(x, n=10000):
    x = np.asarray(x, float); r = np.random.default_rng(0); m = r.choice(x, (n, len(x))).mean(1); return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(x), values=x.tolist())
def J(f):
    p = os.path.join(LG, f); return json.load(open(p)) if os.path.exists(p) else {}
CA = {f"conf:{k}": v for k, v in J("c_carrier_f2.json").items()}   # review-3 fix: confirmation units only (development units set the thresholds)
if CA:
    out["effects"]["carrier_DAS_suff"] = boot([v["sets"]["DAS"]["SUFF"]["slope"] for v in CA.values()])
    out["effects"]["carrier_ENC_minus_DEC_m"] = boot([v["sets"]["ENC_m"]["SUFF"]["slope"] - v["sets"]["DEC_m"]["SUFF"]["slope"] for v in CA.values()])
    out["effects"]["carrier_DAS_minus_SRCF"] = boot([v["sets"]["DAS"]["SUFF"]["slope"] - v["sets"]["DAS"]["SRCF"]["slope"] for v in CA.values()])
    out["effects"]["carrier_reverse_remaining"] = boot([v["sets"]["DAS"]["NEC"]["reverse"]["remaining_slope"] for v in CA.values()])
    out["effects"]["trace_keys_plus_queries"] = boot([v["TRACE"]["keys_and_queries"] for v in CA.values()])
RT = J("d_routes2_f2.json")
if RT: out["effects"]["routes_max_single_open"] = boot([max(r["block"][k]["rel_slope"] for k in ("QR", "QO", "RO")) / r["block"]["none"]["rel_slope"] for r in RT.values()])
DR = J("d_rows_f2.json")
if DR:
    out["effects"]["rows_TRUE_R2"] = boot([r["arms"]["TRUE"]["R2"] for r in DR.values() if r["mixture_heads"]])
    out["effects"]["rows_SHUFD_vs_borrowed_R2"] = boot([r["arms"]["SHUFD_vs_borrowed"]["R2"] for r in DR.values() if r["mixture_heads"]])
E4 = J("e4_noninv_f2.json").get("models", {})
if E4: out["effects"]["E4_SYM1_MOVE_reduction"] = boot([r["SYM1"]["MOVE_reduction"] for r in E4.values()])
# readout-map factorial: per parent, W effect on final lik_RO1 holding L, D fixed (paired cells)
KO = json.load(open(os.path.join(LG, "known_outcomes.json"))); fin = {}
for k, v in KO.items(): p, L, W, D = k.split("|"); fin[(p, int(L), int(W), int(D))] = v["lik_RO1"]
for f in ("unit_metrics_f2.jsonl", "unit_metrics_fx_dev13d.jsonl"):
    p = os.path.join(LG, f)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r["step"] == 4000 and r["factors"][4] in ("", "rep"): fin[tuple(r["factors"][:4])] = r["lik_RO1"]
pairs = []
for (p, L, W, D), v in fin.items():
    for (p2, L2, W2, D2), v2 in fin.items():
        if p2 == p and L2 == L and D2 == D and W2 > W: pairs.append(dict(parent=p, L=L, D=D, W=(W, W2), dlik=v2 - v))
out["factorial"]["W_only_pairs"] = pairs
for fac, idx in (("L", 1), ("D", 3)):
    pr = []
    for a, v in fin.items():
        for b, v2 in fin.items():
            if a[0] == b[0] and all(a[j] == b[j] for j in (1, 2, 3) if j != idx) and b[idx] > a[idx]: pr.append(abs(v2 - v))
    out["factorial"][f"{fac}_only_abs_dlik"] = boot(pr) if pr else None
out["factorial"]["W_only_abs_dlik"] = boot([abs(x["dlik"]) for x in pairs]) if pairs else None
json.dump(out, open(os.path.join(LG, "uncertainty_13d.json"), "w"), indent=1, default=float)
print(json.dumps({k: (v if k != "effects" else {a: (round(b["mean"], 3), round(b["lo"], 3), round(b["hi"], 3), b["n"]) for a, b in v.items()}) for k, v in out.items() if k != "factorial"}, default=str)[:3000])
print({k: (v if not isinstance(v, dict) else (round(v["mean"], 3), v["n"])) for k, v in out["factorial"].items() if k != "W_only_pairs"})
