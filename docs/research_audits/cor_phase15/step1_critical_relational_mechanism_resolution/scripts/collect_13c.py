"""XIII-C: collect the confirmation measurements into ONE bundle (logs/bundle_13c<TAG>.json) that score_13c.py scores.
Reads only the confirmation-stream outputs (tags *_conf):
  logs/unit_metrics_conf.jsonl (stream 88_100_000; steps 546 and 4000),  logs/fair_fits_conf.json (stream 88_300_000),
  logs/adapt3_conf/*.json (stream 88_000_000),  logs/c_causal_conf.json (88_500_000),  logs/d_rows_conf.json (88_600_000),
  logs/d_routes_conf.json (88_700_000),  logs/d_das_conf.json (88_800_000).
Missing inputs are recorded as null (the scorer treats a missing required input as FAIL of that arm, never as a pass -- RB rule).
Usage: collect_13c.py [TAG]"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "..", "logs"); TAG = sys.argv[1] if len(sys.argv) > 1 else ""
def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []
def js(p): return json.load(open(p)) if os.path.exists(p) else {}
um = {}
for r in jl(os.path.join(L, f"unit_metrics_conf{TAG}.jsonl")): um.setdefault(r["unit"], {})[str(r["step"])] = {k: v for k, v in r.items() if isinstance(v, (int, float))}
ad = {}
for p in glob.glob(os.path.join(L, f"adapt3_conf{TAG}", "*.json")):
    d = json.load(open(p)); ad[os.path.basename(p).split("_b0.6")[0]] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, dict) or kk == "J_DM"}) for k, v in d.items()}
bundle = dict(unit_metrics=um, fair_fits={k: dict(rho_PM=v["fams"]["PM"]["params"]["rho"], KL=v["KL_model_vs_bayes_test"], boot=v.get("bootstrap", {}).get("PM")) for k, v in js(os.path.join(L, f"fair_fits_conf{TAG}.json")).items() if k != "BAYES_SELF"},
              adapt3=ad, c_causal=js(os.path.join(L, f"c_causal_conf{TAG}.json")), d_rows=js(os.path.join(L, f"d_rows_conf{TAG}.json")),
              d_routes=js(os.path.join(L, f"d_routes_conf{TAG}.json")), d_das=js(os.path.join(L, f"d_das_conf{TAG}.json")))
json.dump(bundle, open(os.path.join(L, f"bundle_13c{TAG}.json"), "w"), indent=1); print("bundle", {k: len(v) for k, v in bundle.items()})
