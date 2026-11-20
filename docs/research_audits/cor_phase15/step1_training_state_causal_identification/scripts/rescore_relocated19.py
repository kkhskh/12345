"""Rescore the frozen confirmation from a relocated copy of the package (clean extraction).  The frozen scorer (scripts/score19.py, unchanged)
checks that every run's recorded branch source (an absolute path written by the harness at run time) lies inside the package and equals the
intended state.  In a relocated copy those absolute paths still name the original location.  This wrapper (not part of the freeze) maps the
original package root to the current one when run_config.json files are read, and nothing else; every other check is the frozen code path.
Usage: rescore_relocated19.py ORIGINAL_ROOT OUT.json"""
import os, sys, json, types
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..")); sys.path.insert(0, H)
import score19
ORIG, out = os.path.abspath(sys.argv[1]), sys.argv[2]
shim = types.ModuleType("json"); shim.__dict__.update(json.__dict__)
def load(f, *a, **k):
    d = json.load(f, *a, **k)
    if isinstance(d, dict) and isinstance(d.get("branch_dir"), str) and d["branch_dir"].startswith(ORIG): d["branch_dir"] = R + d["branch_dir"][len(ORIG):]
    return d
shim.load = load; score19.json = shim
D = json.load(open(os.path.join(R, "logs", "conf", "design19.json"))); recs = [json.loads(l) for l in open(os.path.join(R, "logs", "conf", "measure_c19.jsonl")) if l.strip()]
errs, M, dup = score19.provenance(D, recs, os.path.join(R, "checkpoints", "ft"), files=True)
if errs: json.dump(dict(verdict="PROVENANCE_FAIL", errors=errs), open(out, "w"), indent=1); print("PROVENANCE_FAIL", errs[:5]); sys.exit(2)
S = score19.score(D, M, dict(score19.CFG), score19.matching(D, os.path.join(R, "checkpoints", "ft"))); S["verdict"] = "SCORED (relocated)"
json.dump(S, open(out, "w"), indent=1); print("SCORED (relocated)", {k: v["band"] for k, v in S["hypotheses"].items()} == {k: v["band"] for k, v in json.load(open(os.path.join(R, "logs", "conf", "score_c19.json")))["hypotheses"].items()})
