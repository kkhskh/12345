"""XIII-E A5 frozen rule: continuation set = the 8 preselected runs (W'=601, L'=701 for every anchor) + the 2 other runs whose mix_mass at 1211
is closest to 0.5 (ties: smaller |lik_RO1 - 0.4|).  Requires 1211 metrics for all 64 runs.  -> logs/manifest_A_cont.json and queue_Acont.txt"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LD = os.path.join(HERE, "..", "logs")
D = json.load(open(os.path.join(LD, "A_design.json"))); UM = {}
for l in open(os.path.join(LD, "unit_metrics_A.jsonl")): x = json.loads(l); UM[(x["unit"], x["step"])] = x
missing = [r["spec"] for r in D["runs"] if (r["spec"], 1211) not in UM]
if missing: print("REFUSED: missing 1211 metrics", missing[:3]); sys.exit(2)
fixed = D["continuation"]["fixed"]; rest = [r for r in D["runs"] if r["tag"] not in fixed]
rest.sort(key=lambda r: (abs(UM[(r["spec"], 1211)]["mix_mass"] - 0.5), abs(UM[(r["spec"], 1211)]["lik_RO1"] - 0.4)))
final = fixed + [r["tag"] for r in rest[:2]]
json.dump(dict(continuation_final=final, rule=D["continuation"]["rule_extra"]), open(os.path.join(LD, "manifest_A_cont.json"), "w"), indent=1)
q = []
for t in final:
    r = next(x for x in D["runs"] if x["tag"] == t); q.append(f"FX_BRANCH={t}:1211 python3 scripts/ft_factorial.py {r['parent']} {r['L']} {r['W']} {r['D']}")
open(os.path.join(LD, "queue_Acont.txt"), "w").write("\n".join(q) + "\n"); print(final)
