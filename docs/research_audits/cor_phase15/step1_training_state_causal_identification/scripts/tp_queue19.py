"""XIX: build a transplant queue for matched step-100 state pairs.  Each line: compose19.py (state construction) && xix_train.py branch
continuation (plain training, steps 100 -> STOP) && frozen measurement.  Usage: tp_queue19.py SPEC.json OUTQUEUE
SPEC: {"parent", "stream", "measure_out", "stop", "pairs": [{"name", "L", "W", "D", "B": dir, "I": dir}], "arms": [{"name", "base": "B"|"I", "groups", "opt", "perturb"}]}"""
import json, sys
S = json.load(open(sys.argv[1])); lines = []
X18 = "../step1_pathway_selection_causal_identification/scripts/measure18.py"
for p in S["pairs"]:
    for a in S["arms"]:
        base, other = (p["B"], p["I"]) if a["base"] == "B" else (p["I"], p["B"])
        if a.get("donor"): other = p[a["donor"]]                       # foreign-donor control: groups from another trajectory's state
        tag = f"tp_{p['name']}_{a['name']}"; suf = f"_{p['name']}_{a['name']}"
        comp = f"python3 scripts/compose19.py {base} {other} {tag} --params-from-I {a['groups']} --opt {a['opt']}" + (f" --perturb {a['perturb']}" if a.get("perturb") is not None else "") + (f" --alpha {a['alpha']}" if a.get("alpha") is not None else "") + (" --foreign-donor" if a.get("donor") else "")
        run = f"FX_BRANCH={tag}:100 FX_STOP={S['stop']} FX_TAGSUF={suf} python3 scripts/xix_train.py {S['parent']} {p['L']} {p['W']} {p['D']}"
        rd = f"checkpoints/ft/{S['parent']}_fx_L{p['L']}_W{p['W']}_D{p['D']}_stop{S['stop']}_br100{suf}"
        lines.append(f"{comp} && {run} && python3 {X18} {rd} {S['parent']} {S['stream']} {S['measure_out']}")
open(sys.argv[2], "w").write("\n".join(lines) + "\n"); print(len(lines), "runs")
