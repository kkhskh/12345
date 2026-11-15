"""XIII-D: final organisation (step 4000) of every non-intervened fine-tune measured so far, keyed by (parent, L, W, D).
Sources: XIII-C logs/unit_metrics*.jsonl (confirmation stream 88_100_000 preferred, else development stream 86_100_000) and this package's
logs/unit_metrics*.jsonl.  -> logs/known_outcomes.json"""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, ".."); X13C = os.path.join(PKG, "..", "step1_critical_relational_mechanism_resolution")
rows = {}
files = sorted(glob.glob(os.path.join(X13C, "logs", "unit_metrics*.jsonl")) + glob.glob(os.path.join(PKG, "logs", "unit_metrics*.jsonl")), key=lambda p: ("_conf" in p or "_expl" in p or PKG in p, p))
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r["step"] != 4000: continue
        p = r["unit"].split(":")
        if p[1] == "s": key = (p[0], int(p[2]), int(p[2]), int(p[2]))
        elif len(p) == 5: key = (p[0], int(p[2]), int(p[3]), int(p[4]))
        else: continue                                                     # interventions / replays excluded
        rows[key] = dict(lik_RO1=r["lik_RO1"], mix_mass=r["mix_mass"], route_Q=r.get("route_Q"), KL=r.get("KL"), src=os.path.basename(f))
json.dump({"|".join(map(str, k)): v for k, v in sorted(rows.items())}, open(os.path.join(PKG, "logs", "known_outcomes.json"), "w"), indent=1)
for k, v in sorted(rows.items()): print(k, round(v["lik_RO1"], 3), round(v["mix_mass"], 3), v["src"])
