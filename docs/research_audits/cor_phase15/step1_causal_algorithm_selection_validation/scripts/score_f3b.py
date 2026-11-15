"""XIII-D freeze-3b scorer (P10).  Reads LOGDIR/unit_metrics_f3b.jsonl -> LOGDIR/score_f3b.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
P = json.load(open(os.path.join(HERE, "predictions_13d_f3b.json")))["predictions"][0]; M = {}
try:
    for l in open(os.path.join(LD, "unit_metrics_f3b.jsonl")): r = json.loads(l); M[(r["unit"], r["step"])] = r
except Exception: pass
def cls(u):
    r = M.get((u, 1211))
    if r is None: return None, None
    c = "EARLY" if (r["mix_mass"] < 0.5 and r["lik_RO1"] >= 0.4) else "NOT-EARLY" if r["mix_mass"] >= 0.5 else "UNCLASSIFIED"
    return c, dict(lik=r["lik_RO1"], mix=r["mix_mass"], KL=r["KL"], competent=r["KL"] <= 0.05)
refs_ok = all(cls(u)[0] == w for u, w in P["refs"].items()); arms = []
for k, (u, want) in P["cells"].items():
    c, d = cls(u); arms.append(dict(arm=k, unit=u, want=want, got=c, detail=d, diagnostic=bool(refs_ok and d and d["competent"]), passed=(c == want) if c else False))
diag = [a for a in arms if a["diagnostic"]]; missing = any(a["got"] is None for a in arms)
v = "FALSIFIED" if missing else ("NON-DIAGNOSTIC" if not diag else ("CONFIRMED" if all(a["passed"] for a in diag) else "FALSIFIED"))
desc = {u: cls(u) for u in ("uP163:fx:21:20:21:stop1211", "uP164:s:21", "uP164:s:20")}
json.dump(dict(scored=dict(P10=dict(arms=arms, refs_ok=refs_ok, verdict=v)), summary=dict(P10=v), descriptive=desc), open(os.path.join(LD, "score_f3b.json"), "w"), indent=1, default=str)
print(dict(P10=v), refs_ok, [(a["arm"], a["got"]) for a in arms])
