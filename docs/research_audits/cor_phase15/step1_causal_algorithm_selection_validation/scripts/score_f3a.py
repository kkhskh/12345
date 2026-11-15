"""XIII-D freeze-3a scorer (P5a-c).  Reads LOGDIR/adapt4_f3a/uP165_s_20_b0.6_lr0.001.json and ..._lr0.003.json -> LOGDIR/score_f3a.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); LD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "logs")
P = json.load(open(os.path.join(HERE, "predictions_13d_f3a.json"))); U = P["unit"].replace(":", "_")
def load(lr):
    p = os.path.join(LD, "adapt4_f3a", f"{U}_b0.6_lr{lr}.json")
    try: return json.load(open(p))
    except Exception: return {}
D = dict(lr1=load("0.001"), lr3=load("0.003")); res = {}
for p in P["predictions"]:
    arms = []
    for lr, a, c, t in p["rules"]:
        try: v = float(D[lr][a]["J_DM"]["slope"]); ok = v >= t if c == ">=" else v < t
        except Exception: v, ok = None, None
        arms.append(dict(arm=f"{lr}:{a}{c}{t}", diagnostic=True, passed=bool(ok) if ok is not None else None, missing=ok is None, slope=v,
                         acc=D[lr].get(a, {}).get("acc") if isinstance(D[lr].get(a), dict) else None))
    res[p["id"]] = dict(arms=arms, verdict="CONFIRMED" if all(x["passed"] for x in arms) else "FALSIFIED")
desc = {f"{lr}:{a}": (D[lr].get(a, {}).get("J_DM", {}).get("slope"), D[lr].get(a, {}).get("acc")) for lr, a in (("lr1", "L0attn"), ("lr1", "L0mlp"))}
json.dump(dict(scored=res, summary={k: v["verdict"] for k, v in res.items()}, descriptive=desc), open(os.path.join(LD, "score_f3a.json"), "w"), indent=1)
print({k: v["verdict"] for k, v in res.items()}, desc)
