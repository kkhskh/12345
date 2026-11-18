"""EXPLORATORY (post-scoring; review C finding on EXTENDED section 6): optimizer and confidence summaries from the inline trajectories.
Plateau = points at step >= 500 with F_G < 0.2 and (if acquired) step < t_A - 100; transition = t_A - 100 .. t_A + 100;
confidence pre = t_A - 150 .. t_A - 50, post = t_A + 50 .. t_A + 150.  -> logs/expl/optimizer_summary.json"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, ".."); S = json.load(open(os.path.join(P, "logs", "score_XVI.json"))); out = {}
for k, m in S["models"].items():
    L = [json.loads(l) for l in open(os.path.join(P, k, "traj.jsonl"))]; tA = m["t_A"]
    pl = [x for x in L if x["step"] >= 500 and x["F_G"] < 0.2 and (tA is None or x["step"] < tA - 100)]; tr = [x for x in L if tA is not None and tA - 100 <= x["step"] <= tA + 100]
    f = lambda Z, key: float(np.mean([x[key] for x in Z if x[key] is not None])) if Z else None
    out[k.split("/")[-1]] = dict(t_A=tA, plateau_points=len(pl), plateau_grad_norm=f(pl, "grad_norm"), plateau_update_norm=f(pl, "update_norm"), plateau_conf=f(pl, "conf"),
        transition_grad_norm=f(tr, "grad_norm"), transition_update_norm=f(tr, "update_norm"),
        conf_pre=f([x for x in L if tA and tA - 150 <= x["step"] < tA - 50], "conf"), conf_post=f([x for x in L if tA and tA + 50 <= x["step"] <= tA + 150], "conf"),
        param_norm_start=L[0]["param_norm"], param_norm_end=L[-1]["param_norm"])
json.dump(out, open(os.path.join(P, "logs", "expl", "optimizer_summary.json"), "w"), indent=1)
r = lambda v: "-" if v is None else f"{v:.3f}"
for k, v in out.items(): print(k.ljust(16), v["t_A"], "plateau gn", r(v["plateau_grad_norm"]), "un", r(v["plateau_update_norm"]), "| trans gn", r(v["transition_grad_norm"]), "un", r(v["transition_update_norm"]), "| conf", r(v["conf_pre"]), "->", r(v["conf_post"]))
