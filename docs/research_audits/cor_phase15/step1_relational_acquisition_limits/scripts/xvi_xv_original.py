"""EXPLORATORY: the frozen XVI acquisition and stage definitions (score_XVI.model) applied to XV's original-schedule runs, from the
XVI diagnostics recomputed offline on XV's snapshots (logs/expl/xvdiag_XV_*.json; snapshots every 25 steps to 2000, every 250 to 4000,
so the 3-consecutive-point rule spans 50 steps early and 500 steps late).  Exposure: XV's exact cosine schedule.  -> logs/expl/xv_original_arm.json"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); import optax
s = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, 4000); cum = {}; c = 0.0
for i in range(4000): c += float(s(i)); cum[i + 1] = c
out = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "logs", "expl", "xvdiag_XV_*.json"))):
    d = json.load(open(f)); R = d["refs"]; L = d["traj"]; KLoa, KLo1, Aref = R["ordera"]["KL"], R["order1"]["KL"], R["ordera"]["A_conf"]
    acq = lambda x: x["F_G"] >= 0.5 and x["KL"] <= 0.75 * KLoa
    iA = next((i for i in range(len(L) - 2) if acq(L[i]) and acq(L[i + 1]) and acq(L[i + 2])), None)
    r = dict(t_A=L[iA]["step"] if iA is not None else None, E_A=cum[L[iA]["step"]] if iA is not None else None, t_first=next((x["step"] for x in L if x["F_G"] >= 0.2), None),
             max_FG=max(x["F_G"] for x in L), final=dict(KL=L[-1]["KL"], F_G=L[-1]["F_G"], A_conf=L[-1]["A_conf"]), KL_oa=KLoa)
    if iA is not None:
        ok = [x["F_G"] <= 0.1 and x["A_conf"] is not None and x["A_conf"] >= 0.5 * Aref and x["KL"] <= KLo1 - 0.5 * (KLo1 - KLoa) for x in L[:iA]]; run = best = 0
        for v in ok: run = run + 1 if v else 0; best = max(best, run)
        r["stage"], r["stage_run"] = best >= 4, best
        lo = max([L[j]["step"] for j in range(iA + 1) if L[j]["F_G"] <= 0.1], default=0); hi = next((L[j]["step"] for j in range(iA, len(L)) if L[j]["F_G"] >= 0.9), None); r["width"] = hi - lo if hi else None
    out[os.path.basename(f)[10:-5]] = r
json.dump(dict(total_exposure=cum[4000], runs=out), open(os.path.join(HERE, "..", "logs", "expl", "xv_original_arm.json"), "w"), indent=1)
for k, v in out.items(): print(k, v["t_A"], None if v["E_A"] is None else round(v["E_A"], 2), "first", v["t_first"], "maxFG %.2f" % v["max_FG"], "stage", v.get("stage"), v.get("stage_run"), "width", v.get("width"), "final KL %.3f FG %.2f" % (v["final"]["KL"], v["final"]["F_G"]))
print("total XV exposure", round(cum[4000], 4))
