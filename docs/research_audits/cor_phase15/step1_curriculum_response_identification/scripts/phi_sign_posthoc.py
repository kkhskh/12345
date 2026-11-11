"""EXPLORATORY (post hoc, found after the confirmatory outcomes): monotonicity of the fitted reliability evidence at perfect
agreement, s = phi_rel(5) - phi_rel(4), from the frozen Bayes-mixture fits (model_*.json), for every development and cohort
model, with KL; Fisher exact test of sign by curriculum among full-budget models (cohort only, and cohort + development).
Not a confirmatory test: it is a candidate prospective prediction (OPEN_QUESTIONS.md)."""
import glob, json, os
from scipy.stats import fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); rows = []
for f in sorted(glob.glob(os.path.join(HERE, "..", "logs", "*", "model_*.json"))):
    d = json.load(open(f)); th = d["fits"]["mix"]["theta"]; m = d["model"]; ph = f.split(os.sep)[-2]
    rows.append(dict(phase=ph, model=m, s=th[4] - th[3], KL=d["quality"]["KL_bayes"], full_budget=not m.startswith("b")))
out = dict(rows=rows)
for tag, sel in (("cohort", lambda r: r["phase"] == "conf"), ("cohort+dev", lambda r: True)):
    T = [r for r in rows if sel(r) and r["full_budget"] and ("T" in r["model"][:2])]; P = [r for r in rows if sel(r) and r["full_budget"] and ("P" in r["model"][:2])]
    tab = [[sum(r["s"] > 0 for r in T), sum(r["s"] <= 0 for r in T)], [sum(r["s"] > 0 for r in P), sum(r["s"] <= 0 for r in P)]]
    out[tag] = dict(table_T_pos_neg_P_pos_neg=tab, fisher_p=float(fisher_exact(tab)[1]))
json.dump(out, open(os.path.join(HERE, "..", "logs", "phi_sign_posthoc.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
