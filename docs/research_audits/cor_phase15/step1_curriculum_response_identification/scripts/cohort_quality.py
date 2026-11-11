"""Archived form of the inline computation that produced logs/conf/cohort_quality.json (success rule and b* selection).
[R: ST9] Medians of T1 budget KL were computed over all eight parents in that file; the frozen text specifies successful parents.
Both give b* = 500; this script reports both."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "..")
rows = {}
for a in ("T1", "P"):
    for s in range(60, 68):
        m = f"c{'T' if a == 'T1' else 'P'}{s}"; rows[m] = (json.load(open(f"{B}/checkpoints/ft_{a}_s{s}.json"))["eval"]["acc"], json.load(open(f"{B}/logs/conf/model_{m}.json"))["quality"]["KL_bayes"])
for s in range(60, 68):
    for b in (250, 500):
        m = f"bcT{s}_b{b}"; rows[m] = (json.load(open(f"{B}/checkpoints/budget/cT{s}_b{b}.json"))["eval"]["acc"], json.load(open(f"{B}/logs/conf/model_{m}.json"))["quality"]["KL_bayes"])
succ = lambda m: rows[m][0] >= 0.6595
okT = [s for s in range(60, 68) if succ(f"cT{s}")]; Pm = np.median([rows[f"cP{s}"][1] for s in range(60, 68) if succ(f"cP{s}")])
out = {}
for tag, ss in (("successful_parents", okT), ("all_parents", list(range(60, 68)))):
    med = {b: float(np.median([rows[f"bcT{s}_b{b}"][1] for s in ss])) for b in (250, 500)}; med[2000] = float(np.median([rows[f"cT{s}"][1] for s in ss]))
    out[tag] = dict(T_median_KL=med, b_star=min(med, key=lambda b: abs(np.log(med[b]) - np.log(Pm))))
out["P_median_KL_successful"] = float(Pm); out["success"] = {m: bool(succ(m)) for m in rows}
json.dump(out, open(f"{B}/logs/conf/cohort_quality_archived.json", "w"), indent=1); print(out["successful_parents"], out["all_parents"], Pm)
