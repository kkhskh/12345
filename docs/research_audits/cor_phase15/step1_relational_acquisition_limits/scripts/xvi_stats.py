"""XVI post-scoring statistics (descriptive and exploratory; the frozen verdicts are in logs/score_XVI.json).  -> logs/xvi_stats.json
  P3: unpaired Mann-Whitney (exact) and paired sign summary; observed ratio range.
  Arms on the exposure axis: fresh constant (XVI), continuation (XVI, total exposure incl. XV's 2.0), XV original cosine (exploratory, from
  logs/expl/xv_original_arm.json): acquisition proportion by exposure 1.0, 2.0, 3.0, 4.0, 5.8 (the continuations end at 5.8995) (censoring respected: a censored run counts as
  not acquired only if its observed exposure reaches the threshold; otherwise it is excluded at that threshold).
  Complexity (exploratory): arity-2 (XV, cosine) vs arity-3 (XVI fresh constant, same nu = 1/3 and matched value) exposure at acquisition."""
import json, os
import numpy as np
from scipy.stats import mannwhitneyu, binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); LOG = os.path.join(HERE, "..", "logs")
S = json.load(open(os.path.join(LOG, "score_XVI.json"))); X = json.load(open(os.path.join(LOG, "expl", "xv_original_arm.json")))
M = {k.split("/")[-1]: v for k, v in S["models"].items()}; out = {}
tl = [M[f"WB_low_s{s}"]["t_A"] for s in (501, 502, 503, 504)]; th = [M[f"WB_high_s{s}"]["t_A"] for s in (501, 502, 503, 504)]
r = [float(np.log(a / b)) for a, b in zip(tl, th)]
out["P3"] = dict(t_low=tl, t_high=th, ratios=[float(np.exp(x)) for x in r], mean_t_low=float(np.mean(tl)), mean_t_high=float(np.mean(th)),
                 mannwhitney_two_sided_p=float(mannwhitneyu(tl, th, alternative="two-sided", method="exact").pvalue),
                 sign_test_p=float(binomtest(sum(x > 0 for x in r), len(r), 0.5).pvalue), pairs_above_ln2=sum(x >= np.log(2) for x in r),
                 xv_A_same_condition_cosine_t_A=[X["runs"][f"A_s20{i}"]["t_A"] for i in (1, 2, 3, 4)])
arms = {"XVI_fresh_const_C": [(M[f"WA_const_s{s}"]["E_total_A"], M[f"WA_const_s{s}"]["E_total_end"]) for s in (401, 402, 403, 404)],
        "XVI_cont_C": [(M[f"WA_cont_C_s{s}"]["E_total_A"], M[f"WA_cont_C_s{s}"]["E_total_end"]) for s in (201, 202, 203)],
        "XVI_cont_D": [(M[f"WA_cont_D_s{s}"]["E_total_A"], M[f"WA_cont_D_s{s}"]["E_total_end"]) for s in (201, 202, 204)],
        "XV_cosine_C": [(X["runs"][f"C_s20{i}"]["E_A"], X["total_exposure"]) for i in (1, 2, 3, 4)],
        "XV_cosine_D": [(X["runs"][f"D_s20{i}"]["E_A"], X["total_exposure"]) for i in (1, 2, 3, 4)]}
out["acquisition_by_exposure"] = {a: {str(e): (lambda ok, n: f"{ok}/{n}")(sum(x is not None and x <= e for x, end in v), sum((x is not None and x <= e) or end >= e - 1e-6 for x, end in v)) for e in (1.0, 2.0, 3.0, 4.0, 5.8)} for a, v in arms.items()}
a2 = [X["runs"][f"B_s20{i}"]["E_A"] for i in (1, 2, 3, 4)]; a3 = [M[f"WA_const_s{s}"]["E_A"] for s in (401, 402, 403, 404)]
out["complexity_exposure_nu_1_3"] = dict(arity2_XV_B_cosine=a2, arity3_XVI_fresh_const=a3, ratio_of_medians=float(np.median(a3) / np.median(a2)), ratio_range=[min(a3) / max(a2), max(a3) / min(a2)])
a2d = [X["runs"][f"A_s20{i}"]["E_A"] for i in (1, 2, 3, 4)]; out["complexity_exposure_nu_0_05"] = dict(arity2_XV_A_cosine=a2d, arity3_cont_D_total=[x for x, _ in arms["XVI_cont_D"]], arity3_XV_D_cosine=[x for x, _ in arms["XV_cosine_D"]])
json.dump(out, open(os.path.join(LOG, "xvi_stats.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
