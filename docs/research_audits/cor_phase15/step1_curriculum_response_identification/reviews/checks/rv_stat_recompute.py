"""Statistical reviewer: recompute frozen verdicts from logs/conf/*.json (pair similarities, per-model values) and probe
inference sensitivity.  Reads only archived JSON; no model forward passes.  Output: logs/review_checks/rv_stat_recompute.out"""
import itertools, json, os
import numpy as np
from scipy.stats import wilcoxon, fisher_exact, norm
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
L = lambda f: json.load(open(os.path.join(H, "logs", "conf", f + ".json")))
out = []; P = lambda *a: out.append(" ".join(str(x) for x in a))


def smat(d, key="P1_primary_N_input_clogit_pearson"):
    names = d["models"]; n = len(names); S = np.eye(n); pr = d[key]["pairs"] if key in d else d["robustness"][key]["pairs"]
    for i, j in itertools.combinations(range(n), 2): S[i, j] = S[j, i] = pr[f"{names[i]}~{names[j]}"]
    arm = np.array([1 if "T" in m[:3] else 0 for m in names]); return names, S, arm


def coh(S, lab):
    t = [S[i, j] for i, j in itertools.combinations(np.where(lab == 1)[0], 2)]
    p = [S[i, j] for i, j in itertools.combinations(np.where(lab == 0)[0], 2)]
    return np.mean(t) - np.mean(p), np.mean(t), np.mean(p)


def perm(S, arm):
    n = len(arm); obs = coh(S, arm)[0]; null = []
    for c in itertools.combinations(range(n), arm.sum()):
        l = np.zeros(n, int); l[list(c)] = 1; null.append(coh(S, l)[0])
    null = np.array(null); return obs, (null >= obs - 1e-12).mean(), (np.abs(null) >= abs(obs) - 1e-12).mean(), len(null)


def boot(S, arm, seed, reps=4000, distinct=False):
    r = np.random.default_rng(seed); T_, P_ = np.where(arm == 1)[0], np.where(arm == 0)[0]; v = []
    for _ in range(reps):
        dt, dp = r.choice(T_, len(T_)), r.choice(P_, len(P_))
        if distinct: dt, dp = np.unique(dt), np.unique(dp)
        ct = [S[i, j] for i, j in itertools.combinations(dt, 2) if i != j]; cp = [S[i, j] for i, j in itertools.combinations(dp, 2) if i != j]
        if ct and cp: v.append(np.mean(ct) - np.mean(cp))
    return np.array(v)


# ---------------------------------------------------------------- 1. P1 recompute and inference sensitivity
d = L("primary"); names, S, arm = smat(d); obs, p1, p2, n = perm(S, arm); c = coh(S, arm)
P("== P1 primary (6+6)"); P("coh_T %.4f coh_P %.4f DELTA %.4f | perm one-sided p %.4f two-sided %.4f n_relabel %d" % (c[1], c[2], obs, p1, p2, n))
P("logged:", d["P1_primary_N_input_clogit_pearson"]["coherence"], d["P1_primary_N_input_clogit_pearson"]["ci95"])
for sd in (0, 1, 2, 3):
    v = boot(S, arm, sd); P("bootstrap seed %d: 95%% CI [%.3f, %.3f]  90%% CI [%.3f, %.3f]  P(boot<=0)=%.3f  SE %.3f" %
                            (sd, *np.percentile(v, [2.5, 97.5]), *np.percentile(v, [5, 95]), (v <= 0).mean(), v.std()))
v = boot(S, arm, 0, distinct=True); P("bootstrap distinct-models-only 95%% CI [%.3f, %.3f]" % tuple(np.percentile(v, [2.5, 97.5])))
# leave-one-model-out influence
for k in range(len(names)):
    keep = np.array([i for i in range(len(names)) if i != k]); P("  drop %s: DELTA %.3f" % (names[k], coh(S[np.ix_(keep, keep)], arm[keep])[0]))
# jackknife-of-models SE (delete one model per arm, U-statistic style)
jk = []
for k in range(len(names)):
    keep = np.array([i for i in range(len(names)) if i != k]); jk.append(coh(S[np.ix_(keep, keep)], arm[keep])[0])
jk = np.array(jk); nn = len(jk); se_jk = np.sqrt((nn - 1) / nn * ((jk - jk.mean()) ** 2).sum())
P("jackknife SE %.3f -> normal one-sided p %.4f; approx power at alpha=.025 (one-sided) for true DELTA 0.149: %.2f, 0.30: %.2f, 0.45: %.2f" %
  (se_jk, 1 - norm.cdf(obs / se_jk), *(norm.cdf(x / se_jk - 1.96) for x in (0.149, 0.30, 0.45))))
# same-seed cross-arm pairs (shared T4-transfer seed) vs different-seed cross pairs
ss, ds = [], []
for i, j in itertools.combinations(range(len(names)), 2):
    if arm[i] != arm[j]: (ss if names[i][2:] == names[j][2:] else ds).append(S[i, j])
P("cross-arm similarity: same seed (shared transfer RNG) mean %.3f (n=%d) vs different seed %.3f (n=%d)" % (np.mean(ss), len(ss), np.mean(ds), len(ds)))
# T-T pair list
P("T-T pairs:", sorted(round(S[i, j], 3) for i, j in itertools.combinations(np.where(arm == 1)[0], 2)))
P("P-P pairs:", sorted(round(S[i, j], 3) for i, j in itertools.combinations(np.where(arm == 0)[0], 2)))

# ---------------------------------------------------------------- 2. all runs, budget arms
for f in ("allruns", "budget_b500", "budget_b250"):
    dd = L(f); nm_, S_, a_ = smat(dd); o, q1, q2, nr = perm(S_, a_); v = boot(S_, a_, 0)
    P("== %s: DELTA %.4f coh_T %.4f coh_P %.4f perm1 %.4f perm2 %.4f (n %d) boot95 [%.3f, %.3f]" % (f, o, *coh(S_, a_)[1:], q1, q2, nr, *np.percentile(v, [2.5, 97.5])))

# ---------------------------------------------------------------- 3. b* rule
cq = L("cohort_quality"); rows = cq["rows"]; succT = [61, 62, 63, 64, 65, 67]; succP = [61, 62, 63, 64, 66, 67]
Pm = np.median([rows[f"cP{s}"][1] for s in succP]); Pm_all = np.median([rows[f"cP{s}"][1] for s in range(60, 68)])
P("== b*: P-count median KL successful %.4f (all 8: %.4f); logged P_median %.4f, logged T medians %s" % (Pm, Pm_all, cq["P_median_KL"], cq["T_median_KL"]))
for tag, seeds in (("successful", succT), ("all8", range(60, 68))):
    med = {b: np.median([rows[(f"cT{s}" if b == 2000 else f"bcT{s}_b{b}")][1] for s in seeds]) for b in (250, 500, 2000)}
    dist = {b: abs(np.log(m) - np.log(Pm)) for b, m in med.items()}
    P("  T medians (%s): %s  |log dist| %s  -> b* %d" % (tag, {b: round(m, 4) for b, m in med.items()}, {b: round(x, 3) for b, x in dist.items()}, min(dist, key=dist.get)))
P("  transfer acc of b500 variants (successful parents):", {s: round(rows[f"bcT{s}_b500"][0], 4) for s in succT}, "threshold 0.6595")
P("  KL b500 T:", sorted(round(rows[f"bcT{s}_b500"][1], 3) for s in succT), " P:", sorted(round(rows[f"cP{s}"][1], 3) for s in succP))

# ---------------------------------------------------------------- 4. P2(a) overlap
kl = d["P2_adjusted"]["KL_by_arm"]; lo, hi = max(min(kl["T1"]), min(kl["P-count"])), min(max(kl["T1"]), max(kl["P-count"]))
P("== P2a overlap region [%.3f, %.3f]: T in region %d, P in region %d" % (lo, hi, sum(lo <= x <= hi for x in kl["T1"]), sum(lo <= x <= hi for x in kl["P-count"])))
P("   adjusted DELTA %.4f p %.3f CI %s" % (d["P2_adjusted"]["DELTA_adj_KL"], d["P2_adjusted"]["perm_p_KL"], d["P2_adjusted"]["ci95_KL"]))

# ---------------------------------------------------------------- 5. P3, P5
op = d["P3_operator_families"]["per_model"]; b5 = L("budget_b500")["P3_operator_families"]["per_model"]
for tag, dd in (("2000", op), ("b500", b5)):
    T = [v["mix_rel"] for k, v in dd.items() if "T" in k[:3]]; Pp = [v["mix_rel"] for k, v in dd.items() if "T" not in k[:3]]
    raw_T = [v["mix"] for k, v in dd.items() if "T" in k[:3]]; raw_P = [v["mix"] for k, v in dd.items() if "T" not in k[:3]]
    ceil = {k: round(v["ceiling"], 4) for k, v in dd.items()}
    # rank-based (Mann-Whitney exact via permutation of ranks) on mix_rel and on raw mix r2
    def mw(a, b):
        allv = np.r_[a, b]; rk = allv.argsort().argsort(); obs = rk[:len(a)].mean() - rk[len(a):].mean(); nul = []
        for cc in itertools.combinations(range(len(allv)), len(a)):
            l = np.zeros(len(allv), bool); l[list(cc)] = True; nul.append(rk[l].mean() - rk[~l].mean())
        return (np.abs(nul) >= abs(obs) - 1e-12).mean()
    P("== P3b %s: mix_rel T %s P %s | rank perm p2 %.4f | raw mix r2 rank p2 %.4f | ceilings %s" % (tag, np.round(T, 3).tolist(), np.round(Pp, 3).tolist(), mw(T, Pp), mw(raw_T, raw_P), ceil))
ex = np.array(d["P5_pathways"]["excess_N_vs_R"]); P("== P5a Wilcoxon one-sided recomputed p %.5f (logged %.5f); T>0.1: %d/6, P>0.1: %d/6" %
                                                      (wilcoxon(ex, alternative="greater").pvalue, d["P5_pathways"]["wilcoxon_p_excess_gt0"], (ex[arm == 1] > 0.1).sum(), (ex[arm == 0] > 0.1).sum()))
P("   P5a P-count only Wilcoxon one-sided p %.4f; T1 only p %.4f" % (wilcoxon(ex[arm == 0], alternative="greater").pvalue, wilcoxon(ex[arm == 1], alternative="greater").pvalue))
nul = []
for cc in itertools.combinations(range(12), 6):
    l = np.zeros(12, bool); l[list(cc)] = True; nul.append(ex[l].mean() - ex[~l].mean())
o5 = ex[arm == 1].mean() - ex[arm == 0].mean(); P("   post hoc arm difference in excess %.3f, exact perm one-sided p %.4f (NOT frozen)" % (o5, (np.array(nul) >= o5 - 1e-12).mean()))
pb = L("p5b"); f = pb["f_Cm"]; fT = [f[m] for m in names if "T" in m[:3]]; fP = [f[m] for m in names if "T" not in m[:3]]
nul = []
allf = np.r_[fT, fP]
for cc in itertools.combinations(range(12), 6):
    l = np.zeros(12, bool); l[list(cc)] = True; nul.append(allf[l].mean() - allf[~l].mean())
o = np.mean(fT) - np.mean(fP); P("== P5b diff %.4f two-sided p %.4f (logged %.4f, %.4f)" % (o, (np.abs(nul) >= abs(o) - 1e-12).mean(), pb["diff_T_minus_P"], pb["perm_p_two_sided"]))
npj = L("null_primary"); P("== P3c TP %.4f TT %.4f PP %.4f (criterion >= 0.9)" % (npj["mean_coef_corr_TP"], npj["mean_coef_corr_TT"], npj["mean_coef_corr_PP"]))
P("== P1b", {k: v for k, v in d["P1b_one_factor_N"].items() if k not in ("pairs", "loadings")})
P("== P4 Nabs", d["family_Nabs"]["coherence"]["DELTA"], d["family_Nabs"]["ci95"])

# ---------------------------------------------------------------- 6. phi-sign post hoc: the same quality confound
ph = json.load(open(os.path.join(H, "logs", "phi_sign_posthoc.json")))["rows"]
cf = {r["model"]: r for r in ph if r["phase"] == "conf"}
for b in (250, 500):
    s = [cf[f"bcT{x}_b{b}"]["s"] for x in range(60, 68)]; P("== phi-sign T1 budget %d (8 parents): s>0 in %d/8; values %s" % (b, sum(x > 0 for x in s), np.round(s, 3).tolist()))
sT = [cf[f"cT{x}"]["s"] for x in range(60, 68)]; sP = [cf[f"cP{x}"]["s"] for x in range(60, 68)]
P("   T1 full budget s:", np.round(sT, 3).tolist(), "P-count:", np.round(sP, 3).tolist())
s5 = [cf[f"bcT{x}_b500"]["s"] for x in succT]; sPs = [cf[f"cP{x}"]["s"] for x in succP]
tab = [[sum(x > 0 for x in s5), sum(x <= 0 for x in s5)], [sum(x > 0 for x in sPs), sum(x <= 0 for x in sPs)]]
P("   quality-matched (b500 successful parents vs successful P) table", tab, "Fisher p %.3f" % fisher_exact(tab)[1])
allr = [r for r in ph if r["phase"] == "conf"]; kls = np.array([r["KL"] for r in allr]); ss_ = np.array([r["s"] for r in allr])
P("   corr(s, log KL) over all 32 conf models %.3f" % np.corrcoef(ss_, np.log(kls))[0, 1])

# ---------------------------------------------------------------- 7. seed-paired (restricted) permutation: cT_s and cP_s share init + transfer seed
d = L("primary"); names, S, arm = smat(d); seeds = [m[-2:] for m in names]; obs = coh(S, arm)[0]
units = sorted(set(seeds)); null = []
for bits in itertools.product((0, 1), repeat=len(units)):
    perm_idx = list(range(len(names)))
    for u, b in zip(units, bits):
        if b:
            ii = [k for k in range(len(names)) if seeds[k] == u]
            if len(ii) == 2: perm_idx[ii[0]], perm_idx[ii[1]] = ii[1], ii[0]
    lab = arm[perm_idx]
    if lab.sum() == arm.sum(): null.append(coh(S, lab)[0])
null = np.array(null); P("== seed-paired permutation (swap arm labels within complete seed pairs): distinct relabellings %d, one-sided p %.4f" % (len(np.unique(np.round(null, 12))), (null >= obs - 1e-12).mean()))
# ---------------------------------------------------------------- 8. P2(b) sensitivity: apply the success threshold to the b500 variants too
d5 = L("budget_b500"); n5, S5, a5 = smat(d5); keep = np.array([k for k, m in enumerate(n5) if m != "bcT67_b500"])
o, q1, q2, nr = perm(S5[np.ix_(keep, keep)], a5[keep]); v = boot(S5[np.ix_(keep, keep)], a5[keep], 0)
P("== P2(b) without bcT67_b500 (transfer acc 0.651 < 0.6595): DELTA_q %.4f perm1 %.3f perm2 %.3f boot95 [%.3f, %.3f]" % (o, q1, q2, *np.percentile(v, [2.5, 97.5])))
P("   two-sided perm p for the primary-budget reversal (Delta_q=-0.10): %.3f" % perm(S5, a5)[2])
open(os.path.join(H, "logs", "review_checks", "rv_stat_recompute.out"), "w").write("\n".join(out) + "\n"); print("\n".join(out[-4:]))
# ---------------------------------------------------------------- 9. P6: within-parent paired bootstrap of coh_T(2000) - coh_T(500) (parents resampled jointly)
nA, SA, aA = smat(L("primary")); nB, SB, aB = smat(L("budget_b500"))
tA = np.where(aA == 1)[0]; tB = np.where(aB == 1)[0]; assert [nA[i][-2:] for i in tA] == [nB[i][3:5] for i in tB]
r = np.random.default_rng(5); dv = []
for _ in range(4000):
    s_ = r.choice(6, 6); pr = [(i, j) for i, j in itertools.combinations(s_, 2) if i != j]
    if pr: dv.append(np.mean([SA[tA[i], tA[j]] for i, j in pr]) - np.mean([SB[tB[i], tB[j]] for i, j in pr]))
P("== P6 paired parent bootstrap coh_T(2000)-coh_T(500): %.3f, 95%% CI [%.3f, %.3f]" % (np.mean([SA[tA[i], tA[j]] for i, j in itertools.combinations(range(6), 2)]) - np.mean([SB[tB[i], tB[j]] for i, j in itertools.combinations(range(6), 2)]), *np.percentile(dv, [2.5, 97.5])))
open(os.path.join(H, "logs", "review_checks", "rv_stat_recompute.out"), "w").write("\n".join(out) + "\n"); print(out[-1])
