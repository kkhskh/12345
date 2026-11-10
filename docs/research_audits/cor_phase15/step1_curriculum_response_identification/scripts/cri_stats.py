"""Model-level statistics for Investigation V.  Usage: cri_stats.py PHASE OUTNAME MODEL [MODEL ...]
Arms are read from model names (P*/cP* = P-count, T*/cT* = T1).  The statistical unit is the trained model.
For a response family F, a model's response is the change of centred logits at the scored decision, Δ_F = c_F − c_base,
on the generator-defined valid set ok_F (no model-dependent selection).  Pair similarity s(m, m′) = Pearson r over all
valid entries.  coh_A = mean s over pairs within arm A; DELTA = coh_T − coh_P.
Inference: (i) exact or Monte-Carlo permutation of arm labels over models (one-sided, DELTA > 0, the direction observed in
Investigation IV), (ii) model bootstrap: models resampled with replacement within arm, coherence over pairs of distinct
draws, 4000 replicates, percentile 95 % CI.  Pairwise dependence is respected because both procedures resample models.
Writes logs/PHASE/OUTNAME.json."""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.stats import spearmanr, rankdata
PH, OUTNAME, names = sys.argv[1], sys.argv[2], sys.argv[3:]
D = os.path.join(HERE, "..", "logs", PH); Gd = np.load(os.path.join(D, "generator.npz"))
Mz = {m: np.load(os.path.join(D, f"model_{m}.npz")) for m in names}
Mj = {m: json.load(open(os.path.join(D, f"model_{m}.json"))) for m in names}
import cri as _cri
arm = np.array([1 if _cri.kind_of(m) == "T" else 0 for m in names]); nm = len(names)
rng = np.random.default_rng(0)


def soft(c): e = np.exp(c - c.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def resp(m, F, space="clogit"):
    z = Mz[m]; b, a = z["c_base"], z[f"c_{F}"]
    if space == "clogit": return a - b
    if space == "margin": return np.stack([a[:, 0] - a[:, 1], a[:, 0] - a[:, 2], a[:, 1] - a[:, 2]], 1) - np.stack([b[:, 0] - b[:, 1], b[:, 0] - b[:, 2], b[:, 1] - b[:, 2]], 1)
    if space == "prob": return soft(a) - soft(b)
    if space == "zscore": d = a - b; return d / (d.std() + 1e-12)
    raise ValueError(space)


def sim_matrix(R, mask, metric="pearson"):
    S = np.eye(nm)
    for i, j in itertools.combinations(range(nm), 2):
        x, y = R[i][mask].ravel(), R[j][mask].ravel()
        if metric == "pearson": v = np.corrcoef(x, y)[0, 1]
        elif metric == "spearman": v = spearmanr(x, y)[0]
        elif metric == "sign": v = float(np.mean(np.sign(R[i][mask].argmax(-1) - 1e-9) == np.sign(R[j][mask].argmax(-1) - 1e-9)) * 0 + np.mean(R[i][mask].argmax(-1) == R[j][mask].argmax(-1)))
        elif metric == "cosine": v = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        S[i, j] = S[j, i] = v
    return S


def coh(S, lab, sel=None):
    idx = np.arange(nm) if sel is None else sel
    out = {}
    for a, nmk in ((1, "T"), (0, "P")):
        ii = [k for k in idx if lab[k] == a]
        prs = [(i, j) for i, j in itertools.combinations(ii, 2) if i != j]
        out[nmk] = float(np.mean([S[i, j] for i, j in prs])) if prs else np.nan
    cr = [S[i, j] for i in idx for j in idx if lab[i] == 1 and lab[j] == 0]
    out["cross"] = float(np.mean(cr)) if cr else np.nan
    out["DELTA"] = out["T"] - out["P"]; return out


def coh_boot(S, reps=4000):
    T_ = np.where(arm == 1)[0]; P_ = np.where(arm == 0)[0]; vals = []
    for _ in range(reps):
        dt, dp = rng.choice(T_, len(T_)), rng.choice(P_, len(P_))
        ct = [S[i, j] for i, j in itertools.combinations(dt, 2) if i != j]; cp = [S[i, j] for i, j in itertools.combinations(dp, 2) if i != j]
        if ct and cp: vals.append(np.mean(ct) - np.mean(cp))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def perm_p(S, stat_fn=lambda S, lab: coh(S, lab)["DELTA"], reps=20000):
    obs = stat_fn(S, arm); nT = arm.sum(); allc = list(itertools.combinations(range(nm), nT))
    labs = []
    if len(allc) <= reps:
        for c in allc: l = np.zeros(nm, int); l[list(c)] = 1; labs.append(l)
    else:
        for _ in range(reps): l = np.zeros(nm, int); l[rng.choice(nm, nT, replace=False)] = 1; labs.append(l)
    null = np.array([stat_fn(S, l) for l in labs]); return float(obs), float((null >= obs - 1e-12).mean()), len(labs)


def family_block(F, space="clogit", metric="pearson"):
    mask = Gd[f"ok_{F.replace('abs', '')}"].astype(bool); R = [resp(m, F, space) for m in names]; S = sim_matrix(R, mask, metric)
    c = coh(S, arm); obs, p, nperm = perm_p(S); ci = coh_boot(S)
    return dict(coherence=c, DELTA=c["DELTA"], ci95=ci, perm_p_one_sided=p, n_perm=nperm, n_valid=int(mask.sum()),
                pairs={f"{names[i]}~{names[j]}": float(S[i, j]) for i, j in itertools.combinations(range(nm), 2)}), S


OUT = dict(models=names, arm={m: ("T1" if a else "P-count") for m, a in zip(names, arm)})
# ---------------------------------------------------------------- P1 primary + output-space and metric robustness
blk, S_N = family_block("N"); OUT["P1_primary_N_input_clogit_pearson"] = blk
OUT["robustness"] = {}
for space in ("margin", "prob", "zscore"):
    OUT["robustness"][f"N|{space}|pearson"] = family_block("N", space)[0]
for metric in ("spearman", "cosine", "sign"):
    OUT["robustness"][f"N|clogit|{metric}"] = family_block("N", "clogit", metric)[0]
for F in ("N2", "R", "R2", "C1", "C2", "C3", "Nabs", "Rabs"):
    OUT[f"family_{F}"] = family_block(F)[0]
# ---------------------------------------------------------------- P1b: one-common-factor account of the pair correlations
def one_factor(F="N"):
    """r_m = λ_m f + ε_m with one factor f shared by ALL models (both arms).  For each pair (m, m′) the factor is estimated
    without m and m′ (mean of the other models' standardized responses); λ_m = corr(r_m, f_{-mm′}).  Predicted pair
    correlation λ_m λ_m′.  EXCESS = mean over within-arm pairs of (observed − predicted) minus the same over cross pairs.
    A curriculum-specific shared component predicts EXCESS > 0; a single common component plus model-specific noise predicts 0."""
    mask = Gd[f"ok_{F.replace('abs', '')}"].astype(bool); Rr = [resp(m, F)[mask].ravel() for m in names]
    Z = [(x - x.mean()) / (x.std() + 1e-12) for x in Rr]; res = {}; lam = {}
    for i, j in itertools.combinations(range(nm), 2):
        oth = [k for k in range(nm) if k not in (i, j)]; f = np.mean([Z[k] for k in oth], 0)
        li, lj = np.corrcoef(Z[i], f)[0, 1], np.corrcoef(Z[j], f)[0, 1]; obs = np.corrcoef(Z[i], Z[j])[0, 1]
        res[(i, j)] = (obs, li * lj); lam.setdefault(i, []).append(li); lam.setdefault(j, []).append(lj)
    def excess(lab):
        w = [o - p_ for (i, j), (o, p_) in res.items() if lab[i] == lab[j]]; c = [o - p_ for (i, j), (o, p_) in res.items() if lab[i] != lab[j]]
        return float(np.mean(w) - np.mean(c))
    def excess_T(lab):
        w = [o - p_ for (i, j), (o, p_) in res.items() if lab[i] == 1 and lab[j] == 1]; return float(np.mean(w))
    S_dummy = np.zeros((nm, nm))
    o1, p1, n1 = perm_p(S_dummy, lambda S, lab: excess(lab)); o2, p2, _ = perm_p(S_dummy, lambda S, lab: excess_T(lab))
    return dict(EXCESS_within_minus_cross=o1, perm_p=p1, EXCESS_TT=o2, perm_p_TT=p2, n_perm=n1,
                pairs={f"{names[i]}~{names[j]}": dict(observed=float(o), predicted=float(p_)) for (i, j), (o, p_) in res.items()},
                loadings={names[i]: float(np.mean(v)) for i, v in lam.items()})
OUT["P1b_one_factor_N"] = one_factor("N"); OUT["P1b_one_factor_Nabs"] = one_factor("Nabs")

# ---------------------------------------------------------------- magnitudes (5.1)
mag = {}
cBn, cB0 = Gd["cB_N"], Gd["cB_base"]
assert np.abs(cBn - cB0)[Gd["ok_N"].astype(bool)].max() < 1e-9           # Bayes response exactly 0 on N
for m in names:
    mk = Gd["ok_N"].astype(bool); dN = resp(m, "N")[mk]; mkR = Gd["ok_R"].astype(bool); dR = resp(m, "R")[mkR]
    dBR = (Gd["cB_R"] - Gd["cB_base"])[mkR]
    mag[m] = dict(mean_abs_N=float(np.abs(dN).mean()), sd_N=float(dN.std()), frac_N_below_0_05=float((np.abs(dN).max(-1) < 0.05).mean()),
                  mean_abs_R=float(np.abs(dR).mean()), mean_abs_R_bayes=float(np.abs(dBR).mean()),
                  ratio_N_to_R_bayes=float(np.abs(dN).mean() / np.abs(dBR).mean()))
OUT["magnitudes"] = mag
# ---------------------------------------------------------------- P2: control for approximation quality
qual = {m: Mj[m]["quality"] for m in names}; KL = np.array([qual[m]["KL_bayes"] for m in names]); LE = np.array([qual[m]["mean_abs_logodds_err"] for m in names])
def adj_delta(S, lab, cov=KL):
    pr = list(itertools.combinations(range(nm), 2)); y = np.array([S[i, j] for i, j in pr])
    TT = np.array([lab[i] == 1 and lab[j] == 1 for i, j in pr], float); PP = np.array([lab[i] == 0 and lab[j] == 0 for i, j in pr], float)
    Xd = np.c_[np.ones(len(pr)), TT, PP, [cov[i] + cov[j] for i, j in pr], [abs(cov[i] - cov[j]) for i, j in pr]]
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]; return float(b[1] - b[2])
obs, p, npm = perm_p(S_N, lambda S, lab: adj_delta(S, lab, KL)); obs2, p2, _ = perm_p(S_N, lambda S, lab: adj_delta(S, lab, LE))
bt = []
for _ in range(2000):
    T_ = rng.choice(np.where(arm == 1)[0], arm.sum()); P_ = rng.choice(np.where(arm == 0)[0], (arm == 0).sum()); sel = np.r_[T_, P_]
    Ss = S_N[np.ix_(sel, sel)]; lab = arm[sel]
    pr = [(i, j) for i, j in itertools.combinations(range(len(sel)), 2) if sel[i] != sel[j]]
    y = np.array([Ss[i, j] for i, j in pr]); cv = KL[sel]
    TT = np.array([lab[i] == 1 and lab[j] == 1 for i, j in pr], float); PP = np.array([lab[i] == 0 and lab[j] == 0 for i, j in pr], float)
    Xd = np.c_[np.ones(len(pr)), TT, PP, [cv[i] + cv[j] for i, j in pr], [abs(cv[i] - cv[j]) for i, j in pr]]
    if np.linalg.matrix_rank(Xd) == Xd.shape[1]: b = np.linalg.lstsq(Xd, y, rcond=None)[0]; bt.append(b[1] - b[2])
OUT["P2_adjusted"] = dict(DELTA_adj_KL=obs, perm_p_KL=p, DELTA_adj_logodds_err=obs2, perm_p_logodds_err=p2, n_perm=npm,
                          ci95_KL=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                          quality=qual, KL_by_arm={"T1": KL[arm == 1].tolist(), "P-count": KL[arm == 0].tolist()})
# ---------------------------------------------------------------- P3: Bayes-relevant response errors and operator families
mkR = Gd["ok_R"].astype(bool); dBR = (Gd["cB_R"] - Gd["cB_base"])[mkR]; Rerr = []
for m in names:
    d = resp(m, "R")[mkR]; beta = (d * dBR).sum() / (dBR ** 2).sum(); Rerr.append(np.zeros_like(resp(m, "R")))
    Rerr[-1][mkR] = d - beta * dBR
S_Re = sim_matrix(Rerr, mkR); c = coh(S_Re, arm); o, p, _ = perm_p(S_Re)
OUT["P3_R_error_coherence"] = dict(coherence=c, DELTA=c["DELTA"], ci95=coh_boot(S_Re), perm_p_one_sided=p)
fam_stats = {}
for m in names:
    f = Mj[m]["fits"]; ceil = max(f["split_half_r2_within"], 1e-6)
    fam_stats[m] = dict(mix=f["mix"]["r2_within"], vote=f["vote"]["r2_within"], exact=f["exact_bayes_scaled"]["r2_within"], ceiling=ceil,
                        mix_rel=f["mix"]["r2_within"] / ceil, vote_rel=f["vote"]["r2_within"] / ceil, exact_rel=f["exact_bayes_scaled"]["r2_within"] / ceil,
                        mix_minus_vote_rel=(f["mix"]["r2_within"] - f["vote"]["r2_within"]) / ceil)
def armdiff(key):
    v = np.array([fam_stats[m][key] for m in names]); obs = v[arm == 1].mean() - v[arm == 0].mean(); nT = arm.sum()
    allc = list(itertools.combinations(range(nm), nT)); null = []
    for cc in allc:
        l = np.zeros(nm, bool); l[list(cc)] = True; null.append(v[l].mean() - v[~l].mean())
    null = np.array(null); return dict(T=v[arm == 1].tolist(), P=v[arm == 0].tolist(), diff=float(obs), perm_p_two_sided=float((np.abs(null) >= abs(obs) - 1e-12).mean()))
OUT["P3_operator_families"] = dict(per_model=fam_stats, **{k: armdiff(k) for k in ("mix_rel", "vote_rel", "exact_rel", "mix_minus_vote_rel")})
# fitted-family account of null responses on the surface: pairs of pi-equivalent vectors within an episode
import cri as R_
E = R_.E; V = Gd["surfV"]; NS = V.shape[0]; pairs = []
for i in range(NS):
    P = E.pi_of_n(V[i])
    for a, b in itertools.combinations(range(36), 2):
        if np.abs(P[a] - P[b]).max() < 1e-12: pairs.append((i, a, b))
pairs = np.array(pairs); nul = {}
obsR, fitR = [], []
for m in names:
    z = Mz[m]; sv = z["surf"]; pm = z["surf_pred_mix"]
    o = sv[pairs[:, 0], pairs[:, 2]] - sv[pairs[:, 0], pairs[:, 1]]; f = pm[pairs[:, 0], pairs[:, 2]] - pm[pairs[:, 0], pairs[:, 1]]
    obsR.append(o); fitR.append(f)
    nul[m] = dict(r2_fit_explains_observed_surface_null=float(1 - ((o - f) ** 2).sum() / ((o - o.mean(0)) ** 2).sum()),
                  mean_abs_obs=float(np.abs(o).mean()), mean_abs_fit=float(np.abs(f).mean()))
allmask = np.ones(len(pairs), bool)
So = sim_matrix(obsR, allmask); Sf = sim_matrix(fitR, allmask)
OUT["surface_null"] = dict(n_pairs=int(len(pairs)), per_model=nul, observed=dict(coherence=coh(So, arm), ci95=coh_boot(So), perm_p=perm_p(So)[1]),
                           mix_fit_predicted=dict(coherence=coh(Sf, arm), ci95=coh_boot(Sf), perm_p=perm_p(Sf)[1]))
# ---------------------------------------------------------------- P5: pathway profiles (sufficiency of patched position sets), N vs R
pw = {}
for m in names:
    z = Mz[m]; prof = {}
    for F in ("N", "R"):
        mk = Gd[f"ok_{F}"].astype(bool); full = (z[f"c_{F}"] - z["c_base"])[mk]
        for sname in ("Rq", "Qs", "Oth"):
            d = (z[f"c_{F}_patch_{sname}"] - z["c_base"])[mk]; prof[f"{F}|{sname}"] = float((d * full).sum() / (full * full).sum())
        h = mk.copy(); idx = np.where(mk)[0]; half = np.zeros(len(mk), bool); half[idx[::2]] = True
        for tag, hm in (("a", half), ("b", mk & ~half)):
            fl = (z[f"c_{F}"] - z["c_base"])[hm]
            for sname in ("Rq", "Qs", "Oth"):
                d = (z[f"c_{F}_patch_{sname}"] - z["c_base"])[hm]; prof[f"{F}{tag}|{sname}"] = float((d * fl).sum() / (fl * fl).sum())
    dist_NR = sum(abs(prof[f"N|{s}"] - prof[f"R|{s}"]) for s in ("Rq", "Qs", "Oth"))
    dist_split = 0.5 * sum(abs(prof[f"Na|{s}"] - prof[f"Nb|{s}"]) + abs(prof[f"Ra|{s}"] - prof[f"Rb|{s}"]) for s in ("Rq", "Qs", "Oth"))
    pw[m] = dict(profile=prof, dist_N_vs_R=dist_NR, dist_split_half_null=dist_split)
dd = np.array([pw[m]["dist_N_vs_R"] - pw[m]["dist_split_half_null"] for m in names])
from scipy.stats import wilcoxon
OUT["P5_pathways"] = dict(per_model=pw, excess_N_vs_R=dd.tolist(), wilcoxon_p_excess_gt0=float(wilcoxon(dd, alternative="greater").pvalue) if nm >= 6 else None,
                          frac_models_excess_gt_0_1=float((dd > 0.1).mean()))
json.dump(OUT, open(os.path.join(D, f"{OUTNAME}.json"), "w"), indent=1, default=float)
p1 = OUT["P1_primary_N_input_clogit_pearson"]
print("P1", p1["coherence"], p1["ci95"], p1["perm_p_one_sided"]); print("P1b", {k: v for k, v in OUT["P1b_one_factor_N"].items() if k != "pairs"}); print("P2", {k: v for k, v in OUT["P2_adjusted"].items() if k not in ("quality", "KL_by_arm")})
print("P3 Rerr", OUT["P3_R_error_coherence"]["coherence"], OUT["P3_R_error_coherence"]["ci95"]); print("P3 fam", {k: (round(v["diff"], 3), round(v["perm_p_two_sided"], 3)) for k, v in OUT["P3_operator_families"].items() if k != "per_model"})
print("Nabs", OUT["family_Nabs"]["coherence"], OUT["family_Nabs"]["ci95"]); print("surface", OUT["surface_null"]["observed"]["coherence"], OUT["surface_null"]["mix_fit_predicted"]["coherence"])
print("P5", [round(x, 3) for x in OUT["P5_pathways"]["excess_N_vs_R"]], OUT["P5_pathways"]["wilcoxon_p_excess_gt0"])
