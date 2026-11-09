"""PP1 counterexamples II (mandate §2.3), constructed models only (no training):
 (a) TEMP3: Bayes log-probs / 3 (accuracy identical to Bayes, strongly miscalibrated)       -> predicts sign is preserved
 (b) TEMP1/3: Bayes x 3 (overconfident)                                                      -> sign preserved
 (c) CX1 (from pp1_calibration): argmax-preserving margin-dependent distortion, same accuracy -> sign broken
 (d) CX3 density-ratio counterexample: Bayes everywhere except on the agreement-count pattern classes with the largest
     (edit-endpoint frequency / natural frequency) ratio, chosen so the corrupted region has natural mass <= 2%;
     on that region the margin between the base top-2 candidates is reflected (logits of c1, c2 swapped).
     Selection uses a SEPARATE edit sample (stream 6_170_000) from the scored population (6_100_000).
Reports natural accuracy, ECE (10 bins), natural KL, and sign agreement on the frozen PP1 edit population."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas")
sys.path.insert(0, os.path.join(AT, "src"))
import numpy as np, gen_family as G
f4 = G.Fam(pc=1.0, rho=1.0); K, Q = 3, 6

def edits(NB, seed):
    d = G.sample(f4, NB, np.random.default_rng(seed)); X0 = d["X"]; bi, qq, Xe = [], [], []
    for q in range(Q):
        for k in range(Q):
            if k == q: continue
            for g, (a, b) in enumerate(G.GROUPS):
                X = X0.copy(); ag = X[:, k, a] == X[:, k, b]; X[:, k, b] = np.where(ag, (X[:, k, a] + 1) % K, X[:, k, a])
                Xe.append(X); bi.append(np.arange(NB)); qq.append(np.full(NB, q))
    return X0, np.concatenate(Xe), np.concatenate(bi), np.concatenate(qq)
def pattern(X):  # sorted tuple of per-group total agreement counts (0..6)
    A = np.stack([(X[:, :, a] == X[:, :, b]).sum(1) for a, b in G.GROUPS], -1); A = np.sort(A, -1)
    return A[:, 0] * 49 + A[:, 1] * 7 + A[:, 2]
def logpost(X):
    with np.errstate(divide="ignore", invalid="ignore"): return np.log(G.posterior(f4, X)["post_y"])

# region selection on a separate edit sample
Xs0, Xse, _, _ = edits(300, 6_170_000)
Dn = G.sample(f4, 20000, np.random.default_rng(6_150_000)); pn_pat = pattern(Dn["X"])
fin = np.isfinite(logpost(Xse)).all((1, 2)); pe_pat = pattern(Xse[fin])
nat = np.bincount(pn_pat, minlength=343) / len(pn_pat); ed = np.bincount(pe_pat, minlength=343) / len(pe_pat)
ratio = np.where(nat > 0, ed / np.maximum(nat, 1e-9), 0); ratio[ed == 0] = 0
R, mass = [], 0.0
for c in np.argsort(-ratio):
    if ratio[c] <= 1 or mass + nat[c] > 0.02: continue
    R.append(int(c)); mass += nat[c]
R = np.array(R)

# scored population (identical to frozen PP1)
X0, Xe, bi, qq = edits(1500, 6_100_000); n = len(Xe)
lb0, lbe = logpost(X0), logpost(Xe)
order = np.argsort(-np.nan_to_num(lb0, nan=-1e9), -1); c1, c2 = order[bi, qq, 0], order[bi, qq, 1]
mg = lambda l, rows: l[rows, qq, c1] - l[rows, qq, c2]
dm = mg(lbe, np.arange(n)) - mg(lb0, bi); scored = np.isfinite(lbe).all((1, 2)) & (np.abs(dm) > 0.5)
Pn = G.posterior(f4, Dn["X"])["post_y"]; lbn = np.log(Pn)
def norm(z): z = np.nan_to_num(z, nan=-50.0, neginf=-50.0); return z - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) - z.max(-1, keepdims=True)
def temp(T): return lambda l, X: norm(np.nan_to_num(l, nan=-50, neginf=-50) / T)
def cx1(l, X):
    l = np.nan_to_num(l, nan=-50.0, neginf=-50.0); s = np.sort(l, -1); mt = s[..., -1] - s[..., -2]
    return norm((l - l.mean(-1, keepdims=True)) / (0.05 + mt ** 2)[..., None])
def cx3(l, X):
    l = np.nan_to_num(l, nan=-50.0, neginf=-50.0).copy(); inR = np.isin(pattern(X), R)
    s = np.argsort(-l, -1); i = np.where(inR)[0][:, None]; qa = np.arange(Q)[None, :]
    a, b = s[i, qa, 0], s[i, qa, 1]; la, lb_ = l[i, qa, a].copy(), l[i, qa, b].copy(); l[i, qa, a] = lb_; l[i, qa, b] = la
    return norm(l)
def ece(P, Y, nb=10):
    conf = P.max(-1).ravel(); cor = (P.argmax(-1) == Y).ravel(); e = 0
    for j in range(nb):
        m = (conf > j / nb) & (conf <= (j + 1) / nb)
        if m.any(): e += m.mean() * abs(conf[m].mean() - cor[m].mean())
    return float(e)
res = dict(R_patterns=[tuple(int(v) for v in (c // 49, (c // 7) % 7, c % 7)) for c in R], R_natural_mass=float(mass),
           R_frac_scored_edit_endpoints=float(np.isin(pattern(Xe), R)[scored].mean()), R_frac_scored_base=float(np.isin(pattern(X0), R)[bi][scored].mean()))
for nm, fn in [("BAYES", temp(1.0)), ("TEMP3_underconfident", temp(3.0)), ("TEMP1/3_overconfident", temp(1 / 3)), ("CX1_margin_distortion", cx1), ("CX3_density_ratio_region", cx3)]:
    ln, l0, le = fn(lbn, Dn["X"]), fn(lb0, X0), fn(lbe, Xe); dmh = mg(le, np.arange(n)) - mg(l0, bi)
    res[nm] = dict(natural_acc=float((ln.argmax(-1) == Dn["Y"]).mean()), natural_ECE=ece(np.exp(ln), Dn["Y"]),
                   natural_KL=float((Pn * (lbn - ln)).sum(-1).mean()), sign_agreement=float((np.sign(dmh) == np.sign(dm))[scored].mean()))
    print(nm, res[nm], flush=True)
print({k: v for k, v in res.items() if k.startswith("R_")})
json.dump(res, open(os.path.join(HERE, "..", "logs", "pp1_cx_matched.json"), "w"), indent=1)
