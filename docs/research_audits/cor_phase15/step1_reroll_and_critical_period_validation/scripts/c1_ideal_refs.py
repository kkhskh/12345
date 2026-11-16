"""XIII-E C1: ideal-model, null and decoy reference values for every XIII-D causal metric (pure numpy, exact posteriors), with bootstrap
uncertainty over pairs (500 resamples).  Mechanisms (the 'model' is exact Bayes in every case; only the role of the patched subspace differs):
  IDEAL  the patched subspace carries exactly the sufficient statistic m (two-colluder, beta = 0.6) or the agreement counts n_g (C = 1 E1 setting)
         or the role responsibilities (C = 1 row setting), and nothing else;
  NULL   the patched subspace carries nothing the decision uses (interventions on it leave the output unchanged);
  DECOY  the patched subspace carries a decoy statistic (raw first-source cross-equality counts) that the exact decision ignores.
For an exact-Bayes model NULL and DECOY give identical values on every metric scored against m-changing pairs; they differ only on metrics
evaluated on DECOY pairs, where an m-carrier gives 0 by sufficiency.  Adapted from the XIII-D review-1 scripts (ideal_carrier.py,
srcfree_e1.py, shufd_ideal.py).  -> logs/c1_ideal_refs.json"""
import json, os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import genlib as GL, xii_bayes as XB
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def fid(o, p): o, p = o.ravel(), p.ravel(); b = (o * p).sum() / (p * p).sum(); ss = ((o - o.mean()) ** 2).sum(); return float(b), float(1 - ((o - b * p) ** 2).sum() / ss)
def boot(fn, n, B=500, seed=0):
    r = np.random.default_rng(seed); v = [fn(r.integers(0, n, n)) for _ in range(B)]; return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out = {}
# ---------------- two-colluder carrier metrics (C-workstream)
E2 = GL.Env(3, 2, beta=0.6); K = 3; PAIRS = [(0, 1), (0, 2), (1, 2)]; Hm = {0: 2, 1: 1, 2: 0}; cA = np.log(1 + 0.6 * K ** 3 / 0.4)
def mstat(X):
    a = X.reshape(len(X), 6, 3, 2); o = np.zeros((len(X), 6, 3))
    for j, (g, h) in enumerate(PAIRS):
        eq = (a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0]); o[:, :, j] = eq.sum(1, keepdims=True) - eq
    return o
def logL(X, q):
    n = len(X); idx = np.arange(n); rel, _, _ = GL._group_lik(E2, X); relm = rel.mean(-1); a_ = X.reshape(*X.shape[:2], 3, 2); o = []
    for g, hs in E2.roles:
        s = a_[:, :, list(hs), :].reshape(n, 6, -1); al = (s == s[..., :1]).all(-1); colH = 0.6 * al / K + 0.4 / K ** 4
        lq = np.log(relm[:, :, g]) + np.log(colH); msk = np.ones_like(lq, bool); msk[idx, q] = False; tot = np.where(msk, lq, 0).sum(1)
        o.append(tot[:, None] + np.log(rel[idx, q, g, :]) + np.log(colH[idx, q])[:, None])
    return np.stack(o), [g for g, _ in E2.roles]
def post_m(X, q, mnew):
    L, gs = logL(X, q); mold = mstat(X)[np.arange(len(X)), q]
    L = L + np.stack([cA * (mnew[:, Hm[g]] - mold[:, Hm[g]]) for g in gs])[:, :, None]; mx = L.max((0, 2), keepdims=True); P = np.exp(L - mx).sum(0); return P / P.sum(-1, keepdims=True)
r = np.random.default_rng(1); n0 = 6000; X = GL.sample(E2, n0, r)["X"]; q = r.integers(0, 6, n0); h = r.integers(0, 3, n0); X2 = X.copy()
perms = [np.array(p) for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
for j in range(n0):
    pm = perms[r.integers(5)]; cols = [2 * h[j], 2 * h[j] + 1]; oth = [t for t in range(6) if t != q[j]]; X2[j][np.ix_(oth, cols)] = pm[X[j][np.ix_(oth, cols)]]
ii = np.arange(n0); ma, mb = mstat(X)[ii, q], mstat(X2)[ii, q]; dm = np.abs(mb - ma).sum(-1) > 0
X, X2, q, ma, mb = X[dm][:2400], X2[dm][:2400], q[dm][:2400], ma[dm][:2400], mb[dm][:2400]; n = len(q)
Ba, Bb = XB.posterior_shared(E2, X, q), XB.posterior_shared(E2, X2, q); dB = cen(Bb) - cen(Ba)
out["propS_check"] = float(np.abs(post_m(X, q, mb) - Bb).max())
ru = np.random.default_rng(2); un = np.array([ru.choice(np.where(q == q[j])[0][np.where(q == q[j])[0] != j]) for j in range(n)])
nat = GL.sample(E2, 20000, np.random.default_rng(3)); qn = np.random.default_rng(4).integers(0, 6, 20000); mbar = mstat(nat["X"])[np.arange(20000), qn].mean(0); Mbar = np.repeat(mbar[None], n, 0)
D = {}
D["SUFF"] = cen(post_m(X, q, mb)) - cen(Ba)
D["unrelated_donor(SRCF)"] = cen(post_m(X, q, ma[un])) - cen(Ba)
D["reverse(donor run with base projection)"] = cen(post_m(X2, q, ma)) - cen(Ba)
D["restore(base meanrep, donor intact)"] = cen(Bb) - cen(post_m(X, q, Mbar))
D["meanrep_both_remaining"] = cen(post_m(X2, q, Mbar)) - cen(post_m(X, q, Mbar))
C = {}
for k, v in D.items():
    s_, r2_ = fid(v, dB); C[k] = dict(ideal_slope=s_, ideal_R2=r2_, ideal_slope_CI=boot(lambda i: fid(v[i], dB[i])[0], n), null_slope=0.0 if k in ("SUFF", "unrelated_donor(SRCF)") else 1.0 if k.startswith("reverse") or k.startswith("restore") or k.startswith("meanrep") else None)
C["DAS_minus_SRCF_excess"] = dict(ideal=C["SUFF"]["ideal_slope"] - C["unrelated_donor(SRCF)"]["ideal_slope"], null=0.0)
Xn, qn2 = nat["X"][:5000], qn[:5000]; Bn = XB.posterior_shared(E2, Xn, qn2)
kl = lambda P, Q: (P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)
dmr = kl(Bn, post_m(Xn, qn2, np.repeat(mbar[None], 5000, 0))); dz = kl(Bn, post_m(Xn, qn2, np.zeros((5000, 3))))
C["meanrep_damage_KL"] = dict(ideal=float(dmr.mean()), ideal_CI=boot(lambda i: dmr[i].mean(), 5000), null=0.0)
C["zero_ablation_damage_KL"] = dict(ideal=float(dz.mean()), ideal_CI=boot(lambda i: dz[i].mean(), 5000), null=0.0)
# null pairs: m unchanged -> an ideal m-carrier transfers exactly 0 (Proposition S); a decoy carrier transfers 0 on J_DM pairs as well
C["null_JD0_transfer"] = dict(ideal=0.0, null=0.0, decoy=0.0, note="exact by sufficiency; a decoy carrier also transfers 0 under an exact-Bayes model, so null preservation cannot separate an m-carrier from a decoy carrier")
out["carrier"] = C
# ---------------- E1 eligibility own_share (C = 1, beta = 1)
env = GL.Env(3, 1); a_ = 0.8; noi = 1 / K ** 2
r_ag = (a_ * a_ + (K - 1) * ((1 - a_) / (K - 1)) ** 2) / K; r_dis = (2 * a_ * (1 - a_) / (K - 1) + (K - 2) * ((1 - a_) / (K - 1)) ** 2) / K
def post_from(cnt, Xq):
    A = Xq.reshape(-1, 3, 2); ag = A[..., 0] == A[..., 1]; y = np.arange(K)
    p1 = np.where(A[..., 0, None] == y, a_, (1 - a_) / (K - 1)); p2 = np.where(A[..., 1, None] == y, a_, (1 - a_) / (K - 1)); Lg = p1 * p2 / noi
    W = (r_ag / noi) ** cnt * (r_dis / noi) ** (5 - cnt); e = (cnt == 5) & ag; m = e.sum(1, keepdims=True) - e
    P = (W[..., None] * m[..., None] * Lg).sum(1); return P / P.sum(-1, keepdims=True)
rng = np.random.default_rng(11); Xc = GL.sample(env, 60000, rng)["X"]; qc = rng.integers(0, 6, len(Xc)); a6 = Xc.reshape(len(Xc), 6, 3, 2); eq = a6[..., 0] == a6[..., 1]
fam = {k: ([], [], []) for k in ("E5", "E3", "QF5", "QF4")}
for i in range(len(Xc)):
    qq = qc[i]; cnt = eq[i].sum(0) - eq[i, qq]; agq = eq[i, qq]; kind = ["E5", "E3", "QF5", "QF4"][rng.integers(4)]
    if len(fam[kind][2]) >= 1500: continue
    x = Xc[i].copy()
    if kind in ("E5", "E3"):
        lo, hi = (4, 5) if kind == "E5" else (3, 4); hs = [hh for hh in range(3) if agq[hh] and cnt[hh] in (lo, hi)]
        if not hs: continue
        hh = hs[rng.integers(len(hs))]; others = [p for p in range(6) if p != qq]
        if cnt[hh] == hi: cand = [p for p in others if eq[i, p, hh]]; p = cand[rng.integers(len(cand))]; x[p, 2 * hh + 1] = (x[p, 2 * hh] + rng.integers(1, 3)) % 3
        else: cand = [p for p in others if not eq[i, p, hh]]; p = cand[rng.integers(len(cand))]; x[p, 2 * hh + 1] = x[p, 2 * hh]
    else:
        n_ = 5 if kind == "QF5" else 4; hs = [hh for hh in range(3) if cnt[hh] == n_]
        if not hs: continue
        hh = hs[rng.integers(len(hs))]; x[qq, 2 * hh + 1] = (x[qq, 2 * hh] + rng.integers(1, 3)) % 3 if agq[hh] else x[qq, 2 * hh]
    e2 = x.reshape(6, 3, 2)
    if not (eq[i].all(0).any() and (e2[..., 0] == e2[..., 1]).all(0).any()): continue
    fam[kind][0].append(Xc[i]); fam[kind][1].append(x); fam[kind][2].append(qq)
def counts(X_, q_): e = (X_.reshape(len(X_), 6, 3, 2)[..., 0] == X_.reshape(len(X_), 6, 3, 2)[..., 1]); return e.sum(1) - e[np.arange(len(X_)), q_]
E1 = {}
for k, (Xb, Xd, qq) in fam.items():
    Xb, Xd, qq = np.array(Xb), np.array(Xd), np.array(qq); m_ = len(qq); i = np.arange(m_); Xbq, Xdq = Xb.reshape(m_, 6, 6)[i, qq], Xd.reshape(m_, 6, 6)[i, qq]
    Bb_ = post_from(counts(Xb, qq), Xbq); Bd_ = post_from(counts(Xd, qq), Xdq); own = cen(Bd_) - cen(Bb_)
    r7 = np.random.default_rng(7); un_ = np.array([r7.choice(np.where(qq == qq[j])[0][np.where(qq == qq[j])[0] != j]) for j in range(m_)])
    Bs = post_from(counts(Xb, qq)[un_], Xbq); ok = np.isfinite(Bs).all(1); d_ = cen(Bs) - cen(Bb_)
    sh = lambda ix: float((d_[ix] * own[ix]).sum() / (own[ix] ** 2).sum()); ix = np.where(ok)[0]
    cosv = float((d_[ok] * own[ok]).sum() / np.sqrt((d_[ok] ** 2).sum() * (own[ok] ** 2).sum()))
    E1[k] = dict(ideal_DAS_own_share=1.0, ideal_SRCFREE_own_share=sh(ix), ideal_SRCFREE_CI=boot(lambda j: sh(ix[j]), len(ix)), ideal_SRCFREE_cosine=cosv,
                 ideal_excess=1.0 - sh(ix), null_DAS_own_share=0.0, null_SRCFREE_own_share=0.0, mean_abs_own=float(np.abs(own).mean()), mean_abs_srcfree=float(np.abs(d_[ok]).mean()))
out["E1_own_share"] = E1
# ---------------- E2 attention-row controls (C = 1, beta = 1)
def parts(X_, q_):
    n_ = len(X_); i = np.arange(n_); rel, col, no = GL._group_lik(env, X_); relm = rel.mean(-1); lr = np.log(relm / no)
    W = np.exp(lr.sum(1) - lr[i, q_]); Lg = rel[i, q_] / no; m_, e_ = GL.eligibility(X_, 3, q_); return W, Lg, m_
def resp(X_, q_):
    W, Lg, m_ = parts(X_, q_); rr = W * m_ * Lg.sum(-1); return rr / np.maximum(rr.sum(1, keepdims=True), 1e-300), Lg / Lg.sum(-1, keepdims=True)
def outp(rr, ell): P = (rr[..., None] * ell).sum(1); return P / P.sum(-1, keepdims=True)
r5 = np.random.default_rng(5); Nn = 6000; Xr = GL.sample(env, Nn, r5)["X"]; qr = r5.integers(0, 6, Nn); Xd_ = Xr.copy(); hr = r5.integers(0, 3, Nn); qp = (qr + r5.integers(1, 6, Nn)) % 6
for i in range(Nn):
    a0, a1 = Xr[i, qp[i], 2 * hr[i]], Xr[i, qp[i], 2 * hr[i] + 1]; Xd_[i, qp[i], 2 * hr[i] + 1] = a0 if a0 != a1 else (a0 + r5.integers(1, 3)) % 3
PB, PBd = GL.posterior_closed(env, Xr, qr), GL.posterior_closed(env, Xd_, qr); keep = np.where(np.abs(cen(PBd) - cen(PB)).max(-1) > 0.05)[0][:800]
Xr, Xd_, qr = Xr[keep], Xd_[keep], qr[keep]; nn = len(qr); i = np.arange(nn); PB, PBd = PB[keep], PBd[keep]; dBr = cen(PBd) - cen(PB)
rb, ell = resp(Xr, qr); rd, _ = resp(Xd_, qr); r6 = np.random.default_rng(6); perm = np.array([r6.choice(np.where((qr == qr[j]) & (i != j))[0]) for j in range(nn)])
rn = np.clip(rb + (rd - rb)[perm], 0, None); rn = rn / np.maximum(rn.sum(1, keepdims=True), 1e-300); dS = cen(outp(rn, ell)) - cen(PB); dT = cen(outp(rd, ell)) - cen(PB)
out["E2_rows"] = dict(TRUE_ideal_R2=fid(dT, dBr)[1], SHUFD_vs_base_ideal_R2=fid(dS, dBr)[1], SHUFD_vs_borrowed_ideal_R2=fid(dS, dBr[perm])[1],
                      SHUFD_vs_borrowed_null_R2=0.0, note="ideal and null both give ~0 on the borrowed-pair arm: no power by construction (rows act through the base episode's values)",
                      SHUFD_vs_base_CI=boot(lambda j: fid(dS[j], dBr[j])[1], nn))
json.dump(out, open(os.path.join(HERE, "..", "logs", "c1_ideal_refs.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)[:3000])
