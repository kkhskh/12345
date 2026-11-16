"""XIII-E Workstream C2-C4 (analysis only, existing XIII-D carrier checkpoints C-C1 = uP160 s23, C-C2 = uP165 s21, early01 adapted).
Same site, pairs and direction-set construction as XIII-D c_carrier.py (copied verbatim below, with fresh streams), plus:
 C2  decoder quality: held-out R2 (independent natural episodes, stream ST+50) of (a) the token-level ridge decoder itself (its own predictions),
     (b) linear decoders fitted on each 3-d projection (DEC, ENC, CCD, DAS, DAS_DECOY, RAND, VMRAND); signal and noise covariance along each set
     (variance of E[x|m] projected vs residual variance projected); patch displacement norm on J_DM pairs; SUFF for variance-matched random
     subspaces VMRAND (directions drawn from N(0, Sigma_site), orthonormalised; 3 seeds).
 C3  principal angles (cosines) and projection overlap ||P1'P2||_F^2/k between every pair of sets, in the Euclidean metric and in the
     Sigma-whitened metric; second DAS seed (stability); TRACE key/query split for ENC_m, DAS_DECOY and the second DAS seed.
 C4  k = 1..6 sweep of DAS (fit on the first half of J_DM, evaluate on the second): SUFF, reverse remaining slope, null JD0/DEC, unrelated-donor
     control, meanrep damage KL, decoy-trained DAS (same k) SUFF, seed stability (mean principal cosine between two DAS seeds).
Usage: CA_CKDIR=adapt4_carrier CA_STREAM=96200000 python3 scripts/c_audit.py SPEC -> logs/c_audit.json"""
import json, os, sys, itertools, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import xc, ftlib as F, native as NV, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore")
sp = sys.argv[1]; UN = sp.replace(":", "_"); M0 = xc.load(sp); E2 = GL.Env(3, 2, beta=0.6); ST = int(os.environ.get("CA_STREAM", 96_200_000)); K = 3
cands = ([os.path.join(NV.X13D, "checkpoints", os.environ["CA_CKDIR"], f"{UN}_early01_b0.6_lr0.001.npz")] if os.environ.get("CA_CKDIR") else []) + glob.glob(os.path.join(HERE, "..", "checkpoints", "adapt3*", f"{UN}_early01_b0.6_lr0.001.npz")) + glob.glob(os.path.join(NV.X13C, "checkpoints", "adapt3", f"{UN}_early01_b0.6_lr0.001.npz"))
print("checkpoint:", cands[0], flush=True); z = np.load(cands[0]); MA = dict(M0); MA["layers"] = [dict(l) for l in M0["layers"]]
for k_, v in z.items():
    top, *rest = k_.split("/")
    if top in ("l0", "l1", "l2") and rest: MA["layers"][int(top[1])][rest[0]] = jnp.asarray(v)
    elif top in ("gf", "bf", "WA"): MA[top] = jnp.asarray(v)
PAIRS = [(0, 1), (0, 2), (1, 2)]
def mstat(X, decoy=False):
    a = X.reshape(len(X), 6, 3, 2); out = np.zeros((len(X), 6, 3))
    for j, (g, h) in enumerate(PAIRS):
        eq = (a[:, :, g, 0] == a[:, :, h, 0]) if decoy else ((a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0]))
        out[:, :, j] = eq.sum(1, keepdims=True) - eq
    return out
def logr(X, q):
    rel, _, _ = GL._group_lik(E2, X); relm = rel.mean(-1); a_ = X.reshape(len(X), 6, 3, 2); i = np.arange(len(X)); T = []
    for g in range(3):
        H = [h for h in range(3) if h != g]; s = a_[:, :, H, :].reshape(len(X), 6, 4); al = (s == s[..., :1]).all(-1); colH = 0.6 * al / 3 + 0.4 / 81
        lr = np.log(relm[:, :, g])                                            # role likelihood: Y marginalised at every question (incl. q)
        T.append(lr.sum(1) + np.log(colH).sum(1))
    T = np.stack(T, 1); T = T - T.max(1, keepdims=True); return T - np.log(np.exp(T).sum(1, keepdims=True))
# ---------------- pairs
def jpairs(seed, n):
    r = np.random.default_rng(seed); d = GL.sample(E2, n, r); X = d["X"]; q = r.integers(0, 6, n); h = r.integers(0, 3, n); X2 = X.copy()
    perms = [np.array(p) for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
    for j in range(n):
        pm = perms[r.integers(5)]; cols = [2 * h[j], 2 * h[j] + 1]; oth = [t for t in range(6) if t != q[j]]; X2[j][np.ix_(oth, cols)] = pm[X[j][np.ix_(oth, cols)]]
    ii = np.arange(n); dm = np.abs(mstat(X2)[ii, q] - mstat(X)[ii, q]).sum(-1) > 0
    dB = RP.cen(XB.posterior_shared(E2, X2, q)) - RP.cen(XB.posterior_shared(E2, X, q)); return X, X2, q, dB, dm
X, X2, q, dB, dm = jpairs(ST, 3000)
JDM = (X[dm][:1200], X2[dm][:1200], q[dm][:1200], dB[dm][:1200]); JD0 = (X[~dm][:300], X2[~dm][:300], q[~dm][:300], dB[~dm][:300])
rd = np.random.default_rng(ST + 1); dd = GL.sample(E2, 4000, rd); Xd = dd["X"]; qd = rd.integers(0, 6, 4000); Xd2 = Xd.copy(); keep = np.zeros(4000, bool)
for j in range(4000):
    a = Xd[j].reshape(6, 3, 2); cs = [(p, g) for p in range(6) if p != qd[j] for g in range(3) if a[p, g, 0] != a[p, g, 1]]
    if not cs: continue
    p, g = cs[rd.integers(len(cs))]; x2 = Xd[j].copy(); x2[p, 2 * g] = 3 - a[p, g, 0] - a[p, g, 1]
    if (mstat(x2[None], True)[0, qd[j]] == mstat(Xd[j][None], True)[0, qd[j]]).all(): continue
    Xd2[j] = x2; keep[j] = True
DEC = (Xd[keep][:300], Xd2[keep][:300], qd[keep][:300], RP.cen(XB.posterior_shared(E2, Xd2[keep][:300], qd[keep][:300])) - RP.cen(XB.posterior_shared(E2, Xd[keep][:300], qd[keep][:300])))
n = len(JDM[2]); half = n // 2; TR, TE = np.arange(half), np.arange(half, n)
# ---------------- model helpers
fwd_rec = jax.jit(lambda X_: (lambda r: (F.forward(MA, X_, rec=r), r["resid2"])[1])({}))
def resid2(X_): return np.concatenate([np.asarray(fwd_rec(jnp.asarray(X_[a:a + 500]))) for a in range(0, len(X_), 500)])
def logits(X_, fn=None):
    f = jax.jit(lambda X__: F.forward(MA, X__)) if fn is None else None
    return np.concatenate([np.asarray(F.forward(MA, jnp.asarray(X_[a:a + 500]), resid_patch=None if fn is None else {2: fn(a, min(a + 500, len(X_)))})) for a in range(0, len(X_), 500)])
def site_mask(q_):
    m = np.zeros((len(q_), 43), bool); i = np.arange(len(q_)); m[i[:, None], q_[:, None] * 6 + np.arange(6)[None]] = True; m[i, 36 + q_] = True; return m
def patched(Xb, q_, Zd, P, mode="inter", mu=None, Zsrc=None):
    """forward of Xb with the site modified along span(P): inter = x + (zd - x)PP^T; ablate = x - xPP^T; meanrep = x + (mu - x)PP^T"""
    m = site_mask(q_); Pj = jnp.asarray(P); i = np.arange(len(q_))
    def fn(a, b):
        mm = jnp.asarray(m[a:b])[..., None]
        if mode == "inter": zd = jnp.asarray(Zd[a:b]); return lambda x: jnp.where(mm, x + ((zd - x) @ Pj) @ Pj.T, x)
        if mode == "ablate": return lambda x: jnp.where(mm, x - (x @ Pj) @ Pj.T, x)
        if mode == "meanrep": mu_ = jnp.asarray(mu); return lambda x: jnp.where(mm, x + ((mu_ - x) @ Pj) @ Pj.T, x)
        if mode == "restore": mu_ = jnp.asarray(mu); zs = jnp.asarray(Zsrc[a:b]); return lambda x: jnp.where(mm, x + ((mu_ - x) @ Pj) @ Pj.T + ((zs - mu_) @ Pj) @ Pj.T, x)
    return logits(Xb, fn)[i, q_]
def dlt(La, Lb): return RP.cen(RP.sm(Lb)) - RP.cen(RP.sm(La))
# base/donor states and outputs
S = {}
for nm, (Xa, Xb_, qq, dBq) in dict(JDM=JDM, JD0=JD0, DEC=DEC).items():
    i = np.arange(len(qq)); La = logits(Xa)[i, qq]; Lb = logits(Xb_)[i, qq]
    S[nm] = dict(Xa=Xa, Xb=Xb_, q=qq, dB=dBq, Za=resid2(Xa), Zb=resid2(Xb_), La=La, Lb=Lb, own=dlt(La, Lb))
ru = np.random.default_rng(ST + 2); un = np.array([ru.choice(np.where(JDM[2] == JDM[2][j])[0][np.where(JDM[2] == JDM[2][j])[0] != j]) for j in range(n)])
# ---------------- direction sets from natural episodes (token-level at the site)
dn = GL.sample(E2, 3000, np.random.default_rng(ST + 3)); Xn = dn["X"]; qn = np.random.default_rng(ST + 4).integers(0, 6, 3000); Zn = resid2(Xn); inn = np.arange(3000)
toks = np.concatenate([qn[:, None] * 6 + np.arange(6)[None], (36 + qn)[:, None]], 1)                        # (3000, 7) site tokens
Xs = Zn[inn[:, None], toks].reshape(-1, 64); mu_site = Xs.mean(0)
Tm = np.repeat(mstat(Xn)[inn, qn], 7, 0); Tr = np.repeat(logr(Xn, qn), 7, 0)
def dirs_for(T):
    Xc = Xs - Xs.mean(0); Tc = T - T.mean(0); W = np.linalg.solve(Xc.T @ Xc + 1.0 * len(Xc) * 1e-3 * np.eye(64), Xc.T @ Tc)
    Yh = Xc @ W; A = (Xc.T @ Xc / len(Xc)) @ W @ np.linalg.pinv(np.atleast_2d(np.cov(Yh.T)))
    ccd = []
    for j in range(T.shape[1]):
        lo, hi = np.percentile(T[:, j], [33, 67]); ccd.append(Xs[T[:, j] >= hi].mean(0) - Xs[T[:, j] <= lo].mean(0))
    o = lambda B: np.linalg.qr(B)[0][:, :K]
    return o(W), o(A), o(np.stack(ccd, 1))
DIRS = {}
DIRS["DEC_m"], DIRS["ENC_m"], DIRS["CCD_m"] = dirs_for(Tm)
DIRS["DEC_r"], DIRS["ENC_r"], DIRS["CCD_r"] = dirs_for(Tr)
for j in range(5): DIRS[f"RAND{j}"] = np.linalg.qr(np.random.default_rng(500 + j).normal(size=(64, K)))[0]
def train_das(nm, seed, idx, k=None):
    k = k or K; d = S[nm]; P = jax.random.normal(jax.random.PRNGKey(seed), (64, k)); opt = optax.adam(3e-2); st = opt.init(P); m_all = site_mask(d["q"]); tgt = RP.sm(d["Lb"])
    def lossf(P_, Xa, mm, zb, qq, tg):
        R = jnp.linalg.qr(P_)[0]; L = F.forward(MA, Xa, resid_patch={2: lambda x: jnp.where(mm[..., None], x + ((zb - x) @ R) @ R.T, x)})[jnp.arange(Xa.shape[0]), qq]
        return -(tg * jax.nn.log_softmax(L, -1)).sum(-1).mean()
    g = jax.jit(jax.value_and_grad(lossf)); r = np.random.default_rng(seed)
    for s in range(200):
        j = r.choice(idx, 128, replace=False); l_, gr = g(P, jnp.asarray(d["Xa"][j]), jnp.asarray(m_all[j]), jnp.asarray(d["Zb"][j]), jnp.asarray(d["q"][j]), jnp.asarray(tgt[j]))
        up, st = opt.update(gr, st); P = optax.apply_updates(P, up)
    return np.asarray(jnp.linalg.qr(P)[0])
DIRS["DAS"] = train_das("JDM", 11, TR)
DIRS["DAS_DECOY"] = train_das("DEC", 12, np.arange(len(S["DEC"]["q"])))
res = dict(unit=sp)
d = S["JDM"]; Sig = np.cov(Xs.T); w, V = np.linalg.eigh(Sig); Wh = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-8))) @ V.T   # whitening
for j in range(3):
    R_ = np.random.default_rng(900 + j).multivariate_normal(np.zeros(64), Sig, size=K).T; DIRS[f"VMRAND{j}"] = np.linalg.qr(R_)[0]
DIRS["DAS_s2"] = train_das("JDM", 21, TR)
def suff(P):
    Lp = patched(d["Xa"][TE], d["q"][TE], d["Zb"][TE], P); dp = dlt(d["La"][TE], Lp); f = RP.fid(dp, d["dB"][TE]); return f["slope"], f["R2"], float(np.abs(dp).mean())
# ---- C2 decoder quality
dv = GL.sample(E2, 2000, np.random.default_rng(ST + 50)); Xv = dv["X"]; qv = np.random.default_rng(ST + 51).integers(0, 6, 2000); Zv = resid2(Xv); iv = np.arange(2000)
tv = np.concatenate([qv[:, None] * 6 + np.arange(6)[None], (36 + qv)[:, None]], 1); Xvs = Zv[iv[:, None], tv].reshape(-1, 64)
TmV = np.repeat(mstat(Xv)[iv, qv], 7, 0); TrV = np.repeat(logr(Xv, qv), 7, 0)
def r2(Y, Yh): return float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
def ridge(Xa_, T): Xc = Xa_ - Xa_.mean(0); Tc = T - T.mean(0); W = np.linalg.solve(Xc.T @ Xc + len(Xc) * 1e-3 * np.eye(Xa_.shape[1]), Xc.T @ Tc); return W, Xa_.mean(0), T.mean(0)
c2 = {}
for tn, (T, TV) in dict(m=(Tm, TmV), logr=(Tr, TrV)).items():
    W, mx, mt = ridge(Xs, T); c2[f"token_ridge_decoder_R2_{tn}"] = r2(TV, (Xvs - mx) @ W + mt)            # the decoder itself, held out
    # site-pooled (7 tokens) decoding from each 3-d projection, held out
    for nm in ("DEC_m", "ENC_m", "CCD_m", "DEC_r", "ENC_r", "DAS", "DAS_DECOY", "RAND0", "VMRAND0"):
        P = DIRS[nm]; Ftr = (Zn[inn[:, None], toks] @ P).reshape(3000, -1); Fte = (Zv[iv[:, None], tv] @ P).reshape(2000, -1)
        Ttr = mstat(Xn)[inn, qn] if tn == "m" else logr(Xn, qn); Tte = mstat(Xv)[iv, qv] if tn == "m" else logr(Xv, qv)
        W2, mx2, mt2 = ridge(Ftr, Ttr); c2[f"proj_R2_{tn}_{nm}"] = r2(Tte, (Fte - mx2) @ W2 + mt2)
Mbin = mstat(Xn)[inn, qn]; cond_mean = np.zeros_like(Xs); Tm_tok = np.repeat(Mbin, 7, 0); key = [tuple(r) for r in Tm_tok.astype(int)]
grp = {}
for i_, k_ in enumerate(key): grp.setdefault(k_, []).append(i_)
for k_, ix in grp.items(): cond_mean[ix] = Xs[ix].mean(0)
SigS = np.cov(cond_mean.T); SigN = np.cov((Xs - cond_mean).T)
geo = {}
for nm, P in DIRS.items():
    Zd = (d["Zb"][TE] - d["Za"][TE]); mm = site_mask(d["q"][TE]); disp = float(np.linalg.norm((Zd[mm] @ P), axis=1).mean())
    geo[nm] = dict(var_along=float(np.trace(P.T @ Sig @ P)), signal_var=float(np.trace(P.T @ SigS @ P)), noise_var=float(np.trace(P.T @ SigN @ P)), patch_displacement=disp)
for nm in ("VMRAND0", "VMRAND1", "VMRAND2", "DAS_s2"):
    s_, r_, ma_ = suff(DIRS[nm]); geo[nm].update(SUFF_slope=s_, SUFF_R2=r_, SUFF_mean_abs=ma_)
for nm in ("DEC_m", "ENC_m", "DAS", "RAND0"):
    s_, r_, ma_ = suff(DIRS[nm]); geo[nm].update(SUFF_slope=s_, SUFF_R2=r_, SUFF_mean_abs=ma_)
res["C2"] = dict(decoding=c2, geometry=geo); print("C2", {k: round(v, 3) for k, v in c2.items()}, flush=True)
# ---- C3 principal angles
names = ["DAS", "DAS_s2", "DAS_DECOY", "ENC_m", "ENC_r", "CCD_m", "DEC_m", "DEC_r", "RAND0", "VMRAND0"]
def pang(A, B, M=None):
    if M is not None: A = np.linalg.qr(M @ A)[0]; B = np.linalg.qr(M @ B)[0]
    s = np.linalg.svd(A.T @ B, compute_uv=False); return [float(x) for x in np.clip(s, 0, 1)], float((s ** 2).sum() / min(A.shape[1], B.shape[1]))
c3 = {}
for i_, a in enumerate(names):
    for b in names[i_ + 1:]:
        ce, oe = pang(DIRS[a], DIRS[b]); cw, ow = pang(DIRS[a], DIRS[b], np.linalg.cholesky(Sig + 1e-8 * np.eye(64)).T)
        c3[f"{a}|{b}"] = dict(cos=ce, overlap=oe, cos_cov=cw, overlap_cov=ow)
res["C3"] = c3; print("C3", {k: round(v["overlap"], 3) for k, v in c3.items() if k.startswith("DAS|")}, flush=True)
# trace split for other sufficient sets
def trace_split(P):
    Pj = jnp.asarray(P); Zmix_ = jnp.where(m, Za_j + ((jnp.asarray(Zb_) - Za_j) @ Pj) @ Pj.T, Za_j); Lall_, _ = l2(Zmix_, Zmix_, Zmix_, Zmix_); full_ = dlt(L0_, Lall_); T_ = (full_ * full_).sum()
    sh = lambda L: float((dlt(L0_, L) * full_).sum() / T_)
    return dict(keys_only=sh(l2(Za_j, Zmix_, Za_j)[0]), queries_only=sh(l2(Zmix_, Za_j, Za_j)[0]), values_only=sh(l2(Za_j, Za_j, Zmix_)[0]), keys_and_queries=sh(l2(Zmix_, Zmix_, Za_j)[0]))
lay = MA["layers"][2]; Za, Zb_, qq = d["Za"][TE], d["Zb"][TE], d["q"][TE]; ii = np.arange(len(qq)); m = jnp.asarray(site_mask(qq))[..., None]; Za_j = jnp.asarray(Za)
def l2(zq, zk, zv, zs=None):
    def ln1(z): return F.N.ln(z, lay["g1"], lay["b1"])
    qh = jnp.einsum("ntd,hde->nhte", ln1(zq), lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", ln1(zk), lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", ln1(zv), lay["WV"])
    sc = jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]); A = jax.nn.softmax(sc, -1)
    xa = (Za_j if zs is None else zs) + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
    hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); xo = xa + hid @ lay["W2"] + lay["c2"]
    return np.asarray(F.N.ln(xo, MA["gf"], MA["bf"])[:, 36:42] @ MA["WA"])[ii, qq], np.asarray(sc)
L0_, _ = l2(Za_j, Za_j, Za_j)
res["TRACE_split"] = {nm: trace_split(DIRS[nm]) for nm in ("DAS", "DAS_s2", "ENC_m", "DAS_DECOY")}; print("TRACE", res["TRACE_split"], flush=True)
# ---- C4 k sweep
ru2 = np.random.default_rng(ST + 2); un = np.array([ru2.choice(np.where(d["q"] == d["q"][j])[0][np.where(d["q"] == d["q"][j])[0] != j]) for j in range(len(d["q"]))])
Ln0 = logits(Xn[:800])[inn[:800], qn[:800]]; Pn0 = RP.sm(Ln0)
def damage(P):
    Q = RP.sm(patched(Xn[:800], qn[:800], None, P, "meanrep", mu=mu_site)); return float((Pn0 * (np.log(Pn0 + 1e-12) - np.log(Q + 1e-12))).sum(-1).mean())
c4 = {}
for k in range(1, 7):
    Pa = train_das("JDM", 11, TR, k); Pb = train_das("JDM", 21, TR, k); Pd = train_das("DEC", 12, np.arange(len(S["DEC"]["q"])), k)
    s_, r_, ma_ = suff(Pa); sd, rd_, _ = suff(Pd)
    nulls = {}
    for nn_ in ("JD0", "DEC"):
        e = S[nn_]; dq = dlt(e["La"], patched(e["Xa"], e["q"], e["Zb"], Pa)); nulls[nn_] = float(np.abs(dq).mean() / max(ma_, 1e-9))
    rv = RP.fid(dlt(d["La"][TE], patched(d["Xb"][TE], d["q"][TE], d["Za"][TE], Pa)), d["dB"][TE])["slope"]
    sf = RP.fid(dlt(d["La"][TE], patched(d["Xa"][TE], d["q"][TE], d["Za"][un][TE], Pa)), d["dB"][TE])
    c4[k] = dict(SUFF_slope=s_, SUFF_R2=r_, reverse_remaining=rv, null_rel_JD0=nulls["JD0"], null_rel_DEC=nulls["DEC"], unrelated_donor_slope=sf["slope"], unrelated_donor_R2=sf["R2"],
                 meanrep_damage_KL=damage(Pa), decoy_DAS_SUFF_slope=sd, seed_stability_mean_cos=float(np.mean(pang(Pa, Pb)[0])))
    print("k", k, {a: round(b, 3) for a, b in c4[k].items()}, flush=True)
res["C4"] = c4
outp = os.path.join(HERE, "..", "logs", "c_audit.json"); R = json.load(open(outp)) if os.path.exists(outp) else {}
R[sp] = res; json.dump(R, open(outp, "w"), indent=1, default=float)
