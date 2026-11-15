"""XIII-D Workstream C2-C6: what does the two-colluder repair carrier encode, is it NECESSARY, does it preserve null-pair decisions, and how does
the frozen layer 2 use it?  Matched site and matched procedure for every direction set.

Model: an early01-adapted model (beta = 0.6) from adapt3 (XIII-C or this package: checkpoints/adapt3*/<unit>_early01_b0.6_lr0.001.npz).
SITE (identical for every direction set): the layer-2 input (resid2) at the 6 source tokens of the target q AND its readout token; one shared
64 x k basis P applied at every site token; interchange x <- x + (x_d - x) P P^T.
Pairs (fresh stream CA_STREAM): J_DM (m-changing relabellings), J_D0 (m-preserving relabellings: exact change 0), DECOY (raw cross-equality
changes, m fixed: exact change 0), SRCFREE donors (an unrelated episode with the same q).
Direction sets (k = 3 unless stated), all fitted on held-out natural beta = 0.6 episodes (token-level samples at the site), never on test pairs:
  DAS     learned interchange subspace (trained on the first half of J_DM pairs to reproduce the adapted model's own donor output; tested on the
          second half and on all null pairs)
  DEC_m   ridge DECODER directions for m (3 targets)          ENC_m  Haufe ENCODING patterns A = Cov(x) W Cov(yhat)^-1 for m
  CCD_m   class-conditional differences: mean(x | m_H >= 3) - mean(x | m_H <= 1) for each pair H
  DEC_r / ENC_r / CCD_r  the same for the exact reliable-group log-posterior vector (log r_0, log r_1, log r_2) (CCD: top vs bottom tercile)
  RAND    5 random orthonormal k-subspaces;  DAS_DECOY  a subspace trained identically but on DECOY pairs (dimension- and budget-matched control)
Tests per direction set:
  SUFF   J_DM held-out: fidelity slope / R2 on the exact change; own-share of the adapted model's own change
  NULL   J_D0, DECOY: mean |Δ| of the patched output relative to mean |Δ| on J_DM (false-positive transfer) and relative to the model's own null response
  SRCF   source-free donor: fidelity to the pair's exact change and magnitude
  NEC    necessity on J_DM held-out (both base and donor runs modified at the site): (a) ABLATE (remove the P-component), (b) MEANREP (set it to the
         dataset mean), (c) REVERSE natural-donor interchange (donor run with the base's P-component), (d) RESTORE (MEANREP then add the donor's P
         component back in the donor run); remaining J effect (fidelity slope) and the damage to unrelated decisions (mean KL between the modified and
         unmodified model on natural episodes)
  CONTENT (DAS only): held-out R2 of linear and quadratic decoders from the site-pooled k-dim projection for m, log r, the exact posterior, and the
         candidate log-evidence at q
  TRACE  (DAS only) layer-2 path patching of the DAS component: into keys only, queries only, values only, or the residual skip only (which then
         feeds MLP2 and the readout); shares of
         the full DAS-interchange effect; change of mixture-head attention scores at the q tokens vs change of exact log r.
Usage: CA_TAG=_dev python3 scripts/c_carrier.py SPEC  -> logs/c_carrier<TAG>.json (keyed by SPEC)"""
import json, os, sys, itertools, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import xc, ftlib as F, native as NV, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore")
sp = sys.argv[1]; UN = sp.replace(":", "_"); M0 = xc.load(sp); E2 = GL.Env(3, 2, beta=0.6); ST = int(os.environ.get("CA_STREAM", 90_200_000)); K = 3
cands = glob.glob(os.path.join(HERE, "..", "checkpoints", "adapt3*", f"{UN}_early01_b0.6_lr0.001.npz")) + glob.glob(os.path.join(NV.X13C, "checkpoints", "adapt3", f"{UN}_early01_b0.6_lr0.001.npz"))
z = np.load(cands[0]); MA = dict(M0); MA["layers"] = [dict(l) for l in M0["layers"]]
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
def train_das(nm, seed, idx):
    d = S[nm]; P = jax.random.normal(jax.random.PRNGKey(seed), (64, K)); opt = optax.adam(3e-2); st = opt.init(P); m_all = site_mask(d["q"]); tgt = RP.sm(d["Lb"])
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
out = dict(unit=sp, n_pairs=dict(JDM=n, JD0=len(JD0[2]), DEC=len(DEC[2])), sets={})
# unrelated-decision damage set
Ln0 = logits(Xn[:800])[inn[:800], qn[:800]]; Pn0 = RP.sm(Ln0)
def damage(P, mode):
    L = patched(Xn[:800], qn[:800], None, P, mode, mu=mu_site); Q = RP.sm(L); return float((Pn0 * (np.log(Pn0 + 1e-12) - np.log(Q + 1e-12))).sum(-1).mean())
d = S["JDM"]
for nm, P in DIRS.items():
    r = {}
    Lp = patched(d["Xa"][TE], d["q"][TE], d["Zb"][TE], P); dp = dlt(d["La"][TE], Lp); f = RP.fid(dp, d["dB"][TE])
    own = d["own"][TE]; r["SUFF"] = dict(slope=f["slope"], R2=f["R2"], own_share=float((dp * own).sum() / (own * own).sum()), mean_abs=float(np.abs(dp).mean()))
    for nn_ in ("JD0", "DEC"):
        e = S[nn_]; Lq = patched(e["Xa"], e["q"], e["Zb"], P); dq = dlt(e["La"], Lq)
        r[f"NULL_{nn_}"] = dict(mean_abs=float(np.abs(dq).mean()), rel_to_JDM_patch=float(np.abs(dq).mean() / max(np.abs(dp).mean(), 1e-9)),
                               rel_to_own_null=float(np.abs(dq).mean() / max(np.abs(e["own"]).mean(), 1e-9)), own_null_mean_abs=float(np.abs(e["own"]).mean()))
    Ls = patched(d["Xa"][TE], d["q"][TE], d["Za"][un][TE], P); ds = dlt(d["La"][TE], Ls); fs = RP.fid(ds, d["dB"][TE])
    r["SRCF"] = dict(slope=fs["slope"], R2=fs["R2"], mean_abs=float(np.abs(ds).mean()))
    if nm in ("DAS", "DEC_m", "ENC_m", "CCD_m", "DEC_r", "ENC_r", "RAND0", "DAS_DECOY"):
        nec = {}
        for mode in ("ablate", "meanrep"):
            La_ = patched(d["Xa"][TE], d["q"][TE], None, P, mode, mu=mu_site); Lb_ = patched(d["Xb"][TE], d["q"][TE], None, P, mode, mu=mu_site)
            nec[mode] = dict(remaining_slope=RP.fid(dlt(La_, Lb_), d["dB"][TE])["slope"], damage_KL=damage(P, mode))
        Lrv = patched(d["Xb"][TE], d["q"][TE], d["Za"][TE], P); nec["reverse"] = dict(remaining_slope=RP.fid(dlt(d["La"][TE], Lrv), d["dB"][TE])["slope"])
        Lr_a = patched(d["Xa"][TE], d["q"][TE], None, P, "meanrep", mu=mu_site); Lr_b = patched(d["Xb"][TE], d["q"][TE], None, P, "restore", mu=mu_site, Zsrc=d["Zb"][TE])
        nec["restore"] = dict(slope=RP.fid(dlt(Lr_a, Lr_b), d["dB"][TE])["slope"]); r["NEC"] = nec
    out["sets"][nm] = r
    print(nm, "SUFF", round(r["SUFF"]["slope"], 3), round(r["SUFF"]["R2"], 3), "NULL_D0", round(r["NULL_JD0"]["rel_to_JDM_patch"], 3), "NULL_DEC", round(r["NULL_DEC"]["rel_to_JDM_patch"], 3),
          "SRCF", round(r["SRCF"]["slope"], 3), {k: {a: round(b, 4) for a, b in v.items()} for k, v in r.get("NEC", {}).items()}, flush=True)
# ---------------- CONTENT of the DAS subspace (site-pooled k-dim projection)
P = DIRS["DAS"]; proj = (Zn[inn[:, None], toks] @ P).reshape(3000, -1)                                        # 7 tokens x 3 dims = 21 features
PBn = XB.posterior_shared(E2, Xn, qn); a6 = Xn.reshape(3000, 6, 6)[inn, qn]; EV = np.where(a6[..., None] == np.arange(3), np.log(0.8), np.log(0.1)).sum(1)
targets = dict(m=mstat(Xn)[inn, qn], logr=logr(Xn, qn), posterior=RP.cen(np.log(PBn + 1e-12)), cand_evidence=EV - EV.mean(-1, keepdims=True))
def r2fit(Fz, T):
    tr, te = np.arange(2000), np.arange(2000, 3000); Fc = np.c_[Fz, np.ones(len(Fz))]
    W = np.linalg.lstsq(Fc[tr], T[tr], rcond=None)[0]; Yh = Fc[te] @ W; return float(1 - ((T[te] - Yh) ** 2).sum() / ((T[te] - T[te].mean(0)) ** 2).sum())
quad = np.c_[proj, (proj[:, :, None] * proj[:, None, :]).reshape(3000, -1)]
out["CONTENT"] = {k: dict(linear=r2fit(proj, T), quadratic=r2fit(quad, T), full_site_linear=r2fit(Zn[inn[:, None], toks].reshape(3000, -1), T)) for k, T in targets.items()}
print("CONTENT", {k: {a: round(b, 3) for a, b in v.items()} for k, v in out["CONTENT"].items()}, flush=True)
# ---------------- TRACE through frozen layer 2: patch the DAS component of the donor into keys / queries / values only, or only the MLP input
lay = MA["layers"][2]; te_ = TE; Za, Zb_, qq = d["Za"][te_], d["Zb"][te_], d["q"][te_]; ii = np.arange(len(qq)); m = jnp.asarray(site_mask(qq))[..., None]; Pj = jnp.asarray(P)
Zmix = jnp.where(m, jnp.asarray(Za) + ((jnp.asarray(Zb_) - jnp.asarray(Za)) @ Pj) @ Pj.T, jnp.asarray(Za))
def l2(zq, zk, zv, zs=None):
    def ln1(z): return F.N.ln(z, lay["g1"], lay["b1"])
    qh = jnp.einsum("ntd,hde->nhte", ln1(zq), lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", ln1(zk), lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", ln1(zv), lay["WV"])
    sc = jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]); A = jax.nn.softmax(sc, -1)
    xa = (jnp.asarray(Za) if zs is None else zs) + jnp.einsum("nhte,hed->ntd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"])
    hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); xo = xa + hid @ lay["W2"] + lay["c2"]
    return np.asarray(F.N.ln(xo, MA["gf"], MA["bf"])[:, 36:42] @ MA["WA"])[ii, qq], np.asarray(sc)
Za_j = jnp.asarray(Za); L0_, sc0 = l2(Za_j, Za_j, Za_j); Lall, scall = l2(Zmix, Zmix, Zmix, Zmix)
full = dlt(L0_, Lall); TEf = (full * full).sum()
def share(L): dd_ = dlt(L0_, L); return float((dd_ * full).sum() / TEf)
tr_ = dict(keys_only=share(l2(Za_j, Zmix, Za_j)[0]), queries_only=share(l2(Zmix, Za_j, Za_j)[0]), values_only=share(l2(Za_j, Za_j, Zmix)[0]),
           keys_and_queries=share(l2(Zmix, Zmix, Za_j)[0]), skip_only=share(l2(Za_j, Za_j, Za_j, Zmix)[0]))
mass_h = []
for h in range(4):
    rows = np.asarray(jax.nn.softmax(jnp.asarray(sc0), -1))[ii, h, 36 + qq]; mass_h.append(float(np.take_along_axis(rows, qq[:, None] * 6 + np.arange(6)[None], 1).sum(1).mean()))
mixh = [h for h in range(4) if mass_h[h] >= 0.8]; dlr = logr(d["Xb"][te_], qq) - logr(d["Xa"][te_], qq); corr = {}
for h in mixh:
    s0 = sc0[ii, h, 36 + qq]; s1 = scall[ii, h, 36 + qq]; dsc = np.stack([(s1 - s0)[ii[:, None], qq[:, None] * 6 + 2 * g + np.arange(2)[None]].mean(1) for g in range(3)], 1)
    corr[str(h)] = float(np.corrcoef((dsc - dsc.mean(1, keepdims=True)).ravel(), (dlr - dlr.mean(1, keepdims=True)).ravel())[0, 1])
tr_["full_check_vs_forward"] = RP.fid(full, d["dB"][te_])["slope"]; tr_["mixture_heads"] = mixh; tr_["corr_dscore_dlogr"] = corr
out["TRACE"] = tr_; print("TRACE", tr_, flush=True)
outp = os.path.join(HERE, "..", "logs", "c_carrier" + os.environ.get("CA_TAG", "") + ".json"); res = json.load(open(outp)) if os.path.exists(outp) else {}
res[sp] = out; json.dump(res, open(outp, "w"), indent=1)
