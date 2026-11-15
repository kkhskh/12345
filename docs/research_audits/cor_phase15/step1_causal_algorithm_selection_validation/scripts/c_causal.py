"""XIII-C Workstream C2/C3/C4: what does a successful two-colluder repair construct, is it USED, and how does the frozen layer 2 use it?
Input: an adapted model from adapt3.py (checkpoints/adapt3/<unit>_<arm>_b0.6_lr1e-3.npz) and its native (C = 1) model.  beta = 0.6.
Pairs (adapt3 constructions, fresh stream CC_STREAM): J_DM (relabelling that changes the sufficient statistic m), J_D0 (does not change m;
exact change 0), DECOY (raw cross-equality changes, m fixed; exact change 0).
C2  linear decodability (ridge, held-out R2) at resid2 (the input of frozen layer 2) of: m_H (3 pairs), the decoy statistic (raw first-source
    cross-equality counts), and the exact per-role log-likelihood of the copying term (= linear in m) -- at RO (readout of q), QS (mean of q's
    source tokens), OS (mean of the other source tokens); native vs adapted.  RESIDUALISED on per-group statistics (self-agreement counts,
    values at q one-hot): R2 beyond what per-group features explain.
C3  causal use: (i) site interchange of resid2 from the J_DM donor at RO / QS / OS / ALL -> fidelity on the exact change; (ii) PROBE-SUBSPACE
    interchange: x <- x + (x_d - x) P P^T at the site, P = orthonormalised 3-dim span of the m-probe weights at that site, vs the same for the
    DECOY-probe span and for random 3-dim subspaces; (iii) PROBE-SUBSPACE MEAN-ABLATION on both base and donor (removes m-information along P):
    the J response that remains.
C3b responsibility hypothesis (see block below): per-token probe of the exact role posterior at q-source tokens, 1-/3-dim interchange.
C4  frozen layer 2: Shapley shares (64 subsets) of the J_DM effect over layer-2 components {A rows at RO (all heads), V inputs at RO / q tokens /
    other tokens, MLP2 at RO, skip at RO} in the adapted model; and the correlation between the change of each group's mixture-head attention
    mass at q and the change of the exact posterior probability that the group is the reliable one.
Usage: CC_TAG=_dev python3 scripts/c_causal.py SPEC ARM  -> logs/c_causal<TAG>.json (keyed SPEC|ARM)"""
import json, os, sys, itertools, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
import xc, ftlib as F, native as NV, genlib as GL, xii_bayes as XB, relpairs as RP
np.seterr(all="ignore")
sp, arm = sys.argv[1], sys.argv[2]; UN = sp.replace(":", "_"); M0 = xc.load(sp); E2 = GL.Env(3, 2, beta=0.6); ST = int(os.environ.get("CC_STREAM", 87_100_000))
def load_adapted():
    z = np.load(os.path.join(HERE, "..", "checkpoints", "adapt3", f"{UN}_{arm}_b0.6_lr0.001.npz")); M = dict(M0); M["layers"] = [dict(l) for l in M0["layers"]]
    for k, v in z.items():
        top, *rest = k.split("/")
        if top in ("l0", "l1", "l2") and rest: M["layers"][int(top[1])][rest[0]] = jnp.asarray(v)
        elif top in ("gf", "bf", "WA"): M[top] = jnp.asarray(v)
    return M
MA = load_adapted(); PAIRS = [(0, 1), (0, 2), (1, 2)]
def mstat(X, decoy=False):
    a = X.reshape(len(X), 6, 3, 2); out = np.zeros((len(X), 6, 3))
    for j, (g, h) in enumerate(PAIRS):
        eq = (a[:, :, g, 0] == a[:, :, h, 0]) if decoy else ((a[:, :, g, 0] == a[:, :, g, 1]) & (a[:, :, h, 0] == a[:, :, h, 1]) & (a[:, :, g, 0] == a[:, :, h, 0]))
        out[:, :, j] = eq.sum(1, keepdims=True) - eq
    return out
def role_post(X, q):
    """exact p(reliable group = g | X) under shared copying (C = 2)."""
    rel, _, _ = GL._group_lik(E2, X); relm = rel.mean(-1); a_ = X.reshape(len(X), 6, 3, 2); i = np.arange(len(X)); T = []
    for g in range(3):
        H = [h for h in range(3) if h != g]; s = a_[:, :, H, :].reshape(len(X), 6, 4); alleq = (s == s[..., :1]).all(-1); colH = 0.6 * alleq / 3 + 0.4 / 81
        T.append(np.log(relm[:, :, g]).sum(1) + np.log(colH).sum(1))
    T = np.stack(T, 1); T = T - T.max(1, keepdims=True); P = np.exp(T); return P / P.sum(1, keepdims=True)
rng = np.random.default_rng(ST); d = GL.sample(E2, 2400, rng); Xj = d["X"]; qj = rng.integers(0, 6, len(Xj)); hj = rng.integers(0, 3, len(Xj))
perms = [np.array(p) for p in itertools.permutations(range(3)) if p != (0, 1, 2)]; Xj2 = Xj.copy()
for i in range(len(Xj)):
    pm = perms[rng.integers(5)]; cols = [2 * hj[i], 2 * hj[i] + 1]; others = [t for t in range(6) if t != qj[i]]; Xj2[i][np.ix_(others, cols)] = pm[Xj[i][np.ix_(others, cols)]]
ii = np.arange(len(Xj)); dm = np.abs(mstat(Xj2)[ii, qj] - mstat(Xj)[ii, qj]).sum(-1) > 0
dB = RP.cen(XB.posterior_shared(E2, Xj2, qj)) - RP.cen(XB.posterior_shared(E2, Xj, qj)); ok = np.isfinite(dB).all(1) & dm
Xb, Xd, q, dB = Xj[ok][:1000], Xj2[ok][:1000], qj[ok][:1000], dB[ok][:1000]; n = len(q); i = np.arange(n)
# probe training set: natural beta = 0.6 episodes
dp = GL.sample(E2, 4000, np.random.default_rng(ST + 1)); Xp = dp["X"]; qp = np.random.default_rng(ST + 2).integers(0, 6, 4000); ip = np.arange(4000)
def pergroup(X, q):
    a = X.reshape(len(X), 6, 3, 2); eq = (a[..., 0] == a[..., 1]); k = np.arange(len(X)); cnt = eq.sum(1) - eq[k, q]
    vq = np.eye(3)[a[k, q]].reshape(len(X), -1); return np.concatenate([np.eye(6)[cnt].reshape(len(X), -1), eq[k, q].astype(float), vq], 1)
def states(M, X):
    f = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), r["resid2"])[1])({})); return np.concatenate([np.asarray(f(jnp.asarray(X[a:a + 500]))) for a in range(0, len(X), 500)])
def site_feat(Z, q, site):
    k = np.arange(len(Z))
    if site == "RO": return Z[k, 36 + q]
    qs = q[:, None] * 6 + np.arange(6)[None]
    if site == "QS": return Z[k[:, None], qs].mean(1)
    m = np.ones((len(Z), 36), bool); m[k[:, None], qs] = False; return (Z[:, :36] * m[..., None]).sum(1) / 30
def ridge(Xf, Y, lam=1.0):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Z = (Xf - mu) / sd; W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0))); return W, mu, sd, Y.mean(0)
def r2(Y, Yh): return float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
out = dict(unit=sp, arm=arm, n_pairs=n, C2={}, C3={}, C4={})
G = pergroup(Xp, qp); tr, te = ip < 3000, ip >= 3000
targets = dict(m=mstat(Xp)[ip, qp], decoy=mstat(Xp, True)[ip, qp])
for nm, M in (("native", M0), ("adapted", MA)):
    Z = states(M, Xp); out["C2"][nm] = {}
    for site in ("RO", "QS", "OS"):
        S = site_feat(Z, qp, site); r = {}
        for tn, Y in targets.items():
            Wg = ridge(G[tr], Y[tr]); res = Y - ((G - Wg[1]) / Wg[2]) @ Wg[0] - Wg[3]               # residual of the target on per-group statistics
            Wf = ridge(S[tr], res[tr]); Yh = ((S[te] - Wf[1]) / Wf[2]) @ Wf[0] + Wf[3]; r[tn] = r2(res[te], Yh)
        out["C2"][nm][site] = r
print("C2", out["C2"], flush=True)
# probe directions (adapted model, residualised m) in RAW resid2 coordinates at each site
ZA = states(MA, Xp); P_dirs = {}
for site in ("RO", "QS", "OS"):
    S = site_feat(ZA, qp, site)
    for tn, Y in targets.items():
        Wg = ridge(G[tr], Y[tr]); res = Y - ((G - Wg[1]) / Wg[2]) @ Wg[0] - Wg[3]; Wf = ridge(S[tr], res[tr]); Wraw = Wf[0] / Wf[2][:, None]
        P_dirs[(site, tn)] = np.linalg.qr(Wraw)[0]
rB, rD = {}, {}; L0 = np.asarray(F.forward(MA, jnp.asarray(Xb), rec=rB))[i, q]; LD = np.asarray(F.forward(MA, jnp.asarray(Xd), rec=rD))[i, q]
z2b, z2d = np.asarray(rB["resid2"]), np.asarray(rD["resid2"]); own = RP.cen(RP.sm(LD)) - RP.cen(RP.sm(L0)); out["C3"]["full"] = RP.fid(own, dB)
def tokmask(site):
    m = np.zeros((n, 43), bool)
    if site == "RO": m[i, 36 + q] = True
    elif site == "QS": m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = True
    elif site == "OS": m[:, :36] = True; m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = False
    else: m[:] = True
    return m
def run(fn): L = np.asarray(F.forward(MA, jnp.asarray(Xb), resid_patch={2: fn}))[i, q]; return RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0)), dB)
for site in ("RO", "QS", "OS", "ALL"):
    m = jnp.asarray(tokmask(site))[..., None]; out["C3"][f"site_{site}"] = run(lambda x: jnp.where(m, jnp.asarray(z2d), x))
mu_site = {site: ZA.reshape(-1, 64).mean(0) for site in ("RO", "QS", "OS")}
for site in ("RO", "QS", "OS"):
    m = jnp.asarray(tokmask(site))[..., None]; r = {}
    for tn in ("m", "decoy"):
        P = jnp.asarray(P_dirs[(site, tn)]); r[f"interchange_{tn}"] = run(lambda x: jnp.where(m, x + ((jnp.asarray(z2d) - x) @ P) @ P.T, x))
    rr = []
    for s in range(3):
        P = jnp.asarray(np.linalg.qr(np.random.default_rng(50 + s).normal(size=(64, 3)))[0]); rr.append(run(lambda x: jnp.where(m, x + ((jnp.asarray(z2d) - x) @ P) @ P.T, x))["slope"])
    r["interchange_random_slope"] = rr
    P = P_dirs[(site, "m")]; mu = mu_site[site]
    def abl(Zs): return Zs - ((Zs - mu) @ P) @ P.T
    Lb_ = np.asarray(F.forward(MA, jnp.asarray(Xb), resid_patch={2: lambda x: jnp.where(m, jnp.asarray(abl(np.asarray(z2b))), x)}))[i, q]
    Ld_ = np.asarray(F.forward(MA, jnp.asarray(Xd), resid_patch={2: lambda x: jnp.where(m, jnp.asarray(abl(np.asarray(z2d))), x)}))[i, q]
    r["ablate_m_remaining"] = RP.fid(RP.cen(RP.sm(Ld_)) - RP.cen(RP.sm(Lb_)), dB); out["C3"][f"probe_{site}"] = r
print("C3", {k: (round(v["slope"], 3) if "slope" in v else {a: (round(b["slope"], 3) if isinstance(b, dict) else [round(x, 3) for x in b]) for a, b in v.items()}) for k, v in out["C3"].items()}, flush=True)

# C3b: RESPONSIBILITY hypothesis (added after the development finding that the m-probe subspace carries ~0 of the J effect although m is
# decodable): early adaptation writes the copying-aware probability that group g is the reliable one into g's q-source tokens, which the frozen
# mixture heads read.  Per-token probe at the six q-source tokens: target log r_g (exact C = 2 role posterior of the token's group), residualised
# on that group's per-group statistics; 1-dim (r only) and 3-dim (log r_g, r_g, sum of g's m_H) probe subspaces; interchange from the J donor
# at the q-source tokens only, vs random subspaces of equal dimension; and mean-ablation of the 1-dim responsibility direction.
def tok_targets(X, q):
    rp = role_post(X, q); mm = mstat(X)[np.arange(len(X)), q]; k = np.arange(len(X)); a = X.reshape(len(X), 6, 3, 2); eq = a[..., 0] == a[..., 1]; cnt = eq.sum(1) - eq[k, q]
    T, Gf = [], []
    for s in range(6):
        g = s // 2; msum = mm[:, [j for j, (u, v) in enumerate(PAIRS) if g in (u, v)]].sum(1)
        T.append(np.stack([np.log(rp[:, g] + 1e-9), rp[:, g], msum], 1))
        Gf.append(np.concatenate([np.eye(6)[cnt[:, g]], eq[k, q, g][:, None].astype(float), np.eye(3)[a[k, q, g, 0]], np.eye(3)[a[k, q, g, 1]], np.eye(2)[[s % 2] * len(X)]], 1))
    return np.stack(T, 1), np.stack(Gf, 1)                                     # (n,6,3), (n,6,F)
Tt, Gt = tok_targets(Xp, qp); Ztok = ZA[ip[:, None], qp[:, None] * 6 + np.arange(6)[None]]          # (4000,6,64)
trt = np.repeat(tr, 6); tet = np.repeat(te, 6); Tt2, Gt2, Z2 = Tt.reshape(-1, 3), Gt.reshape(-1, Gt.shape[-1]), Ztok.reshape(-1, 64)
Wg = ridge(Gt2[trt], Tt2[trt]); res = Tt2 - ((Gt2 - Wg[1]) / Wg[2]) @ Wg[0] - Wg[3]; Wf = ridge(Z2[trt], res[trt]); Yh = ((Z2[tet] - Wf[1]) / Wf[2]) @ Wf[0] + Wf[3]
out["C3"]["resp_probe_R2"] = dict(logr=r2(res[tet][:, :1], Yh[:, :1]), r=r2(res[tet][:, 1:2], Yh[:, 1:2]), msum=r2(res[tet][:, 2:], Yh[:, 2:]))
Wraw = Wf[0] / Wf[2][:, None]; mQS = jnp.asarray(tokmask("QS"))[..., None]; rr = {}
for nm, P in (("resp1", np.linalg.qr(Wraw[:, :1])[0]), ("resp3", np.linalg.qr(Wraw)[0])):
    Pj = jnp.asarray(P); rr[f"interchange_{nm}"] = run(lambda x: jnp.where(mQS, x + ((jnp.asarray(z2d) - x) @ Pj) @ Pj.T, x))
    rr[f"random_{nm}"] = [run(lambda x, Pr=jnp.asarray(np.linalg.qr(np.random.default_rng(70 + s_).normal(size=(64, P.shape[1])))[0]): jnp.where(mQS, x + ((jnp.asarray(z2d) - x) @ Pr) @ Pr.T, x))["slope"] for s_ in range(3)]
P1 = np.linalg.qr(Wraw[:, :1])[0]; mu = Z2.mean(0)
def abl1(Zs): return Zs - ((Zs - mu) @ P1) @ P1.T
Lb_ = np.asarray(F.forward(MA, jnp.asarray(Xb), resid_patch={2: lambda x: jnp.where(mQS, jnp.asarray(abl1(np.asarray(z2b))), x)}))[i, q]
Ld_ = np.asarray(F.forward(MA, jnp.asarray(Xd), resid_patch={2: lambda x: jnp.where(mQS, jnp.asarray(abl1(np.asarray(z2d))), x)}))[i, q]
rr["ablate_resp1_remaining"] = RP.fid(RP.cen(RP.sm(Ld_)) - RP.cen(RP.sm(Lb_)), dB); out["C3"]["resp_QS"] = rr
print("C3b", out["C3"]["resp_probe_R2"], {k: (round(v["slope"], 3) if isinstance(v, dict) else [round(x, 3) for x in v]) for k, v in rr.items()}, flush=True)

# C3c: LEARNED causal subspace (DAS) at the q-source + readout tokens on J_DM pairs (first half train, second half test): is there a low-dim
# causal carrier, and how aligned is it with the decodable (probe) directions?  Alignment = largest cosine between the DAS subspace and the
# responsibility (resp3) / m probe spans (principal angles).
import optax
mQR = jnp.asarray(tokmask("QS") | tokmask("RO"))[..., None]; half = n // 2; trI, teI = np.arange(half), np.arange(half, n)
PD = RP.sm(LD); dcs = {}
def das_pl(Pm, idx):
    Rm = jnp.linalg.qr(Pm)[0]; zb, zd = jnp.asarray(z2b[idx]), jnp.asarray(z2d[idx]); mm = mQR[idx]
    return F.forward(MA, jnp.asarray(Xb[idx]), resid_patch={2: lambda x: jnp.where(mm, x + ((zd - x) @ Rm) @ Rm.T, x)})[jnp.arange(len(idx)), jnp.asarray(q[idx])]
for k in (1, 3, 8):
    Pm = jax.random.normal(jax.random.PRNGKey(300 + k), (64, k)); opt = optax.adam(3e-2); st = opt.init(Pm)
    lossf = lambda Pm_, idx: -(jnp.asarray(PD[idx]) * jax.nn.log_softmax(das_pl(Pm_, idx), -1)).sum(-1).mean()
    rng_ = np.random.default_rng(k)
    for s_ in range(150):
        idx = rng_.choice(trI, 128, replace=False); l_, g_ = jax.value_and_grad(lossf)(Pm, idx); up, st = opt.update(g_, st); Pm = optax.apply_updates(Pm, up)
    Rm = np.asarray(jnp.linalg.qr(Pm)[0]); L = np.asarray(das_pl(Pm, teI)); fte = RP.fid(RP.cen(RP.sm(L)) - RP.cen(RP.sm(L0[teI])), dB[teI])
    cos = {nm: float(np.linalg.svd(Rm.T @ np.linalg.qr(B)[0], compute_uv=False).max()) for nm, B in (("resp3", Wraw), ("m_QS", P_dirs[("QS", "m")]), ("m_RO", P_dirs[("RO", "m")]))}
    rnd = np.linalg.qr(np.random.default_rng(9).normal(size=(64, k)))[0]
    cos_rand = float(np.linalg.svd(rnd.T @ np.linalg.qr(Wraw)[0], compute_uv=False).max())
    dcs[f"k{k}"] = dict(test_fid=fte, max_cos=cos, max_cos_random_baseline=cos_rand)
out["C3"]["das_QSRO"] = dcs
print("C3c", {k: (round(v["test_fid"]["slope"], 3), round(v["test_fid"]["R2"], 3), {a: round(b, 2) for a, b in v["max_cos"].items()}, round(v["max_cos_random_baseline"], 2)) for k, v in dcs.items()}, flush=True)
# C4: layer-2 Shapley over components on J_DM pairs (l2_decomp_all construction) + attention-mass vs exact role-posterior change
lay = MA["layers"][2]; Ad = np.asarray(rD["pattern2"]); mlpd = np.asarray(rD["mlp_out2"]); ro = 36 + q
mR = tokmask("RO"); mQ = tokmask("QS"); mO = tokmask("OS")
def l2run(sub):
    x = jnp.asarray(z2b); z = F.N.ln(x, lay["g1"], lay["b1"]); zd = F.N.ln(jnp.asarray(z2d), lay["g1"], lay["b1"])
    qh = jnp.einsum("ntd,hde->nhte", z, lay["WQ"]); kh = jnp.einsum("ntd,hde->nhte", z, lay["WK"]); vh = jnp.einsum("ntd,hde->nhte", z, lay["WV"]); vd = jnp.einsum("ntd,hde->nhte", zd, lay["WV"])
    for c, mm in (("VR", mR), ("VQ", mQ), ("VO", mO)):
        if c in sub: vh = jnp.where(jnp.asarray(mm)[:, None, :, None], vd, vh)
    A = jax.nn.softmax(jnp.einsum("nhte,nhse->nhts", qh, kh) / jnp.sqrt(qh.shape[-1]), -1); t = jnp.asarray(ro); nn = jnp.arange(n)
    if "A" in sub: A = A.at[nn, :, t].set(jnp.asarray(Ad[i, :, ro]))
    oh = jnp.einsum("nhte,hed->nhtd", jnp.einsum("nhts,nhse->nhte", A, vh), lay["WO"]).sum(1)
    xs = x if "K" not in sub else x.at[nn, t].set(jnp.asarray(z2d[i, ro]))
    xa = xs + oh; hid = jax.nn.gelu(F.N.ln(xa, lay["g2"], lay["b2"]) @ lay["W1"] + lay["c1"], approximate=True); mo = hid @ lay["W2"] + lay["c2"]
    if "M" in sub: mo = mo.at[nn, t].set(jnp.asarray(mlpd[i, ro]))
    return RP.sm(np.asarray(F.N.ln(xa + mo, MA["gf"], MA["bf"])[:, 36:42] @ MA["WA"])[i, q])
comps = ("A", "VR", "VQ", "VO", "M", "K"); P0 = l2run(()); TE = RP.cen(l2run(comps)) - RP.cen(P0); vals = {}
for k in range(7):
    for sub in itertools.combinations(comps, k): nie = RP.cen(l2run(sub)) - RP.cen(P0); vals["+".join(sub)] = float((nie * TE).sum() / (TE * TE).sum())
key = lambda s: "+".join(c for c in comps if c in s)
sh = {c: sum(math.factorial(len(s)) * math.factorial(5 - len(s)) / math.factorial(6) * (vals[key(set(s) | {c})] - vals[key(set(s))]) for k in range(6) for s in itertools.combinations([o for o in comps if o != c], k)) for c in comps}
Ab = np.asarray(rB["pattern2"])
def gm(A_):
    row = A_[i, :, ro]                                                     # (n,4,43): readout-token rows of each head
    return np.stack([row[i, :, q * 6 + 2 * g] + row[i, :, q * 6 + 2 * g + 1] for g in range(3)], -1)   # (n,4,3) mass on each group's q tokens
dmass = gm(Ad) - gm(Ab); drole = role_post(Xd, q) - role_post(Xb, q)
mixh = [h for h in range(4) if np.take_along_axis(Ab[i, h, ro], q[:, None] * 6 + np.arange(6)[None], 1).sum(1).mean() >= 0.8]
out["C4"] = dict(shapley=sh, attention_total=sh["A"] + sh["VR"] + sh["VQ"] + sh["VO"], subset_proj=vals, mixture_heads=mixh,
                 corr_dmass_drole={str(h): float(np.corrcoef(dmass[:, h, :].ravel(), drole.ravel())[0, 1]) for h in range(4)})
print("C4", {k: round(v, 3) for k, v in sh.items()}, out["C4"]["corr_dmass_drole"], "mixture heads", mixh, flush=True)
outp = os.path.join(HERE, "..", "logs", "c_causal" + os.environ.get("CC_TAG", "") + ".json"); res = json.load(open(outp)) if os.path.exists(outp) else {}
res[f"{sp}|{arm}"] = out; json.dump(res, open(outp, "w"), indent=1)
