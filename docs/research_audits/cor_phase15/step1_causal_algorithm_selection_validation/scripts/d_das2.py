"""XIII-D d_das2 = XIII-C d_das + E1: (i) sites that EXCLUDE the directly edited tokens: ALLX2 (all 43 tokens except the source tokens whose
value differs between base and donor) and ALLXQ2 (all tokens except every source token of each edited question); (ii) a graded eligibility
alternative G: e_g = agree_q(g) * (n_g / 5)^4 (graded version of E; 1, 0.41, 0.13, ... for n = 5, 4, 3), scored like E/A1/A2.
--- XIII-C docstring ---
XIII-C Workstream D2/D4 + B6 (R7): eligibility-variable identification with E/A1/A2-DISCRIMINATING on-support families, across neural SITES,
with corrected controls.
Families (math_identifiability.py T6; every edit keeps >= 1 group agreeing on all six questions in base and edit = on-support):
  E5  relational edit at q' != q: a group agreeing at q moves n_h 5 <-> 4        changes E and A1 eligibility, NOT A2
  E3  relational edit at q' != q: a group agreeing at q moves n_h 4 <-> 3        changes A2 only
  QF5 flip agreement at q of a group with n_h = 5                                 changes E and A2, NOT A1
  QF4 flip agreement at q of a group with n_h = 4                                 changes A2 only
Predicted responses for B(=E), A1, A2 (algorithm_discrimination.py rules; W and L terms are shared and exact).
DAS: a k-dim subspace R (64 x k, orthonormalised by QR of a 64 x k parameter, k >= 1) at a SITE; interchange x <- x + (x_d - x) R R^T at the site's
tokens; trained (DD_STEPS Adam steps, batch 128) to make the patched base output match the model's OWN donor output (KL), on a BALANCED mix of the
four families (the objective names no eligibility rule).  Sites: RO2, QS2, OS2, ROQS2 (resid2 at readout / q sources / other sources / both),
ROOS2 (readout + other sources), ALL2 (all 43 tokens), and RO1, QS1 (resid1).  ROOS2 / ALL2 added after development showed that in
early-readout units relational eligibility information reaches layer 2 at the OTHER-question source tokens (d_routes route O ~ 0.45).
Held-out evaluation per family: own_share (projection of the patched change on the model's own change), R2 of the patched change against the E,
A1 and A2 predictions; same for the model's own full change (behaviour).
Controls per (site, k): NOOP = donor := base (genuine no-op; max |Δlogit| reported, must be 0), RAND = 3 random orthonormal k-subspaces,
SRCFREE = the trained subspace patched with the state of an UNRELATED episode with the same q (no information about this edit),
FULLSITE = whole natural replacement of the site (k = 64; site coverage).  DECOY (best site, k = 8): subspace trained under the same budget to make
the patched output match base + A2-predicted change (can the site be made to express A2?).
Usage: DD_TAG=_dev python3 scripts/d_das.py SPEC...  -> logs/d_das<TAG>.json (resumable per unit)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp, optax
import xc, ftlib as F, native as NV, genlib as GL, relpairs as RP
np.seterr(all="ignore")
ST = int(os.environ.get("DD_STREAM", 86_700_000)); STEPS = int(os.environ.get("DD_STEPS", 250)); KS = [int(k) for k in os.environ.get("DD_KS", "2,8").split(",")]
SITES = os.environ.get("DD_SITES", "RO2,QS2,OS2,ROQS2,RO1,QS1").split(",")
env = GL.Env(3, 1)
def parts(X, q):
    rel, col, noi = GL._group_lik(env, X); relm = rel.mean(-1); i = np.arange(len(X)); lr = np.log(relm / noi)
    W = np.exp(lr.sum(1) - lr[i, q]); Lg = rel[i, q] / noi; cnt = GL.counts(X, 3)[i, q]; a_ = X.reshape(len(X), 6, 3, 2)[i, q]; return W, Lg, cnt, a_[..., 0] == a_[..., 1]
def alg_logp(X, q, which):
    W, Lg, cnt, ag = parts(X, q)
    e = ((cnt == 5) & ag) if which == "E" else (cnt == 5) if which == "A1" else (ag * (cnt / 5.0) ** 4) if which == "G" else ((cnt >= 4) & ag)
    f = e.sum(1, keepdims=True) - e; P = np.maximum((W[..., None] * f[..., None] * Lg).sum(1), 1e-30); return np.log(P / P.sum(-1, keepdims=True))
def make_families(seed, nper):
    rng = np.random.default_rng(seed); X = NV.Bm.sample_fast(NV.ENV, 30000, rng)["X"]; q = rng.integers(0, 6, len(X)); fam = {k: ([], [], []) for k in ("E5", "E3", "QF5", "QF4")}
    a6 = X.reshape(len(X), 6, 3, 2); eq = a6[..., 0] == a6[..., 1]
    for i in range(len(X)):
        if min(len(v[2]) for v in fam.values()) >= nper: break
        qq = q[i]; cnt = eq[i].sum(0) - eq[i, qq]; agq = eq[i, qq]
        kind = ["E5", "E3", "QF5", "QF4"][rng.integers(4)]
        if len(fam[kind][2]) >= nper: continue
        x = X[i].copy()
        if kind in ("E5", "E3"):
            lo, hi = (4, 5) if kind == "E5" else (3, 4); hs = [h for h in range(3) if agq[h] and cnt[h] in (lo, hi)]
            if not hs: continue
            h = hs[rng.integers(len(hs))]; others = [p for p in range(6) if p != qq]
            if cnt[h] == hi: cand = [p for p in others if eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2 * h + 1] = (x[p, 2 * h] + rng.integers(1, 3)) % 3
            else: cand = [p for p in others if not eq[i, p, h]]; p = cand[rng.integers(len(cand))]; x[p, 2 * h + 1] = x[p, 2 * h]
        else:
            n_ = 5 if kind == "QF5" else 4; hs = [h for h in range(3) if cnt[h] == n_]
            if not hs: continue
            h = hs[rng.integers(len(hs))]
            x[qq, 2 * h + 1] = (x[qq, 2 * h] + rng.integers(1, 3)) % 3 if agq[h] else x[qq, 2 * h]
        e2 = x.reshape(6, 3, 2); ok_b = eq[i].all(0).any(); ok_e = (e2[..., 0] == e2[..., 1]).all(0).any()
        if not (ok_b and ok_e): continue
        fam[kind][0].append(X[i]); fam[kind][1].append(x); fam[kind][2].append(qq)
    out = {}
    for k, (Xb, Xd, qq) in fam.items():
        Xb, Xd, qq = np.array(Xb), np.array(Xd), np.array(qq)
        pred = {w: alg_logp(Xd, qq, w) - alg_logp(Xb, qq, w) for w in ("E", "A1", "A2", "G")}; pred = {w: v - v.mean(-1, keepdims=True) for w, v in pred.items()}
        PBb = GL.posterior_closed(env, Xb, qq); PBd = GL.posterior_closed(env, Xd, qq); exact = RP.cen(PBd) - RP.cen(PBb)
        out[k] = dict(Xb=Xb, Xd=Xd, q=qq, pred=pred, E_vs_exact=float(np.abs(pred["E"] - exact).max()))
    return out
TR = make_families(ST, 300); TE = make_families(ST + 1, 250)
fam_info = {k: dict(n_train=len(TR[k]["q"]), n_test=len(TE[k]["q"]), E_vs_exact_maxdiff=TE[k]["E_vs_exact"],
                    pairwise_pred_dist={a + "-" + b: float(np.abs(TE[k]["pred"][a] - TE[k]["pred"][b]).mean()) for a, b in (("E", "A1"), ("E", "A2"), ("A1", "A2"))}) for k in TE}
print("families", fam_info, flush=True)
def site_mask(site, q, n, ed=None):
    m = np.zeros((n, 43), bool); i = np.arange(n)
    if site == "ALLX": m[:] = True; m[:, :36] &= ~ed; return m
    if site == "ALLXQ":
        m[:] = True; eq_ = ed.reshape(n, 6, 6).any(-1)                            # edited questions
        m[:, :36] &= ~np.repeat(eq_, 6, axis=1); return m
    if site == "ALL": m[:] = True; return m
    if site == "ROOS":                                                         # readout token + the 30 other-question source tokens
        m[i, 36 + q] = True; o = np.zeros((n, 43), bool); o[:, :36] = True; o[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = False; return m | o
    if site.startswith("RO"): m[i, 36 + q] = True
    if "QS" in site: m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = True
    if site.startswith("OS"): m[:, :36] = True; m[i[:, None], q[:, None] * 6 + np.arange(6)[None]] = False
    return m
def r2(o, p):
    o, p = o.ravel(), p.ravel(); ss = ((o - o.mean()) ** 2).sum(); b = (o * p).sum() / max((p * p).sum(), 1e-12); return float(1 - ((o - b * p) ** 2).sum() / max(ss, 1e-12)) if ss > 1e-10 else 0.0
def corr(o, p): o, p = o.ravel(), p.ravel(); return float(np.corrcoef(o, p)[0, 1]) if o.std() > 1e-9 and p.std() > 1e-9 else 0.0
outp = os.path.join(HERE, "..", "logs", "d_das" + os.environ.get("DD_TAG", "") + ".json"); res = json.load(open(outp)) if os.path.exists(outp) else {}
res["_families"] = fam_info
for sp in sys.argv[1:]:
    if sp in res: continue
    t0 = time.time(); M = xc.load(sp); fwd_rec = jax.jit(lambda X_: (lambda r: (F.forward(M, X_, rec=r), (r["resid1"], r["resid2"]))[1])({}))
    def states(X_):
        a = [fwd_rec(jnp.asarray(X_[i:i + 500])) for i in range(0, len(X_), 500)]; return np.concatenate([np.asarray(x[0]) for x in a]), np.concatenate([np.asarray(x[1]) for x in a])
    fl = jax.jit(lambda X_: F.forward(M, X_))
    D = {}
    for split, FAM in (("tr", TR), ("te", TE)):
        for k, v in FAM.items():
            Xb, Xd, q = v["Xb"], v["Xd"], v["q"]; n = len(q); i = np.arange(n); s1d, s2d = states(Xd)
            ru = np.random.default_rng(7 + len(q)); un = np.array([ru.choice(np.where(q == q[j])[0][np.where(q == q[j])[0] != j]) if (q == q[j]).sum() > 1 else j for j in range(n)])
            Lb = np.asarray(F.batched(fl, Xb))[i, q]; Ld = np.asarray(F.batched(fl, Xd))[i, q]
            D[(split, k)] = dict(Xb=Xb, q=q, s1d=s1d, s2d=s2d, un=un, Lb=Lb, Ld=Ld, own=RP.cen(RP.sm(Ld)) - RP.cen(RP.sm(Lb)), pred=v["pred"], ed=(Xb != Xd).reshape(len(q), 36))
    def _pl(Rm, l, m, Xb, q, sd):
        def fn(x): return jnp.where(m[..., None], x + ((sd - x) @ Rm) @ Rm.T, x)
        return F.forward(M, Xb, resid_patch={l: fn})[jnp.arange(Xb.shape[0]), q]
    _plj = {1: jax.jit(lambda Rm, m, Xb, q, sd: _pl(Rm, 1, m, Xb, q, sd)), 2: jax.jit(lambda Rm, m, Xb, q, sd: _pl(Rm, 2, m, Xb, q, sd))}
    def patched_logits(Rm, site, Xb, q, sd, ed=None):
        l = 1 if site.endswith("1") else 2; m = jnp.asarray(site_mask(site[:-1], q, len(q), ed))
        return _plj[l](Rm, m, jnp.asarray(Xb), jnp.asarray(q), jnp.asarray(sd))
    def ortho(P): return jnp.linalg.qr(P)[0]
    def train(site, k, target="own", seed=0):
        P = jax.random.normal(jax.random.PRNGKey(100 + seed), (64, k)); opt = optax.adam(3e-2); st = opt.init(P)
        keys = [kk for kk in D if kk[0] == "tr"]; rng = np.random.default_rng(seed)
        l = 1 if site.endswith("1") else 2
        def lossf(P, m, Xb, q, sd, tgt):
            lp = jax.nn.log_softmax(_pl(ortho(P), l, m, Xb, q, sd), -1); return -(tgt * lp).sum(-1).mean()
        g = jax.jit(jax.value_and_grad(lossf))
        for s in range(STEPS):
            kk = keys[s % len(keys)]; d = D[kk]; j = rng.choice(len(d["q"]), min(128, len(d["q"])), replace=False)
            sd = d["s1d"] if site.endswith("1") else d["s2d"]
            if target == "own": tgt = RP.sm(d["Ld"][j])
            else: tgt = RP.sm(RP.cen(RP.sm(d["Lb"][j])) + d["pred"]["A2"][j])
            m = jnp.asarray(site_mask(site[:-1], d["q"][j], len(j), d["ed"][j]))
            lo, gr = g(P, m, jnp.asarray(d["Xb"][j]), jnp.asarray(d["q"][j]), jnp.asarray(sd[j]), jnp.asarray(tgt)); up, st = opt.update(gr, st); P = optax.apply_updates(P, up)
        return ortho(P)
    def evaluate(Rm, site, mode="patch"):
        r = {}
        for k in TE:
            d = D[("te", k)]; sd = d["s1d"] if site.endswith("1") else d["s2d"]; Xb, q = d["Xb"], d["q"]
            if mode == "srcfree": sd = sd[d["un"]]
            if mode == "noop": sd = np.asarray((states(Xb)[0] if site.endswith("1") else states(Xb)[1]))
            L = np.concatenate([np.asarray(patched_logits(Rm, site, Xb[a:a + 250], q[a:a + 250], sd[a:a + 250], d["ed"][a:a + 250])) for a in range(0, len(q), 250)])
            dd = RP.cen(RP.sm(L)) - RP.cen(RP.sm(d["Lb"]))
            r[k] = dict(own_share=float((dd * d["own"]).sum() / max((d["own"] ** 2).sum(), 1e-12)), **{f"R2_{w}": r2(dd, d["pred"][w]) for w in ("E", "A1", "A2", "G")},
                        **{f"corr_{w}": corr(dd, d["pred"][w]) for w in ("E", "A1", "A2", "G")}, mean_abs=float(np.abs(dd).mean()))
            if mode == "noop": r[k]["max_abs_dlogit"] = float(np.abs(L - d["Lb"]).max())
        return r
    R = dict(behaviour={k: dict(**{f"R2_{w}": r2(D[("te", k)]["own"], D[("te", k)]["pred"][w]) for w in ("E", "A1", "A2", "G")},
                                **{f"corr_{w}": corr(D[("te", k)]["own"], D[("te", k)]["pred"][w]) for w in ("E", "A1", "A2")}) for k in TE}, sites={})
    for site in SITES:
        R["sites"][site] = dict(FULLSITE=evaluate(jnp.eye(64), site))
        for k in KS:
            Rm = train(site, k); e = dict(DAS=evaluate(Rm, site), SRCFREE=evaluate(Rm, site, "srcfree"))
            e["RAND"] = [evaluate(ortho(jax.random.normal(jax.random.PRNGKey(900 + j), (64, k))), site) for j in range(2)]
            if k == KS[0]: e["NOOP"] = evaluate(Rm, site, "noop")
            R["sites"][site][f"k{k}"] = e
        print(sp, site, "full own", {k: round(v["own_share"], 2) for k, v in R["sites"][site]["FULLSITE"].items()},
              {f"k{k}": {fm: (round(v["own_share"], 2), round(v["R2_E"], 2), round(v["R2_A2"], 2)) for fm, v in R["sites"][site][f"k{k}"]["DAS"].items()} for k in KS}, round(time.time() - t0), flush=True)
    best = max(SITES, key=lambda s: np.mean([v["own_share"] for v in R["sites"][s][f"k{KS[-1]}"]["DAS"].values()]))
    Rd = train(best, KS[-1], target="A2", seed=1); R["decoy_A2"] = dict(site=best, k=KS[-1], DAS=evaluate(Rd, best))
    R["sec"] = time.time() - t0; res[sp] = R; json.dump(res, open(outp, "w"), indent=1)
    print(sp, "behaviour", {k: (round(v["R2_E"], 2), round(v["R2_A1"], 2), round(v["R2_A2"], 2)) for k, v in R["behaviour"].items()}, "best", best, round(R["sec"]), flush=True)
