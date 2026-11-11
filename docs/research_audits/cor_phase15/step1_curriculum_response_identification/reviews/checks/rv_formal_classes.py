"""Formal review check 1: class structure (§1 NATURAL_INTERVENTIONS), null-edit availability, pi-null vs p*-null,
exactness of mix_exact, and the 𝔽_mix / 𝔽_vote separation claims.  Read-only use of the package code."""
import os, sys, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(PKG, "src")); import cri as R; E = R.E
from scipy.stats import binom
np.set_printoptions(precision=4, suppress=True)
print("E.RHO (reliability likelihood ratio, called rho in pi formula) =", E.RHO, " log =", np.log(E.RHO), "; generator rho =", R.E.F4.rho)

# ---- 1. classes over the 91 support vectors
S = R.SUPP; P = E.pi_of_n(S); key = [tuple(np.round(p, 12)) for p in P]
cls = {}
for v, k in zip(S, key): cls.setdefault(k, []).append(tuple(v))
nH = (S == 5).sum(1)
by = {1: set(), 2: set(), 3: set()}
for v, k in zip(S, key): by[int((np.array(v) == 5).sum())].add(k)
print("support vectors", len(S), "distinct pi", len(cls), "by |H|:", {h: len(s) for h, s in by.items()})
bad = []
for k, vs in cls.items():
    if len(vs) > 1:
        for a, b in itertools.combinations(vs, 2):
            a, b = np.array(a), np.array(b); H = np.where(a == 5)[0]
            okk = len(H) == 1 and (b[H] == 5).all() and len(set((b - a)[[g for g in range(3) if g != H[0]]])) == 1
            if not okk: bad.append((a, b))
print("pi-equal pairs not of form |H|=1, common shift:", len(bad))
# minimal (+-1) null edits available as a function of (n_g1, n_g2)
av = lambda a, b: (a <= 3 and b <= 3) or (a >= 1 and b >= 1)
print("no minimal null edit for |H|=1 vectors:", [(a, b) for a in range(5) for b in range(5) if not av(a, b)])

# analytic fraction of decisions admitting a minimal null edit (T4: colluder count 5 a.s.)
ap = R.E.G.agree_probs(R.E.F4) if hasattr(R.E.G, "agree_probs") else None
pr, pn = ap["rel"], ap["noise"]
frac = sum(binom.pmf(a, 5, pr) * binom.pmf(b, 5, pn) for a in range(5) for b in range(5) if av(a, b))
print("analytic P(minimal null edit exists) = %.4f  (doc: 81-82%%)" % frac)

# ---- 2. Monte Carlo: pi-null vs p*-null (p* at q invariant to the whole count vector)
rng = np.random.default_rng(1); N = 20000; d = E.sample(N, 7); X = d["X"]; q = rng.integers(0, 6, N)
Xp, ok = E.pi_preserving_edit(X, 0, rng) if False else (None, None)
n = E.loo(X)[np.arange(N), q]
J = R.J_at(X[np.arange(N)][:, None].reshape(N, 6, 6), 0)  # placeholder overwritten below
J = np.stack([R.E.G.joint(R.E.F4, X[i:i + 1, q[i]:q[i] + 1])[0, 0] for i in range(N)])        # N,C,K
pall = np.einsum("vc,nck->nvk", P, J); pall /= pall.sum(-1, keepdims=True)                      # p*(.|x_q, v) for all 91 v
flat = (np.abs(pall - pall[:, :1]).max((1, 2)) < 1e-12)
print("decisions whose p*(.|x_q, v) is identical for ALL 91 support v (every count edit is Bayes-null): %.4f" % flat.mean())
# Bayes-null single-group edits (|change|=1 in one component, staying in support) that are NOT pi-null
idx = {tuple(v): i for i, v in enumerate(S)}; cnt = 0; tot = 0
for i in range(N):
    v0 = n[i]; i0 = idx[tuple(v0)]
    for g in range(3):
        for s in (-1, 1):
            v1 = v0.copy(); v1[g] += s
            if v1[g] < 0 or v1[g] > 5 or tuple(v1) not in idx: continue
            tot += 1; cnt += np.abs(pall[i, idx[tuple(v1)]] - pall[i, i0]).max() < 1e-12
print("single-component edits in support that leave p* at q unchanged (yet change pi): %d / %d = %.4f" % (cnt, tot, cnt / tot))

# ---- 3. mix_exact == exact Bayes on the stored confirmation surface
g = np.load(os.path.join(PKG, "logs", "conf", "generator.npz")); NS = 120
Xg, qs, sV, cB = g["X"], g["qsel"], g["surfV"][:NS], g["cBsurf"][:NS]
Jq = np.concatenate([R.J_at(Xg[i:i + 1], qs[i]) for i in range(NS)]); Jrep = np.repeat(Jq, 36, 0); Vs = sV.reshape(-1, 3)
Xqr = np.repeat(Xg[np.arange(NS), qs[:NS]], 36, 0); grp = np.repeat(np.arange(NS), 36); tgt = cB.reshape(-1, 3)
ex = R.mix_exact(Jrep, Vs); print("max |mix_exact - exact Bayes| on surface:", np.abs(ex - tgt).max())

# ---- 4. separation: vote fit to exact Bayes; vote fit to shrunk mixture members; mix fit to vote members
_, pv = R.fit_family("vote", tgt, Jrep, Xqr, Vs); print("vote -> exact Bayes R2_within = %.3f (doc ~0.1)" % R.r2(pv, tgt, grp)["r2_within"])
thB = np.r_[np.arange(1, 6) * np.log(E.RHO), np.full(5, -6.0), 0.0]
for t in (1.0, 0.3, 0.1, 0.03):
    th = thB.copy(); th[:10] *= t; f = R.mix_logits(th, Jrep, Vs); _, pv = R.fit_family("vote", f, Jrep, Xqr, Vs)
    print("mix member phi scaled by %.2f: vote R2_within = %.3f ; within-sd %.4f" % (t, R.r2(pv, f, grp)["r2_within"], (f - f.reshape(NS, 36, 3).mean(1).repeat(36, 0)).std()))
rr = np.random.default_rng(3)
for j in range(3):
    a = rr.normal(0, 1, 6); f = R.vote_logits(a, Xqr, Vs); _, pm = R.fit_family("mix", f, Jrep, Xqr, Vs)
    print("random vote member %d: mix R2_within = %.3f" % (j, R.r2(pm, f, grp)["r2_within"]))

# ---- 5. representability is implied by Bayes-closeness: Bayes + arbitrary non-mixture perturbation
for lam in (0.25, 0.5, 1.0):
    a = rr.normal(0, 1, 6); v = R.vote_logits(a, Xqr, Vs); w = lambda Z: Z - Z.reshape(NS, 36, 3).mean(1).repeat(36, 0)
    v = v * lam * np.linalg.norm(w(tgt)) / max(np.linalg.norm(w(v)), 1e-9); f = tgt + v
    _, pm = R.fit_family("mix", f, Jrep, Xqr, Vs); _, pv = R.fit_family("vote", f, Jrep, Xqr, Vs)
    print("Bayes + vote-type component (within-norm ratio %.2f): R2_mix %.3f  R2_vote %.3f" % (lam, R.r2(pm, f, grp)["r2_within"], R.r2(pv, f, grp)["r2_within"]))
for eta in (0.3, 0.6):
    f = tgt + eta * tgt.std() * rr.standard_normal(tgt.shape); f -= f.mean(-1, keepdims=True)
    _, pm = R.fit_family("mix", f, Jrep, Xqr, Vs); _, pv = R.fit_family("vote", f, Jrep, Xqr, Vs)
    print("Bayes + iid lookup-table error eta=%.1f: R2_mix %.3f  R2_vote %.3f" % (eta, R.r2(pm, f, grp)["r2_within"], R.r2(pv, f, grp)["r2_within"]))
