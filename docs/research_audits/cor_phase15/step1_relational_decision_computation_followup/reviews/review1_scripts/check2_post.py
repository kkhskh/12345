import sys, itertools, numpy as np
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, PKG + "/src")
import genlib as GL, xii_bayes as XB
src = open(PKG + "/scripts/censored_bayes.py").read().split("res = {}")[0]
src = src.replace('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = "%s/scripts"' % PKG)
ns = {}; exec(src, ns)
censored_post, lik_q = ns["censored_post"], ns["lik_q"]
K, A = 3, 0.8; pr = A**2 + (1-A)**2/(K-1)

def my_full(X, q, beta, localq=False, pergroup_corr=False):
    """own implementation: sum over roles and y with per-question generator factors."""
    n = len(X); a = X.reshape(n, 6, 3, 2); ag = a[..., 0] == a[..., 1]
    P = np.zeros((n, K))
    for g in range(3):
        H = [h for h in range(3) if h != g]
        cs = a[:, :, H, :].reshape(n, 6, 4); eq4 = (cs == cs[..., :1]).all(-1)
        lw = np.zeros(n)
        for qq in range(6):
            if localq or pergroup_corr:
                rel = np.where(ag[:, qq, g], pr, 1 - pr)
                ind = np.where(ag[:, qq, H[0]], 1/K, 1-1/K) * np.where(ag[:, qq, H[1]], 1/K, 1-1/K)
                col = beta * (ag[:, qq, H[0]] & ag[:, qq, H[1]]) + (1 - beta) * ind
            else:
                rel = np.where(ag[:, qq, g], pr / K, (1 - pr) / (K * (K - 1)))
                col = beta * eq4[:, qq] / K + (1 - beta) / K**4
            lw += np.where(qq != q, np.log(rel) + np.log(col), 0)
        aq = a[np.arange(n), q]
        for y in range(K):
            p1 = lambda v: np.where(v == y, A, (1 - A) / (K - 1))
            r = p1(aq[:, g, 0]) * p1(aq[:, g, 1])
            if pergroup_corr:   # no cross-group value equality at q: colluder pairs at q marginal, but copy event shared
                # p(pairs of H at q | shared copy event) with cross-group equality censored: marginalise over the
                # (unobserved) relation -> use per-group pair probs under the event mixture
                e0 = aq[:, H[0], 0] == aq[:, H[0], 1]; e1 = aq[:, H[1], 0] == aq[:, H[1], 1]
                c = beta * (e0 & e1) / K**2 + (1 - beta) / K**4
            else:
                eqq = eq4[np.arange(n), q]; c = beta * eqq / K + (1 - beta) / K**4
            P[:, y] += np.exp(lw) * r * c
    return P / P.sum(1, keepdims=True)

for beta in (1.0, 0.6):
    env = GL.Env(3, 2, beta=beta); rng = np.random.default_rng(123 + int(10*beta)); n = 40000
    d = GL.sample(env, n, rng); X = d["X"]; q = rng.integers(0, 6, n); y = d["Y"][np.arange(n), q]
    Pf = XB.posterior_shared(env, X, q); Pm = my_full(X, q, beta)
    Pl = censored_post(X, q, env, "LOCALQ"); Plm = my_full(X, q, beta, localq=True)
    Pg = censored_post(X, q, env, "PERGROUP"); Pb = GL.posterior_brute(env, X, q)
    Pc = my_full(X, q, beta, pergroup_corr=True)
    acc = lambda P: (P.argmax(-1) == y).mean()
    print(f"beta {beta}: max|shared - mine| {np.nanmax(np.abs(Pf-Pm)):.2e}; max|LOCALQ - mine| {np.nanmax(np.abs(Pl-Plm)):.2e}; "
          f"max|PERGROUP - posterior_brute| {np.nanmax(np.abs(Pg-Pb)):.2e}; max|FULL-LOCALQ| {np.nanmax(np.abs(Pf-Pl)):.2e}")
    print(f"   acc FULL {acc(Pf):.4f} LOCALQ {acc(Pl):.4f} PERGROUP {acc(Pg):.4f} PERGROUP(shared copy event, bits only at q too) {acc(Pc):.4f} nan {np.isnan(Pf).any()} ")
    # ties at beta 0.6 for PERGROUP?
    s = np.sort(Pg, -1); print("   PERGROUP fraction exact ties top-2:", float((np.abs(s[:, -1]-s[:, -2]) < 1e-12).mean()))
    if beta == 1.0:
        a = X.reshape(n, 6, 3, 2); ag = a[..., 0] == a[..., 1]; idx = np.arange(n)
        Aset = np.zeros((n, 3), bool); Ap = np.zeros((n, 3), bool)
        for g in range(3):
            H = [h for h in range(3) if h != g]; cs = a[:, :, H, :].reshape(n, 6, 4); eq4 = (cs == cs[..., :1]).all(-1)
            Aset[:, g] = eq4.all(1)
            m = np.ones((n, 6), bool); m[idx, q] = False
            Ap[:, g] = (ag[:, :, H].all(-1) | ~m).all(1) & eq4[idx, q]
        assert (Aset <= Ap).all()
        extra = (Ap & ~Aset).any(1)
        aq = a[idx, q].reshape(n, 6); all6 = (aq == aq[:, :1]).all(1)
        print("   frac with A' \\ A nonempty:", extra.mean(), " of which all six equal at q:", all6[extra].mean(), " |A|>=2:", (Aset.sum(1) >= 2).mean())
