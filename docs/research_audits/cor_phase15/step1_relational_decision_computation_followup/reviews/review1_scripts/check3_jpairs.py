import sys, itertools, numpy as np
PKG = "/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation_followup"
sys.path.insert(0, PKG + "/src")
import genlib as GL, xii_bayes as XB
src = open(PKG + "/scripts/censored_bayes.py").read().split("res = {}")[0].replace('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = "%s/scripts"' % PKG)
ns = {}; exec(src, ns); censored_post = ns["censored_post"]
cen = lambda P: np.log(P + 1e-12) - np.log(P + 1e-12).mean(-1, keepdims=True)
def jpairs(beta, n=1500, seed=None):
    E2 = GL.Env(3, 2, beta=beta); ST0 = 76_000_000 + int(100 * beta)
    rj = np.random.default_rng(ST0 + 700_000 if seed is None else seed); dj = GL.sample(E2, n, rj); Xj = dj["X"]; qj = rj.integers(0, 6, n); hj = rj.integers(0, 3, n)
    perms = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]; Xj2 = Xj.copy()
    for i in range(n):
        pm = np.array(perms[rj.integers(len(perms))]); cols = [2 * hj[i], 2 * hj[i] + 1]; others = [t for t in range(6) if t != qj[i]]
        Xj2[i][np.ix_(others, cols)] = pm[Xj[i][np.ix_(others, cols)]]
    return E2, Xj, Xj2, qj, hj, dj["ridx"]
for beta in (0.6, 1.0):
    E2, X, X2, q, h, ridx = jpairs(beta); n = len(X); i = np.arange(n)
    a = X.reshape(n, 6, 3, 2); a2 = X2.reshape(n, 6, 3, 2)
    bits_same = ((a[..., 0] == a[..., 1]) == (a2[..., 0] == a2[..., 1])).all()
    q_same = (X[i, q] == X2[i, q]).all()
    P1, P2 = XB.posterior_shared(E2, X, q), XB.posterior_shared(E2, X2, q)
    dB = cen(P2) - cen(P1); ok = np.isfinite(dB).all(1) & ~np.isnan(P1).any(1) & ~np.isnan(P2).any(1)
    dInd = cen(GL.posterior_brute(E2, X2, q)) - cen(GL.posterior_brute(E2, X, q))
    dL = cen(censored_post(X2, q, E2, "LOCALQ")) - cen(censored_post(X, q, E2, "LOCALQ"))
    # n4 per pair
    def n4(aa):
        out = []
        for H in ((0, 1), (0, 2), (1, 2)):
            cs = aa[:, :, list(H), :].reshape(n, 6, 4); out.append((cs == cs[..., :1]).all(-1).sum(1))
        return np.stack(out, 1)
    print(f"beta {beta}: bits unchanged {bits_same}; values at q unchanged {q_same}; n finite {ok.sum()}/{n}")
    print(f"   max|d product rule| {np.nanmax(np.abs(dInd[ok])) if ok.any() else 'na'}; max|d LOCALQ| {np.nanmax(np.abs(dL[ok])) if ok.any() else 'na'}")
    if ok.any():
        print(f"   exact mean|d cen logp| {np.abs(dB[ok]).mean():.3f}; decision change {(P1[ok].argmax(-1) != P2[ok].argmax(-1)).mean():.3f}; frac n4 changed {(n4(a)!=n4(a2)).any(1).mean():.3f}; frac exact change ~0 (<1e-9) {(np.abs(dB[ok]).max(1)<1e-9).mean():.3f}")
    if beta == 1.0:
        # support at beta=1: unnormalised check via whether any role admissible
        g_true = np.array([E2.roles[r][0] for r in ridx])
        def supp(aa):
            s = np.zeros(n, bool)
            for g in range(3):
                H = [k for k in range(3) if k != g]; cs = aa[:, :, H, :].reshape(n, 6, 4); s |= (cs == cs[..., :1]).all(-1).all(1)
            return s
        s2 = supp(a2)
        print(f"   beta=1: x' on support {s2.mean():.3f}; of those, edited group is the reliable one {(h==g_true)[s2].mean():.3f}; "
              f"max|dB| on on-support pairs {np.abs(dB[s2 & ok]).max() if (s2&ok).any() else 'na'}")
    # on-support at 0.6: likelihood positive for every X trivially; check min unnormalised prob > 0 through finite logs
