"""Workstream C1: verify xii_bayes.posterior_shared against (a) an INDEPENDENT brute-force marginalisation of the sampler's latent variables
(written from the sampler's definition: per question y ~ U(K), copy event cop ~ Bern(beta) SHARED by all colluder sources, copied value c ~ U(K)
SHARED by all colluder sources, per-source reports), exhaustively over every X in a reduced environment (G=3, Q=2, K=3: 3^12 = 531,441 X),
for C in {1, 2} and beta in {1, 0.6}; and (b) Monte-Carlo calibration on genlib.sample at the full size (Q=6).
Also reports how far genlib.posterior_brute (independent-copying model) is from the truth.  -> logs/verify_shared_posterior.json"""
import json, os, sys, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, genlib as GL, xii_bayes as XB
np.seterr(all="ignore"); K, G, A = 3, 3, 0.8
Xq = np.array(list(itertools.product(range(K), repeat=2 * G)))           # all 729 single-question source vectors
def table(roles, beta):
    """p(X_q, y | role) by explicit latent sum; shape (R, 729, K)."""
    out = np.zeros((len(roles), len(Xq), K))
    for r, (g, hs) in enumerate(roles):
        for y in range(K):
            for cop, pc in ((1, beta), (0, 1 - beta)):
                if pc == 0: continue
                for c in range(K):
                    p = np.full(len(Xq), (1 / K) * pc * (1 / K))
                    for s in range(2 * G):
                        gi = s // 2; x = Xq[:, s]
                        if gi == g: p = p * np.where(x == y, A, (1 - A) / (K - 1))
                        elif gi in hs: p = p * (np.where(x == c, 1.0, 0.0) if cop else 1 / K)
                        else: p = p * (1 / K)
                    out[r, :, y] += p
    return out
res = {}; t0 = time.time()
for C in (1, 2):
    for beta in (1.0, 0.6):
        env = GL.Env(3, C, K, 2, A, beta); R = env.roles; T = table(R, beta); Tm = T.sum(-1)       # p(X_q | role)
        # all X = (X_0, X_1): posterior at q = 0:  sum_r T[r, i0, y] * Tm[r, i1]
        num = np.einsum("riy,rj->ijy", T, Tm) / len(R); pX = num.sum(-1); P_true = num / np.maximum(pX[..., None], 1e-300)
        I0, I1 = np.meshgrid(np.arange(729), np.arange(729), indexing="ij")
        X = np.stack([Xq[I0.ravel()], Xq[I1.ravel()]], 1); sup = pX.ravel() > 0
        Ps = XB.posterior_shared(env, X[sup], np.zeros(sup.sum(), int)); Pb = GL.posterior_brute(env, X[sup], np.zeros(sup.sum(), int))
        Pt = P_true.reshape(-1, K)[sup]; w = pX.ravel()[sup]
        res[f"C{C}_beta{beta}"] = dict(total_prob=float(pX.sum()), n_support=int(sup.sum()), shared_max_abs_err=float(np.nanmax(np.abs(Ps - Pt))),
                                       shared_nan=int(np.isnan(Ps).any(-1).sum()), brute_max_abs_err=float(np.nanmax(np.abs(Pb - Pt))),
                                       brute_weighted_mean_abs_err=float((np.abs(Pb - Pt).sum(-1) * w).sum() / w.sum()))
        print(f"C{C} beta{beta}", res[f"C{C}_beta{beta}"], flush=True)
# (b) Monte-Carlo calibration at full size
mc = {}
for C, beta in ((1, 1.0), (2, 1.0), (2, 0.6)):
    env = GL.Env(3, C, K, 6, A, beta); rng = np.random.default_rng(75_000_000 + 10 * C + int(10 * beta)); d = GL.sample(env, 200000, rng)
    q = rng.integers(0, 6, 200000); P = XB.posterior_shared(env, d["X"], q); y = d["Y"][np.arange(200000), q]
    conf = P.max(-1); hit = P.argmax(-1) == y; bins = np.linspace(1 / 3, 1, 9); b = np.clip(np.digitize(conf, bins) - 1, 0, 7)
    ece = float(sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(8) if (b == k).any()))
    nll = float(-np.log(P[np.arange(200000), y] + 1e-12).mean()); Pb = GL.posterior_brute(env, d["X"], q); nllb = float(-np.log(Pb[np.arange(200000), y] + 1e-12).mean())
    mc[f"C{C}_beta{beta}"] = dict(ECE=ece, acc=float(hit.mean()), mean_conf=float(conf.mean()), NLL_shared=nll, NLL_brute=nllb)
    print(f"MC C{C} beta{beta}", mc[f"C{C}_beta{beta}"], flush=True)
json.dump(dict(exhaustive_Q2=res, montecarlo_Q6=mc, seconds=time.time() - t0), open(os.path.join(HERE, "..", "logs", "verify_shared_posterior.json"), "w"), indent=1)
