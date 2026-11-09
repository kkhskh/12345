"""TRANS statistical problem (Workstream C1), exact enumeration over the 36 (sigma, tau) pairs.
For a scored question q, conditions on the numbers of AB and BC questions among the OTHER five questions, and reports:
  Bayes accuracy on q by type; posterior entropy (bits) of sigma, tau and the composite tau.sigma given X_{-q};
  for beta = 0.9 (training value) and beta = 1.0 (noise-free: exact coverage effects).
Noise-free coverage facts (proved in TRANS_COMPOSITION_RESEARCH.md): one observed (a, sigma(a)) pair leaves 2 candidate
permutations of {0,1,2}; two distinct pairs determine sigma.  Writes logs/trans/identifiability.json."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, tasks3 as T3, tasks_ext as TX
K, Q = 3, 6; out = {}
def joint_post(X, beta):
    J = TX.trans_joint(X, beta); lM = np.log(J.sum(-1)); return J, lM
for beta in (0.9, 1.0):
    rng = np.random.default_rng(9_600_000); rows = {}
    d = T3.sample_trans_pairs(20000, rng, beta=beta); X, Y, typ = d["X"], d["Y"], d["typ"]
    J, lM = joint_post(X, beta)
    comp = np.array([[ [k for k, P in enumerate(T3.PERMS) if (P == T3.PERMS[t][T3.PERMS[s]]).all()][0] for s in range(6) for t in range(6)]])[0]
    for q in (0,):
        loo = lM[:, [k for k in range(Q) if k != q]].sum(1); w = np.exp(loo - loo.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)   # N,36
        py = (w[..., None] * J[:, q]).sum(1); acc = py.argmax(-1) == Y[:, q]
        Ws = w.reshape(-1, 6, 6)
        def H(pm): pm = np.clip(pm, 1e-15, 1); return -(pm * np.log2(pm)).sum(-1)
        Hs, Ht = H(Ws.sum(2)), H(Ws.sum(1)); Wc = np.zeros((len(w), 6))
        for j in range(36): Wc[:, comp[j]] += w[:, j]
        Hc = H(Wc)
        nab = (typ[:, 1:] == 0).sum(1); nbc = (typ[:, 1:] == 1).sum(1)
        for a in range(6):
            for b in range(6 - a):
                m = (nab == a) & (nbc == b)
                if m.sum() < 30: continue
                r = dict(n=int(m.sum()), H_sigma=float(Hs[m].mean()), H_tau=float(Ht[m].mean()), H_composite=float(Hc[m].mean()))
                for t, nm in enumerate(["AB", "BC", "C_only"]):
                    mm = m & (typ[:, 0] == t)
                    if mm.sum() >= 10: r[f"acc_{nm}"] = float(acc[mm].mean())
                rows[f"nAB={a},nBC={b}"] = r
    out[f"beta{beta}"] = rows
    print(beta, {k: (v["n"], round(v["H_sigma"], 2), round(v["H_tau"], 2), round(v["H_composite"], 2), round(v.get("acc_C_only", float("nan")), 3)) for k, v in rows.items()}, flush=True)
# noise-free coverage combinatorics: number of permutations of {0,1,2} consistent with m distinct observed pairs
perms = list(itertools.permutations(range(3)))
out["coverage"] = {m: len([p for p in perms if all(p[a] == a for a in range(m))]) for m in range(4)}   # identity w.l.o.g.
print(out["coverage"]); json.dump(out, open(os.path.join(HERE, "..", "logs", "trans", "identifiability.json"), "w"), indent=1)
