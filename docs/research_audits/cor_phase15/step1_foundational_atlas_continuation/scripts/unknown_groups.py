"""Unknown groups (mandate §9): statistical identifiability of group structure, exact Bayes.
 (A) hidden pairing of 6 sources (15 pairings), T1 and T4 roles: posterior mass on the true pairing vs Q (gen_family.posterior_unknown).
 (B) one colluding block of unknown membership and size s in {2,3,4} among 6 sources (all other sources independent reliable,
     accuracy alpha): posterior on the true block, enumerating every block of every size 2..4 (uniform prior), vs Q and rho.
 (C) overlapping groups (misspecified): source 2 copies block {0,1}'s value w.p. 1/2 and block {3,4}'s w.p. 1/2 in each
     question; posterior over candidate blocks, reported as the mass on {0,1,2}, {2,3,4}, {0,1}, {3,4} and {0,1,2,3,4}.
Writes logs/landscape/unknown_groups.json."""
import json, os, sys, itertools
HERE = os.path.dirname(__file__); AT = os.path.join(HERE, "..", "..", "step1_relational_computation_atlas"); sys.path.insert(0, os.path.join(AT, "src"))
import numpy as np, gen_family as G
K = 3; out = {}
rows = []
for nm, pc in (("T1perm", 0.0), ("T4perm", 1.0)):
    for Q in (3, 6, 12):
        f = G.Fam(pc=pc, rho=1.0, perm=True, Q=Q); d = G.sample(f, 1500, np.random.default_rng(8_700_000 + Q))
        pp = G.posterior_unknown(f, d["X"])["pairing_post"]
        true = [tuple(sorted(tuple(sorted((int(d["perms"][i].tolist().index(a)), int(d["perms"][i].tolist().index(b))))) for a, b in G.GROUPS)) for i in range(len(d["X"]))]
        idx = {tuple(sorted(tuple(sorted(p)) for p in P)): j for j, P in enumerate(G.ALL_PAIRINGS)}
        m = np.array([pp[i, idx[true[i]]] for i in range(len(true))])
        rows.append(dict(gen=nm, Q=Q, mass_true_pairing=float(m.mean()), map_correct=float(np.mean([pp[i].argmax() == idx[true[i]] for i in range(len(true))]))))
        print(rows[-1], flush=True)
out["A_hidden_pairing"] = rows
def block_lik(X, block, alpha, rho):
    """P(X | colluding block), per episode, marginalising Y (uniform) per question.  Non-block sources: reliable."""
    N, Q, S = X.shape; ys = np.arange(K); e = (1 - alpha) / (K - 1)
    oth = [s for s in range(S) if s not in block]
    lr = np.prod(np.where(X[:, :, oth, None] == ys, alpha, e), 2)                  # N,Q,K
    xb = X[:, :, list(block)]; same = (xb == xb[:, :, :1]).all(-1)
    lb = rho / K * same + (1 - rho) / K ** len(block)                              # independent of y
    return (np.log((lr / K).sum(-1)) + np.log(lb)).sum(1)
blocks = [b for s in (2, 3, 4) for b in itertools.combinations(range(6), s)]
rowsB = []
for s in (2, 3, 4):
    for rho in (0.5, 1.0):
        for Q in (3, 6, 12):
            rng = np.random.default_rng(8_800_000 + 10 * s + Q); N = 1000; Y = rng.integers(0, K, (N, Q))
            X = np.where(rng.random((N, Q, 6)) < 0.8, Y[..., None], (Y[..., None] + rng.integers(1, K, (N, Q, 6))) % K)
            tb = tuple(range(s)); c = rng.integers(0, K, (N, Q)); cop = rng.random((N, Q)) < rho
            for j in tb: X[:, :, j] = np.where(cop, c, rng.integers(0, K, (N, Q)))
            L = np.stack([block_lik(X, b, 0.8, rho) for b in blocks], 1); L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
            ti = blocks.index(tb)
            rowsB.append(dict(size=s, rho=rho, Q=Q, mass_true_block=float(P[:, ti].mean()), map_correct=float((P.argmax(1) == ti).mean()),
                              mass_on_strict_subsets=float(P[:, [i for i, b in enumerate(blocks) if set(b) < set(tb)]].sum(1).mean()),
                              mass_on_strict_supersets=float(P[:, [i for i, b in enumerate(blocks) if set(b) > set(tb)]].sum(1).mean())))
            print(rowsB[-1], flush=True)
out["B_unknown_block_size"] = rowsB
rowsC = []
for Q in (6, 12, 24):
    rng = np.random.default_rng(8_900_000 + Q); N = 1000; Y = rng.integers(0, K, (N, Q))
    X = np.where(rng.random((N, Q, 6)) < 0.8, Y[..., None], (Y[..., None] + rng.integers(1, K, (N, Q, 6))) % K)
    c1, c2 = rng.integers(0, K, (N, Q)), rng.integers(0, K, (N, Q)); X[:, :, 0] = X[:, :, 1] = c1; X[:, :, 3] = X[:, :, 4] = c2
    X[:, :, 2] = np.where(rng.random((N, Q)) < 0.5, c1, c2)
    L = np.stack([block_lik(X, b, 0.8, 1.0) for b in blocks], 1); L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
    r = dict(Q=Q); top = P.mean(0).argsort()[::-1][:4]; r["top_blocks_mean_mass"] = {str(blocks[i]): float(P[:, i].mean()) for i in top}
    rowsC.append(r); print(r, flush=True)
out["C_overlap_misspecified"] = rowsC
json.dump(out, open(os.path.join(HERE, "..", "logs", "landscape", "unknown_groups.json"), "w"), indent=1)
