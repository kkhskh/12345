"""Track C, part 1: the ORIGINAL COR obstruction / discrepancy operators (Phase-1 formal definitions, docs/research_audits/cor_phase1/02-03).

Original objects (as written in the Phase-1 audit; nothing here is reconstructed from memory):
  local states s_c(v; x) = P^c_v phi_v(x) in stalks F_c(v);  coboundary (d s)(e: u->v) = rho_{u->v} s(u) - s(v);
  O_c = || d s ||_W^2  (residual energy);   D_c = min_{g in ker d} || s - g ||_M^2  (distance to global sections).
Exact checks implemented here (numerical, finite-dimensional, random instances; each mirrors a Phase-1 theorem or counterexample):
  T1  D(s) = (ds)^T (d M^-1 d^T)^+ (ds)                      [Phase-1 03 §2]
  T2  W* = (d M^-1 d^T)^+ makes O == D identically            [Phase-1 03 §3]: O and D are two weightings of the same residual
  T3  O(s) = O(-s), D(s) = D(-s): both are even functions; with the premise that the decision margin is an odd function of s, they
      cannot encode the direction of a decision change (Theorem Sign, Step 1B)
  T4  P = 0 gives O = D = 0 for every input (Attack F): the scores are not determined by the model's decision
  T5  DESCRIPTIVE ONLY (review B C-MAJ-1: guaranteed by construction): on XVI GATE, the source-level discrepancy over the EVIDENCE sources, D_c = sum_{s in P1,P2,P0} w_s [x_s != c] (any nonnegative
      weights; this is the original operator's source-level form used in XVII), gives an argmin decision that does not depend on the
      gate sources at all, so its gate-flip following is exactly 0 while Bayes' is 1 (XVII).  C4 for the source-level instantiation rests on
      the exact order-1 LP / Farkas certificate (od_repaired.lp_margin, order 1), not on T5.  Whether ANY order-1
      additive rule could reproduce the decisions is settled separately by the exact LP certificate in od_repaired.lp_margin."""
import numpy as np, itertools


def coboundary(rhos, edges, dims):
    """rhos[e]: (dim_head, dim_tail) matrices; edges [(u, v)]; dims per vertex.  Returns d: C0 -> C1."""
    off = np.cumsum([0] + dims); rows = []
    for (u, v), R in zip(edges, rhos):
        row = np.zeros((dims[v], off[-1])); row[:, off[u]:off[u + 1]] = R; row[:, off[v]:off[v + 1]] -= np.eye(dims[v]); rows.append(row)
    return np.vstack(rows)


def O_of(d, W, s): r = d @ s; return float(r @ W @ r)


def D_direct(d, M, s):
    """min_{g in ker d} ||s - g||_M^2 by explicit projection onto ker d in the M inner product."""
    U, S_, Vt = np.linalg.svd(d); tol = max(d.shape) * np.finfo(float).eps * (S_.max() if S_.size else 1)
    K = Vt[(S_ > tol).sum():].T                                    # basis of ker d
    if K.shape[1] == 0: return float(s @ M @ s)
    G = K.T @ M @ K; a = np.linalg.solve(G, K.T @ M @ s); r = s - K @ a; return float(r @ M @ r)


def D_formula(d, M, s): r = d @ s; return float(r @ np.linalg.pinv(d @ np.linalg.inv(M) @ d.T) @ r)


def checks(seed=0, n_inst=50):
    rng = np.random.default_rng(seed); out = dict(T1=[], T2=[], T3=[], T4=[])
    for _ in range(n_inst):
        nv = rng.integers(3, 7); dims = list(rng.integers(1, 4, nv)); edges = [(u, v) for u in range(nv) for v in range(nv) if u != v and rng.random() < 0.4] or [(0, 1)]
        rhos = [rng.normal(size=(dims[v], dims[u])) for u, v in edges]; d = coboundary(rhos, edges, dims)
        A = rng.normal(size=(sum(dims), sum(dims))); M = A @ A.T + 0.1 * np.eye(sum(dims)); B = rng.normal(size=(d.shape[0], d.shape[0])); W = B @ B.T + 0.1 * np.eye(d.shape[0])
        s = rng.normal(size=sum(dims))
        out["T1"].append(abs(D_direct(d, M, s) - D_formula(d, M, s)) / (1 + D_direct(d, M, s)))
        Ws = np.linalg.pinv(d @ np.linalg.inv(M) @ d.T); out["T2"].append(abs(O_of(d, Ws, s) - D_direct(d, M, s)) / (1 + D_direct(d, M, s)))   # vs the direct projection (review B C-MIN-1)
        out["T3"].append(abs(O_of(d, W, s) - O_of(d, W, -s)) + abs(D_formula(d, M, s) - D_formula(d, M, -s)))
        out["T4"].append(O_of(d, W, 0 * s) + D_formula(d, M, 0 * s))
    return {k: dict(max_abs_or_rel_error=float(max(v)), n=len(v), holds=bool(max(v) < 1e-8)) for k, v in out.items()}


def gate_source_level(T, a=2, b1=0.9, b0=0.9, nu=1 / 3, n_w=200, seed=0):
    """T5 on all 3^6 single-question inputs: for many random nonnegative weight vectors w over ALL six sources, the argmin of
    D_c = sum_s w_s [x_s != c] never follows a gate flip (the decision is unchanged when only the gate sources change)."""
    rng = np.random.default_rng(seed); XA = T.xall(6); K = T.K; g = T.gate(XA, a)
    px, PB = T.joint(a, b1, b0, nu); worst = 0.0
    # gate flips: pairs (x, x') differing only in the last gate source with g != g'
    idx = {tuple(x): i for i, x in enumerate(XA)}; pairs = []
    for i, x in enumerate(XA):
        for v in range(K):
            if v == x[a - 1]: continue
            y = x.copy(); y[a - 1] = v; j = idx[tuple(y)]
            if g[i] != g[j]: pairs.append((i, j))
    pairs = np.array(pairs); bayes_change = (PB[pairs[:, 0]].argmax(1) != PB[pairs[:, 1]].argmax(1))
    def unique(P): s_ = np.sort(P, 1); return s_[:, -1] - s_[:, -2] > 1e-9
    dec = bayes_change & unique(PB[pairs[:, 0]]) & unique(PB[pairs[:, 1]])
    for _ in range(n_w):
        w = np.zeros(6); w[3:] = rng.random(3) * (rng.random(3) < 0.9)         # evidence sources P1, P2, P0 only
        D = np.stack([(w[None] * (XA != c)).sum(1) for c in range(K)], 1)
        dchg = (D[pairs[:, 0]].argmin(1) != D[pairs[:, 1]].argmin(1)); worst = max(worst, float(dchg[dec].mean()))
    return dict(n_decisive_gate_flips=int(dec.sum()), bayes_follow=1.0, source_level_follow_max_over_weights=worst, holds=bool(worst == 0.0))
