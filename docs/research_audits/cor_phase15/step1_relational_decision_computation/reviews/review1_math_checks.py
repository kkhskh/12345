import sys, itertools, numpy as np
sys.path.insert(0, '/home/claude/12345/docs/research_audits/cor_phase15/step1_relational_decision_computation/src')
import genlib as GL, xii_bayes as XB
np.seterr(all='ignore'); K, G, A = 3, 3, 0.8

# ---------- 1. sampler vs explicit per-question table (my own, independent of package code) ----------
def my_table(roles, beta):
    Xq = np.array(list(itertools.product(range(K), repeat=6)))
    T = np.zeros((len(roles), 729, K))
    for r, (g, hs) in enumerate(roles):
        for y in range(K):
            for cop in (0, 1):
                pc = beta if cop else 1 - beta
                if pc == 0: continue
                for c in range(K):
                    p = np.full(729, pc / K / K)
                    for s in range(6):
                        x = Xq[:, s]; gi = s // 2
                        if gi == g: p = p * np.where(x == y, A, (1 - A) / 2)
                        elif gi in hs: p = p * ((x == c) if cop else 1 / K)
                        else: p = p / K
                    T[r, :, y] += p
    return Xq, T
code = lambda X: (X * (K ** np.arange(5, -1, -1))).sum(-1)
for C, beta in ((1, 1.0), (2, 1.0), (2, 0.6)):
    env = GL.Env(3, C, K, 2, A, beta); Xq, T = my_table(env.roles, beta)
    d = GL.sample(env, 1_000_000, np.random.default_rng(1 + C))
    # joint (X_0, X_1) frequencies vs exact p(X) = mean_r Tm[r,i0] Tm[r,i1]
    Tm = T.sum(-1); i0, i1 = code(d['X'][:, 0]), code(d['X'][:, 1])
    pX = (Tm[:, i0] * Tm[:, i1]).mean(0)
    full = np.einsum('ri,rj->ij', Tm, Tm) / len(env.roles)
    emp = np.bincount(i0 * 729 + i1, minlength=729 ** 2) / 1e6
    # chi-square on cells with expected count >= 20
    e = full.ravel() * 1e6; m = e >= 20
    chi = ((emp.ravel() * 1e6 - e)[m] ** 2 / e[m]).sum()
    print(f'sampler C{C} b{beta}: off-support samples={int((full.ravel()[emp.ravel()>0]==0).sum())}, chi2={chi:.0f} on {m.sum()} cells (mass {full.ravel()[m].sum():.3f})')
    # joint (X_0, Y_0)
    j = np.bincount(i0 * K + d['Y'][:, 0], minlength=729 * K) / 1e6; ej = T.mean(0).ravel()
    print(f'   max |emp - exact| for p(X_0, Y_0): {np.abs(j - ej).max():.2e}  (sd ~ {np.sqrt(ej.max()/1e6):.1e})')

# ---------- 2. exhaustive check at q = 1 (package only tested q = 0) ----------
for C, beta in ((2, 1.0), (2, 0.6)):
    env = GL.Env(3, C, K, 2, A, beta); Xq, T = my_table(env.roles, beta); Tm = T.sum(-1)
    num = np.einsum('rj,riy->jiy', Tm, T) / len(env.roles)          # X_0 = j, X_1 = i, posterior at q=1
    pX = num.sum(-1); I0, I1 = np.meshgrid(np.arange(729), np.arange(729), indexing='ij')
    X = np.stack([Xq[I0.ravel()], Xq[I1.ravel()]], 1); sup = pX.ravel() > 0
    Ps = XB.posterior_shared(env, X[sup], np.ones(sup.sum(), int)); Pt = (num / pX[..., None]).reshape(-1, K)[sup]
    print(f'exhaustive q=1 C{C} b{beta}: max err {np.abs(Ps - Pt).max():.1e}')

# ---------- 3. closed forms R1-R4, my own implementation, at Q=6 ----------
def pieces(X, q):
    n = len(X); a = X.reshape(n, 6, 3, 2); idx = np.arange(n)
    y = np.arange(K)
    rel = np.where(a[..., 0, None] == y, A, (1 - A) / 2) * np.where(a[..., 1, None] == y, A, (1 - A) / 2)   # n,Q,G,K
    ratio = rel.mean(-1) * K * K
    W = np.prod(np.where(np.arange(6)[None, :, None] == q[:, None, None], 1.0, ratio), 1)
    L = rel[idx, q] * K * K
    agree = a[..., 0] == a[..., 1]                                     # n,Q,G
    return a, W, L, agree
def R1(X, q):
    a, W, L, ag = pieces(X, q); e = ag.all(1); S = e.sum(1, keepdims=True)
    P = (W[..., None] * L * (S - e)[..., None]).sum(1); return P / P.sum(-1, keepdims=True), S[:, 0]
def R2(X, q, beta):
    a, W, L, ag = pieces(X, q); c = np.prod(1 - beta + beta * K * ag, 1)
    P = (W[..., None] * L * (c.sum(1, keepdims=True) - c)[..., None]).sum(1); return P / P.sum(-1, keepdims=True)
def R34(X, q, beta):
    a, W, L, ag = pieces(X, q); P = 0
    for g in range(3):
        h1, h2 = [h for h in range(3) if h != g]
        all4 = ag[..., h1] & ag[..., h2] & (a[..., h1, 0] == a[..., h2, 0])
        J = np.prod(1 - beta + beta * K ** 3 * all4, 1)
        P = P + W[:, g, None] * L[:, g] * J[:, None]
    return P / P.sum(-1, keepdims=True)
N = 100000
for C, beta in ((1, 1.0), (1, 0.6), (2, 1.0), (2, 0.6)):
    env = GL.Env(3, C, beta=beta); rng = np.random.default_rng(99 + C + int(10 * beta)); d = GL.sample(env, N, rng)
    X = d['X']; q = rng.integers(0, 6, N); Pt = XB.posterior_shared(env, X, q)
    if C == 1 and beta == 1:
        P1, S = R1(X, q); Pc = GL.posterior_closed(env, X, q); Pd, _ = GL.deepsets_factor(env, X, q)
        print('R1 mine/closed/deepsets max err', np.abs(P1 - Pt).max(), np.abs(Pc - Pt).max(), np.abs(Pd - Pt).max())
    if C == 1 and beta < 1:
        print('R2 graded max err', np.nanmax(np.abs(R2(X, q, beta) - Pt)))
        P1, S = R1(X, q); nanrows = np.isnan(P1).any(-1)
        print(f'  hard rule NaN rows {nanrows.mean():.4f}; acc on defined rows {(P1[~nanrows].argmax(-1) == d["Y"][np.arange(N), q][~nanrows]).mean():.3f}')
    if C == 2:
        P34 = R34(X, q, beta); print(f'R3/4 C2 b{beta} J-form max err', np.nanmax(np.abs(P34 - Pt)))
        Pb = GL.posterior_brute(env, X, q); kl = (Pt * (np.log(Pt + 1e-300) - np.log(Pb + 1e-300))).sum(-1)
        a, W, L, ag = pieces(X, q); e = ag.all(1); S = e.sum(1)
        print(f'  S distribution', np.bincount(S, minlength=4) / N)
        print(f'  product rule: frac KL>1e-9 {np.mean(kl > 1e-9):.3f}; frac KL>0.01 {np.mean(kl > 0.01):.3f}; decision disagree {np.mean(Pb.argmax(-1) != Pt.argmax(-1)):.3f}')
        if beta == 1:
            m3 = S == 3; idx = np.arange(N)
            vq = a[idx, q, :, 0]                                    # value of each group at q (self-agreeing)
            all3 = (vq[:, 0] == vq[:, 1]) & (vq[:, 1] == vq[:, 2])
            print(f'  S=3: frac all three equal at q {all3[m3].mean():.3f}')
            # rule: odd-one-out at q (or any group if all equal) -> p ∝ L_odd
            odd = np.where(vq[:, 1] == vq[:, 2], 0, np.where(vq[:, 0] == vq[:, 2], 1, 2))
            Pr = L[idx, odd]; Pr = Pr / Pr.sum(-1, keepdims=True)
            print(f'  S=3: cross-group-equality-at-q rule max err {np.abs(Pr[m3] - Pt[m3]).max():.1e}')
            # J_H vs e-product: ties in J on S=3 (reliable identical to colluders everywhere)
            Js = []
            for g in range(3):
                h1, h2 = [h for h in range(3) if h != g]
                Js.append((ag[..., h1] & ag[..., h2] & (a[..., h1, 0] == a[..., h2, 0])).all(1))
            Js = np.stack(Js, 1).sum(1); print(f'  S=3: frac with >1 role J=1: {(Js[m3] > 1).mean():.4f}')

# ---------- 5. counterexample: is the B-C coincidence at q'=2 what drives x2? ----------
Av = [0, 1, 2, 0, 1, 2]; B2 = [0, 2, 0, 1, 2, 0]; Cc = [1, 0, 0, 2, 0, 1]
mk = lambda a, b, c: np.array([[[a[t]] * 2 + [b[t]] * 2 + [c[t]] * 2 for t in range(6)]]).reshape(1, 6, 6)
print('B2 vs Cc equal at', [t for t in range(6) if B2[t] == Cc[t]], '; A vs B2 equal at', [t for t in range(6) if Av[t] == B2[t]])
env = GL.Env(3, 2, beta=0.6)
B3 = [0, 2, 1, 1, 2, 0]                                           # B differs from A at t>=1 and from C at every t
print('B3 vs Cc equal at', [t for t in range(6) if B3[t] == Cc[t]], '; A vs B3 equal at', [t for t in range(6) if Av[t] == B3[t]])
for nm, b in (('x2 (doc)', B2), ('x2 w/o B-C coincidence', B3)):
    x = mk(Av, b, Cc); print(nm, np.round(XB.posterior_shared(env, x, np.array([0]))[0], 4), np.round(R34(x, np.array([0]), 0.6)[0], 4))
# ---------- 7. rank at beta=0.6, and rank of the six-question pairwise kernel ----------
for beta in (0.6, 0.1):
    chi = beta * np.eye(K) / K + (1 - beta) / K ** 4
    k6 = 1 - beta + beta * K ** 3 * np.eye(K); k = k6
    for _ in range(5): k = np.kron(k, k6)
    print(f'beta {beta}: rank chi {np.linalg.matrix_rank(chi)}, rank six-question J^beta kernel {np.linalg.matrix_rank(k)} of {K**6}')
