"""Track C, part 2: the REPAIRED (higher-order, candidate-indexed) O/D of the project note `claude/od_repair_proposal.md`.

The companion script `od_repair_check.py` named in that note could NOT be located in the repository, the project, or the session
filesystem (searched 2026-11-20).  Its contents are not reconstructed.  This module is an INDEPENDENT implementation of the definitions
exactly as written in the note (section 2) and re-derives its claims R1-R5 from scratch on the XVI GATE family (xvitask.py, vendored).

Definitions (note §2): trust complex = hyperedges h with candidate-indexed energies E_h(x_h, c), declared from the generator.
  D_c(x) = sum_h E_h(x_h, c);  signed contrast Delta_{c,c'} = D_{c'} - D_c;  O(x) = min_c sum_h E_h(x_h, c) - sum_h min_c E_h(x_h, c).
GATE (arity a): g = [x_0+..+x_{a-1} = 0 mod 3]; e(v, c; b) = -log(b if v == c else (1-b)/2);
  h1 = gate + P1: g * e(x_P1, c; b1);  h2 = gate + P2: g * e(x_P2, c; b1);  h0 = gate + P0: (1-g) * e(x_P0, c; b0).
Proof (review B C-MAJ-2).  With E_h = -log of the generative model's likelihood factor for h, an OBSERVED gate g(x) and a uniform class
prior: D_c(x) = sum_h E_h(x_h, c) = -log p(x | c) + const(x), hence Delta_{c,c'} = log p(c|x) - log p(c'|x) (R1); argmin D = Bayes argmax
(R3); deleting h shifts Delta by -(E_h(c') - E_h(c)) (R5); deleting h equals Bayes with h's evidence source marginalised out of the joint
whenever that source enters only through h (M2).  Scope: observed selector variable, each evidence source in exactly one factor, uniform
prior.  M1 (O is a function of the Bayes posterior) holds on GATE only (3 posterior classes); it is false in general (review B §3.2).
Checks: R1 Delta == log posterior ratio; R2 hyperedge order a+1 (the least-squares projection onto order <= a functions is reported as a DESCRIPTIVE approximation only;
representability of the DECISIONS is settled by the exact LP certificate lp_margin with a rational Farkas verification); R3 decisions; R4 O >= 0 and O = 0 => deletion invariance; R5 exact signed deletion effect; and two analyses
the note did not contain: (M1) is O a function of the Bayes posterior? (M2) is the factor-deletion (FD) destination identical to the
prediction of standard Bayesian inference with the factor removed?"""
import numpy as np, itertools
CONDITIONS = [(2, 0.9, 0.9, 1 / 3), (3, 0.9, 0.9, 1 / 3), (2, 0.9, 0.9, 0.05), (3, 0.9, 0.9, 0.05), (2, 0.9, 0.55, 0.05)]   # XVI (a, b1, b0, nu)


def energies(T, XA, a, b1, b0):
    K = T.K; S = XA.shape[1]; P1, P2, P0 = S - 3, S - 2, S - 1; g = T.gate(XA, a).astype(float)
    def e(v, b): return np.stack([-np.log(np.where(v == c, b, (1 - b) / (K - 1))) for c in range(K)], 1)
    return dict(h1=g[:, None] * e(XA[:, P1], b1), h2=g[:, None] * e(XA[:, P2], b1), h0=(1 - g)[:, None] * e(XA[:, P0], b0))


def D_of(E, keep=None): return sum(v for k, v in E.items() if keep is None or k in keep)


def O_of(E, keep=None):
    ks = [k for k in E if keep is None or k in keep]; return D_of(E, ks).min(1) - sum(E[k].min(1) for k in ks)


def order_projection_residual(T, XA, a, PB):
    """Least-squares projection of the Bayes log-odds (c vs c', all candidate pairs) onto the span of all functions of <= a input
    coordinates (an additive complex whose hyperedges have order <= a).  Exact linear algebra over the full input space."""
    S = XA.shape[1]; K = T.K; cols = []
    for k in range(a + 1):
        for sub in itertools.combinations(range(S), k):
            for vals in itertools.product(range(K), repeat=k): cols.append(np.all(XA[:, list(sub)] == np.array(vals), 1).astype(float) if k else np.ones(len(XA)))
    A = np.stack(cols, 1); res = []; follow = []
    L = np.log(PB + 1e-300)
    pred = np.zeros_like(L)
    for c in range(K):
        coef, *_ = np.linalg.lstsq(A, L[:, c] - L[:, 0], rcond=None); pred[:, c] = A @ coef; res.append(float(np.abs(A @ coef - (L[:, c] - L[:, 0])).max()))
    return float(max(res)), pred


def lp_margin(T, XA, PB, order):
    """max t s.t. some additive function F(x, c) = sum_{|h| <= order} phi_h(x_h, c), |phi| <= 1, satisfies F(x, c*) - F(x, c) >= t for every
    input with a unique Bayes argmax c* and every c != c*.  t > 0 <=> an order-<=order additive complex reproduces every Bayes decision."""
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    S = XA.shape[1]; K = T.K; feats = []
    for k in range(order + 1):
        for sub in itertools.combinations(range(S), k): feats.append(sub)
    # feature index: (sub, vals, c)
    keyidx = {}; nf = 0
    for sub in feats:
        for vals in itertools.product(range(K), repeat=len(sub)):
            for c in range(K): keyidx[(sub, vals, c)] = nf; nf += 1
    u = unique(PB); rows = []
    for i in np.where(u)[0]:
        cs = int(PB[i].argmax())
        for c in range(K):
            if c != cs: rows.append((i, cs, c))
    A = lil_matrix((len(rows), nf + 1))
    for r, (i, cs, c) in enumerate(rows):
        x = XA[i]
        for sub in feats:
            vals = tuple(int(x[j]) for j in sub); A[r, keyidx[(sub, vals, cs)]] -= 1; A[r, keyidx[(sub, vals, c)]] += 1
        A[r, nf] = 1.0                                                 # -(F(c*) - F(c)) + t <= 0
    cobj = np.zeros(nf + 1); cobj[nf] = -1.0
    res = linprog(cobj, A_ub=A.tocsr(), b_ub=np.zeros(len(rows)), bounds=[(-1, 1)] * nf + [(None, 1)], method="highs")
    out = dict(max_margin=float(-res.fun) if res.success else None, status=res.message, n_constraints=len(rows), n_vars=nf, representable=bool(res.success and -res.fun > 1e-9))
    if not out["representable"]: out["farkas_exact"] = farkas_exact(A.tocsr()[:, :nf])
    return out


def farkas_exact(Mx):
    """Gordan certificate that no strictly separating additive function exists: y >= 0, sum y = 1, M^T y = 0 (rows of M = constraint
    difference vectors).  Found by LP, rationalised (limit_denominator), and verified EXACTLY in rational arithmetic."""
    from scipy.optimize import linprog
    from fractions import Fraction
    m, n = Mx.shape; Aeq = np.vstack([Mx.T.toarray(), np.ones((1, m))]); beq = np.concatenate([np.zeros(n), [1.0]])
    r = linprog(np.zeros(m), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * m, method="highs")
    if not r.success: return dict(found=False, status=r.message)
    y = [Fraction(v).limit_denominator(10 ** 6) if v > 1e-12 else Fraction(0) for v in r.x]; Mc = Mx.tocoo(); acc = [Fraction(0)] * n
    for i_, j_, v in zip(Mc.row, Mc.col, Mc.data):
        if y[i_]: acc[j_] += y[i_] * Fraction(int(round(v)))
    return dict(found=True, verified_exact=bool(all(a == 0 for a in acc) and all(v >= 0 for v in y) and sum(y) > 0), support=int(sum(v > 0 for v in y)))


def gate_flip_pairs(T, XA, a):
    idx = {tuple(x): i for i, x in enumerate(XA)}; g = T.gate(XA, a); pairs = []
    for i, x in enumerate(XA):
        for v in range(T.K):
            if v == x[a - 1]: continue
            y = x.copy(); y[a - 1] = v; j = idx[tuple(y)]
            if g[i] != g[j]: pairs.append((i, j))
    return np.array(pairs)


def unique(P, tol=1e-9): s = np.sort(P, 1); return s[:, -1] - s[:, -2] > tol


def check_condition(T, a, b1, b0, nu):
    XA = T.xall(6); px, PB = T.joint(a, b1, b0, nu); E = energies(T, XA, a, b1, b0); D = D_of(E); K = T.K; out = {}
    L = np.log(PB); R1 = max(float(np.abs((D[:, c] - D[:, cp]) - (L[:, cp] - L[:, c])).max()) for c in range(K) for cp in range(K))
    out["R1_max_err_signed_identity"] = R1
    pairs = gate_flip_pairs(T, XA, a); dec = (PB[pairs[:, 0]].argmax(1) != PB[pairs[:, 1]].argmax(1)) & unique(PB[pairs[:, 0]]) & unique(PB[pairs[:, 1]])
    fol = lambda P: float(((P[pairs[:, 0]].argmax(1) != P[pairs[:, 1]].argmax(1)) & (P[pairs[:, 1]].argmax(1) == PB[pairs[:, 1]].argmax(1)))[dec].mean())
    res_a, pred_a = order_projection_residual(T, XA, a, PB); res_a1, pred_a1 = order_projection_residual(T, XA, a + 1, PB)
    out["R2_hyperedge_orders"] = dict(h1=a + 1, h2=a + 1, h0=a + 1)
    out["R2_order_le_a_best_fit_max_residual"] = res_a; out["R2_order_le_a_gate_flip_following"] = fol(np.exp(pred_a - pred_a.max(1, keepdims=True)))
    out["R2_order_le_a_plus_1_max_residual"] = res_a1
    out["R3_accuracy_bayes"] = float((px[:, None] * PB * (PB == PB.max(1, keepdims=True)) / (PB == PB.max(1, keepdims=True)).sum(1, keepdims=True)).sum())
    out["R3_repaired_D_argmin_equals_bayes_argmax"] = bool(np.all((-D).argmax(1)[unique(PB)] == PB.argmax(1)[unique(PB)]))
    out["R3_gate_flip_following_repaired"] = fol(np.exp(-D - (-D).max(1, keepdims=True))); out["R3_n_decisive_gate_flips"] = int(dec.sum())
    O = O_of(E); out["R4_O_min"] = float(O.min()); viol = 0; ncase = 0
    for r in range(1, 3):
        for keep in itertools.combinations(list(E), r):
            m = O < 1e-12; full = (-D).argmax(1); sub = (-D_of(E, keep)).argmax(1)
            u = m & unique(np.exp(-D)) & unique(np.exp(-D_of(E, keep))); ncase += int(u.sum()); viol += int((full[u] != sub[u]).sum())
    out["R4_O0_deletion_invariance_violations"] = viol; out["R4_cases"] = ncase
    r5 = 0.0
    for h in E:
        Dm = D_of(E, [k for k in E if k != h])
        for c in range(K):
            for cp in range(K): r5 = max(r5, float(np.abs(((Dm[:, cp] - Dm[:, c]) - (D[:, cp] - D[:, c])) + (E[h][:, cp] - E[h][:, c])).max()))
    out["R5_max_err_deletion_effect"] = r5
    pos = O > 1e-12; P1, P2 = 3, 4; g = T.gate(XA, a)
    out["O_positive_mass"] = float(px[pos].sum()); out["O_positive_iff_trusted_disagreement"] = bool(np.all(pos == (g & (XA[:, P1] != XA[:, P2]))))
    top2 = np.sort(PB, 1)[:, -1] - np.sort(PB, 1)[:, -2]; out["mean_bayes_top2_gap_O_pos"] = float((px[pos] * top2[pos]).sum() / px[pos].sum()); out["mean_bayes_top2_gap_O_zero"] = float((px[~pos] * top2[~pos]).sum() / px[~pos].sum())
    # M1: is O a (deterministic) function of the Bayes posterior vector?
    keyP = np.round(np.sort(PB, 1), 10); groups = {}
    for k_, o in zip(map(tuple, keyP), O): groups.setdefault(k_, []).append(o)
    out["M1_O_spread_within_equal_posterior_groups"] = float(max(max(v) - min(v) for v in groups.values())); out["M1_n_posterior_groups"] = len(groups)
    out["M1_O_is_function_of_sorted_posterior"] = bool(out["M1_O_spread_within_equal_posterior_groups"] < 1e-9)
    out["M1_O_pos_iff_bayes_top2_tie"] = bool(np.all(pos == (top2 < 1e-9)))
    # M2: FD destination (delete hyperedge h / marginalise g) vs exact Bayes of the modified generative model
    out["M2_deleted_h_destination_equals_bayes_without_factor"] = {}
    _, PBy = T.joint(a, b1, b0, nu); pxy = px[:, None] * PBy; K3 = T.K ** np.arange(XA.shape[1] - 1, -1, -1); code = XA @ K3
    for h in E:                                         # independent reference (review B C-MAJ-2): marginalise h's evidence source out of the JOINT
        Dm = D_of(E, [k for k in E if k != h]); Pm = np.exp(-(Dm - Dm.min(1, keepdims=True))); Pm /= Pm.sum(1, keepdims=True)
        src = {"h1": 3, "h2": 4, "h0": 5}[h]; rest = code - XA[:, src] * K3[src]; agg = {}
        for r_, v in zip(rest, pxy): agg[r_] = agg.get(r_, 0) + v
        Pb = np.array([agg[r_] / agg[r_].sum() for r_ in rest])
        # deleting h1 (or h2) also removes the source when g = 0, where it carries no evidence; deleting h0 removes P0 when g = 1 likewise
        out["M2_deleted_h_destination_equals_bayes_without_factor"][h] = float(np.abs(Pm - Pb).max())
    PG = T.gatefree(XA, a, b1, b0, nu); chg = PB.argmax(1) != PG.argmax(1); out["latent_marginalisation_change_mass_raw_argmax"] = float(px[chg].sum())
    u2 = unique(PB) & unique(PG); out["latent_marginalisation_change_mass_unique_both"] = float(px[chg & u2].sum())
    out["n_decisive_gate_flips_ordered"] = int(dec.sum()); out["n_decisive_gate_flips_unordered"] = int(dec.sum()) // 2; out["n_decisive_gate_flips_note_value"] = 162; out["latent_change_mass_note_values"] = {"nu=1/3": 0.111, "nu=0.05": 0.017}
    # decision representability by additive complexes of bounded order (exact LP certificate; the note's Proposition S / R2)
    out["LP_max_margin_by_order"] = {k: lp_margin(T, XA, PB, k) for k in sorted({1, a, a + 1})}
    return out


def run(T):
    return {f"{a}:{b1}:{b0}:{round(nu, 4)}": check_condition(T, a, b1, b0, nu) for a, b1, b0, nu in CONDITIONS}
