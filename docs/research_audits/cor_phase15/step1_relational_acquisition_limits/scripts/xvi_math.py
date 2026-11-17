"""XVI exact references for task conditions a:b1:b0:nu[:S] (src/xvitask.py), by enumeration of all K^S inputs of one question.
Per condition: Bayes NLL / accuracy; gate-free mixture; expected-NLL-optimal order-k log-linear rules for k = 1 .. kmax (default a + 1):
NLL, accuracy, KL from Bayes, gate-flip slope G (exact, all on-support flips); the minimal decision order (smallest k with KL < 1e-6);
mutual information of g with every single source, every proper subset of gate sources, the full gate-source tuple, and the pair-agreement
indicator; source marginals; competing-cue statistics: agreement overlap accuracy (best prediction of g from [P1 == P2]) and the gains
   gain_gate          = NLL(order-a optimum) - NLL(Bayes)        (the value of the gate: the best rule that cannot use it, minus Bayes)
   gain_intermediate  = NLL(order-1 optimum) - NLL(order-a optimum) (the value of the gate-free agreement strategy beyond source-wise rules)
   acc_gain_gate      = acc(Bayes) - acc(order-a optimum).
Usage: xvi_math.py OUT.json SPEC [SPEC ...] [--kmax k]"""
import itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, xvitask as T, xvi_math_core as M
K = 3


def cen(P): L = np.log(P + 1e-15); return L - L.mean(-1, keepdims=True)


def flips(XA, a):
    idx = {tuple(x): i for i, x in enumerate(XA)}; out = []
    for i, x in enumerate(XA):
        part = x[:a - 1].sum() % K; g = (x[:a].sum() % K) == 0
        news = [(-part + 1) % K, (-part + 2) % K] if g else [(-part) % K]
        for v in news: y = x.copy(); y[a - 1] = v; out.append((i, idx[tuple(y)], 1 / len(news)))
    return np.array([(i, j) for i, j, _ in out]), np.array([w for _, _, w in out])


def slope(P, PB, px, F):
    (ij, w) = F; i, j = ij[:, 0], ij[:, 1]; dB = cen(PB[j]) - cen(PB[i]); dm = cen(P[j]) - cen(P[i])
    return float((px[i] * w * (dm * dB).sum(1)).sum() / (px[i] * w * (dB * dB).sum(1)).sum())


def MI(px, gv, f):
    keys = [f(x) for x in XA_]; j = {}
    for k, g, p in zip(keys, gv, px): j[(k, g)] = j.get((k, g), 0) + p
    pf, pg = {}, {}
    for (u, g), p in j.items(): pf[u] = pf.get(u, 0) + p; pg[g] = pg.get(g, 0) + p
    return float(sum(p * np.log(p / (pf[u] * pg[g])) for (u, g), p in j.items() if p > 0))


def analyse(spec, kmax=None, fl=True):
    global XA_
    sp = spec.split(":"); a, b1, b0, nu = int(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]); S = int(sp[4]) if len(sp) > 4 else 6
    XA = T.xall(S); XA_ = XA; px, PB = T.joint(a, b1, b0, nu, S); r = dict(a=a, b1=b1, b0=b0, nu=nu, S=S)
    nll = lambda P: float(-(px[:, None] * PB * np.log(P + 1e-15)).sum()); acc = lambda P: float((px * PB[np.arange(len(P)), P.argmax(1)]).sum())
    F = flips(XA, a) if fl else None; GF = T.gatefree(XA, a, b1, b0, nu)
    r["bayes"] = dict(NLL=nll(PB), acc=acc(PB)); r["gatefree"] = dict(NLL=nll(GF), acc=acc(GF), G=slope(GF, PB, px, F) if fl else None)
    kmax = kmax or a + 1
    for k in range(1, kmax + 1):
        P, ok = M.fit(k, px, PB, XA); r[f"order{k}"] = dict(NLL=nll(P), acc=acc(P), KL=nll(P) - nll(PB), G=slope(P, PB, px, F) if fl else None, converged=ok)
    r["min_decision_order"] = next((k for k in range(1, kmax + 1) if r[f"order{k}"]["KL"] < 1e-6), None)
    if a <= kmax: r["order_a_has_zero_G"] = bool(abs(r[f"order{a}"]["G"]) < 1e-6) if fl else None
    gv = (XA[:, :a].sum(1) % K) == 0
    r["MI_g_single"] = {f"x{s}": MI(px, gv, lambda x, s=s: x[s]) for s in range(S)}
    r["MI_g_proper_gate_subsets_max"] = max(MI(px, gv, lambda x, A=A: tuple(x[list(A)])) for m in range(1, a) for A in itertools.combinations(range(a), m))
    r["MI_g_all_gate_sources"] = MI(px, gv, lambda x: tuple(x[:a])); r["MI_g_agree"] = MI(px, gv, lambda x: x[S - 3] == x[S - 2])
    ag = XA[:, S - 3] == XA[:, S - 2]; pa1 = float((px * ag * gv).sum() / (px * gv).sum()); pa0 = float((px * ag * ~gv).sum() / (px * ~gv).sum())
    r["P_agree_g1"], r["P_agree_g0"] = pa1, pa0; r["cue_overlap_acc"] = 0.5 * (max(pa1, pa0) + max(1 - pa1, 1 - pa0))
    r["source_marginals_max_dev"] = float(max(abs((px * (XA[:, s] == v)).sum() - 1 / K) for s in range(S) for v in range(K)))
    if a <= kmax:
        r["gain_gate"] = r[f"order{a}"]["NLL"] - r["bayes"]["NLL"]; r["gain_intermediate"] = r["order1"]["NLL"] - r[f"order{a}"]["NLL"]
        r["acc_gain_gate"] = r["bayes"]["acc"] - r[f"order{a}"]["acc"]; r["gain_gate_vs_gatefree_mixture"] = r["gatefree"]["NLL"] - r["bayes"]["NLL"]
    return r


if __name__ == "__main__":
    args = sys.argv[2:]; kmax = None
    if "--kmax" in args: i = args.index("--kmax"); kmax = int(args[i + 1]); args = args[:i] + args[i + 2:]
    out = {}
    for spec in args:
        r = analyse(spec, kmax); out[spec] = r
        print(spec, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
