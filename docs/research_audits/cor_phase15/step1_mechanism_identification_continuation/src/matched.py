"""§5 definitions experiments and §11 matched relation-versus-aggregate interventions.

D-experiments (generator only, plus the frozen engineered circuits):
  D1-D5 same-value/different-decision cases; K=3 dot product vs argmax agreement.
Matched experiments on frozen engineered circuits A (designed), A-sum (sum-only), C' and B:
  X1 raw-sum matching (1,1) vs (3,-1): reader-site nuisance variables differ.
  X2 post-LN aggregate matching at A's reader (exact, by bisection); reachability.
  X3 matching ALL non-relational reader coordinates of A: proved impossible to differ (injectivity).
  X4 C': matching the relation-blind pathway (block-1 MLP ablated) -> decision difference.
  X5 sufficient-statistic impossibility: the sum-only reader gives identical outputs on matched pairs.
  X6 B (QK gate): matched value aggregate with different agreement.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath  # noqa: E402
from cprime import closed_form  # noqa: E402
from semantics import candidate_relative_margin_agreement, sample_k, semantic_agreement  # noqa: E402

LOG, OUT = [], {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def bisect(f, lo, hi, it=100):
    flo = f(lo)
    assert (flo > 0) != (f(hi) > 0), (lo, hi, flo, f(hi))
    for _ in range(it):
        mid = 0.5 * (lo + hi); fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    mA, BA, embA, coordsA = cbmath.agreement_family()
    mS, BS, embS, coordsS = cbmath.sum_only_family()
    e1, e2 = BA[0], BA[1]

    def A_state(a, b):
        tr = cbmath.forward(mA, embA(a, b), 2)
        u, v = coordsA(tr)
        return u, v, cbmath.margin(tr)

    def S_state(a, b):
        tr = cbmath.forward(mS, embS(a, b), 2)
        u, v = coordsS(tr)
        return u, v, cbmath.margin(tr)

    # ---------------- D experiments
    log("D. same-value / different-decision cases for each consistency definition")
    rows = {}
    # D1 semantic agreement, binary: (1.5,1.2) and (-1.5,-1.2) both agree; T3 targets c vs c'
    rows["D1"] = dict(pair=[(1.5, 1.2), (-1.5, -1.2)], value=[1, 1], T3_target=[1, -1],
                      Cprime_m=[closed_form(1.5, 1.2)["m"], closed_form(-1.5, -1.2)["m"]])
    # D2 representation inner product (binary scalar): (1,1) vs (-1,-1) both +1
    rows["D2"] = dict(pair=[(1, 1), (-1, -1)], value=[1, 1], T3_target=[1, -1],
                      Cprime_m=[closed_form(1, 1)["m"], closed_form(-1, -1)["m"]])
    # D3 computational compatibility for A (sign uv at the reader): equal for (2,0.5) & (-2,-0.5);
    # A's decision is equal (by A1*) but C' decision differs
    rows["D3"] = dict(pair=[(2, 0.5), (-2, -0.5)], A_m=[A_state(2, .5)[2], A_state(-2, -.5)[2]],
                      Cprime_m=[closed_form(2, .5)["m"], closed_form(-2, -.5)["m"]])
    # D4 causal compatibility under single-source ablation (set source to 0) in C':
    def d4(a, b):
        base = np.sign(closed_form(a, b)["m"])
        return int(np.sign(closed_form(0, b)["m"]) == base and np.sign(closed_form(a, 0)["m"]) == base)
    g = np.linspace(-3, 3, 41)
    agree = np.array([[a * b > 0 for b in g] for a in g]).ravel()
    comp = np.array([[d4(a, b) for b in g] for a in g]).ravel()
    rows["D4"] = dict(frac_agree_cells_compatible=float(comp[agree].mean()),
                      frac_conflict_cells_compatible=float(comp[~agree].mean()))
    for k, v in rows.items():
        log("  ", k, json.dumps(v, default=float))
    OUT["D"] = rows

    # K=3 dot vs argmax
    rng = np.random.default_rng(21)
    bk = sample_k("T5b", 50000, rng)
    sem = semantic_agreement(bk.zA, bk.zB).astype(bool)
    dot = (bk.zA * bk.zB).sum(1)
    zc = lambda z: z - z.mean(1, keepdims=True)
    dotc = (zc(bk.zA) * zc(bk.zB)).sum(1)
    crm = candidate_relative_margin_agreement(bk.zA, bk.zB)
    res = dict(P_agree=float(sem.mean()),
               P_dot_neg_given_agree=float((dot[sem] < 0).mean()),
               P_dot_pos_given_disagree=float((dot[~sem] > 0).mean()),
               P_centred_dot_neg_given_agree=float((dotc[sem] < 0).mean()),
               P_centred_dot_pos_given_disagree=float((dotc[~sem] > 0).mean()),
               sign_dot_accuracy_for_agreement=float(((dot > 0) == sem).mean()),
               sign_centred_dot_accuracy=float(((dotc > 0) == sem).mean()),
               crm_sign_accuracy=float(((crm > 0) == sem).mean()))
    log("D-K3 (T5 generator, 50000 draws, seed 21):", json.dumps(res))
    OUT["D_K3"] = res
    # explicit example: agree on candidate 0 with negative inner product
    zA = np.array([1.0, -3.0, 0.0]); zB = np.array([1.0, 0.0, -3.0])
    log(f"   explicit: zA={zA}, zB={zB}: argmax both 0, <zA,zB>={zA @ zB}, centred dot="
        f"{zc(zA[None])[0] @ zc(zB[None])[0]:.3f}")

    # ---------------- X1 raw-sum matching
    log("X1 raw-sum matched pair (1,1) vs (3,-1) at A's reader site")
    for ab in [(1, 1), (3, -1)]:
        u, v, m = A_state(*ab)
        log(f"   {ab}: u={u:.5f} v={v:.5f}  S^=u+v={u + v:.5f}  |(u,v)|={math.hypot(u, v):.5f}  m_A={m:+.5f}")
    OUT["X1"] = {str(ab): A_state(*ab) for ab in [(1, 1), (3, -1)]}

    # ---------------- X2 post-LN aggregate matching (exact)
    log("X2 exact post-LN aggregate matching S^(a',b') = S^(a,b), opposite agreement, A vs sum-only")
    X2 = []
    for a, b in [(1, 1), (0.5, 0.5), (2, 1), (1.5, 0.3)]:
        u0, v0, m0 = A_state(a, b)
        S0 = u0 + v0
        for a2 in [2.5, 3.0, 4.0]:
            f = lambda bb: sum(A_state(a2, bb)[:2]) - S0
            try:
                b2 = bisect(f, -6.0, 0.0)
            except AssertionError:
                log(f"   ({a},{b}) -> a'={a2}: no conflicting b' reaches S^={S0:.4f} (unreachable)")
                X2.append(dict(base=(a, b), a2=a2, reachable=False)); continue
            u2, v2, m2 = A_state(a2, b2)
            s0, s2 = S_state(a, b)[2], S_state(a2, b2)[2]
            log(f"   ({a},{b}) S^={S0:.5f} m_A={m0:+.4f} m_sum={s0:+.4f}  |  ({a2},{b2:.5f}) S^={u2 + v2:.5f} "
                f"m_A={m2:+.4f} m_sum={s2:+.4f}")
            X2.append(dict(base=(a, b), a2=a2, b2=b2, mA=(m0, m2), msum=(s0, s2), reachable=True))
    OUT["X2"] = X2
    assert all(abs(r["msum"][0] - r["msum"][1]) < 1e-9 for r in X2 if r["reachable"])

    # ---------------- X3 injectivity
    log("X3 A's reader input (u,v) determines (x,y), hence the relation: matched non-relational AND "
        "relational coordinates are the same state. Numerical inversion check:")
    rng = np.random.default_rng(22)
    errs = []
    d = 6; rho = 3.0
    for _ in range(200):
        x, y = rng.normal(0, 2, 2)
        sig = math.sqrt((x * x + y * y + rho * rho) / d + 1e-5)
        u, v = x / sig, y / sig
        n2 = u * u + v * v                     # = (x^2+y^2)/sig^2 ; solve sig^2 = (x^2+y^2+rho^2)/d+eps
        sig2 = (rho * rho / d + 1e-5) / (1 - n2 / d)
        errs.append(max(abs(u * math.sqrt(sig2) - x), abs(v * math.sqrt(sig2) - y)))
    log(f"   inversion max error {max(errs):.2e} (requires u^2+v^2 < d, always true)")
    OUT["X3_inv_err"] = max(errs)

    # ---------------- X4 C': matched relation-blind pathway
    log("X4 C': pairs with equal relation-blind margin m_blind (block-1 MLP ablated) but opposite agreement")
    mC, DC, embC = cbmath.rereading()
    mCb, _, embCb = cbmath.rereading(block1_mlp=False)
    full = lambda a, b: cbmath.margin(cbmath.forward(mC, embC(a, b), 3))
    blind = lambda a, b: cbmath.margin(cbmath.forward(mCb, embCb(a, b), 3))
    X4 = []
    for a, b in [(1.0, 1.0), (1.5, 0.5), (0.6, 0.6), (2.0, 2.0)]:
        t = blind(a, b)
        for a2 in [2.0, 3.0, 4.0]:
            f = lambda bb: blind(a2, bb) - t
            try:
                b2 = bisect(f, -6.0, -1e-6)
            except AssertionError:
                log(f"   ({a},{b}) -> a'={a2}: unreachable"); X4.append(dict(base=(a, b), a2=a2, reachable=False)); continue
            X4.append(dict(base=(a, b), a2=a2, b2=b2, blind=(t, blind(a2, b2)), full=(full(a, b), full(a2, b2)),
                           reachable=True))
            log(f"   ({a},{b}) m_blind={t:+.4f} m={full(a, b):+.4f}  |  ({a2},{b2:+.4f}) m_blind={blind(a2, b2):+.4f} "
                f"m={full(a2, b2):+.4f}  decision flip: {np.sign(full(a, b)) != np.sign(full(a2, b2))}")
    OUT["X4"] = X4

    # ---------------- X6 B: matched value aggregate
    log("X6 B (QK gate): matched value-path input (B's value coordinate y and A's direct vote x) cannot "
        "differ in agreement; the relation enters only through the score. Matched direct-path margin:")
    mB, DB, embB = cbmath.qk_gate()
    mBo, _, embBo = cbmath.qk_gate(qk_off=True)
    def Bm(a, b, model=mB, emb=embB):
        X, r = emb(a, b); return cbmath.margin(cbmath.forward(model, X, r))
    X6 = []
    for a, b in [(1.0, 1.0), (1.5, 0.7)]:
        t = Bm(a, b, mBo, embBo)
        for a2 in [2.5, 3.5]:
            f = lambda bb: Bm(a2, bb, mBo, embBo) - t
            try:
                b2 = bisect(f, -6.0, -1e-6)
            except AssertionError:
                log(f"   ({a},{b}) -> a'={a2}: unreachable under uniform routing"); X6.append(dict(base=(a, b), a2=a2, reachable=False)); continue
            X6.append(dict(base=(a, b), a2=a2, b2=b2, uniform=(t, Bm(a2, b2, mBo, embBo)), full=(Bm(a, b), Bm(a2, b2)),
                           reachable=True))
            log(f"   ({a},{b}) m_uniform={t:+.4f} m={Bm(a, b):+.4f} | ({a2},{b2:+.4f}) m_uniform={Bm(a2, b2, mBo, embBo):+.4f} "
                f"m={Bm(a2, b2):+.4f}")
    OUT["X6"] = X6

    root = HERE.parent
    (root / "logs" / "matched.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "matched.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
