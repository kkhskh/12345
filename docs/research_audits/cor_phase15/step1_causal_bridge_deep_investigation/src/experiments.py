"""Run every preregistered experiment (PREREGISTRATION.md) and record PASS/FAIL.

Usage: python src/experiments.py   (writes logs/experiments.txt and logs/results.json)
A FAIL is recorded, not repaired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cbmath import (  # noqa: E402
    ACTS, agreement_family, compose, even_part, forward, gauge_head_redistribution, gauge_qk,
    gauge_relu_rescale, gauge_residual_rotation, gauge_split_neuron, gauge_vo, is_relation_blind,
    layernorm, margin, mean_zero_basis, positive_control, qk_gate, random_orthogonal_fixing_one,
    reconciliation, replace_site, sum_only_family,
)

OUT = []
RES = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def record(pid, ok, detail):
    RES[pid] = {"pass": bool(ok), "detail": detail}
    log(f"[{'PASS' if ok else 'FAIL'}] {pid}: {detail}")


BIN = [(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
GRID = [v for v in np.linspace(-1.5, 1.5, 13) if abs(v) > 1e-9]


def readout_hook_mid(model, tr, r):
    """Margin that the residual before the MLP would produce (decomposition)."""
    h = tr.mid[0][r]
    if model.final_ln:
        h = layernorm(h, model.gf, model.bf, model.eps)
    lg = h @ model.WU + model.bU
    return float(lg[0] - lg[1])


# ======================================================== Mechanism A
def mech_A():
    log("\n=== Mechanism A: agreement-sensitive MLP ===")
    tables = {}
    for fl in (True, False):
        model, (e1, e2, e3), embed = positive_control(final_ln=fl)
        rows = []
        for a, b in BIN:
            tr = forward(model, embed(a, b), 2)
            rows.append(dict(a=a, b=b, agree=a == b, margin=margin(tr), pre_mlp=readout_hook_mid(model, tr, 2),
                             attn=tr.pattern[0][0][2].tolist(), mid=(tr.mid[0][2] @ e1, tr.mid[0][2] @ e2),
                             q=float(tr.mlp_out[0][2] @ e3)))
        tables[fl] = rows
        log(f"positive control final_ln={fl}")
        for rw in rows:
            log(f"  a={rw['a']:+.0f} b={rw['b']:+.0f} agree={int(rw['agree'])} margin={rw['margin']:+.10f} "
                f"pre-MLP margin={rw['pre_mlp']:+.1e} q={rw['q']:+.10f} read<-(x,y)=({rw['mid'][0]:+.6f},{rw['mid'][1]:+.6f}) attn={np.round(rw['attn'],4).tolist()}")
    ok = all(np.sign(r_["margin"]) == (1 if r_["agree"] else -1) for fl in tables for r_ in tables[fl])
    record("PA1", ok, "sign(m)=+ iff a=b in all four cells, with and without final LN")
    ok = all(abs(r_["pre_mlp"]) < 1e-12 for fl in tables for r_ in tables[fl])
    record("PA2", ok, "pre-MLP readout margin is 0 in every cell")

    model, (e1, e2, e3), embed = positive_control(final_ln=True)
    rng = np.random.default_rng(0)
    okc = True
    for _ in range(50):
        w = rng.dirichlet([1, 1, 1])
        def hk(li, site, arr, w=w):
            if site == "pattern0":
                arr = arr.copy(); arr[2] = w
            return arr
        for a, b in BIN:
            okc &= np.sign(margin(forward(model, embed(a, b), 2, hk))) == (1 if a == b else -1)
    record("PA3", okc, "50 random clamped attention rows (any positive weights on both sources): sign unchanged in all cells")

    base = {(a, b): margin(forward(model, embed(a, b), 2)) for a, b in BIN}
    ok = all(np.sign(base[(-a, b)]) == -np.sign(base[(a, b)]) and np.sign(base[(a, -b)]) == -np.sign(base[(a, b)])
             and np.isclose(base[(-a, -b)], base[(a, b)]) for a, b in BIN)
    record("PA4", ok, "flip A or flip B reverses sign(m); flipping both leaves m unchanged")
    log("  ablation: MLP removed ->", [round(margin(forward(model, embed(a, b), 2, replace_site(0, 'mlp_out', np.zeros((3, 4))))), 12) for a, b in BIN])

    # independent pure-python implementation
    import positive_control_ref as ref
    refm = {k: ref.margin(*k) for k in BIN}
    log("  pure-python reference margins:", {k: round(v, 10) for k, v in refm.items()})
    RES["PA_ref_maxdiff"] = max(abs(refm[k] - base[k]) for k in BIN)
    log(f"  max |numpy - pure python| = {RES['PA_ref_maxdiff']:.2e}")

    # PA5: law for many activations, fully recomputed continuous block
    viol = {}
    for act in ("gelu", "gelu_erf", "relu", "silu", "softplus"):
        m, _, emb, coords = agreement_family(act=act)
        bad = 0; relu_err = 0.0
        for a in GRID:
            for b in GRID:
                tr = forward(m, emb(a, b), 2)
                x, y = coords(tr)
                mm = margin(tr)
                bad += np.sign(mm) != np.sign(x * y)
                if act == "relu":
                    relu_err = max(relu_err, abs(mm - 2 * np.sign(x * y) * min(abs(x), abs(y))))
        viol[act] = int(bad)
        if act == "relu":
            RES["PA5_relu_formula_maxerr"] = relu_err
    record("PA5", all(v == 0 for v in viol.values()) and RES["PA5_relu_formula_maxerr"] < 1e-12,
           f"sign(m)=sign(xy) on 144-cell grid for {list(viol)}; violations={viol}; ReLU formula max err={RES['PA5_relu_formula_maxerr']:.1e}")

    mags = {}
    for act in ("tanh", "sigmoid"):
        m, _, emb, _ = agreement_family(act=act)
        mags[act] = max(abs(margin(forward(m, emb(a, b), 2))) for a in GRID for b in GRID)
    record("PA6", all(v < 1e-12 for v in mags.values()), f"max |m| for odd/constant-sum activations: {mags}")

    def q(x, y, act="gelu"):
        return even_part(act, x + y) - even_part(act, x - y)
    small = [q(t, t * 0.7) / (4 * np.sqrt(2 / np.pi) * t * t * 0.7) for t in (1e-2, 1e-3)]
    large = [q(t, 0.6 * t) / (2 * min(t, 0.6 * t)) for t in (20.0, 60.0)]
    record("PA7", abs(small[-1] - 1) < 1e-3 and abs(large[-1] - 1) < 1e-3,
           f"small-signal ratio to 4sqrt(2/pi)xy -> {small}; large-signal ratio to 2 sgn(xy) min -> {large}")

    # PA8 matched-sum: intervene on the read residual (site 'mid') with exact (x,y)
    m, (e1, e2, e3, e4, e5), emb, coords = agreement_family(act="gelu")
    def at(x, y):
        tr0 = forward(m, emb(1.0, 1.0), 2)
        mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * e5
        return margin(forward(m, emb(1.0, 1.0), 2, replace_site(0, "mid", mid)))
    trip = [at(1, 1), at(2, 0), at(3, -1)]
    record("PA8", trip[0] > 0 and abs(trip[1]) < 1e-12 and trip[2] < 0,
           f"matched S=2 triple (1,1),(2,0),(3,-1) via residual intervention: {np.round(trip,6).tolist()}")

    ms, _, emb_s, _ = sum_only_family(act="gelu")
    # binary-cell existence of an offset: min over agree > max over disagree
    ms_d, (f1, f2, f3, f4, f5), _, _ = sum_only_family(act="gelu")
    def at_s(x, y):
        tr0 = forward(ms, emb_s(1.0, 1.0), 2)
        mid = tr0.mid[0].copy(); mid[2] = x * f1 + y * f2 + 3.0 * f5
        return margin(forward(ms, emb_s(1.0, 1.0), 2, replace_site(0, "mid", mid)))
    sb = {(a, b): margin(forward(ms, emb_s(a, b), 2)) for a, b in BIN}
    agree_vals = [sb[k] for k in BIN if k[0] == k[1]]; dis_vals = [sb[k] for k in BIN if k[0] != k[1]]
    trip_s = [at_s(1, 1), at_s(2, 0), at_s(3, -1)]
    record("PA9", min(agree_vals) > max(dis_vals) and np.ptp(trip_s) < 1e-12,
           f"sum-only reader separates binary agree/disagree (min agree {min(agree_vals):.4f} > max dis {max(dis_vals):.4f}) yet is constant on the matched-sum triple {np.round(trip_s,6).tolist()}")

    # exact behavioural equivalence on the binary domain (d=4, no final LN)
    model_d, (e1, e2, e3), embed = positive_control(final_ln=False)
    model_s, _, _ = positive_control(final_ln=False)
    L = model_s.layers[0]
    s2 = np.sqrt(0.5 + model_s.eps)
    L.Win = s2 * np.column_stack([e1 + e2, -(e1 + e2)]); L.bin = np.zeros(2)
    L.Wout = np.stack([2 * e3, 2 * e3])
    h2 = float(even_part("gelu", np.array(2.0)))
    model_s.bU = np.array([-h2, h2])          # offset -2 h(2) on the margin
    eqd = max(abs(margin(forward(model_d, embed(a, b), 2)) - margin(forward(model_s, embed(a, b), 2))) for a, b in BIN)
    RES["PA9_binary_equivalence_maxdiff"] = eqd
    log(f"  designed vs sum-only implementation, binary cells: max |m_designed - m_sum| = {eqd:.2e} (identical decisions and margins)")

    # interaction and agreement contrast do not separate the two
    def I(f, a, b):
        return f(a, b) - f(a, 0) - f(0, b) + f(0, 0)
    fq = lambda x, y: q(x, y)
    fs = lambda x, y: even_part("gelu", x + y)
    cells = [(0.5, 0.8), (0.5, -0.8), (1.2, 1.0), (1.2, -1.0)]
    log("  interaction I sign: designed", [int(np.sign(I(fq, *c))) for c in cells], " sum-only", [int(np.sign(I(fs, *c))) for c in cells])
    Eoo = lambda f, x, y: (f(x, y) - f(-x, y) - f(x, -y) + f(-x, -y)) / 4
    log("  odd-odd component: designed", [round(float(Eoo(fq, *c)), 6) for c in cells], " sum-only", [round(float(Eoo(fs, *c)), 6) for c in cells])

    # PA10 removal of conditions
    errs = {}
    e1_, e2_ = mean_zero_basis(6)[:2]
    variants = {
        "A1 skewed read": dict(read_dirs=[2 * e1_ + e2_, -(2 * e1_ + e2_), e1_ - e2_, -(e1_ - e2_)]),
        "A1' read offset": dict(read_offset=[0.3, 0.3, 0.0, 0.0]),
        "A2 unequal out": dict(out_w=(1.3, 1.3, -1, -1)),
        "A3 tanh": dict(act="tanh"),
        "A4 direct path": dict(readout_mix=0.5),
        "A4 LN bias": dict(final_ln=True, kappa_bias=0.3),
    }
    for nm, kw in variants.items():
        mm, _, emb, _ = agreement_family(**kw)
        errs[nm] = sum(np.sign(margin(forward(mm, emb(a, b), 2))) != np.sign(a * b) for a in GRID for b in GRID) / len(GRID) ** 2
    record("PA10", all(v > 0 for v in errs.values()), f"fraction of grid cells with sign(m)!=sign(ab): {{{', '.join(f'{k}: {v:.3f}' for k,v in errs.items())}}}")

    # PA11 perturbations
    m0, _, emb, coords = agreement_family(act="gelu")
    summ = {}
    for eta in (0.01, 0.03, 0.1, 0.3):
        bin_ok = 0; grid_err = []; near_axis = []
        for s in range(20):
            rng = np.random.default_rng(100 + s)
            mp = __import__("copy").deepcopy(m0)
            Lp = mp.layers[0]
            for nm in ("WV", "WO", "Win", "Wout"):
                A = getattr(Lp, nm)
                setattr(Lp, nm, A + eta * rng.normal(0, 1, A.shape) * np.abs(A).max())
            mp.WU = mp.WU + eta * rng.normal(0, 1, mp.WU.shape) * np.abs(mp.WU).max()
            bin_ok += all(np.sign(margin(forward(mp, emb(a, b), 2))) == np.sign(a * b) for a, b in BIN)
            bad = [(a, b) for a in GRID for b in GRID if np.sign(margin(forward(mp, emb(a, b), 2))) != np.sign(a * b)]
            grid_err.append(len(bad) / len(GRID) ** 2)
            near_axis += [min(abs(a), abs(b)) for a, b in bad]
        summ[eta] = dict(binary_all_correct=f"{bin_ok}/20", grid_err_mean=round(float(np.mean(grid_err)), 4),
                         median_min_abs_evidence_at_errors=(round(float(np.median(near_axis)), 3) if near_axis else None))
    log("  perturbations:", summ)
    record("PA11", summ[0.01]["binary_all_correct"] == "20/20" and
           all(v["median_min_abs_evidence_at_errors"] is None or v["median_min_abs_evidence_at_errors"] <= 0.75 for v in summ.values()),
           "binary cells survive small perturbations; errors concentrate where the weaker source is small (grid values in [0.25,1.5])")

    m, _, emb, coords = agreement_family(act="gelu", final_ln=True)
    bad = sum(np.sign(margin(forward(m, emb(a, b), 2))) != np.sign(a * b) for a in GRID for b in GRID)
    record("PA12", bad == 0, f"continuous family with final LN: violations {bad}/144")

    m, _, emb, _ = agreement_family(act="gelu")
    dev = max(abs(margin(forward(m, emb(a, b), 2)) - margin(forward(m, emb(-a, -b), 2))) for a in GRID for b in GRID)
    record("PA13", dev < 1e-12, f"max |m(a,b) - m(-a,-b)| = {dev:.1e}: agreeing on c' pushes toward c (not candidate-relative)")

    # section-4 quantities on the binary cells
    model, (e1, e2, e3), embed = positive_control(final_ln=False)
    log("  candidate quantities on binary cells (source writes at read: a e1, b e2):")
    for a, b in BIN:
        u, v = a * e1, b * e2
        log(f"    a={a:+.0f} b={b:+.0f}: cos(src writes)={u@v/np.linalg.norm(u)/np.linalg.norm(v):+.3f} "
            f"|u-v|={np.linalg.norm(u-v):.3f} |x-y| (pairing e1<->e2)={abs(a-b):.0f} xy={a*b:+.0f} "
            f"q={q(np.array(a),np.array(b)):+.4f}")


# ======================================================== Mechanism B
def mech_B():
    log("\n=== Mechanism B: query-key agreement gate ===")
    model, D, embed = qk_gate()
    fU = D["fU"]
    LV = [-1.0, -0.5, 0.5, 1.0]

    def parts(tr, r):
        h2 = 2 * float(tr.head_out[1][0][r] @ fU)
        h1 = 2 * float(tr.head_out[0][0][r] @ fU)
        return h1, h2

    rows = []
    mono_ok = True
    for a in LV:
        for b in LV:
            X, r = embed(a, b)
            tr = forward(model, X, r)
            s = tr.scores[1][0][r]
            zq = layernorm(tr.mid[0][r], model.layers[1].g1, model.layers[1].b1, model.eps) @ D["fQ"]
            zk = layernorm(tr.mid[0][1], model.layers[1].g1, model.layers[1].b1, model.eps) @ D["fB"]
            mono_ok &= np.isclose(s[1], 4.0 * zq * zk / np.sqrt(2)) and abs(s[0]) < 1e-12 and abs(s[2]) < 1e-12
            rows.append(dict(a=a, b=b, xy=zq * zk, wB=tr.pattern[1][0][r][1], m=margin(tr), parts=parts(tr, r)))
    order = sorted(rows, key=lambda d: d["xy"])
    mono_ok &= all(order[i]["wB"] <= order[i + 1]["wB"] + 1e-15 for i in range(len(order) - 1))
    record("PB1", mono_ok, "score(read->B) = lam x^ y^/sqrt2 exactly, other scores 0; B's weight is monotone in x^y^")
    for rw in rows:
        log(f"  a={rw['a']:+.1f} b={rw['b']:+.1f} x^y^={rw['xy']:+.4f} w_B={rw['wB']:.4f} H1 vote={rw['parts'][0]:+.4f} H2 vote={rw['parts'][1]:+.4f} m={rw['m']:+.4f}")
    record("PB2", all(np.sign(r_["m"]) == np.sign(r_["a"]) for r_ in rows if r_["a"] * r_["b"] > 0), "agreeing cells: sign(m) = shared sign")

    # PB3 dissection
    b = 1.0
    X0, r = embed(0.0, b); tr0 = forward(model, X0, r)
    pat0 = tr0.pattern[1][0].copy(); V0 = tr0.V[1][0].copy()
    var_full, var_pat, var_val = [], [], []
    for a in LV:
        X, r = embed(a, b)
        var_full.append(parts(forward(model, X, r), r)[1])
        var_pat.append(parts(forward(model, X, r, replace_site(1, "pattern0", pat0)), r)[1])
        var_val.append(parts(forward(model, X, r, replace_site(1, "V0", V0)), r)[1])
    spread = lambda v: float(np.ptp(v))
    log(f"  B's vote across a in {LV} (b=+1): full {np.round(var_full,4).tolist()} | routing fixed {np.round(var_pat,4).tolist()} | values fixed {np.round(var_val,4).tolist()}")
    record("PB3", spread(var_pat) < 0.05 * spread(var_full) and spread(var_val) > 0.5 * spread(var_full),
           f"spread of B's vote over a: full {spread(var_full):.4f}, routing-fixed {spread(var_pat):.4f} (prediction: ~0), values-fixed {spread(var_val):.4f}")

    # PB4 query-only / key-only substitution
    a, b = 1.0, 1.0
    Xb, r = embed(a, b); trb = forward(model, Xb, r)
    Xq, _ = embed(-a, b); trq = forward(model, Xq, r)
    Xk, _ = embed(a, -b); trk = forward(model, Xk, r)
    Qcf = trb.Q[1][0].copy(); Qcf[r] = trq.Q[1][0][r]
    Kcf = trb.K[1][0].copy(); Kcf[1] = trk.K[1][0][1]
    mb = margin(trb)
    dQ = margin(forward(model, Xb, r, replace_site(1, "Q0", Qcf))) - mb
    dK = margin(forward(model, Xb, r, replace_site(1, "K0", Kcf))) - mb
    dQK = margin(forward(model, Xb, r, compose(replace_site(1, "Q0", Qcf), replace_site(1, "K0", Kcf)))) - mb
    record("PB4", abs(dQ) > 1e-3 and abs(dK) > 1e-3 and abs(dQK - dQ - dK) > 1e-3,
           f"query-only {dQ:+.4f}, key-only {dK:+.4f}, both {dQK:+.4f}, non-additivity {dQK-dQ-dK:+.4f}")

    mo, _, _ = qk_gate(qk_off=True)
    rev = []
    for a in np.linspace(-1.5, 1.5, 31):
        for b in np.linspace(-1.5, 1.5, 31):
            if a * b >= 0:
                continue
            X, r = embed(a, b)
            g, f = margin(forward(model, X, r)), margin(forward(mo, X, r))
            if np.sign(f) == np.sign(b) and np.sign(g) == np.sign(a):
                rev.append((round(a, 2), round(b, 2), round(f, 4), round(g, 4)))
    record("PB5", len(rev) > 0, f"{len(rev)} conflicting cells where fixed routing follows B but the gate follows A; e.g. {rev[:3]}")

    T = np.array([[margin(forward(mo, *embed(a, b))) for b in LV] for a in LV])
    add_res = float(np.abs(T - T[:, [0]] - T[[0], :] + T[0, 0]).max())
    Tg = np.array([[margin(forward(model, *embed(a, b))) for b in LV] for a in LV])
    record("PB6", is_relation_blind(T) and not is_relation_blind(Tg),
           f"qk_off: relation-blind={is_relation_blind(T)}, additivity residual {add_res:.2e}; gated: relation-blind={is_relation_blind(Tg)}")

    dev = max(abs(margin(forward(model, *embed(a, b, layout="ARB"))) - margin(forward(model, *embed(a, -b, layout="ARB")))) for a in LV for b in LV)
    record("PB7", dev < 1e-12, f"B after the read position: max |m(b) - m(-b)| = {dev:.1e}")

    out = []
    for a in (1.0, -1.0):
        X, r = embed(a, 1.0, layout="ABBR", b2=-1.0)
        tr = forward(model, X, r)
        w = tr.pattern[1][0][r]
        out.append((a, round(w[1], 4), round(w[2], 4), round(margin(tr), 4), round(margin(forward(mo, X, r)), 4)))
    ok = all((w1 > w2) == (a > 0) for a, w1, w2, _, _ in out) and all(abs(g) > abs(f) for _, _, _, g, f in out)
    record("PB8", ok, f"(a, w_B+, w_B-, m_gate, m_fixed) = {out}")

    # PB9 downstream re-reading and final LN
    found = []
    for name, mm in (("reread MLP", qk_gate(reread_mlp=6.0)[0]), ("final LN+bias", qk_gate(final_ln=True, kappa_bias=1.0)[0])):
        for a0 in LV:
            for a1 in LV:
                for b in LV:
                    if a0 == a1:
                        continue
                    X0, r = embed(a0, b); X1, _ = embed(a1, b)
                    t0, t1 = forward(mm, X0, r), forward(mm, X1, r)
                    dh = 2 * float((t1.head_out[1][0][r] - t0.head_out[1][0][r]) @ fU)
                    dm = margin(t1) - margin(t0)
                    if abs(dh) > 1e-3 and np.sign(dh) != np.sign(dm):
                        found.append((name, a0, a1, b, round(dh, 4), round(dm, 4)))
    names = {f[0] for f in found}
    record("PB9", len(names) >= 1, f"cells where sign(total change) != sign(gated head's direct change): {len(found)}; kinds {sorted(names)}; e.g. {found[:2]}")


# ======================================================== Mechanism C
def mech_C():
    log("\n=== Mechanism C: two-block reconciliation ===")
    model, D, embed = reconciliation()
    e1, e2, e3 = D["e1"], D["e2"], D["e3"]
    LV = list(np.linspace(-1.5, 1.5, 13))

    def S_hat(tr):
        z = layernorm(tr.resid_in[1][2], model.layers[1].g2, model.layers[1].b2, model.eps)
        return float(z @ (e1 + e2)), float(z @ e3)
    bad = 0; n = 0
    for a in LV:
        for b in LV:
            tr = forward(model, embed(a, b), 2)
            s, _ = S_hat(tr)
            if abs(s) > 1e-9:
                n += 1; bad += np.sign(margin(tr)) != np.sign(s)
    record("PC1", bad == 0, f"sign(m) = sign(S^) on {n} grid cells with S^ != 0 (violations {bad})")

    def x_of(a, kappa=2.0, d=8, eps=1e-5):
        return a / np.sqrt((a * a + kappa * kappa) / d + eps)
    def a_of(x, kappa=2.0, d=8, eps=1e-5):
        # invert x = a/sqrt((a^2+k^2)/d+eps)
        c = x * x / d
        return np.sign(x) * np.sqrt(x * x * (kappa * kappa / d + eps) / (1 - c))
    # matched-sum pairs in INPUT space (exact inversion of the source LN)
    pairs = []
    for xa, ya, xd, yd in ((0.6, 0.6, 1.5, -0.3), (0.4, 0.4, 1.2, -0.4), (-0.6, -0.6, -1.5, 0.3)):
        pairs.append(((a_of(xa), a_of(ya)), (a_of(xd), a_of(yd))))
    res = []
    for (ag, dg) in pairs:
        tra, trd = forward(model, embed(*ag), 2), forward(model, embed(*dg), 2)
        res.append((round(S_hat(tra)[0], 6), round(S_hat(trd)[0], 6), round(margin(tra), 4), round(margin(trd), 4)))
    log("  matched-sum pairs (S^_agree, S^_dis, m_agree, m_dis):", res)
    amp_ok = all(abs(ma) > abs(md) and np.sign(ma) == np.sign(md) for _, _, ma, md in res)
    # PC2: existence of a prior interval giving a reversal
    ma, md = res[0][2], res[0][3]
    prior = 0.5 * (ma + md)
    mp, _, embp = reconciliation(prior=prior)
    m_ag, m_dg = margin(forward(mp, embp(*pairs[0][0]), 2)), margin(forward(mp, embp(*pairs[0][1]), 2))
    record("PC2", amp_ok and m_ag > 0 > m_dg,
           f"agreement amplifies the S-favoured preference at fixed S ({amp_ok}); with any prior in ({md:.4f},{ma:.4f}) [here {prior:.4f}] the decision is c under agreement ({m_ag:+.4f}) and c' under disagreement ({m_dg:+.4f})")

    ag, dg = pairs[0]
    tra, trd = forward(model, embed(*ag), 2), forward(model, embed(*dg), 2)
    patched = trd.mlp_out[0].copy(); patched[2] = tra.mlp_out[0][2]
    m_patch = margin(forward(model, embed(*dg), 2, replace_site(0, "mlp_out", patched)))
    mab, _, embab = reconciliation(block1_mlp=False)
    gap_on = margin(tra) - margin(trd)
    gap_off = margin(forward(mab, embab(*ag), 2)) - margin(forward(mab, embab(*dg), 2))
    record("PC3", abs(m_patch - margin(tra)) < 0.25 * abs(gap_on) and abs(gap_off) < 0.25 * abs(gap_on),
           f"interchange of block-1 MLP output (dis<-agree): m {margin(trd):+.4f} -> {m_patch:+.4f} (agree run {margin(tra):+.4f}); agreement gap at fixed S with block-1 MLP {gap_on:+.4f}, ablated {gap_off:+.4f}")

    mb_, _, embb = reconciliation(bottleneck=True)
    devs = [abs(margin(forward(mb_, embb(*ag), 2)) - margin(forward(mb_, embb(*dg), 2))) for ag, dg in pairs]
    record("PC4", max(devs) < 1e-10, f"bottleneck: matched-sum pairs give identical margins (max diff {max(devs):.1e})")

    flips = 0
    for a in LV:
        for b in LV:
            m0 = margin(forward(model, embed(a, b), 2)); m1 = margin(forward(model, embed(a, b, extra=2.0), 2))
            flips += (np.sign(m0) != np.sign(m1)) and abs(m0) > 1e-9
    record("PC5", flips == 0, f"unread residual component: sign flips {flips}/169")

    # PC6 alternative decomposition: one wide ReLU layer fitted to the C margin function of (x, y)
    rng = np.random.default_rng(7)
    def fC(x, y):
        tr0 = forward(model, embed(1.0, 1.0), 2)
        mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * D["anR"]
        return margin(forward(model, embed(1.0, 1.0), 2, replace_site(0, "mid", mid)))
    tr_pts = rng.uniform(-1.5, 1.5, (600, 2)); te_pts = rng.uniform(-1.5, 1.5, (300, 2))
    ytr = np.array([fC(*p) for p in tr_pts]); yte = np.array([fC(*p) for p in te_pts])
    W = rng.normal(0, 1.5, (2, 300)); bb = rng.uniform(-2, 2, 300)
    F = lambda P: np.maximum(P @ W + bb, 0)
    coef, *_ = np.linalg.lstsq(np.column_stack([F(tr_pts), np.ones(len(tr_pts))]), ytr, rcond=None)
    pred = np.column_stack([F(te_pts), np.ones(len(te_pts))]) @ coef
    rel = float(np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte))
    sign_agree = float(np.mean(np.sign(pred) == np.sign(yte)))
    record("PC6", rel < 0.1, f"single wide ReLU layer on (x,y): held-out relative RMSE {rel:.4f}, sign agreement {sign_agree:.3f}; no q variable in it")


# ======================================================== Identifiability
def ident():
    log("\n=== Identifiability ===")
    model, D, embed = qk_gate(n_heads2=2)
    fU = D["fU"]
    rng = np.random.default_rng(3)
    E = rng.normal(0, 1.0, model.layers[1].WO[0].shape)
    mg = gauge_head_redistribution(model, 1, 0, 1, E)
    LV = [-1.0, -0.5, 0.5, 1.0]
    dl = max(np.abs(forward(model, *embed(a, b)).logits - forward(mg, *embed(a, b)).logits).max() for a in LV for b in LV)
    X, r = embed(1.0, 1.0); Xc, _ = embed(-1.0, 1.0)
    t, tg = forward(model, X, r), forward(mg, X, r)
    tc, tcg = forward(model, Xc, r), forward(mg, Xc, r)
    de = [2 * float(t.head_out[1][h][r] @ fU) for h in (0, 1)]
    deg = [2 * float(tg.head_out[1][h][r] @ fU) for h in (0, 1)]
    def patch_effect(mm, tb, tcf, h):
        ho = tb.head_out[1].copy(); ho[h] = tcf.head_out[1][h]
        return margin(forward(mm, X, r, replace_site(1, "head_out", ho))) - margin(tb)
    pe = [patch_effect(model, t, tc, h) for h in (0, 1)]
    peg = [patch_effect(mg, tg, tcg, h) for h in (0, 1)]
    record("PI1", dl < 1e-12 and max(abs(np.array(de) - deg)) > 1e-3 and max(abs(np.array(pe) - peg)) > 1e-3,
           f"max |logit change| {dl:.1e}; per-head direct effects {np.round(de,4).tolist()} -> {np.round(deg,4).tolist()}; per-head patch effects {np.round(pe,4).tolist()} -> {np.round(peg,4).tolist()}; sum of patch effects {sum(pe):+.4f} vs {sum(peg):+.4f}")

    m1, D1, emb1 = qk_gate()
    Mx = rng.normal(0, 1, (2, 2)) + 2 * np.eye(2); N = rng.normal(0, 1, (2, 2)) + 2 * np.eye(2)
    mq = gauge_qk(m1, 1, 0, Mx); mv = gauge_vo(m1, 1, 0, N)
    X, r = emb1(0.7, -0.4)
    a0, aq, av = forward(m1, X, r), forward(mq, X, r), forward(mv, X, r)
    record("PI2", np.allclose(a0.scores[1][0][r][: r + 1], aq.scores[1][0][r][: r + 1]) and np.allclose(a0.logits, aq.logits) and np.allclose(a0.logits, av.logits)
           and not np.allclose(a0.Q[1][0], aq.Q[1][0]) and not np.allclose(a0.V[1][0], av.V[1][0]),
           "QK and VO gauges: scores, head outputs and logits fixed; Q, K, V vectors changed")

    mc, Dc, embc = reconciliation()
    mr = gauge_relu_rescale(mc, 1, np.array([0.3, 4.0, 2.0, 0.5]))
    dl = max(np.abs(forward(mc, embc(a, b), 2).logits - forward(mr, embc(a, b), 2).logits).max() for a in LV for b in LV)
    record("PI3", dl < 1e-12 and not np.allclose(mc.layers[1].Wout, mr.layers[1].Wout), f"ReLU rescaling: max |logit change| {dl:.1e}; per-unit output weights changed")

    mp, (e1, e2, e3), embp = positive_control(final_ln=True)
    R = random_orthogonal_fixing_one(4, 11)
    mR, mapX = gauge_residual_rotation(mp, R)
    dl = max(np.abs(forward(mp, embp(a, b), 2).logits - forward(mR, mapX(embp(a, b)), 2).logits).max() for a, b in BIN)
    pre = max(np.abs(forward(mp, embp(a, b), 2).mlp_pre[0] - forward(mR, mapX(embp(a, b)), 2).mlp_pre[0]).max() for a, b in BIN)
    raw = max(abs(forward(mp, embp(a, b), 2).mid[0][2] @ e1 - forward(mR, mapX(embp(a, b)), 2).mid[0][2] @ e1) for a, b in BIN)
    record("PI4", dl < 1e-12 and pre < 1e-12 and raw > 1e-3,
           f"residual rotation R1=1: logits change {dl:.1e}; reader preactivations (source coordinates) change {pre:.1e}; raw residual coordinate on fixed e1 changes {raw:.3f}")

    ms = gauge_split_neuron(mp, 0, 0, 0.5)
    dl = max(np.abs(forward(mp, embp(a, b), 2).logits - forward(ms, embp(a, b), 2).logits).max() for a, b in BIN)
    record("PI5", dl < 1e-12 and ms.layers[0].Wout.shape[0] == 5, f"neuron split: logits change {dl:.1e}; unit 0's output weight halved (attribution split over two units)")


def posthoc():
    """NOT preregistered. Added after PA9 and PA10 failed; labelled post hoc everywhere."""
    log("\n=== POST-HOC analyses (not preregistered; added after PA9/PA10 failed) ===")
    from scipy.optimize import brentq
    d = 6
    # PA9-posthoc: match on the reader's ACTUAL input, S^ = <e1+e2, LN(mid)>, by scaling (x, y)
    for name, fam in (("designed", agreement_family), ("sum-only", sum_only_family)):
        m, basis, emb, coords = fam(act="gelu")
        e1, e2, e5 = basis[0], basis[1], basis[4]
        tr0 = forward(m, emb(1.0, 1.0), 2)
        def run(x, y):
            mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * e5
            t = forward(m, emb(1.0, 1.0), 2, replace_site(0, "mid", mid))
            z = layernorm(t.mid[0][2], m.layers[0].g2, m.layers[0].b2, m.eps)
            return margin(t), float(z @ (e1 + e2)), float(z @ (e1 - e2))
        target = run(1.0, 1.0)[1]
        out = []
        for dx, dy in ((1, 1), (2, 0), (3, -1)):
            t = brentq(lambda t: run(t * dx, t * dy)[1] - target, 1e-3, 10)
            mm, sh, dd = run(t * dx, t * dy)
            out.append((round(sh, 9), round(dd, 4), round(mm, 6)))
        log(f"  {name}: matched on post-LN S^ -> (S^, D^, m) = {out}")
        RES[f"PA9_posthoc_{name}"] = out
    # PA10-posthoc: dense grid including small evidence for A2 (unequal output weights)
    mm, _, emb, _ = agreement_family(out_w=(1.3, 1.3, -1, -1))
    fine = [v for v in np.linspace(-1.5, 1.5, 61) if abs(v) > 1e-9]
    bad = [(round(a, 3), round(b, 3)) for a in fine for b in fine if np.sign(margin(forward(mm, emb(a, b), 2))) != np.sign(a * b)]
    log(f"  A2 unequal output weights on a 60x60 grid incl. small values: {len(bad)} sign errors; smallest-|b| examples {sorted(bad, key=lambda t: abs(t[1]))[:4]}")
    RES["PA10_posthoc_A2_errors"] = len(bad)


def supplementary():
    """NOT preregistered: Mechanism A variants required by mandate section 5A that the
    preregistration did not list. Reported descriptively; no PASS/FAIL."""
    log("\n=== SUPPLEMENTARY (not preregistered) ===")
    G = [v for v in np.linspace(-1.5, 1.5, 13) if abs(v) > 1e-9]
    def err(mm, emb, ref=lambda a, b: np.sign(a * b)):
        return np.mean([np.sign(margin(forward(mm, emb(a, b), 2))) != ref(a, b) for a in G for b in G])
    # width: extra random units of increasing output scale
    out = {}
    for w in (4, 16, 64):
        for sc in (0.05, 0.2, 1.0):
            mm, _, emb, _ = agreement_family(width_extra=w, extra_seed=w)
            Lw = mm.layers[0]; Lw.Wout[4:] *= sc
            out[(w, sc)] = round(float(err(mm, emb)), 3)
    log("  extra random units (width, output scale) -> grid error rate:", out)
    # non-identical encodings: B written along cos(t) e1 + sin(t) e2 (non-orthogonal sources), unequal norms
    d = 6
    Bs = mean_zero_basis(d); e1, e2, e3, e4, e5 = Bs[:5]
    for t in (np.pi / 2, np.pi / 3, np.pi / 6):
        mm, _, emb0, _ = agreement_family()
        dirB = np.cos(t) * e1 + np.sin(t) * e2
        mm.layers[0].WO[0] = 3.0 * (np.outer(e1, e1) + np.outer(e2, dirB))
        log(f"  B encoded at angle {np.degrees(t):.0f} deg to A: grid error rate {err(mm, emb0):.3f}")
    mm, _, emb0, _ = agreement_family()
    mm.layers[0].WO[0] = 3.0 * (np.outer(e1, e1) + 3.0 * np.outer(e2, e2))
    log(f"  B's write 3x larger than A's: grid error rate {err(mm, emb0):.3f} (Corollary A1.3 predicts 0)")
    # noise: random components on the read token along unread and read directions
    rng = np.random.default_rng(5)
    for nm, direc in (("unread e4", e4), ("read e1-e2", (e1 - e2) / np.sqrt(2))):
        bad = 0; tot = 0
        for a in G:
            for b in G:
                X = agreement_family()[2](a, b); X[2] = X[2] + rng.normal(0, 0.5) * direc
                mm = agreement_family()[0]
                bad += np.sign(margin(forward(mm, X, 2))) != np.sign(a * b); tot += 1
        log(f"  noise sd 0.5 on read token along {nm}: grid error rate {bad / tot:.3f}")
    # generic MLPs reading the two source coordinates: how special is the agreement law?
    n_bin = n_law = n_suminsuff = 0
    N = 400
    for k in range(N):
        rr = np.random.default_rng(1000 + k)
        mm, basis, emb, coords = agreement_family()
        L0 = mm.layers[0]
        Wsrc = rr.normal(0, 1, (2, 4))
        L0.Win = np.column_stack([Wsrc[0, j] * e1 + Wsrc[1, j] * e2 for j in range(4)])
        L0.bin = rr.normal(0, 0.5, 4)
        L0.Wout = np.stack([rr.normal() * e3 for _ in range(4)])
        mfun = lambda a, b: margin(forward(mm, emb(a, b), 2))
        cell = {(a, b): mfun(a, b) for a in (-1.0, 1.0) for b in (-1.0, 1.0)}
        n_bin += all(np.sign(cell[(a, b)]) == np.sign(a * b) for a, b in cell) or all(np.sign(cell[(a, b)]) == -np.sign(a * b) for a, b in cell)
        n_law += err(mm, emb) == 0 or err(mm, emb, ref=lambda a, b: -np.sign(a * b)) == 0
        tr0 = forward(mm, emb(1.0, 1.0), 2)
        def at(x, y):
            mid = tr0.mid[0].copy(); mid[2] = x * e1 + y * e2 + 3.0 * e5
            return margin(forward(mm, emb(1.0, 1.0), 2, replace_site(0, "mid", mid)))
        vals = [at(1, 1), at(2, 0), at(3, -1)]   # same post-LN norm? not matched; report D-dependence via S fixed after scaling below
        n_suminsuff += np.ptp(vals) > 1e-6
    # mirror of Mechanism C: the gate opens on DISagreement (gamma < 0); same architecture
    def a_of(x, kappa=2.0, d=8, eps=1e-5):
        return np.sign(x) * np.sqrt(x * x * (kappa * kappa / d + eps) / (1 - x * x / d))
    ag, dg = (a_of(0.6), a_of(0.6)), (a_of(1.5), a_of(-0.3))
    for gam in (1.5, -1.5):
        mm, _, emb = reconciliation(gamma=gam)
        log(f"  Mechanism C with gamma={gam:+.1f}: m(agree)={margin(forward(mm, emb(*ag), 2)):+.4f} m(disagree)={margin(forward(mm, emb(*dg), 2)):+.4f} at matched S")
    log(f"  random 4-unit GELU readers of (x,y), N={N}: binary sign pattern = +-agreement {n_bin}/{N}; "
        f"agreement sign law on the whole grid {n_law}/{N}; output varies across the raw matched-sum triple {n_suminsuff}/{N}")


if __name__ == "__main__":
    mech_A(); mech_B(); mech_C(); ident(); posthoc(); supplementary()
    n_pass = sum(1 for k, v in RES.items() if isinstance(v, dict) and v.get("pass"))
    n_tot = sum(1 for v in RES.values() if isinstance(v, dict) and "pass" in v)
    log(f"\nSUMMARY: {n_pass}/{n_tot} preregistered predictions passed")
    logs = HERE.parent / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "experiments.txt").write_text("\n".join(OUT) + "\n")
    (logs / "results.json").write_text(json.dumps(RES, indent=1, default=float) + "\n")
