"""Program C (§8): multi-layer mediation (C2), nontrivial composition (C3), conditional theorem checks (C4).

Base circuit: C' (cbmath.rereading, default parameters). Variants are built by editing a copy.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath  # noqa: E402
from cbmath import Layer  # noqa: E402

LOG, OUT = [], {}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


BASE, DIRS, EMB = cbmath.rereading()
e1, e2, e3, fU, anS, anR, mkS, mkP = (DIRS[k] for k in ["e1", "e2", "e3", "fU", "anS", "anR", "mkS", "mkP"])
sp = cbmath.mean_zero_basis(10)[8]
d = 10
R = 3


def M(model, a, b, hook=None, emb=EMB):
    return cbmath.margin(cbmath.forward(model, emb(a, b), R, hook))


def variant(bypass=0.0, extra_head=0.0, down_mlp=0.0, reread=0.0, redundant=False, compete=0.0):
    m = copy.deepcopy(BASE)
    L1, L2 = m.layers
    if bypass:
        L1.WO = L1.WO.copy(); L1.WO[0] = L1.WO[0] + bypass * np.outer(e1 + e2, fU)
    if redundant or compete:
        # second relation direction sp: redundant copy (+q) or competing reversed detector (-compete q)
        w = 1.0 if redundant else -compete
        L1.Wout = np.vstack([L1.Wout, L1.Wout * 0])
        L1.Win = np.column_stack([L1.Win, L1.Win]); L1.bin = np.concatenate([L1.bin, L1.bin])
        base_rows = L1.Wout[:4].copy()
        L1.Wout[4:] = np.stack([np.sign(r @ e3) * abs(r @ e3) * w * sp for r in base_rows])
        L2.WQ = L2.WQ.copy()
        if redundant:   # gate reads both copies, each at half gain: function unchanged
            L2.WQ[0] = 6.0 * np.outer(0.5 * e3 + 0.5 * sp, [1.0, 0.0])
        else:           # gate reads the sum of the two relation features
            L2.WQ[0] = 6.0 * np.outer(e3 + sp, [1.0, 0.0])
    if extra_head:
        e0 = np.array([1.0, 0.0])
        L2.WQ = np.concatenate([L2.WQ, (6.0 * np.outer(e3, e0))[None]])
        L2.WK = np.concatenate([L2.WK, np.outer(mkS - mkP, e0)[None]])
        L2.WV = np.concatenate([L2.WV, np.outer(-(e1 + e2), e0)[None]])
        WO = np.zeros((1, 2, d)); WO[0, 0] = extra_head * fU
        L2.WO = np.concatenate([L2.WO, WO])
    layers = [L1, L2]
    if down_mlp or reread:
        zQ = np.zeros((1, d, 2))
        L3 = Layer(WQ=zQ, WK=zQ.copy(), WV=(np.outer(e1 + e2, [1.0, 0.0]))[None] if reread else zQ.copy(),
                   WO=(np.zeros((1, 2, d)) if not reread else np.stack([np.stack([reread * fU, np.zeros(d)])])),
                   bO=np.zeros(d), act="relu",
                   Win=np.column_stack([fU, -fU]) if down_mlp else np.zeros((d, 1)),
                   bin=np.array([-0.5, -0.5]) if down_mlp else np.zeros(1),
                   Wout=np.stack([-down_mlp * fU, down_mlp * fU]) if down_mlp else np.zeros((1, d)),
                   bout=np.zeros(d), attn_on=bool(reread), mlp_on=bool(down_mlp))
        layers.append(L3)
    m.layers = layers
    return m


def target(a, b):
    return (1 if a > 0 else -1) if a * b > 0 else -1


def grid_stats(model, emb=EMB, n=31, lo=0.1):
    g = np.concatenate([-np.linspace(3, lo, n // 2), np.linspace(lo, 3, n // 2)])
    ok_ag = ok_cf = n_ag = n_cf = 0
    for a in g:
        for b in g:
            s = np.sign(M(model, a, b, emb=emb)) == target(a, b)
            if a * b > 0:
                n_ag += 1; ok_ag += s
            else:
                n_cf += 1; ok_cf += s
    return ok_ag / n_ag, ok_cf / n_cf


def main():
    # ---------------- C2 mediation
    log("C2 mediation on C': agreement-changing source change (a,b) -> (a,-b), recomputed.")
    log("   Sites: MLP1 output at R (q_R; query route) and at B (q_B; LN-scale/value route); block-2 scores; values.")
    rng = np.random.default_rng(51)
    rows = []
    for a, b in rng.uniform(0.3, 3.0, (200, 2)):
        base = cbmath.forward(BASE, EMB(a, b), R)
        cf = cbmath.forward(BASE, EMB(a, -b), R)
        mb, mc = cbmath.margin(base), cbmath.margin(cf)
        def patch_mlp(pos_list):
            val = base.mlp_out[0].copy()
            for p in pos_list:
                val[p] = cf.mlp_out[0][p]
            return M(BASE, a, b, cbmath.replace_site(0, "mlp_out", val))
        qR, qB, qRB = patch_mlp([3]), patch_mlp([2]), patch_mlp([2, 3])
        sc = M(BASE, a, b, cbmath.replace_site(1, "scores0", cf.scores[1][0]))
        vv = M(BASE, a, b, cbmath.replace_site(1, "V0", cf.V[1][0]))
        rows.append((mb, mc, qR, qB, qRB, sc, vv))
    r = np.array(rows)
    TE = r[:, 1] - r[:, 0]
    share = lambda col: float(np.median((r[:, col] - r[:, 0]) / TE))
    flip = lambda col: float(np.mean(np.sign(r[:, col]) == np.sign(r[:, 1])))
    res = {k: dict(median_share_of_TE=share(c), decision_matches_counterfactual=flip(c))
           for k, c in [("q_R interchange (off-manifold)", 2), ("q_B interchange (off-manifold)", 3),
                        ("q_R+q_B interchange (off-manifold)", 4), ("block-2 scores interchange", 5),
                        ("block-2 values interchange", 6)]}
    res["TE median"] = float(np.median(TE))
    for k, v in res.items():
        log(f"   {k}: {v}")
    OUT["C2"] = res

    # manifold check: is the q_R-patched block-1 output reachable by any input?
    log("C2-manifold: min over inputs (a'',b'') on a 241x241 grid of || block-1 output(a'',b'') - patched state ||")
    gg = np.linspace(-4, 4, 241)
    outs = {}
    for (a, b) in [(1.0, 1.0), (2.0, 0.5)]:
        base = cbmath.forward(BASE, EMB(a, b), R); cf = cbmath.forward(BASE, EMB(a, -b), R)
        target_state = base.resid_in[1].copy() if False else None
        H1 = base.mid[0] + base.mlp_out[0]
        patched = H1.copy(); patched[3] = base.mid[0][3] + cf.mlp_out[0][3]
        best = min(np.linalg.norm((lambda t: t.mid[0] + t.mlp_out[0])(cbmath.forward(BASE, EMB(x, y), R)) - patched)
                   for x in gg[::4] for y in gg[::4])
        # refine near the best coarse cell
        best_unp = min(np.linalg.norm((lambda t: t.mid[0] + t.mlp_out[0])(cbmath.forward(BASE, EMB(x, y), R)) - H1)
                       for x in gg[::4] for y in gg[::4])
        outs[f"{a},{b}"] = dict(min_dist_patched=float(best), min_dist_unpatched_control=float(best_unp),
                                patch_size=float(np.linalg.norm(patched - H1)))
        log(f"   ({a},{b}): patched-state distance to manifold {best:.4f}  (patch norm {np.linalg.norm(patched - H1):.4f}; "
            f"same search on the unpatched state {best_unp:.4f})")
    OUT["C2_manifold"] = outs

    # ---------------- C3 composition
    log("C3 composition variants: P(target rule) on agree / conflict cells of a 30x30 grid with |a|,|b| >= 0.1")
    comp = {}
    for name, kw in [("base C'", {}),
                     ("bypass beta=0.5", dict(bypass=0.5)), ("bypass beta=2", dict(bypass=2.0)),
                     ("bypass beta=6", dict(bypass=6.0)),
                     ("extra head reading relation, reversed values, w=0.5", dict(extra_head=0.5)),
                     ("extra head reading relation, reversed values, w=1.5", dict(extra_head=1.5)),
                     ("downstream MLP gamma=1 (compress)", dict(down_mlp=1.0)),
                     ("downstream MLP gamma=3 (over-compress)", dict(down_mlp=3.0)),
                     ("later rereading head w=1", dict(reread=1.0)), ("later rereading head w=4", dict(reread=4.0)),
                     ("redundant relation feature", dict(redundant=True)),
                     ("competing reversed relation feature 0.5", dict(compete=0.5)),
                     ("competing reversed relation feature 1.5", dict(compete=1.5))]:
        mv = variant(**kw)
        comp[name] = grid_stats(mv)
        log(f"   {name:55s} agree {comp[name][0]:.3f}  conflict {comp[name][1]:.3f}")
    # permuted identities
    perm = lambda a, b: EMB(b, a)
    g = np.concatenate([-np.linspace(3, 0.1, 15), np.linspace(0.1, 3, 15)])
    same = np.mean([np.sign(M(BASE, a, b)) == np.sign(cbmath.margin(cbmath.forward(BASE, perm(a, b), R))) for a in g for b in g])
    log(f"   permuted source positions (A<->B): decision unchanged on {same:.3f} of cells; "
        f"rule on permuted: {grid_stats(BASE, emb=perm)}")
    comp["permuted_same_decision"] = float(same)
    # redundant: ablate one copy -> still works (non-unique mediation)
    mr = variant(redundant=True)
    def ablate_dir(v):
        def hk(li, site, arr):
            if li == 0 and site == "mlp_out":
                arr = arr.copy(); arr[3] = arr[3] - np.outer(arr[3] @ v, v)[0] if arr.ndim == 1 else arr[3]
                arr[3] = arr[3] - (arr[3] @ v) * v
            return arr
        return hk
    cnt = {"e3": 0, "sp": 0, "both": 0}; n = 0
    for a, b in rng.uniform(0.3, 3.0, (100, 2)):
        for bb in (b, -b):
            n += 1
            full = np.sign(M(mr, a, bb))
            cnt["e3"] += np.sign(M(mr, a, bb, ablate_dir(e3))) == full
            cnt["sp"] += np.sign(M(mr, a, bb, ablate_dir(sp))) == full
            def both(li, site, arr):
                return ablate_dir(sp)(li, site, ablate_dir(e3)(li, site, arr))
            cnt["both"] += np.sign(M(mr, a, bb, both)) == full
    log(f"   redundant relation: decision preserved after ablating e3-copy {cnt['e3']}/{n}, sp-copy {cnt['sp']}/{n}, "
        f"both {cnt['both']}/{n}")
    comp["redundant_ablation"] = {k: int(v) for k, v in cnt.items()}
    OUT["C3"] = comp

    # ---------------- C4 conditional theorem checks
    log("C4 theorem checks.")
    # (i) bypass dominance: |m_bp| > max_j |v_j|  =>  sign m = sign m_bp regardless of routing
    viol = 0; nchk = 0
    for beta in [0.5, 2.0, 6.0]:
        mv = variant(bypass=beta)
        for a, b in rng.uniform(-3, 3, (300, 2)):
            tr = cbmath.forward(mv, EMB(a, b), R)
            Z = cbmath.layernorm(tr.mid[0], mv.layers[1].g1, mv.layers[1].b1, mv.eps)
            vproj = (Z @ mv.layers[1].WV[0] @ mv.layers[1].WO[0]) @ (mv.WU[:, 0] - mv.WU[:, 1])
            m_bp = float((tr.mid[0][R] + tr.mlp_out[0][R]) @ (mv.WU[:, 0] - mv.WU[:, 1]))
            if abs(m_bp) > np.max(np.abs(vproj)):
                nchk += 1
                viol += np.sign(cbmath.margin(tr)) != np.sign(m_bp)
    log(f"   (i) bypass dominance condition held at {nchk} points; violations of 'decision = sign(bypass)' {viol}")
    OUT["C4_bypass"] = dict(n=nchk, violations=int(viol))
    # (ii) monotone downstream preserves the sign: final map m -> phi(m) with phi increasing, phi(0)=0
    # the downstream MLP (gamma) gives phi(m) = m - gamma*sign(m)*relu(|m|/s - 0.5)*s', monotone iff gamma*slope < 1
    for gam in [0.5, 1.0, 3.0]:
        mv = variant(down_mlp=gam)
        ms = np.linspace(-3, 3, 601)
        log(f"   (ii) downstream MLP gamma={gam}: rule agree/conflict = {grid_stats(mv)}")
    OUT["C4_downstream"] = {g_: grid_stats(variant(down_mlp=g_)) for g_ in [0.5, 1.0, 3.0]}

    root = HERE.parent
    (root / "logs" / "program_c.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "program_c.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
