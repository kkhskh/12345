"""Program A (§6): six frozen readers on the same shared-LN2 read state, matched intervention families.

Read state: z = LN2(x e1 + y e2 + rho e5), (u, v) = (z.e1, z.e2), x = a/sigma_A, y = b/sigma_B.
Readers (each a frozen MLP block writing onto the margin direction):
  R1 designed agreement     q = h(u+v) - h(u-v)
  R2 net evidence           S = gelu(u+v) - gelu(-(u+v))  ( = u+v )
  R3 generic interaction    random GELU MLP of width 8 on (u,v) (seed 7), centred
  R4 reversed polarity      -q
  R5 magnitude              h(u) + h(v) - tau, tau = median over the generator
  R6 agreement + support    q + (u+v)
Intervention families (generator-level, recomputed through the whole block):
  I-pol   (a,b) -> (-a,-b)          agreement kept, support flipped
  I-flip  (a,b) -> (a,-b)           agreement flipped, |a|,|b| kept
  I-mag   (a,b) -> (2a, 2b)         support and agreement kept, strength changed
  I-matchS  matched post-LN S^ with opposite agreement (exact, bisection; reachable pairs only)
Plus: unit-group decomposition for R6 (ablate the q-group / the S-group), final LN effect,
architectural vs semantic perturbations.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlp_norm import gelu  # noqa: E402

LOG, OUT = [], {}
D, RHO, KAP, EPS = 6, 3.0, 2.0, 1e-5


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def h(t):
    return gelu(t) + gelu(-t)


def state(a, b, mix=0.0):
    """Reader coordinates. mix: semantic drift, source A also written onto e2 with weight mix."""
    sig = lambda t: t / math.sqrt((t * t + KAP ** 2) / D + EPS)
    x, y = sig(a), sig(b)
    X, Y = x, y + mix * x
    s = math.sqrt((X * X + Y * Y + RHO ** 2) / D + EPS)
    return X / s, Y / s


rng7 = np.random.default_rng(7)
W1 = rng7.normal(0, 1.5, (2, 8)); B1 = rng7.normal(0, 0.5, 8); W2 = rng7.normal(0, 1, 8)
G0 = float(np.mean([gelu(np.array([u, v]) @ W1 + B1) @ W2 for u, v in rng7.normal(0, 0.8, (2000, 2))]))


def readers(u, v, tau):
    q = h(u + v) - h(u - v)
    S = u + v
    gen = float(gelu(np.array([u, v]) @ W1 + B1) @ W2) - G0
    return {"R1 designed": q, "R2 net": S, "R3 generic": gen, "R4 reversed": -q,
            "R5 magnitude": h(u) + h(v) - tau, "R6 agree+support": q + S}


def main():
    rng = np.random.default_rng(31)
    pts = rng.uniform(-3, 3, (3000, 2))
    tau = float(np.median([h(state(a, b)[0]) + h(state(a, b)[1]) for a, b in pts]))
    names = list(readers(0, 0, tau).keys())
    m = lambda a, b, mix=0.0: readers(*state(a, b, mix), tau)

    # signature: fraction of points where the decision sign is preserved (+1) / flipped (-1)
    log("A-sig. decision response to intervention families (3000 points, seed 31): "
        "P(sign unchanged) under I-pol, I-flip, I-mag")
    sig = {}
    for n in names:
        keep = {}
        for fam, T in [("I-pol", lambda a, b: (-a, -b)), ("I-flip", lambda a, b: (a, -b)),
                       ("I-mag", lambda a, b: (2 * a, 2 * b))]:
            keep[fam] = float(np.mean([np.sign(m(a, b)[n]) == np.sign(m(*T(a, b))[n]) for a, b in pts]))
        sig[n] = keep
        log(f"   {n:18s} " + "  ".join(f"{k}={v:.3f}" for k, v in keep.items()))
    OUT["signature"] = sig

    # sign law with respect to semantic agreement
    log("A-law. P(sign m = sign(ab)) on the same points")
    law = {n: float(np.mean([np.sign(m(a, b)[n]) == np.sign(a * b) for a, b in pts])) for n in names}
    for n in names:
        log(f"   {n:18s} {law[n]:.3f}")
    OUT["law"] = law

    # matched post-LN aggregate
    log("A-matchS. matched S^ with opposite agreement (reachable pairs only): mean |delta m| and decision flips")
    pairs = []
    for a, b in rng.uniform(0.1, 1.5, (400, 2)):
        u0, v0 = state(a, b); S0 = u0 + v0
        a2 = float(rng.uniform(1.0, 4.0))
        f = lambda bb: sum(state(a2, bb)) - S0
        if (f(-8.0) > 0) == (f(-1e-9) > 0):
            continue
        lo, hi = -8.0, -1e-9
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(mid) > 0) == (f(lo) > 0):
                lo = mid
            else:
                hi = mid
        pairs.append(((a, b), (a2, (lo + hi) / 2)))
    ms = {}
    for n in names:
        dm = [m(*p)[n] - m(*q)[n] for p, q in pairs]
        fl = [np.sign(m(*p)[n]) != np.sign(m(*q)[n]) for p, q in pairs]
        ms[n] = dict(mean_abs_delta=float(np.mean(np.abs(dm))), flip=float(np.mean(fl)))
        log(f"   {n:18s} mean|dm|={ms[n]['mean_abs_delta']:.4f}  decision flips={ms[n]['flip']:.3f}  (n={len(pairs)})")
    OUT["matchS"] = ms

    # R6 unit-group decomposition (interventionally meaningful: each group can be ablated)
    log("A-dec. R6 = q-group + S-group. Group ablation is a real intervention; the decomposition is additive "
        "in the pre-final-LN logit difference only.")
    agree_cells = [(a, b) for a, b in pts if a * b > 0]
    conf_cells = [(a, b) for a, b in pts if a * b < 0]
    def flips_by_group(cells):
        full = [np.sign(m(a, b)["R6 agree+support"]) for a, b in cells]
        noq = [np.sign(m(a, b)["R2 net"]) for a, b in cells]
        noS = [np.sign(m(a, b)["R1 designed"]) for a, b in cells]
        return float(np.mean(np.array(full) != np.array(noq))), float(np.mean(np.array(full) != np.array(noS)))
    fa, fc = flips_by_group(agree_cells), flips_by_group(conf_cells)
    log(f"   agree cells: ablating q changes decision {fa[0]:.3f}, ablating S {fa[1]:.3f}; "
        f"conflict cells: ablating q {fc[0]:.3f}, ablating S {fc[1]:.3f}")
    OUT["R6_groups"] = dict(agree=fa, conflict=fc)
    # relation term changes the decision sign exactly when |q| > |S| and sign q != sign S
    viol = 0
    for a, b in pts:
        u, v = state(a, b); q = h(u + v) - h(u - v); S = u + v
        pred = np.sign(q + S) != np.sign(S)
        cond = (np.sign(q) != np.sign(S)) and abs(q) > abs(S)
        viol += pred != cond
    log(f"   'q reverses the net-evidence decision iff sign q != sign S and |q| > |S|': violations {viol}/{len(pts)}")
    OUT["R6_reversal_rule_violations"] = int(viol)
    # in which cells? conflicting cells where the stronger source supports c': q<0, S<0 -> no flip;
    # agree cells for c': q>0, S<0, |q|>|S|?
    cnt = {"agree c'": 0, "agree c'_n": 0}
    for a, b in pts:
        if a < 0 and b < 0:
            u, v = state(a, b); q = h(u + v) - h(u - v); cnt["agree c'_n"] += 1
            cnt["agree c'"] += abs(q) > abs(u + v)
    log(f"   agreement-for-c' cells where q overrides net evidence (R6 says c): {cnt['agree c\'']}/{cnt['agree c\'_n']}")
    OUT["R6_override_agree_cprime"] = cnt

    # final LN: margin direction du is mean-zero and LN bias is 0 -> sign preserved, magnitude not additive
    log("A-fLN. with a final LN (unit gain, zero bias) on a residual h = w + m*du (|du|=1, du mean-zero, w ⟂ du):")
    log("   sign(du . LN(h)) = sign(m) exactly; du.LN(h) = m/sigma(h) is not additive across unit groups "
        "(sigma depends on all contributions).")
    w = 3.0
    for mm in [0.5, 1.0, 2.0]:
        out = lambda t: t / math.sqrt((w * w + t * t) / D + EPS)
        log(f"   m={mm}: LN margin {out(mm):.4f}; sum of LN margins of halves {2 * out(mm / 2):.4f}")

    # architectural vs semantic perturbation of R1
    log("A-stab. R1 designed reader: architectural perturbation (weight noise) vs semantic drift (encoding mix)")
    st = {}
    for sd in [0.0, 0.05, 0.1, 0.2]:
        oks = []
        for s in range(20):
            r = np.random.default_rng(100 + s)
            dirs = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float) + r.normal(0, sd, (4, 2))
            ow = np.array([1, 1, -1, -1]) + r.normal(0, sd, 4)
            ok = np.mean([np.sign(ow @ gelu(dirs @ np.array(state(a, b)))) == np.sign(a * b) for a, b in pts[:600]])
            oks.append(ok)
        st[f"weight_noise_{sd}"] = (float(np.min(oks)), float(np.mean(oks)))
        log(f"   weight noise sd={sd}: P(sign law) min {np.min(oks):.3f} mean {np.mean(oks):.3f} over 20 seeds")
    for mix in [0.0, 0.1, 0.3, 0.6]:
        ok = np.mean([np.sign(m(a, b, mix)["R1 designed"]) == np.sign(a * b) for a, b in pts[:600]])
        st[f"semantic_mix_{mix}"] = float(ok)
        log(f"   semantic drift mix={mix} (weights unchanged): P(sign law) {ok:.3f}")
    OUT["stability"] = st

    root = HERE.parent
    (root / "logs" / "program_a.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "program_a.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
