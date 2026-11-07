"""Experiments for CPRIME_CORRECTED_OPERATING_REGIONS.md.

E1  closed form == recomputed forward pass (fresh random points, seed 101).
E2  audit counterexamples reproduced.
E3  certified sign map of m on [-3,3]^2 by interval branch-and-bound, per quadrant,
    compared with the original C'1 target rule T(a,b) = sign(a) if ab>0, -1 if ab<0.
E4  exact decision boundary {m=0} by bracketing + bisection along rays.
E5  analytic sufficient conditions (Theorem C'1*), derived from the score/value structure,
    checked on fresh points (seed 202) for soundness, and their coverage measured.
E6  certified rectangular operating regions: smallest evidence floor tau (on a 0.01 grid)
    such that the rule is CERTIFIED on {|a|,|b| in [tau,3]} per quadrant, and the
    conflict-ratio region {|weak|/|strong| >= r}.
E7  mechanism of the two failure modes: routing gain G and value mass at the failure points.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cprime import P, area, certify, closed_form, forward_margin  # noqa: E402

OUT = {}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def target(a, b):
    if a * b > 0:
        return 1 if a > 0 else -1
    if a * b < 0:
        return -1
    return 0


# ---------------------------------------------------------------- analytic sufficient conditions


def suff(a, b, p=P()):
    """Theorem C'1*: returns +1 / -1 if a sufficient condition certifies the sign of m, else 0.

    Write F = e^{sP} vP + e^{sA} vA + e^{sB} vB + vR  (sR = 0), with vP = -pi mu / sigma_P < 0.
    (S+)  G >= 0, vA, vB, vR >= 0, and vA + vB + vR > (pi mu/sigma_P) e^{-G mu / sigma_P}   =>  m > 0
          (because e^{sA}, e^{sB} >= 1 when G >= 0).
    (S-)  G <= 0 and  (pi mu/sigma_P) e^{|G| mu/sigma_P} > vA^+ + vB^+ + vR^+                  =>  m < 0
          (because e^{sA}, e^{sB} <= 1 when G <= 0, and negative values only help).
    (S-0) all of vA, vB, vR <= 0                                                               =>  m < 0
          (every value is <= 0 and vP < 0).
    """
    c = closed_form(a, b, p)
    G, v = c["G"], c["v"]
    sP = math.sqrt((p.mu ** 2 + p.kappa ** 2) / p.d + p.eps)
    prior = p.prior_v * p.mu / sP
    if v["A"] <= 0 and v["B"] <= 0 and v["R"] <= 0:
        return -1
    if G >= 0 and min(v["A"], v["B"], v["R"]) >= 0 and v["A"] + v["B"] + v["R"] > prior * math.exp(-G * p.mu / sP):
        return 1
    pos = sum(max(v[k], 0.0) for k in "ABR")
    if G <= 0 and prior * math.exp(-G * p.mu / sP) > pos:
        return -1
    return 0


def main():
    p = P()
    rng = np.random.default_rng(101)
    pts = rng.uniform(-3, 3, size=(2000, 2))
    d = max(abs(closed_form(a, b)["m"] - forward_margin(a, b)) for a, b in pts[:400])
    log("E1 closed form vs forward (400 fresh points, seed 101): max |diff| =", f"{d:.3e}")
    OUT["E1_max_diff"] = d

    ce = {(0.01, 0.01): -0.6308419542, (0.05, 0.05): -0.3576101163, (0.08, 0.08): -0.0605622099,
          (1.0, -0.001): 0.1946247153, (0.0, 0.0): -0.6577909845, (1.0, 0.0): 0.2651922423}
    log("E2 audit counterexamples (closed form):")
    OUT["E2"] = {}
    for (a, b), ref in ce.items():
        m = closed_form(a, b)["m"]
        log(f"   m({a},{b}) = {m:+.10f}  audit {ref:+.10f}  target {target(a, b):+d}")
        OUT["E2"][f"{a},{b}"] = m
        assert abs(m - ref) < 1e-9

    # E3 certified sign map per quadrant
    log("E3 certified sign map, min box width 2e-3 (interval arithmetic with outward rounding)")
    OUT["E3"] = {}
    quads = {"Q++ agree": (0, 3, 0, 3), "Q-- agree": (-3, 0, -3, 0), "Q+- conflict": (0, 3, -3, 0),
             "Q-+ conflict": (-3, 0, 0, 3)}
    for name, box in quads.items():
        pos, neg, und = certify(box, min_w=2e-3)
        tot = (box[1] - box[0]) * (box[3] - box[2])
        want = "pos" if name == "Q++ agree" else "neg"
        right = area(pos) if want == "pos" else area(neg)
        wrong = area(neg) if want == "pos" else area(pos)
        log(f"   {name}: certified correct {right / tot:.4%}, certified WRONG {wrong / tot:.4%}, "
            f"undecided {area(und) / tot:.4%}  (target sign {want})")
        OUT["E3"][name] = dict(correct=right / tot, wrong=wrong / tot, undecided=area(und) / tot)

    # E4 exact boundary along rays from the origin (agree quadrant) and along b for fixed a (conflict)
    log("E4 exact boundary {m=0}")
    OUT["E4"] = {}

    def bisect(f, lo, hi, it=80):
        flo = f(lo)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rays = {}
    for th_deg in [5, 15, 30, 45, 60, 75, 85]:
        th = math.radians(th_deg)
        f = lambda r: closed_form(r * math.cos(th), r * math.sin(th))["m"]
        grid = np.linspace(1e-4, 4.2, 800)
        vals = [f(r) for r in grid]
        roots = [bisect(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if (vals[i] > 0) != (vals[i + 1] > 0)]
        rays[th_deg] = roots
        log(f"   agree ray theta={th_deg:2d} deg: roots r* = {[round(r, 6) for r in roots]}  "
            f"(m<0 for r<r*, m>0 beyond)")
    OUT["E4"]["agree_rays"] = rays
    conf = {}
    for a in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        f = lambda b: closed_form(a, b)["m"]
        grid = np.linspace(-3, -1e-6, 1200)
        vals = [f(b) for b in grid]
        roots = [bisect(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if (vals[i] > 0) != (vals[i + 1] > 0)]
        conf[a] = roots
        log(f"   conflict a={a}: roots b* = {[round(r, 6) for r in roots]}  ratio |b*|/a = {[round(-r / a, 4) for r in roots]}")
    OUT["E4"]["conflict_a_pos"] = conf
    conf2 = {}
    for b in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        f = lambda a: closed_form(a, b)["m"]
        grid = np.linspace(-3, -1e-6, 1200)
        vals = [f(a) for a in grid]
        roots = [bisect(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if (vals[i] > 0) != (vals[i + 1] > 0)]
        conf2[b] = roots
        log(f"   conflict b={b}: roots a* = {[round(r, 6) for r in roots]}")
    OUT["E4"]["conflict_b_pos"] = conf2

    # E5 analytic sufficient conditions on fresh points
    rng = np.random.default_rng(202)
    pts = rng.uniform(-3, 3, size=(20000, 2))
    viol = cover = correct = 0
    for a, b in pts:
        s = suff(a, b)
        m = closed_form(a, b)["m"]
        sg = 1 if m > 0 else -1
        if s != 0:
            if s != sg:
                viol += 1
        if sg == target(a, b):
            correct += 1
            if s == sg:
                cover += 1
    log(f"E5 Theorem C'1* on 20000 fresh points (seed 202): soundness violations {viol}; "
        f"target rule correct at {correct / len(pts):.2%} of points; sufficient conditions cover "
        f"{cover / max(correct, 1):.2%} of the correct points")
    OUT["E5"] = dict(violations=viol, rule_correct=correct / len(pts), coverage=cover / max(correct, 1))

    # E6 certified floors
    log("E6 certified rectangular operating regions (step 0.01)")
    OUT["E6"] = {}
    for name, sa, sb, want in [("Q++", 1, 1, "pos"), ("Q--", -1, -1, "neg"), ("Q+-", 1, -1, "neg"), ("Q-+", -1, 1, "neg")]:
        best = None
        for tau in np.arange(0.0, 3.0, 0.01):
            a0, a1 = sorted([sa * tau, sa * 3.0]); b0, b1 = sorted([sb * tau, sb * 3.0])
            pos, neg, und = certify((a0, a1, b0, b1), min_w=5e-3, max_boxes=60000)
            bad = (neg if want == "pos" else pos) + und
            if not bad:
                best = round(float(tau), 2); break
        log(f"   {name}: rule CERTIFIED on |a|,|b| in [tau,3] for tau = {best}")
        OUT["E6"][name] = best
    # conflict ratio region: |weak| >= r |strong| in Q+-, strong in [0.2,3]
    for r in [0.05, 0.1, 0.2, 0.3, 0.5]:
        ok = True
        for a in np.linspace(0.2, 3.0, 57):
            for sign_case in [0, 1]:
                # weak b: |b| in [r a, a];   weak a: a in [r|b|,|b|]
                if sign_case == 0:
                    pos, neg, und = certify((a, a + 0.05, -a, -r * a), min_w=5e-3, max_boxes=20000)
                else:
                    pos, neg, und = certify((r * a, a, -a - 0.05, -a), min_w=5e-3, max_boxes=20000)
                if pos or und:
                    ok = False; break
            if not ok:
                break
        log(f"   Q+- conflict with |weak|/|strong| >= {r} and strong in [0.2,3.05]: certified m<0: {ok}")
        OUT["E6"][f"Q+-_ratio_{r}"] = ok

    # E7 failure mechanism
    log("E7 failure mechanism: routing gain G, prior term, evidence value mass")
    OUT["E7"] = {}
    sP = math.sqrt((p.mu ** 2 + p.kappa ** 2) / p.d + p.eps)
    for a, b in [(0.01, 0.01), (0.08, 0.08), (0.2, 0.2), (1.0, -0.001), (1.0, -0.05), (1.0, -0.3), (1.0, -1.0)]:
        c = closed_form(a, b)
        mass = sum(c["v"][k] for k in "ABR")
        log(f"   ({a},{b}): qR={c['qR']:+.4e} G={c['G']:+.4e} alphaP={c['alpha']['P']:.4f} "
            f"prior|vP|={-c['v']['P']:.4f} evidence vA+vB+vR={mass:+.4f} m={c['m']:+.4f}")
        OUT["E7"][f"{a},{b}"] = dict(qR=c["qR"], G=c["G"], alphaP=c["alpha"]["P"], mass=mass, m=c["m"])





def param_sweep():
    """E8: boundary vs parameters, and soundness of C'1* at fresh random parameters (seed 303)."""
    import dataclasses
    log("E8 parameter dependence of the agreement floor r*(45 deg) and conflict ratio at a=2")
    res = {}
    for lam in [3.0, 6.0, 12.0, 24.0]:
        for pv in [0.25, 1.0, 2.0]:
            p = dataclasses.replace(P(), lam=lam, prior_v=pv)
            f = lambda r: closed_form(r / math.sqrt(2), r / math.sqrt(2), p)["m"]
            g = np.linspace(1e-4, 3, 600); vals = [f(r) for r in g]
            r45 = next((float(g[i + 1]) for i in range(len(g) - 1) if vals[i] < 0 <= vals[i + 1]), None)
            fc = lambda b: closed_form(2.0, b, p)["m"]
            gb = np.linspace(-2, -1e-6, 4000); vb = [fc(b) for b in gb]
            bc = next((float(gb[i]) for i in range(len(gb) - 1) if (vb[i] < 0) != (vb[i + 1] < 0)), None)
            log(f"   lam={lam:5.1f} prior_v={pv:4.2f}: agree floor r*~{r45}  conflict root b*(a=2)~{bc}")
            res[f"{lam},{pv}"] = (r45, bc)
    OUT["E8_sweep"] = res
    rng = np.random.default_rng(303)
    viol = 0; n = 0; cov = []
    for _ in range(40):
        p = dataclasses.replace(P(), lam=float(rng.uniform(1, 20)), mu=float(rng.uniform(1, 5)),
                                kappa=float(rng.uniform(0.5, 3)), rho=float(rng.uniform(1, 5)),
                                lam1=float(rng.uniform(0.5, 3)), c1=float(rng.uniform(0.5, 3)),
                                wv=float(rng.uniform(0.3, 2)), prior_v=float(rng.uniform(0.2, 2)))
        c_ok = c_all = 0
        for a, b in rng.uniform(-3, 3, size=(300, 2)):
            s = suff(a, b, p); m = closed_form(a, b, p)["m"]; sg = 1 if m > 0 else -1
            n += 1
            viol += (s != 0 and s != sg)
            c_all += 1; c_ok += (s != 0)
        cov.append(c_ok / c_all)
    log(f"   C'1* at 40 fresh parameter settings x 300 points: violations {viol}/{n}; "
        f"decided fraction median {np.median(cov):.3f} min {min(cov):.3f}")
    OUT["E8_fresh_params"] = dict(violations=int(viol), n=n, decided_median=float(np.median(cov)), decided_min=float(min(cov)))


if __name__ == "__main__":
    main()
    param_sweep()
    root = Path(__file__).resolve().parents[1]
    (root / "logs" / "cprime_regions.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "cprime_regions.json").write_text(json.dumps(OUT, indent=1, default=str))
