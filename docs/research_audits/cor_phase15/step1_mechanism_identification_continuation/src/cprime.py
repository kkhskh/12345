"""C' (conditional rereading) in exact closed form, plus certified sign regions.

The closed form is derived in CPRIME_CORRECTED_OPERATING_REGIONS.md section 1 and checked against
the full recomputed forward pass of step1_causal_bridge_deep_investigation/src/cbmath.py::rereading.

Certification: a small interval-arithmetic implementation of the same closed form, with
outward rounding (np.nextafter) on every operation, used in a branch-and-bound over boxes
of (a, b). A box is POS if the interval of m is > 0, NEG if < 0, otherwise it is split
down to a minimum width, and leftover boxes are UNDECIDED.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath  # noqa: E402

C = 0.044715
K = math.sqrt(2.0 / math.pi)


def gelu(t):
    return 0.5 * t * (1.0 + math.tanh(K * (t + C * t ** 3)))


def heven(t):
    """h(t) = gelu(t) + gelu(-t) = t * tanh(K (t + C t^3))."""
    return t * math.tanh(K * (t + C * t ** 3))


@dataclass(frozen=True)
class P:
    lam: float = 6.0
    mu: float = 3.0
    kappa: float = 2.0
    rho: float = 3.0
    lam1: float = 1.5
    c1: float = 1.5
    wv: float = 1.0
    prior_v: float = 1.0
    d: int = 10
    eps: float = 1e-5


def closed_form(a: float, b: float, p: P = P()) -> dict:
    d, e = p.d, p.eps
    sig = lambda sq: math.sqrt(sq / d + e)
    sP = sig(p.mu ** 2 + p.kappa ** 2)
    sA = sig(a * a + p.kappa ** 2 + p.mu ** 2)
    sB = sig(b * b + p.kappa ** 2 + p.mu ** 2)
    x, y = a / sA, b / sB                      # block-1 LN coordinates of the sources
    # block-1 residuals after the uniform head
    A1 = a + 2 * x                             # e1 coordinate at A
    B1u, B1v = 4 * x / 3, b + 4 * y / 3        # e1, e2 coordinates at B
    # block-1 MLP (agreement) at B and at read
    sBm = sig(B1u ** 2 + B1v ** 2 + p.kappa ** 2 + p.mu ** 2)
    uB, vB = B1u / sBm, B1v / sBm
    qB = heven(p.c1 * (uB + vB)) - heven(p.c1 * (uB - vB))
    sRm = sig(p.rho ** 2 + x * x + y * y)
    uR, vR = x / sRm, y / sRm
    qR = heven(p.c1 * (uR + vR)) - heven(p.c1 * (uR - vR))
    # block-2 LN scales
    sA2 = sig(A1 ** 2 + p.kappa ** 2 + p.mu ** 2)
    sB2 = sig(B1u ** 2 + B1v ** 2 + p.kappa ** 2 + p.mu ** 2 + (p.lam1 * qB) ** 2)
    sR2 = sig(p.rho ** 2 + x * x + y * y + (p.lam1 * qR) ** 2)
    qhat = p.lam1 * qR / sR2                   # the agreement feature read by the block-2 query
    G = p.lam * qhat / math.sqrt(2.0)          # routing gain
    s = {"P": -G * p.mu / sP, "A": G * p.mu / sA2, "B": G * p.mu / sB2, "R": 0.0}
    v = {"P": -p.prior_v * p.mu / sP, "A": p.wv * A1 / sA2, "B": p.wv * (B1u + B1v) / sB2,
         "R": p.wv * (x + y) / sR2}
    mx = max(s.values())
    w = {k: math.exp(s[k] - mx) for k in s}
    Z = sum(w.values())
    m = sum(w[k] * v[k] for k in s) / Z
    F = sum(w[k] * v[k] for k in s)  # = e^{-max s} * sum_j e^{s_j} v_j : same sign as m (post-review: shifted, no overflow)
    return dict(m=m, F=F, qR=qR, qhat=qhat, G=G, s=s, v=v, alpha={k: w[k] / Z for k in s}, x=x, y=y)


def forward_margin(a: float, b: float, p: P = P()) -> float:
    model, D, emb = cbmath.rereading(lam=p.lam, mu=p.mu, kappa=p.kappa, rho=p.rho, lam1=p.lam1, c1=p.c1,
                                     wv=p.wv, prior_v=p.prior_v, d=p.d, eps=p.eps)
    return cbmath.margin(cbmath.forward(model, emb(a, b), 3))


# ------------------------------------------------------------------ interval arithmetic


PAD = 8e-16   # post-review (R1 F6): ~4 ulps relative padding per operation (single nextafter was insufficient)


def _dn(x):
    x = float(x)
    return np.nextafter(x - abs(x) * PAD, -np.inf)


def _up(x):
    x = float(x)
    return np.nextafter(x + abs(x) * PAD, np.inf)


def C_(v):
    """an exactly-enclosed constant computed in floating point"""
    return I(_dn(v), _up(v))


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        hi = lo if hi is None else hi
        assert lo <= hi, (lo, hi)
        self.lo, self.hi = float(lo), float(hi)

    def __add__(s, o):
        o = o if isinstance(o, I) else I(o)
        return I(_dn(s.lo + o.lo), _up(s.hi + o.hi))
    __radd__ = __add__

    def __neg__(s):
        return I(-s.hi, -s.lo)

    def __sub__(s, o):
        return s + (-(o if isinstance(o, I) else I(o)))

    def __rsub__(s, o):
        return I(o) - s

    def __mul__(s, o):
        o = o if isinstance(o, I) else I(o)
        ps = [s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi]
        return I(_dn(min(ps)), _up(max(ps)))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = o if isinstance(o, I) else I(o)
        assert o.lo > 0 or o.hi < 0, "division by interval containing 0"
        return s * I(_dn(1.0 / o.hi), _up(1.0 / o.lo))

    def __rtruediv__(s, o):
        return I(o) / s

    def sq(s):
        if s.lo >= 0:
            return I(_dn(s.lo * s.lo), _up(s.hi * s.hi))
        if s.hi <= 0:
            return I(_dn(s.hi * s.hi), _up(s.lo * s.lo))
        return I(0.0, _up(max(s.lo * s.lo, s.hi * s.hi)))

    def sqrt(s):
        return I(_dn(math.sqrt(max(s.lo, 0.0))), _up(math.sqrt(s.hi)))

    def exp(s):
        return I(_dn(math.exp(s.lo)), _up(math.exp(s.hi)))

    def __repr__(s):
        return f"[{s.lo:.6g},{s.hi:.6g}]"


def iheven(t: I) -> I:
    """h is even and strictly increasing in |t| (proved in MLP_NORMALIZATION_THEOREMS.md)."""
    if t.lo >= 0:
        lo, hi = heven(t.lo), heven(t.hi)
    elif t.hi <= 0:
        lo, hi = heven(t.hi), heven(t.lo)
    else:
        lo, hi = 0.0, max(heven(t.lo), heven(t.hi))
    pad = 4e-15 * (1 + abs(hi))
    return I(lo - pad, hi + pad)


def interval_m(a: I, b: I, p: P = P()) -> I:
    d, e = p.d, p.eps
    sig = lambda sq: (sq / d + e).sqrt()
    k2m2 = p.kappa ** 2 + p.mu ** 2
    sP = (C_(p.mu ** 2 + p.kappa ** 2) / d + e).sqrt()
    sA = sig(a.sq() + k2m2)
    sB = sig(b.sq() + k2m2)
    # x = a / sA is monotone increasing in a (derivative k2m2/d+e > 0 over sA^3): use endpoint evaluation
    fx = lambda t: t / math.sqrt((t * t + k2m2) / d + e)
    x = I(_dn(_dn(fx(a.lo))), _up(_up(fx(a.hi))))
    y = I(_dn(_dn(fx(b.lo))), _up(_up(fx(b.hi))))
    A1 = a + 2 * x
    B1u = (4.0 / 3.0) * x
    B1v = b + (4.0 / 3.0) * y
    sBm = sig(B1u.sq() + B1v.sq() + k2m2)
    uB, vB = B1u / sBm, B1v / sBm
    qB = iheven(p.c1 * (uB + vB)) - iheven(p.c1 * (uB - vB))
    sRm = sig(p.rho ** 2 + x.sq() + y.sq())
    uR, vR = x / sRm, y / sRm
    qR = iheven(p.c1 * (uR + vR)) - iheven(p.c1 * (uR - vR))
    sA2 = sig(A1.sq() + k2m2)
    sB2 = sig(B1u.sq() + B1v.sq() + k2m2 + (p.lam1 * qB).sq())
    sR2 = sig(p.rho ** 2 + x.sq() + y.sq() + (p.lam1 * qR).sq())
    G = (C_(p.lam) / C_(2.0).sqrt()) * (p.lam1 * qR / sR2)
    sPs, sAs, sBs = -(G * (p.mu / sP)), G * p.mu / sA2, G * p.mu / sB2
    vP = -(p.prior_v * p.mu / sP)
    vA = p.wv * A1 / sA2
    vB = p.wv * (B1u + B1v) / sB2
    vR = p.wv * (x + y) / sR2
    return sPs.exp() * vP + sAs.exp() * vA + sBs.exp() * vB + vR   # F = Z * m, same sign as m


def certify(box, p: P = P(), min_w: float = 1e-3, max_boxes: int = 400000):
    """Branch-and-bound. Returns lists of POS, NEG, UNDECIDED boxes (a0,a1,b0,b1)."""
    pos, neg, und = [], [], []
    stack = [box]
    n = 0
    while stack:
        a0, a1, b0, b1 = stack.pop()
        n += 1
        if n > max_boxes:
            und.append((a0, a1, b0, b1)); continue
        F = interval_m(I(a0, a1), I(b0, b1), p)
        if F.lo > 0:
            pos.append((a0, a1, b0, b1))
        elif F.hi < 0:
            neg.append((a0, a1, b0, b1))
        elif max(a1 - a0, b1 - b0) <= min_w:
            und.append((a0, a1, b0, b1))
        else:
            if a1 - a0 >= b1 - b0:
                am = 0.5 * (a0 + a1); stack += [(a0, am, b0, b1), (am, a1, b0, b1)]
            else:
                bm = 0.5 * (b0 + b1); stack += [(a0, a1, b0, bm), (a0, a1, bm, b1)]
    return pos, neg, und


def area(boxes):
    return sum((a1 - a0) * (b1 - b0) for a0, a1, b0, b1 in boxes)
