"""Per-component decoding accuracy of n^{(-q)} from: nothing, H={h: n_h=5} only, pi only (natural distribution, exact)."""
import itertools
from fractions import Fraction as Fr
from math import comb
al = Fr(4, 5); e = (1 - al) / 2; a_rel = al**2 + 2 * e**2; a_no = Fr(1, 3)
CFG = [(g, h) for g in range(3) for h in range(3) if g != h]; f1 = (al**2 + 2*e**2)/3; f0 = (2*al*e + e**2)/3; rho = f1/f0
supp = [v for v in itertools.product(range(6), repeat=3) if 5 in v]
def pn(v):
    t = Fr(0)
    for g, h in CFG:
        p = Fr(1, 6)
        for k in range(3):
            if k == h: p *= int(v[k] == 5)
            else: a = a_rel if k == g else a_no; p *= comb(5, v[k]) * a**v[k] * (1-a)**(5-v[k])
        t += p
    return t
P = {v: pn(v) for v in supp}
def pi(v):
    w = [rho**v[g] * int(v[h] == 5) for g, h in CFG]; s = sum(w); return tuple(x/s for x in w)
def acc(key):
    out = []
    for c in range(3):
        groups = {}
        for v in supp: groups.setdefault(key(v), {}).setdefault(v[c], Fr(0)); groups[key(v)][v[c]] += P[v]
        out.append(round(float(sum(max(d.values()) for d in groups.values())), 4))
    return out
print("no information   :", acc(lambda v: 0))
print("H only           :", acc(lambda v: tuple(int(x == 5) for x in v)))
print("pi only          :", acc(pi))
print("n (sanity)       :", acc(lambda v: v))
