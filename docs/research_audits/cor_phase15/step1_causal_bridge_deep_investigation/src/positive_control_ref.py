"""Independent pure-Python implementation of the positive control (no numpy).

Written from the specification in the mandate (section 3), not from cbmath.py.
d = 4, tokens [a e1, b e2, 0], uniform causal attention, 4 GELU units, readout +-e3,
final LayerNorm with eps = 1e-5.
"""
import math

D = 4
EPS = 1e-5


def _basis():
    # Gram-Schmidt on the columns of the centring matrix
    cols = [[(1.0 if i == j else 0.0) - 1.0 / D for i in range(D)] for j in range(D - 1)]
    out = []
    for v in cols:
        w = v[:]
        for u in out:
            p = sum(wi * ui for wi, ui in zip(w, u))
            w = [wi - p * ui for wi, ui in zip(w, u)]
        n = math.sqrt(sum(wi * wi for wi in w))
        out.append([wi / n for wi in w])
    return out


E1, E2, E3 = _basis()


def ln(x):
    mu = sum(x) / D
    c = [xi - mu for xi in x]
    var = sum(ci * ci for ci in c) / D
    s = math.sqrt(var + EPS)
    return [ci / s for ci in c]


def gelu(t):
    return 0.5 * t * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t ** 3)))


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def add(*vs):
    return [sum(c) for c in zip(*vs)]


def scale(k, v):
    return [k * vi for vi in v]


def margin(a, b, final_ln=True):
    toks = [scale(a, E1), scale(b, E2), [0.0] * D]
    zs = [ln(t) for t in toks]
    vo = 3.0 * math.sqrt(0.25 + EPS)
    # uniform attention over the three positions (W_Q = W_K = 0), value/output = vo * I
    attn = scale(vo / 3.0, add(*zs))
    mid = add(toks[2], attn)
    z2 = ln(mid)
    s2 = math.sqrt(0.5 + EPS)
    p = s2 * dot(z2, add(E1, E2))
    m_ = s2 * dot(z2, [x - y for x, y in zip(E1, E2)])
    q = gelu(p) + gelu(-p) - gelu(m_) - gelu(-m_)
    final = add(mid, scale(q, E3))
    if final_ln:
        final = ln(final)
    return 2.0 * dot(final, E3)  # logits along +e3 and -e3


if __name__ == "__main__":
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            print(a, b, round(margin(a, b), 10))
