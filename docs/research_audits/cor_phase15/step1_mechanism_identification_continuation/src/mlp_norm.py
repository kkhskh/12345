"""Experiments for MLP_NORMALIZATION_THEOREMS.md (correction 2.2).

Read-token MLP under shared LN2. The pre-LN2 read residual is
    r = x e1 + y e2 + w,          w orthogonal to e1, e2, mean-zero
and LN2(r) = D r / sigma(r) + beta, sigma(r) = sqrt(||r - mean||^2/d + eps), D = diag(g).
The four units read  +/-(e1+e2), +/-(e1-e2)  scaled by c, with output weights (w1, w1, -w2, -w2).

M1  A0 failure reproduced; exact ratio invariance u/v = x/y.
M2  sign law sign q = sign(xy) under shared LN, random anchors/distractor content w (fresh seed 11).
M3  normalization-created interaction: a sum-only reader's margin depends on xy at fixed x+y.
M4  cross-talk cone: predicted from the read-direction perturbation; checked on fresh points (seed 12).
M5  unequal output weights: ReLU ratio cone is scale free; GELU cone moves with the LN scale.
M6  general LN gain and bias: predicted cross-talk and offsets from (D, beta); checked (seed 13).
M7  cross-check against cbmath.agreement_family forward pass.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "step1_causal_bridge_deep_investigation" / "src"))
import cbmath  # noqa: E402

LOG, OUT = [], {}
K = math.sqrt(2 / math.pi)


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def gelu(t):
    return 0.5 * t * (1 + np.tanh(K * (t + 0.044715 * t ** 3)))


def relu(t):
    return np.maximum(t, 0.0)


def ln(r, g=None, beta=None, eps=1e-5):
    d = r.shape[-1]
    mu = r.mean(-1, keepdims=True)
    z = (r - mu) / np.sqrt(((r - mu) ** 2).mean(-1, keepdims=True) + eps)
    if g is not None:
        z = z * g
    if beta is not None:
        z = z + beta
    return z


def q_of(z, e1, e2, c=1.5, act=gelu, w1=1.0, w2=1.0, dirs=None):
    if dirs is None:
        dirs = [e1 + e2, -(e1 + e2), e1 - e2, -(e1 - e2)]
    p = [c * (z @ dd) for dd in dirs]
    return w1 * (act(p[0]) + act(p[1])) - w2 * (act(p[2]) + act(p[3]))


def main():
    d = 6
    Bs = cbmath.mean_zero_basis(d)
    e1, e2, e3, e4, e5 = Bs[:5]
    rho = 3.0

    # M1
    log("M1 shared LN2: coordinate of source A at the read token depends on b; ratio is invariant")
    rows = []
    for b in [0.2, 0.5, 1.5]:
        m, _, emb, coords = cbmath.agreement_family()
        tr = cbmath.forward(m, emb(0.8, b), 2)
        u, v = coords(tr)
        sig = lambda t: t / math.sqrt((t * t + 4.0) / d + 1e-5)
        x, y = sig(0.8), sig(b)
        rows.append((b, u, v, u / v, x / y))
        log(f"   a=0.8 b={b}: u={u:.6f} v={v:.6f}  u/v={u / v:.12f}  x/y={x / y:.12f}")
    OUT["M1"] = rows
    assert max(abs(r[3] - r[4]) for r in rows) < 1e-9

    # M2 sign law under shared LN with random anchors and distractor content
    rng = np.random.default_rng(11)
    viol = 0; n = 0
    Wperp = Bs[2:]  # e3..e5 span (mean-zero, orthonormal, orthogonal to e1,e2)
    for _ in range(20000):
        x, y = rng.normal(0, 2, 2)
        w = Wperp.T @ rng.normal(0, rng.uniform(0, 5), Wperp.shape[0])
        r = x * e1 + y * e2 + w
        z = ln(r)
        for act in (gelu, relu):
            for c in (0.3, 1.5, 6.0):
                q = q_of(z, e1, e2, c=c, act=act)
                n += 1
                if abs(x * y) > 1e-9 and np.sign(q) != np.sign(x * y):
                    viol += 1
    log(f"M2 sign q = sign(xy) under shared LN2, random distractor content, GELU/ReLU, c in (0.3,1.5,6): "
        f"violations {viol}/{n}")
    OUT["M2"] = dict(violations=viol, n=n)

    # M3 normalization-created interaction in a sum-only reader
    log("M3 sum-only reader s(z) = c<z, e1+e2>: sign = sign(x+y) always; magnitude depends on xy at fixed x+y")
    res = []
    for S in [0.5, 1.0, 2.0]:
        vals = []
        for delta in [0.0, 0.5, 1.0, 2.0, 4.0]:
            x, y = (S + delta) / 2, (S - delta) / 2          # x+y = S, xy decreasing in delta
            z = ln(x * e1 + y * e2 + rho * e5)
            vals.append((x * y, float(z @ (e1 + e2))))
        res.append((S, vals))
        log(f"   S={S}: (xy, <LN(r), e1+e2>) = " + ", ".join(f"({a:+.3f}, {b:.4f})" for a, b in vals))
    OUT["M3"] = res

    # M4 cross-talk cone
    log("M4 cross-talk: read directions e1' = e1 + t12 e2, e2' = e2 + t21 e1.  Predicted law:")
    log("   sign q = sign(u'v'),  u' = u + t12 v,  v' = v + t21 u;  equals sign(uv) iff "
        "|v/u| in cone ( |t21| , 1/|t12| ) for same-sign t's (general: see doc)")
    rng = np.random.default_rng(12)
    pv = 0; nn = 0; fail_true = 0
    for _ in range(200):
        t12, t21 = rng.uniform(-0.6, 0.6, 2)
        f1, f2 = e1 + t12 * e2, e2 + t21 * e1
        dirs = [f1 + f2, -(f1 + f2), f1 - f2, -(f1 - f2)]
        for _ in range(100):
            x, y = rng.normal(0, 2, 2)
            z = ln(x * e1 + y * e2 + rho * e5)
            u, v = z @ e1, z @ e2
            q = q_of(z, e1, e2, dirs=dirs)
            up, vp = u + t12 * v, v + t21 * u
            pred = np.sign(up * vp)
            nn += 1
            pv += (np.sign(q) != pred)
            fail_true += (np.sign(q) != np.sign(x * y))
    log(f"   predicted-law violations {pv}/{nn}; original-law (sign xy) failures {fail_true}/{nn}")
    OUT["M4"] = dict(pred_violations=int(pv), n=nn, original_failures=int(fail_true))

    # M5 unequal weights: ReLU scale-free cone, GELU scale-dependent
    log("M5 unequal output weights w2/w1 = r (here r=2): predicted boundary")
    log("   ReLU (h(t)=|t|):  q>0  iff |u+v| > r|u-v|        (homogeneous; independent of LN scale)")
    log("   GELU:             q>0  iff h(c|u+v|) > r h(c|u-v|) (depends on c*||(u,v)||, hence on sigma)")
    r_ = 2.0
    out5 = {}
    for name, act in (("relu", relu), ("gelu", gelu)):
        ts = []
        for scale in (0.2, 1.0, 5.0):
            # boundary in t = v/u (u>0, agreement t>0): smallest t with q>0
            tgrid = np.linspace(1e-3, 1, 4000)
            qv = []
            for t in tgrid:
                u = 1.0; v = t
                z = scale * (u * e1 + v * e2)
                qv.append(q_of(z, e1, e2, act=act, w1=1.0, w2=r_))
            qv = np.array(qv)
            tstar = float(tgrid[np.argmax(qv > 0)]) if (qv > 0).any() else None
            ts.append(tstar)
        out5[name] = ts
        log(f"   {name}: agreement boundary t*=v/u at input scale 0.2, 1, 5: {ts}")
    log(f"   ReLU analytic t* = (r-1)/(r+1) = {(r_ - 1) / (r_ + 1):.4f};  GELU small-scale analytic "
        f"(sqrt r -1)/(sqrt r +1) = {(math.sqrt(r_) - 1) / (math.sqrt(r_) + 1):.4f}")
    OUT["M5"] = out5

    # M6 general LN gain and bias
    log("M6 LN gain g and bias beta: u = (x e1'De1 + y e1'De2 + e1'Dw)/sigma + e1'beta  (cross-talk + offset)")
    rng = np.random.default_rng(13)
    pv = 0; nn = 0; ft = 0
    for _ in range(200):
        g = 1 + rng.normal(0, 0.3, d)
        beta = rng.normal(0, 0.1, d)
        D = np.diag(g)
        for _ in range(50):
            x, y = rng.normal(0, 2, 2)
            w = rho * e5
            r = x * e1 + y * e2 + w
            sigma = math.sqrt(((r - r.mean()) ** 2).mean() + 1e-5)
            u_pred = (x * e1 @ D @ e1 + y * e1 @ D @ e2 + e1 @ D @ w) / sigma + e1 @ beta
            v_pred = (x * e2 @ D @ e1 + y * e2 @ D @ e2 + e2 @ D @ w) / sigma + e2 @ beta
            z = ln(r, g, beta)
            q = q_of(z, e1, e2)
            nn += 1
            pv += (np.sign(q) != np.sign(u_pred * v_pred))
            ft += (np.sign(q) != np.sign(x * y))
    log(f"   predicted-law violations {pv}/{nn}; original-law failures {ft}/{nn}")
    OUT["M6"] = dict(pred_violations=int(pv), n=nn, original_failures=int(ft))

    # M7 cross-check against the model forward pass
    rng = np.random.default_rng(14)
    mx = 0.0
    m, _, emb, coords = cbmath.agreement_family()
    for a, b in rng.uniform(-3, 3, (300, 2)):
        tr = cbmath.forward(m, emb(a, b), 2)
        z = cbmath.layernorm(tr.mid[0][2], m.layers[0].g2, m.layers[0].b2, 1e-5)
        mlp = q_of(z, e1, e2, c=1.0)
        mx = max(mx, abs(cbmath.margin(tr) - mlp))
    log(f"M7 read-token q (c=1) vs cbmath.agreement_family margin, 300 fresh points: max diff {mx:.2e}")
    OUT["M7_max_diff"] = mx

    root = Path(__file__).resolve().parents[1]
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "mlp_norm.txt").write_text("\n".join(LOG) + "\n")
    (root / "logs" / "mlp_norm.json").write_text(json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
