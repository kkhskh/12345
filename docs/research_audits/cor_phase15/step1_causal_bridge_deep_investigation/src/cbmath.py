"""Exact pre-norm Transformer blocks, mechanism constructions, and gauge maps.

Row convention: a residual state is a row vector; X has shape (T, d).
Every intervention is a hook that replaces the value of a named site.
All downstream structural equations are then recomputed.

Sites per layer li:
  resid_in; Q{h}, K{h}, V{h} (per-head vectors before scores);
  scores{h} (masked, pre-softmax); pattern{h} (post-softmax);
  head_out (H, T, d); mid; mlp_pre (T, dm); mlp_out (T, d).
After the last layer: final.

No pretrained model is used anywhere.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------- activations


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


ACTS = {
    "gelu": gelu_tanh,
    "gelu_erf": gelu_erf,
    "relu": lambda x: np.maximum(x, 0.0),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "softplus": lambda x: np.logaddexp(0.0, x),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def even_part(act: str, t):
    """h(t) = phi(t) + phi(-t)."""
    f = ACTS[act]
    return f(t) + f(-t)


# ---------------------------------------------------------------- architecture


def layernorm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


@dataclass
class Layer:
    WQ: np.ndarray  # (H, d, dh)
    WK: np.ndarray
    WV: np.ndarray
    WO: np.ndarray  # (H, dh, d)
    bO: np.ndarray
    Win: np.ndarray  # (d, dm)
    bin: np.ndarray
    Wout: np.ndarray  # (dm, d)
    bout: np.ndarray
    act: str = "gelu"
    g1: Optional[np.ndarray] = None
    b1: Optional[np.ndarray] = None
    g2: Optional[np.ndarray] = None
    b2: Optional[np.ndarray] = None
    attn_on: bool = True
    mlp_on: bool = True

    def __post_init__(self):
        d = self.bO.shape[0]
        for nm in ("g1", "g2"):
            if getattr(self, nm) is None:
                setattr(self, nm, np.ones(d))
        for nm in ("b1", "b2"):
            if getattr(self, nm) is None:
                setattr(self, nm, np.zeros(d))


@dataclass
class Model:
    layers: list
    WU: np.ndarray  # (d, V)
    bU: np.ndarray
    final_ln: bool = False
    gf: Optional[np.ndarray] = None
    bf: Optional[np.ndarray] = None
    eps: float = 1e-5

    def __post_init__(self):
        d = self.WU.shape[0]
        if self.gf is None:
            self.gf = np.ones(d)
        if self.bf is None:
            self.bf = np.zeros(d)

    @property
    def d(self):
        return self.WU.shape[0]


@dataclass
class Trace:
    resid_in: list = field(default_factory=list)
    Q: list = field(default_factory=list)
    K: list = field(default_factory=list)
    V: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    pattern: list = field(default_factory=list)
    VO: list = field(default_factory=list)
    head_out: list = field(default_factory=list)
    mid: list = field(default_factory=list)
    mlp_pre: list = field(default_factory=list)
    mlp_out: list = field(default_factory=list)
    final: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None


Hook = Optional[Callable[[int, str, np.ndarray], np.ndarray]]


def _h(hook, li, site, arr):
    return arr if hook is None else hook(li, site, arr)


def forward(model: Model, X: np.ndarray, r: int, hook: Hook = None) -> Trace:
    tr = Trace()
    T, d = X.shape
    H = X.copy()
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    for li, L in enumerate(model.layers):
        H = _h(hook, li, "resid_in", H)
        tr.resid_in.append(H.copy())
        nH = L.WQ.shape[0]
        outs = np.zeros((nH, T, d))
        Qs, Ks, Vs, Ss, As, VOs = [], [], [], [], [], []
        if L.attn_on:
            Z = layernorm(H, L.g1, L.b1, model.eps)
            for h in range(nH):
                Q = _h(hook, li, f"Q{h}", Z @ L.WQ[h])
                K = _h(hook, li, f"K{h}", Z @ L.WK[h])
                Vv = _h(hook, li, f"V{h}", Z @ L.WV[h])
                S = Q @ K.T / np.sqrt(Q.shape[1])
                S = np.where(mask, -np.inf, S)
                S = _h(hook, li, f"scores{h}", S)
                E = np.exp(S - S.max(axis=1, keepdims=True))
                A = E / E.sum(axis=1, keepdims=True)
                A = _h(hook, li, f"pattern{h}", A)
                VO = Vv @ L.WO[h]
                outs[h] = A @ VO
                Qs.append(Q); Ks.append(K); Vs.append(Vv); Ss.append(S); As.append(A); VOs.append(VO)
        outs = _h(hook, li, "head_out", outs)
        tr.Q.append(Qs); tr.K.append(Ks); tr.V.append(Vs); tr.scores.append(Ss)
        tr.pattern.append(As); tr.VO.append(VOs); tr.head_out.append(outs)
        Hm = H + outs.sum(axis=0) + (L.bO if L.attn_on else 0.0)
        Hm = _h(hook, li, "mid", Hm)
        tr.mid.append(Hm.copy())
        if L.mlp_on:
            Z2 = layernorm(Hm, L.g2, L.b2, model.eps)
            pre = _h(hook, li, "mlp_pre", Z2 @ L.Win + L.bin)
            M = ACTS[L.act](pre) @ L.Wout + L.bout
        else:
            pre = np.zeros((T, L.Win.shape[1]))
            M = np.zeros_like(Hm)
        tr.mlp_pre.append(pre)
        M = _h(hook, li, "mlp_out", M)
        tr.mlp_out.append(M)
        H = Hm + M
    H = _h(hook, len(model.layers), "final", H)
    tr.final = H
    hr = H[r]
    if model.final_ln:
        hr = layernorm(hr, model.gf, model.bf, model.eps)
    tr.logits = hr @ model.WU + model.bU
    return tr


def margin(tr: Trace, c: int = 0, cp: int = 1) -> float:
    return float(tr.logits[c] - tr.logits[cp])


def replace_site(li_target: int, site_target: str, value: np.ndarray) -> Hook:
    def hook(li, site, arr):
        return value.copy() if (li == li_target and site == site_target) else arr
    return hook


def compose(*hooks) -> Hook:
    def hook(li, site, arr):
        for hk in hooks:
            if hk is not None:
                arr = hk(li, site, arr)
        return arr
    return hook


def mean_zero_basis(d: int) -> np.ndarray:
    """(d-1, d) orthonormal rows, each orthogonal to 1."""
    P = np.eye(d) - np.ones((d, d)) / d
    Qm, _ = np.linalg.qr(P[:, : d - 1])
    B = Qm.T
    assert np.allclose(B @ np.ones(d), 0, atol=1e-12) and np.allclose(B @ B.T, np.eye(d - 1), atol=1e-12)
    return B


def _zeros_attn(d, H=1, dh=1):
    return np.zeros((H, d, dh)), np.zeros((H, d, dh)), np.zeros((H, d, dh)), np.zeros((H, dh, d))


# ====================================================================
# Mechanism A: agreement-sensitive MLP
# ====================================================================


def positive_control(final_ln: bool = True, act: str = "gelu", eps: float = 1e-5):
    """Mandate section 3: d=4, 3 tokens (A, B, read), uniform attention (WQ=WK=0),
    4-unit MLP with preactivations a+b, -(a+b), a-b, -(a-b), readout along e3.
    Returns (model, basis(e1,e2,e3), embed(a,b) -> X)."""
    d = 4
    e1, e2, e3 = mean_zero_basis(d)
    WQ = np.zeros((1, d, d)); WK = np.zeros((1, d, d))
    WV = np.eye(d)[None]
    WO = (3.0 * np.sqrt(0.25 + eps) * np.eye(d))[None]   # read receives exactly a e1 + b e2 (binary)
    s2 = np.sqrt(0.5 + eps)                               # LN2 scale of a e1 + b e2 for binary a, b
    Win = s2 * np.column_stack([e1 + e2, -(e1 + e2), e1 - e2, -(e1 - e2)])
    Wout = np.stack([e3, e3, -e3, -e3])
    L = Layer(WQ=WQ, WK=WK, WV=WV, WO=WO, bO=np.zeros(d), Win=Win, bin=np.zeros(4),
              Wout=Wout, bout=np.zeros(d), act=act)
    WU = np.stack([e3, -e3], axis=1)
    model = Model(layers=[L], WU=WU, bU=np.zeros(2), final_ln=final_ln, eps=eps)

    def embed(a, b):
        return np.stack([a * e1, b * e2, np.zeros(d)])
    return model, (e1, e2, e3), embed


def agreement_family(act="gelu", d=6, kappa=2.0, rho=3.0, read_dirs=None, out_w=(1, 1, -1, -1),
                     read_offset=None, readout_mix=0.0, final_ln=False, kappa_bias=0.0,
                     width_extra=0, extra_seed=0, eps=1e-5):
    """Continuous-evidence version of the positive control.

    Tokens: A = a e1 + kappa e4, B = b e2 + kappa e4 (source anchor), read = rho e5.
    The anchors keep LN1 from normalizing away the evidence magnitude:
    after the uniform head, read has x e1 + y e2 + rho e5, with x = a / sigma(a)
    and sigma(a) = sqrt((a^2 + kappa^2)/d + eps). Then sign(x) = sign(a).

    Options that remove the theorem's conditions one at a time:
      read_dirs   : override the four MLP read directions (A1)
      out_w       : output weights of the four units (A2)
      act         : activation (A3)
      readout_mix : the readout gains a component along e1, giving a direct path (A4)
      kappa_bias  : final-LN bias with Delta u . beta != 0 (A4)
      read_offset : constant added to every preactivation (bias in the read directions, A1')
      width_extra : extra random units (generic MLP width)
    """
    Bs = mean_zero_basis(d)
    e1, e2, e3, e4, e5 = Bs[:5]
    WQ = np.zeros((1, d, d)); WK = np.zeros((1, d, d))
    WV = np.eye(d)[None]
    WO = (3.0 * (np.outer(e1, e1) + np.outer(e2, e2)))[None]
    if read_dirs is None:
        read_dirs = [e1 + e2, -(e1 + e2), e1 - e2, -(e1 - e2)]
    Win = np.column_stack(read_dirs)
    Wout = np.stack([w * e3 for w in out_w])
    bin_ = np.zeros(len(read_dirs)) if read_offset is None else np.array(read_offset, float)
    if width_extra:
        rng = np.random.default_rng(extra_seed)
        Win = np.column_stack([Win, rng.normal(0, 1, (d, width_extra))])
        Wout = np.vstack([Wout, rng.normal(0, 1, (width_extra, d))])
        bin_ = np.concatenate([bin_, rng.normal(0, 0.5, width_extra)])
    L = Layer(WQ=WQ, WK=WK, WV=WV, WO=WO, bO=np.zeros(d), Win=Win, bin=bin_,
              Wout=Wout, bout=np.zeros(d), act=act)
    du = e3 + readout_mix * e1
    WU = np.stack([du / 2, -du / 2], axis=1)  # margin = du . (final)
    bf = np.zeros(d)
    if kappa_bias:
        bf = kappa_bias * du / (du @ du)
    model = Model(layers=[L], WU=WU, bU=np.zeros(2), final_ln=final_ln, bf=bf, eps=eps)

    def embed(a, b):
        return np.stack([a * e1 + kappa * e4, b * e2 + kappa * e4, rho * e5])

    def coords(tr):
        z = layernorm(tr.mid[0][2], L.g2, L.b2, eps)
        return float(z @ e1), float(z @ e2)
    return model, (e1, e2, e3, e4, e5), embed, coords


def sum_only_family(act="gelu", **kw):
    """Negative control: the MLP reads only the net evidence S (units +S, -S)."""
    d = kw.get("d", 6)
    e1, e2 = mean_zero_basis(d)[:2]
    return agreement_family(act=act, read_dirs=[e1 + e2, -(e1 + e2)], out_w=(1, 1), **kw)


# ====================================================================
# Mechanism B: query-key agreement gate (two layers, dynamic attention)
# ====================================================================


def qk_gate(lam=4.0, L1=12.0, kappa=1.5, mu=3.0, rho=3.0, wQ=2.0, wU=0.6, kB=1.5, d=8,
            qk_off=False, reread_mlp=0.0, final_ln=False, kappa_bias=0.0, n_heads2=1, eps=1e-5):
    """Positions: A (0), B (1), read (2).

    Layer 1, head H1: read attends to A (A carries a position marker). It copies A's
    evidence onto fQ (later a query feature) and onto fU (A's direct vote, weight wU).
    Layer 2, head H2: query reads fQ and key reads fB, so score(read->B) = lam' * x * y.
    The value at B is B's evidence, written onto fU with weight kB.
    No MLPs unless reread_mlp > 0. Readout: m = 2 fU . h_read.

    qk_off     : W_Q of H2 set to 0 (uniform routing; negative control)
    reread_mlp : a layer-2 MLP that re-reads the fU coordinate and subtracts steeply
                 above a threshold (downstream rereading counterexample)
    n_heads2   : split H2 into identical heads (for the head-redistribution gauge)
    """
    Bs = mean_zero_basis(d)
    fA, fB, fQ, fU, anS, anR, mkA = Bs[:7]
    dh = 2
    e0 = np.array([1.0, 0.0])
    # layer 1
    WQ1 = (L1 * np.outer(anR, e0))[None]
    WK1 = np.outer(mkA, e0)[None]
    WV1 = np.outer(fA, e0)[None]
    WO1 = np.zeros((1, dh, d)); WO1[0, 0] = wQ * fQ + wU * fU
    lay1 = Layer(WQ=WQ1, WK=WK1, WV=WV1, WO=WO1, bO=np.zeros(d), Win=np.zeros((d, 1)), bin=np.zeros(1),
                 Wout=np.zeros((1, d)), bout=np.zeros(d), mlp_on=False)
    # layer 2
    WQ2 = (0.0 if qk_off else lam) * np.outer(fQ, e0)
    WK2 = np.outer(fB, e0)
    WV2 = np.outer(fB, e0)
    WO2 = np.zeros((dh, d)); WO2[0] = kB * fU
    H2 = n_heads2
    lay2 = Layer(WQ=np.stack([WQ2] * H2), WK=np.stack([WK2] * H2), WV=np.stack([WV2] * H2),
                 WO=np.stack([WO2 / H2] * H2), bO=np.zeros(d), act="relu",
                 Win=(np.outer(fU, [1.0]) if reread_mlp else np.zeros((d, 1))),
                 bin=np.array([-0.15]) if reread_mlp else np.zeros(1),
                 Wout=(np.array([-reread_mlp * fU]) if reread_mlp else np.zeros((1, d))),
                 bout=np.zeros(d), mlp_on=bool(reread_mlp))
    WU = np.stack([fU, -fU], axis=1)
    bf = np.zeros(d)
    if kappa_bias:
        bf = kappa_bias * fU / 2
    model = Model(layers=[lay1, lay2], WU=WU, bU=np.zeros(2), final_ln=final_ln, bf=bf, eps=eps)
    dirs = dict(fA=fA, fB=fB, fQ=fQ, fU=fU, anS=anS, anR=anR, mkA=mkA)

    def embed(a, b, layout="ABR", b2=None):
        A = a * fA + kappa * anS + mu * mkA
        Bt = b * fB + kappa * anS
        R = rho * anR
        if layout == "ABR":
            return np.stack([A, Bt, R]), 2
        if layout == "ARB":            # B after the read position: masked
            return np.stack([A, R, Bt]), 1
        if layout == "ABBR":           # competition: two B-type sources
            B2 = b2 * fB + kappa * anS
            return np.stack([A, Bt, B2, R]), 3
        raise ValueError(layout)
    return model, dirs, embed


# ====================================================================
# Mechanism C: two-block reconciliation (candidate-relative amplification)
# ====================================================================


def reconciliation(beta=1.0, gamma=1.5, theta=0.6, lam1=1.0, lam2=2.0, eps_dir=0.25, prior=0.0,
                   kappa=2.0, rho=3.0, d=8, bottleneck=False, block1_mlp=True, eps=1e-5):
    """Block 1: a uniform head copies x e1 + y e2 to read; a GELU MLP writes lam1 * q e3
    (agreement, as in Mechanism A).
    Block 2: attention off; a ReLU MLP with units
        ReLU( beta S^ + gamma q^ - theta ),  ReLU( -beta S^ + gamma q^ - theta )   -> +lam2 fU, -lam2 fU
        ReLU( eps S^ ), ReLU( -eps S^ )                                             -> +fU, -fU
    where S^ = <e1+e2, LN h>, q^ = <e3, LN h>.  Unembedding bias -prior on the margin.
    bottleneck : block 1 writes both sources onto e1 (source identity lost)
    block1_mlp : False ablates the agreement computation"""
    Bs = mean_zero_basis(d)
    e1, e2, e3, fU, anS, anR, sp = Bs[:7]
    WQ = np.zeros((1, d, d)); WK = np.zeros((1, d, d)); WV = np.eye(d)[None]
    if bottleneck:
        WO1 = (3.0 * (np.outer(e1, (e1 + e2) / 2) + np.outer(e2, (e1 + e2) / 2)))[None]
    else:
        WO1 = (3.0 * (np.outer(e1, e1) + np.outer(e2, e2)))[None]
    Win1 = np.column_stack([e1 + e2, -(e1 + e2), e1 - e2, -(e1 - e2)]) * 1.5
    Wout1 = np.stack([lam1 * e3, lam1 * e3, -lam1 * e3, -lam1 * e3])
    b1 = Layer(WQ=WQ, WK=WK, WV=WV, WO=WO1, bO=np.zeros(d), Win=Win1, bin=np.zeros(4),
               Wout=Wout1, bout=np.zeros(d), act="gelu", mlp_on=block1_mlp)
    s = e1 + e2
    Win2 = np.column_stack([beta * s + gamma * e3, -beta * s + gamma * e3, eps_dir * s, -eps_dir * s])
    bin2 = np.array([-theta, -theta, 0.0, 0.0])
    Wout2 = np.stack([lam2 * fU, -lam2 * fU, fU, -fU])
    b2 = Layer(WQ=np.zeros((1, d, 1)), WK=np.zeros((1, d, 1)), WV=np.zeros((1, d, 1)), WO=np.zeros((1, 1, d)),
               bO=np.zeros(d), Win=Win2, bin=bin2, Wout=Wout2, bout=np.zeros(d), act="relu", attn_on=False)
    WU = np.stack([fU / 2, -fU / 2], axis=1)
    bU = np.array([-prior / 2, prior / 2])
    model = Model(layers=[b1, b2], WU=WU, bU=bU, final_ln=False, eps=eps)
    dirs = dict(e1=e1, e2=e2, e3=e3, fU=fU, anS=anS, anR=anR, sp=sp)

    def embed(a, b, extra=0.0):
        return np.stack([a * e1 + kappa * anS, b * e2 + kappa * anS, rho * anR + extra * sp])
    return model, dirs, embed


# ====================================================================
# Ordinal relation-blindness (from the previous package, unchanged in substance)
# ====================================================================


def rb_margin(table, tol=1e-12):
    nA, nB = table.shape
    cells = [(i, j) for i in range(nA) for j in range(nB)]
    nv = nA + nB + 1
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for p in range(len(cells)):
        for q in range(len(cells)):
            if p == q:
                continue
            (i, j), (k, l) = cells[p], cells[q]
            diff = table[i, j] - table[k, l]
            row = np.zeros(nv)
            row[i] += 1; row[nA + j] += 1; row[k] -= 1; row[nA + l] -= 1
            if diff > tol:
                rr = -row.copy(); rr[-1] = 1.0
                A_ub.append(rr); b_ub.append(0.0)
            elif abs(diff) <= tol and p < q:
                A_eq.append(row); b_eq.append(0.0)
    for idx in (0, nA):
        row = np.zeros(nv); row[idx] = 1.0
        A_eq.append(row); b_eq.append(0.0)
    c = np.zeros(nv); c[-1] = -1.0
    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(-1, 1)] * (nA + nB) + [(None, 1)], method="highs")
    return float(res.x[-1]) if res.success else -np.inf


def is_relation_blind(table):
    return rb_margin(table) > 1e-9


# ====================================================================
# Function-preserving reparameterizations (gauges)
# ====================================================================


def gauge_head_redistribution(model: Model, li: int, h1: int, h2: int, E: np.ndarray) -> Model:
    m = copy.deepcopy(model)
    L = m.layers[li]
    assert np.allclose(L.WQ[h1], L.WQ[h2]) and np.allclose(L.WK[h1], L.WK[h2]) and np.allclose(L.WV[h1], L.WV[h2])
    L.WO[h1] = L.WO[h1] + E
    L.WO[h2] = L.WO[h2] - E
    return m


def gauge_qk(model: Model, li: int, h: int, Mx: np.ndarray) -> Model:
    m = copy.deepcopy(model)
    L = m.layers[li]
    L.WQ[h] = L.WQ[h] @ Mx
    L.WK[h] = L.WK[h] @ np.linalg.inv(Mx).T
    return m


def gauge_vo(model: Model, li: int, h: int, N: np.ndarray) -> Model:
    m = copy.deepcopy(model)
    L = m.layers[li]
    L.WV[h] = L.WV[h] @ N
    L.WO[h] = np.linalg.inv(N) @ L.WO[h]
    return m


def gauge_relu_rescale(model: Model, li: int, scales: np.ndarray) -> Model:
    m = copy.deepcopy(model)
    L = m.layers[li]
    assert L.act == "relu" and np.all(scales > 0)
    L.Win = L.Win * scales[None, :]
    L.bin = L.bin * scales
    L.Wout = L.Wout / scales[:, None]
    return m


def gauge_split_neuron(model: Model, li: int, k: int, frac: float = 0.5) -> Model:
    m = copy.deepcopy(model)
    L = m.layers[li]
    L.Win = np.column_stack([L.Win, L.Win[:, k]])
    L.bin = np.concatenate([L.bin, [L.bin[k]]])
    Wout = L.Wout.copy()
    Wout[k] = frac * L.Wout[k]
    L.Wout = np.vstack([Wout, (1 - frac) * L.Wout[k]])
    return m


def random_orthogonal_fixing_one(d: int, seed: int) -> np.ndarray:
    """R orthogonal with R 1 = 1 (acts as a rotation on the mean-zero subspace)."""
    rng = np.random.default_rng(seed)
    B = mean_zero_basis(d)
    Qr, _ = np.linalg.qr(rng.normal(size=(d - 1, d - 1)))
    u = np.ones(d) / np.sqrt(d)
    R = B.T @ Qr @ B + np.outer(u, u)
    assert np.allclose(R @ R.T, np.eye(d), atol=1e-12) and np.allclose(R @ np.ones(d), np.ones(d))
    return R


def gauge_residual_rotation(model: Model, R: np.ndarray):
    """h' = h R^T. Valid when every LN gain is uniform (gamma = c 1) and R fixes 1.
    Returns (model', map for inputs X -> X R^T)."""
    m = copy.deepcopy(model)
    for L in m.layers:
        for g in (L.g1, L.g2):
            assert np.allclose(g, g[0])
        L.WQ = np.einsum("ij,hjk->hik", R, L.WQ)
        L.WK = np.einsum("ij,hjk->hik", R, L.WK)
        L.WV = np.einsum("ij,hjk->hik", R, L.WV)
        L.WO = np.einsum("hkj,ij->hki", L.WO, R)
        L.bO = L.bO @ R.T
        L.b1 = L.b1 @ R.T; L.b2 = L.b2 @ R.T
        L.Win = R @ L.Win
        L.Wout = L.Wout @ R.T
        L.bout = L.bout @ R.T
    assert np.allclose(m.gf, m.gf[0])
    m.bf = m.bf @ R.T
    m.WU = R @ m.WU
    return m, (lambda X: X @ R.T)
