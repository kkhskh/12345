"""Verification code for the Step 1 foundational bridge adjudication.

Everything here is exact finite arithmetic on an explicit pre-norm Transformer
(GPT-2 layout: LN -> multi-head causal attention -> residual add -> LN -> MLP ->
residual add, repeated; final LN; unembedding with optional bias).

The functions are used by tests/test_bridge.py to check the theorems stated in
THEOREM_REGISTER.md.  The proofs are in proofs/PROOFS.md; this file only
checks them numerically and builds the counterexamples.

No pretrained model is loaded.  No historical evaluation data is read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------------


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


@dataclass
class Layer:
    WQ: np.ndarray  # (H, d, dh)
    WK: np.ndarray
    WV: np.ndarray
    WO: np.ndarray  # (H, dh, d)
    bO: np.ndarray  # (d,)
    g1: np.ndarray
    b1: np.ndarray
    g2: np.ndarray
    b2: np.ndarray
    Win: np.ndarray  # (d, dm)
    bin: np.ndarray
    Wout: np.ndarray  # (dm, d)
    bout: np.ndarray
    mlp_on: bool = True


@dataclass
class Model:
    layers: list
    gf: np.ndarray
    bf: np.ndarray
    WU: np.ndarray  # (d, V)
    bU: np.ndarray  # (V,)
    eps: float = 1e-5

    @property
    def d(self) -> int:
        return self.gf.shape[0]


def random_model(seed: int, d: int = 8, H: int = 2, dh: int = 4, dm: int = 16, V: int = 5,
                 n_layers: int = 1, scale: float = 0.6, eps: float = 1e-5,
                 unembed_bias: bool = False) -> Model:
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n_layers):
        layers.append(Layer(
            WQ=rng.normal(0, scale, (H, d, dh)), WK=rng.normal(0, scale, (H, d, dh)),
            WV=rng.normal(0, scale, (H, d, dh)), WO=rng.normal(0, scale, (H, dh, d)),
            bO=rng.normal(0, 0.1, d),
            g1=1 + rng.normal(0, 0.1, d), b1=rng.normal(0, 0.1, d),
            g2=1 + rng.normal(0, 0.1, d), b2=rng.normal(0, 0.1, d),
            Win=rng.normal(0, scale, (d, dm)), bin=rng.normal(0, 0.1, dm),
            Wout=rng.normal(0, scale, (dm, d)), bout=rng.normal(0, 0.1, d)))
    return Model(layers=layers, gf=1 + rng.normal(0, 0.1, d), bf=rng.normal(0, 0.3, d),
                 WU=rng.normal(0, 1.0, (d, V)),
                 bU=(rng.normal(0, 0.3, V) if unembed_bias else np.zeros(V)), eps=eps)


@dataclass
class Trace:
    """All intermediate quantities of one forward pass."""
    resid: list = field(default_factory=list)       # residual entering each layer (T,d)
    head_out: list = field(default_factory=list)    # per layer: (H,T,d) head writes
    scores: list = field(default_factory=list)      # per layer: (H,T,T) masked scores
    values: list = field(default_factory=list)      # per layer: (H,T,d) value vectors after W_O
    mid: list = field(default_factory=list)         # residual after attention (T,d)
    mlp_out: list = field(default_factory=list)     # (T,d)
    final: Optional[np.ndarray] = None              # final residual (T,d)
    logits: Optional[np.ndarray] = None             # (V,) at read position


# An intervention hook receives (layer_index, site_name, array) and returns an array.
Hook = Optional[Callable[[int, str, np.ndarray], np.ndarray]]


def forward(model: Model, X: np.ndarray, r: int, hook: Hook = None) -> Trace:
    """Recomputed forward pass.  `hook` may replace a site; everything
    downstream of the replaced site is recomputed from the structural equations."""
    tr = Trace()
    T, d = X.shape
    Hres = X.copy()
    if hook is not None:
        Hres = hook(-1, "embed", Hres)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    for li, L in enumerate(model.layers):
        tr.resid.append(Hres.copy())
        Z = layernorm(Hres, L.g1, L.b1, model.eps)
        nH = L.WQ.shape[0]
        outs = np.zeros((nH, T, d))
        sc_all = np.zeros((nH, T, T))
        val_all = np.zeros((nH, T, d))
        for h in range(nH):
            Q = Z @ L.WQ[h]
            K = Z @ L.WK[h]
            Vv = Z @ L.WV[h]
            S = Q @ K.T / np.sqrt(Q.shape[1])
            S = np.where(mask, -np.inf, S)
            if hook is not None:
                S = hook(li, f"scores{h}", S)
            Smax = S.max(axis=1, keepdims=True)
            E = np.exp(S - Smax)
            A = E / E.sum(axis=1, keepdims=True)
            VO = Vv @ L.WO[h]
            outs[h] = A @ VO
            sc_all[h] = S
            val_all[h] = VO
        if hook is not None:
            outs = hook(li, "head_out", outs)
        tr.head_out.append(outs)
        tr.scores.append(sc_all)
        tr.values.append(val_all)
        Hmid = Hres + outs.sum(axis=0) + L.bO
        if hook is not None:
            Hmid = hook(li, "mid", Hmid)
        tr.mid.append(Hmid.copy())
        if L.mlp_on:
            Z2 = layernorm(Hmid, L.g2, L.b2, model.eps)
            M = gelu(Z2 @ L.Win + L.bin) @ L.Wout + L.bout
        else:
            M = np.zeros_like(Hmid)
        if hook is not None:
            M = hook(li, "mlp_out", M)
        tr.mlp_out.append(M)
        Hres = Hmid + M
    if hook is not None:
        Hres = hook(len(model.layers), "final", Hres)
    tr.final = Hres
    tr.logits = layernorm(Hres[r], model.gf, model.bf, model.eps) @ model.WU + model.bU
    return tr


def margin(logits: np.ndarray, c: int, cp: int) -> float:
    return float(logits[c] - logits[cp])


# ----------------------------------------------------------------------------
# Final-readout decomposition  m = a(h)/sigma(h) + kappa   (Theorem NC)
# ----------------------------------------------------------------------------


def centered_functional(model: Model, c: int, cp: int) -> np.ndarray:
    """u~ = P_c (gamma ⊙ (u_c - u_c')),  P_c = I - 11^T/d.  Then a(h) = u~ · h."""
    d = model.d
    du = model.WU[:, c] - model.WU[:, cp]
    v = model.gf * du
    return v - v.mean()


def kappa(model: Model, c: int, cp: int) -> float:
    du = model.WU[:, c] - model.WU[:, cp]
    return float(du @ model.bf + model.bU[c] - model.bU[cp])


def sigma(model: Model, h: np.ndarray) -> float:
    hc = h - h.mean()
    return float(np.sqrt((hc @ hc) / h.shape[0] + model.eps))


def decision_null_basis(model: Model, cands: list) -> np.ndarray:
    """Orthonormal basis of N_C = {v : 1^T v = 0, (gamma⊙(u_c-u_c'))^T v = 0 for c,c' in C}."""
    d = model.d
    rows = [np.ones(d)]
    c0 = cands[0]
    for c in cands[1:]:
        rows.append(model.gf * (model.WU[:, c] - model.WU[:, c0]))
    Arow = np.array(rows)
    # null space via SVD
    _, s, Vt = np.linalg.svd(Arow)
    rank = int((s > 1e-10 * s.max()).sum())
    return Vt[rank:].T  # (d, d-rank)


# ----------------------------------------------------------------------------
# Softmax consensus identity  (Theorem CI)
# ----------------------------------------------------------------------------


def consensus_update(w: np.ndarray, v: np.ndarray, j: int, w_new: float, v_new: float) -> tuple:
    """Weighted mean M = sum w v / sum w.  Replace (w_j, v_j) by (w_new, v_new).
    Returns (M, M_new, identity_value) where identity_value is
    [w_new (v_new - M) - w_j (v_j - M)] / Z_new, which Theorem CI says equals M_new - M."""
    Z = w.sum()
    M = float(w @ v / Z)
    w2 = w.copy(); v2 = v.copy()
    w2[j] = w_new; v2[j] = v_new
    Z2 = w2.sum()
    M2 = float(w2 @ v2 / Z2)
    ident = (w_new * (v_new - M) - w[j] * (v[j] - M)) / Z2
    return M, M2, float(ident)


def head_direct_quantities(model: Model, tr: Trace, layer: int, h: int, r: int, ut: np.ndarray):
    """Unnormalised weights w_j = exp(S_rj - max) and scalar values v_j = u~·(V_j W_O) at read r."""
    S = tr.scores[layer][h][r]
    finite = np.isfinite(S)
    w = np.where(finite, np.exp(np.where(finite, S, 0) - S[finite].max()), 0.0)
    v = tr.values[layer][h] @ ut
    return w, v


# ----------------------------------------------------------------------------
# Ordinal relation-blindness test (Theorem RB; Scott 1964 LP criterion)
# ----------------------------------------------------------------------------


def rb_margin(table: np.ndarray, tol: float = 1e-12) -> tuple:
    """Largest t such that there are f, g with
        m_ij > m_kl  =>  f_i + g_j - f_k - g_l >= t
        m_ij = m_kl  =>  f_i + g_j  = f_k + g_l
    with |f|,|g| <= 1.  The table is additively representable up to a strictly
    increasing transformation iff the optimum t is > 0.  Returns (t, f, g)."""
    nA, nB = table.shape
    cells = [(i, j) for i in range(nA) for j in range(nB)]
    nv = nA + nB + 1  # f, g, t
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
                r = -row.copy(); r[-1] = 1.0  # t - (f_i+g_j-f_k-g_l) <= 0
                A_ub.append(r); b_ub.append(0.0)
            elif abs(diff) <= tol and p < q:
                A_eq.append(row); b_eq.append(0.0)
    # fix the additive gauge: f_0 = 0, g_0 = 0
    for idx in (0, nA):
        row = np.zeros(nv); row[idx] = 1.0
        A_eq.append(row); b_eq.append(0.0)
    cobj = np.zeros(nv); cobj[-1] = -1.0
    bounds = [(-1, 1)] * (nA + nB) + [(None, 1)]
    res = linprog(cobj, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
                  bounds=bounds, method="highs")
    if not res.success:
        return -np.inf, None, None
    return float(res.x[-1]), res.x[:nA], res.x[nA:nA + nB]


def is_relation_blind(table: np.ndarray, thresh: float = 1e-9) -> bool:
    t, _, _ = rb_margin(table)
    return t > thresh


def has_crossover(table: np.ndarray, tol: float = 1e-12) -> bool:
    """Independence (single cancellation) violation on either factor."""
    nA, nB = table.shape
    for T_ in (table, table.T):
        n1, n2 = T_.shape
        for i in range(n1):
            for k in range(n1):
                if i == k:
                    continue
                d = T_[i] - T_[k]
                if (d > tol).any() and (d < -tol).any():
                    return True
    return False


# ----------------------------------------------------------------------------
# Mixed repair objective decomposition (Theorem MIX)
# ----------------------------------------------------------------------------


def mixed_step(A: np.ndarray, g: np.ndarray, J: np.ndarray, beta: float) -> tuple:
    """delta*(beta) = -A^{-1}(g - beta J).  Returns (J·delta, consistency term, output term)."""
    Ai = np.linalg.inv(A)
    delta = -Ai @ (g - beta * J)
    cons = float(-J @ Ai @ g)
    outp = float(beta * J @ Ai @ J)
    return float(J @ delta), cons, outp


# ----------------------------------------------------------------------------
# Explicit constructions
# ----------------------------------------------------------------------------

TOKENS = ["BOS", "N", "W", "S", "Q", "OPP"]  # neutral, weak-support, strong-support, query, oppose


def _token_embeddings(d: int, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    E = {t: rng.normal(0, 1.0, d) for t in TOKENS}
    return E


def dilution_model(lam: float = 3.0, mlp: str = "off", mlp_gain: float = 0.0, d: int = 8,
                   eps: float = 1e-5, kappa_bias: float = 0.0) -> tuple:
    """One pre-norm layer, one attention head, read at the last position.

    Keys encode a salience s_t, the query reads salience, values write an evidence
    scalar e_t along a fixed centred direction e_out, and the candidate pair (0,1)
    has centred functional u~ = 2 e_out.  Salience and evidence are set per token:

        token  BOS  N    W    S    Q   OPP
        s      0    0    1    1.5  0   1.5
        e      0    0    0.5  2.0  0   -2.0

    The weights acting on LN1 outputs are solved exactly on the six LN'd token
    vectors, so the specification is in terms of the computation, not fitted to
    any response.  `mlp` = "off" | "random" | "reverse".
    """
    E = _token_embeddings(d)
    s = {"BOS": 0.0, "N": 0.0, "W": 1.0, "S": 1.5, "Q": 0.0, "OPP": 1.5}
    e = {"BOS": 0.0, "N": 0.0, "W": 0.5, "S": 2.0, "Q": 0.0, "OPP": -2.0}
    g1 = np.ones(d); b1 = np.zeros(d)
    Z = np.array([layernorm(E[t], g1, b1, eps) for t in TOKENS])  # (6,d)
    dh = 2
    Kt = np.array([[s[t], 0.0] for t in TOKENS])
    Vt = np.array([[e[t], 0.0] for t in TOKENS])
    Qt = np.array([[0.0, 0.0] if t != "Q" else [lam * np.sqrt(dh), 0.0] for t in TOKENS])
    WK = np.linalg.lstsq(Z, Kt, rcond=None)[0]
    WV = np.linalg.lstsq(Z, Vt, rcond=None)[0]
    WQ = np.linalg.lstsq(Z, Qt, rcond=None)[0]
    for M_, T_ in ((WK, Kt), (WV, Vt), (WQ, Qt)):
        assert np.allclose(Z @ M_, T_, atol=1e-9), "token design not exactly realisable"
    e_out = np.zeros(d); e_out[0], e_out[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    WO = np.zeros((dh, d)); WO[0] = e_out
    dm = 4
    rng = np.random.default_rng(5)
    if mlp == "off":
        Win = np.zeros((d, dm)); Wout = np.zeros((dm, d)); on = False
    elif mlp == "random":
        Win = rng.normal(0, 0.3, (d, dm)); Wout = rng.normal(0, 0.3, (dm, d)) * mlp_gain; on = True
    elif mlp == "reverse":
        # one neuron reads the evidence direction after LN2 and writes -gain * e_out
        Win = np.zeros((d, dm)); Win[:, 0] = e_out * 4.0
        Wout = np.zeros((dm, d)); Wout[0] = -mlp_gain * e_out; on = True
    else:
        raise ValueError(mlp)
    layer = Layer(WQ=WQ[None], WK=WK[None], WV=WV[None], WO=WO[None], bO=np.zeros(d),
                  g1=g1, b1=b1, g2=np.ones(d), b2=np.zeros(d), Win=Win, bin=np.zeros(dm),
                  Wout=Wout, bout=np.zeros(d), mlp_on=on)
    V = 3
    WU = np.zeros((d, V))
    WU[:, 0] = e_out
    WU[:, 1] = -e_out
    WU[:, 2] = np.random.default_rng(3).normal(0, 0.2, d)
    bf = np.zeros(d)
    if kappa_bias:
        # put a final-LN bias whose projection on u_0-u_1 is kappa_bias
        bf = (kappa_bias / 2.0) * e_out
    model = Model(layers=[layer], gf=np.ones(d), bf=bf, WU=WU, bU=np.zeros(V), eps=eps)
    return model, E


def prompt(E: dict, a_tok: str, b_tok: str) -> np.ndarray:
    return np.array([E["BOS"], E[a_tok], E[b_tok], E["Q"]])


def factorial(model: Model, E: dict, A_levels, B_levels, c=0, cp=1) -> np.ndarray:
    T = np.zeros((len(A_levels), len(B_levels)))
    for i, a in enumerate(A_levels):
        for j, b in enumerate(B_levels):
            tr = forward(model, prompt(E, a, b), r=3)
            T[i, j] = margin(tr.logits, c, cp)
    return T


def rb_forces(nA: int, nB: int, strict_pairs: list, p: tuple, q: tuple) -> bool:
    """Prospective prediction from relation-blindness.

    strict_pairs: list of ((i,j),(k,l)) meaning m_ij > m_kl was observed on
    training cells.  Returns True when every additive representation of those
    observations has f_p + g_p > f_q + g_q, i.e. RB forces m_p > m_q on the
    held-out comparison.  Decided by an LP: maximise (f_q+g_q) - (f_p+g_p)
    subject to the training inequalities with unit separation (scale-free, since
    any strict representation can be rescaled)."""
    nv = nA + nB
    A_ub, b_ub = [], []
    for (i, j), (k, l) in strict_pairs:
        row = np.zeros(nv)
        row[i] -= 1; row[nA + j] -= 1; row[k] += 1; row[nA + l] += 1
        A_ub.append(row); b_ub.append(-1.0)  # f_i+g_j - f_k - g_l >= 1
    c = np.zeros(nv)
    c[p[0]] += 1; c[nA + p[1]] += 1; c[q[0]] -= 1; c[nA + q[1]] -= 1
    # minimise (f_p+g_p)-(f_q+g_q); forced iff minimum > 0 (bounded by the gauge box)
    A_eq = np.zeros((2, nv)); A_eq[0, 0] = 1; A_eq[1, nA] = 1
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.zeros(2),
                  bounds=[(-100, 100)] * nv, method="highs")
    if not res.success:
        raise ValueError("training observations are themselves not additively representable")
    return res.fun > 1e-9
